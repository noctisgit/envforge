"""Preference pairs for DPO (train/preference.py): pairing, prompt split, task split, overlap."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from envforge.core.registry import get_pack
from envforge.core.runner import run_many
from envforge.models.scripted import ScriptedModel
from envforge.train import dataset as ds
from envforge.train import preference as pref


def _runs(tmp_path: Path, *, seed_start: int = 10_000, episodes: int = 9) -> list[Path]:
    """A clean "teacher" run and a "student" run that botches every other episode."""
    common = dict(runs_dir=tmp_path / "runs", episodes=episodes, tiers=[1, 2, 3],
                  seed_start=seed_start)  # fmt: skip
    teacher = run_many(get_pack("netops"), ScriptedModel(), run_id="teacher", **common)
    student = run_many(get_pack("netops"), ScriptedModel(error_every=2), run_id="student", **common)
    return [teacher, student]


def _lines(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text().splitlines()] if path.exists() else []


def test_pairs_contrast_a_pass_with_a_failure_of_the_same_task(tmp_path):
    out = tmp_path / "pref"
    rep = pref.build_preference_dataset(_runs(tmp_path), out, val_frac=0.0)
    rows = _lines(out / "train.jsonl")
    assert rep.pairs == len(rows) > 0 and rep.n_val == 0
    assert rep.tasks_with_pairs == len({r["meta"]["task_id"] for r in rows})
    assert rep.tasks_without_rejected > 0  # the student solved some tasks too
    for r in rows:
        assert [m["role"] for m in r["prompt"]] == ["system", "user"]  # the shared opening
        assert r["chosen"][0]["role"] == "assistant" and r["rejected"][0]["role"] == "assistant"
        assert r["chosen"] != r["rejected"]
        assert r["meta"]["chosen"]["reward"] == 1.0 and r["meta"]["rejected"]["reward"] == 0.0
        assert r["meta"]["chosen"]["run_id"] == "teacher"
        assert r["meta"]["rejected"]["run_id"] == "student"
        assert r["tools"] and any(t["function"]["name"] == "finish" for t in r["tools"])
    m = ds.load_manifest(out)
    assert m["kind"] == "preference" and m["counts"]["pairs"] == rep.pairs


def test_split_is_by_task_and_overlap_catches_shared_tasks(tmp_path):
    runs = _runs(tmp_path, episodes=18)
    out = tmp_path / "pref"
    rep = pref.build_preference_dataset(runs, out, val_frac=0.5)
    train, val = _lines(out / "train.jsonl"), _lines(out / "val.jsonl")
    assert rep.n_train == len(train) and rep.n_val == len(val) and train and val
    assert not {r["meta"]["task_id"] for r in train} & {r["meta"]["task_id"] for r in val}
    # an "evaluation" run on the same seeds shares tasks; one on 0+ doesn't
    assert ds.overlap(out, runs[0])
    clean = run_many(get_pack("netops"), ScriptedModel(), runs_dir=tmp_path / "runs",
                     episodes=6, tiers=[1, 2, 3], seed_start=0, run_id="eval")  # fmt: skip
    assert ds.overlap(out, clean) == []


def test_max_pairs_per_task_and_repeated_attempts(tmp_path):
    common = dict(runs_dir=tmp_path / "runs", episodes=4, tiers=[1], seed_start=10_000, repeats=4)
    good = run_many(get_pack("netops"), ScriptedModel(), run_id="good", **common)
    bad = run_many(get_pack("netops"), ScriptedModel(error_every=1), run_id="bad", **common)
    out = tmp_path / "pref"
    rep = pref.build_preference_dataset([good, bad], out, max_pairs_per_task=3, val_frac=0.0)
    assert rep.tasks_with_pairs == 1 and rep.pairs == 3  # 4 passes x 4 failures, capped at 3
    episodes = {(r["meta"]["chosen"]["episode"], r["meta"]["rejected"]["episode"])
                for r in _lines(out / "train.jsonl")}  # fmt: skip
    assert len(episodes) == 3  # distinct episodes on both sides


def test_no_failures_means_no_pairs(tmp_path):
    clean = run_many(get_pack("netops"), ScriptedModel(), runs_dir=tmp_path / "runs",
                     episodes=3, tiers=[1], seed_start=10_000)  # fmt: skip
    with pytest.raises(ValueError, match="both a passing and a failing"):
        pref.build_preference_dataset([clean], tmp_path / "pref")


def test_shared_prefix():
    a = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}, {"x": 1}]
    assert pref.shared_prefix(a, [*a[:2], {"x": 2}]) == 2
    assert pref.shared_prefix(a, a) == 3 and pref.shared_prefix(a, []) == 0

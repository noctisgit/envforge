"""Preference pairs for DPO: the same task done well and done badly.

SFT shows a model what to do. DPO (train/dpo.py) also shows it what *not* to do: for
one task it sees a **chosen** episode (the verifier passed it) next to a **rejected**
one (the verifier failed it) and learns to make the first more likely than the second.

    build_preference_dataset([teacher_run, student_run], out_dir)

The best rejected episodes come from the model you are about to train: its own mistakes
are exactly what it should unlearn. So the usual recipe is to run the SFT adapter on
training seeds (10000+), then pair its failures with the teacher's (or its own) passes.

Each line of train.jsonl / val.jsonl is

    {"prompt":   [system, user],            # the messages both episodes share
     "chosen":   [assistant, tool, ...],    # the rest of the passing episode
     "rejected": [assistant, tool, ...],    # the rest of the failing episode
     "tools": [...], "meta": {...}}

in the same OpenAI chat format as SFT data, so dpo.py masks and tokenises it with
sft.py's code. The split is by task, like dataset.py, and the manifest has the same task
list, so `dataset.overlap()` catches evaluation runs that share tasks with the pairs.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from envforge.core.registry import get_pack
from envforge.core.trajectory import Trajectory, TrajectoryStore, load_run
from envforge.train.dataset import TaskKey, _is_val, manifest_tasks, task_key
from envforge.train.filter import to_chat_example, tool_schemas


@dataclass
class PreferenceReport:
    """What `build_preference_dataset` did, for the CLI and the web preview."""

    out_dir: Path
    episodes_seen: int = 0
    chosen_episodes: int = 0
    rejected_episodes: int = 0
    tasks_with_pairs: int = 0
    tasks_without_rejected: int = 0  # solved every time: nothing to contrast with
    tasks_without_chosen: int = 0  # never solved: no good example to prefer
    pairs: int = 0
    n_train: int = 0
    n_val: int = 0
    by_pack_tier: Counter = field(default_factory=Counter)


def shared_prefix(a: list[dict[str, Any]], b: list[dict[str, Any]]) -> int:
    """How many leading messages two conversations have in common (the DPO prompt)."""
    n = 0
    for x, y in zip(a, b, strict=False):
        if x != y:
            break
        n += 1
    return n


def make_pair(chosen: Trajectory, rejected: Trajectory) -> dict[str, Any] | None:
    """One preference record, or None if the episodes don't diverge after a shared prompt."""
    c = to_chat_example(chosen)["messages"]
    r = to_chat_example(rejected)["messages"]
    n = shared_prefix(c, r)
    if n == 0 or n >= len(c) or n >= len(r):
        return None  # different tasks, or one is a prefix of the other: nothing to compare
    if not any(m["role"] == "assistant" for m in c[n:]) or not any(
        m["role"] == "assistant" for m in r[n:]
    ):
        return None
    side = lambda t: {"run_id": t.run_id, "episode": t.episode, "model": t.model,  # noqa: E731
                      "reward": t.reward, "finished_reason": t.finished_reason}  # fmt: skip
    return {
        "prompt": c[:n],
        "chosen": c[n:],
        "rejected": r[n:],
        "meta": {
            "task_id": chosen.task.id,
            "task_record": chosen.task.hidden.get("task_record"),
            "tier": chosen.task.tier,
            "chosen": side(chosen),
            "rejected": side(rejected),
        },
    }


def build_preference_dataset(
    run_dirs: list[Path],
    out_dir: Path,
    *,
    chosen_min_reward: float = 1.0,
    rejected_max_reward: float = 0.0,
    max_pairs_per_task: int = 4,
    val_frac: float = 0.1,
    pack_resolver: Callable[[str], Any] | None = None,
) -> PreferenceReport:
    """Pair passing and failing episodes of the same task across `run_dirs`.

    An episode is *chosen* with reward >= `chosen_min_reward`, *rejected* with reward <=
    `rejected_max_reward` (episodes that crashed on an infrastructure error are neither:
    they say nothing about the model). A task with k chosen and m rejected episodes gives
    min(max_pairs_per_task, max(k, m)) pairs, cycling through both lists, best chosen
    (highest reward, then shortest) first.
    """
    report = PreferenceReport(out_dir=out_dir)
    chosen: dict[TaskKey, list[Trajectory]] = {}
    rejected: dict[TaskKey, list[Trajectory]] = {}
    sources = []
    for rd in run_dirs:
        trajs = load_run(rd)
        report.episodes_seen += len(trajs)
        sources.append({"run_dir": str(rd), "meta": TrajectoryStore(Path(rd)).read_meta()})
        for t in trajs:
            if t.error:
                continue
            if t.reward >= chosen_min_reward:
                chosen.setdefault(task_key(t), []).append(t)
            elif t.reward <= rejected_max_reward:
                rejected.setdefault(task_key(t), []).append(t)
    report.chosen_episodes = sum(map(len, chosen.values()))
    report.rejected_episodes = sum(map(len, rejected.values()))
    report.tasks_without_rejected = len(set(chosen) - set(rejected))
    report.tasks_without_chosen = len(set(rejected) - set(chosen))

    pairs: dict[TaskKey, list[dict[str, Any]]] = {}
    for k in sorted(set(chosen) & set(rejected)):
        good = sorted(chosen[k], key=lambda t: (-t.reward, len(t.steps), t.run_id, t.episode))
        bad = sorted(rejected[k], key=lambda t: (t.run_id, t.episode))
        n = min(max_pairs_per_task, max(len(good), len(bad)))
        made = [make_pair(good[i % len(good)], bad[i % len(bad)]) for i in range(n)]
        if made := [p for p in made if p is not None]:
            pairs[k] = made
    if not pairs:
        raise ValueError(
            "no task has both a passing and a failing episode: add a run of the model you "
            "want to improve on training seeds (its failures are the rejected examples)"
        )
    report.tasks_with_pairs = len(pairs)

    packs = sorted({k[0] for k in pairs})
    resolve = pack_resolver or get_pack
    tools = tool_schemas(resolve(packs[0])) if len(packs) == 1 else None
    train = [p for k in sorted(pairs) if not _is_val(k, val_frac) for p in pairs[k]]
    val = [p for k in sorted(pairs) if _is_val(k, val_frac) for p in pairs[k]]
    if not train:  # tiny datasets: keep at least one training task
        first = sorted(k for k in pairs if _is_val(k, val_frac))[0]
        train, val = pairs[first], [p for p in val if p not in pairs[first]]
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, split in (("train.jsonl", train), ("val.jsonl", val)):
        with (out_dir / name).open("w", encoding="utf-8") as f:
            for p in split:
                f.write(json.dumps({**p, "tools": tools}) + "\n")

    report.pairs, report.n_train, report.n_val = len(train) + len(val), len(train), len(val)
    for k, ps in pairs.items():
        report.by_pack_tier[(k[0], k[1])] += len(ps)
    manifest = {
        "format_version": 1,
        "kind": "preference",
        "created_at": datetime.now(UTC).isoformat(),
        "chosen_min_reward": chosen_min_reward,
        "rejected_max_reward": rejected_max_reward,
        "max_pairs_per_task": max_pairs_per_task,
        "val_frac": val_frac,
        "sources": sources,
        "counts": {
            "episodes_seen": report.episodes_seen,
            "pairs": report.pairs,
            "train": report.n_train,
            "val": report.n_val,
            "tasks": report.tasks_with_pairs,
        },  # fmt: skip
        "by_pack_tier": {f"{p}/tier{t}": n for (p, t), n in sorted(report.by_pack_tier.items())},
        **manifest_tasks(pairs),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return report

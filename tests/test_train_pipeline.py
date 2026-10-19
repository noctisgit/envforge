"""Phase 3 plumbing, offline: dataset building, contamination checks, remote training
orchestration (with a fake GPU box) and LoRA serving flags."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from envforge.compute.base import CommandResult, ComputeBox, ComputeError
from envforge.compute.vllm import build_docker_run as vllm_docker_run
from envforge.core.registry import get_pack
from envforge.core.runner import run_many
from envforge.models.scripted import ScriptedModel
from envforge.train import dataset as ds
from envforge.train import remote


def _run(tmp_path: Path, seed_start: int, episodes: int = 6, run_id: str | None = None) -> Path:
    return run_many(
        get_pack("netops"),
        ScriptedModel(),
        runs_dir=tmp_path / "runs",
        episodes=episodes,
        tiers=[1, 2, 3],
        seed_start=seed_start,
        run_id=run_id,
    )


# ---- dataset ----------------------------------------------------------------------


def test_build_dataset_dedupes_splits_by_task_and_records_seeds(tmp_path):
    a = _run(tmp_path, seed_start=10_000, run_id="a")
    b = _run(tmp_path, seed_start=10_000, run_id="b")  # the same 6 tasks again
    out = tmp_path / "sft" / "v1"
    rep = ds.build_dataset([a, b], out, val_frac=0.3)
    assert rep.total == 12 and rep.kept == 6 and rep.duplicates == 6
    assert rep.n_train + rep.n_val == 6 and rep.n_train >= 1

    train = [json.loads(x) for x in (out / "train.jsonl").read_text().splitlines()]
    val = [json.loads(x) for x in (out / "val.jsonl").read_text().splitlines()]
    tasks = lambda rows: {r["meta"]["task_id"] for r in rows}  # noqa: E731
    assert not tasks(train) & tasks(val)  # no task on both sides
    assert train[0]["messages"][0]["role"] == "system" and train[0]["tools"]

    m = ds.load_manifest(out)
    assert m["tasks"] == {"netops": [[10_000, 10_005]]}
    assert m["counts"]["duplicates_dropped"] == 6 and m["models"] == {"scripted": 6}


def test_overlap_flags_contaminated_eval_runs(tmp_path):
    train_run = _run(tmp_path, seed_start=10_000, run_id="train")
    out = tmp_path / "sft" / "v1"
    ds.build_dataset([train_run], out)
    clean = _run(tmp_path, seed_start=0, run_id="eval-clean")
    dirty = _run(tmp_path, seed_start=10_003, episodes=3, run_id="eval-dirty")
    assert ds.overlap(out, clean) == []
    hits = ds.overlap(out, dirty)
    assert hits and all(p == "netops" and s >= 10_003 for p, _, s in hits)


def test_dataset_requires_good_episodes(tmp_path):
    run = _run(tmp_path, seed_start=0, episodes=2, run_id="any")
    with pytest.raises(ValueError, match="no episodes"):
        ds.build_dataset([run], tmp_path / "sft" / "x", min_reward=1.5)  # nothing scores 1.5


# ---- remote training --------------------------------------------------------------


class FakeBox(ComputeBox):
    """Records commands; answers `docker inspect` / `test` / `docker logs` from a script."""

    host = "gpu-1"

    def __init__(self, *, exists: bool = False, logs: list[str] | None = None) -> None:
        self.calls: list[str] = []
        self.inputs: list[str | None] = []
        self.puts: list[tuple[str, str]] = []
        self.gets: list[tuple[str, str]] = []
        self.exists = exists
        self.log_polls = list(logs or [])
        self.polls = 0

    def run(self, command, *, input=None, timeout=None, check=False) -> CommandResult:
        cmd = command if isinstance(command, str) else " ".join(command)
        self.calls.append(cmd)
        self.inputs.append(input)
        if cmd.startswith("docker inspect"):
            if not self.exists:
                return CommandResult(1, "", "no such container")
            running = self.polls < len(self.log_polls) - 1
            return CommandResult(0, ("true|0" if running else "false|0") + "\n", "")
        if cmd.startswith("docker logs"):
            text = self.log_polls[min(self.polls, len(self.log_polls) - 1)]
            self.polls += 1
            return CommandResult(0, text, "")
        if cmd.startswith("test "):
            return CommandResult(0 if "adapter_config.json" in cmd else 1, "", "")
        if cmd.startswith("docker run") or "docker run" in cmd:
            self.exists = True
            return CommandResult(0, "abc123\n", "")
        return CommandResult(0, "", "")

    def put(self, local, remote_path, *, recursive=False):
        self.puts.append((str(local), remote_path))

    def get(self, remote_path, local, *, recursive=False):
        self.gets.append((remote_path, str(local)))


def _dataset(tmp_path: Path) -> Path:
    out = tmp_path / "sft" / "v1"
    ds.build_dataset(
        [_run(tmp_path, seed_start=10_000, episodes=12, run_id="t")], out, val_frac=0.3
    )
    return out


def test_start_training_uploads_and_runs_detached_container(tmp_path):
    box = FakeBox()
    name = remote.start_training(
        box, _dataset(tmp_path), base="Qwen/Qwen2.5-7B-Instruct", name="netops-v1",
        args=remote.TrainArgs(max_steps=5), hf_token="hf_secret",
    )  # fmt: skip
    assert name == "envforge-train-netops-v1"
    uploaded = {r for _, r in box.puts}
    assert {
        "/root/envforge/jobs/netops-v1/train.jsonl",
        "/root/envforge/jobs/netops-v1/sft.py",
    } <= uploaded
    run = next(c for c in box.calls if "docker run" in c)
    assert "--detach" in run and "--gpus all" in run and "/root/envforge/adapters:/adapters" in run
    assert "--max-steps 5" in run and "--base Qwen/Qwen2.5-7B-Instruct" in run
    assert "hf_secret" not in run  # the token travels on stdin only
    assert "HF_TOKEN=hf_secret\n" in box.inputs


def test_start_training_refuses_to_clobber(tmp_path):
    with pytest.raises(ComputeError, match="already exists"):
        remote.start_training(FakeBox(exists=True), _dataset(tmp_path), base="m", name="v1")


@pytest.mark.parametrize("bad", ["Has Space", "-rf", "x" * 80, "UPPER", "a/b"])
def test_adapter_names_are_validated(bad):
    with pytest.raises(ValueError):
        remote.check_name(bad)


def test_base_model_cannot_be_a_flag():
    with pytest.raises(ValueError):
        remote.build_docker_run("v1", "--trust-remote-code", remote.TrainArgs(), has_val=False)


def test_follow_parses_progress_until_exit():
    logs = [
        "pip noise\nENVFORGE {\"event\":\"log\",\"step\":1,\"loss\":2.0}\n",
        "pip noise\nENVFORGE {\"event\":\"log\",\"step\":1,\"loss\":2.0}\nENVFORGE {\"event\":\"log\",\"step\":2,\"loss\":1.5}\nnot json ENVFORGE\nENVFORGE {broken\n",
        "ENVFORGE {\"event\":\"log\",\"step\":1,\"loss\":2.0}\nENVFORGE {\"event\":\"log\",\"step\":2,\"loss\":1.5}\nENVFORGE {\"event\":\"done\",\"final_train_loss\":1.4}\n",
    ]  # fmt: skip
    box = FakeBox(exists=True, logs=logs)
    events = list(remote.follow(box, "v1", sleep=lambda s: None))
    assert [e["event"] for e in events] == ["log", "log", "done", "exit"]
    assert events[-1]["code"] == 0


def test_fetch_adapter_copies_to_workspace(tmp_path):
    box = FakeBox(exists=True)
    path = remote.fetch_adapter(box, "v1", tmp_path / "adapters")
    assert path == tmp_path / "adapters" / "v1"
    assert box.gets == [("/root/envforge/adapters/v1", str(tmp_path / "adapters"))]


# ---- serving adapters -------------------------------------------------------------


def test_vllm_serves_lora_adapters_by_name():
    argv = vllm_docker_run(
        "Qwen/Qwen2.5-7B-Instruct", lora_adapters=["netops-v1"], max_lora_rank=16
    )
    s = " ".join(argv)
    assert "/root/envforge/adapters:/adapters:ro" in s
    assert "--enable-lora" in argv and "netops-v1=/adapters/netops-v1" in argv
    assert argv[argv.index("--max-lora-rank") + 1] == "16"
    with pytest.raises(ValueError):
        vllm_docker_run("m", lora_adapters=["bad name"])
    assert "--enable-lora" not in vllm_docker_run("m")

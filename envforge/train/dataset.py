"""Build a training dataset from runs, and keep it honest.

    envforge train dataset data/workspaces/default/runs/<run> ... --name netops-sonnet-v1

writes `data/workspaces/<ws>/sft/<name>/`:

    train.jsonl      chat-format examples (train/filter.py) the SFT script reads
    val.jsonl        a held-out slice of *tasks* (never the same task in both files)
    manifest.json    where every example came from: runs, models, and the exact
                     (pack, tier, seed) tasks used — so evaluation can avoid them

Why the bookkeeping matters: every seed is a different generated task, so a model
evaluated on seeds it was trained on can score well by memorising answers. The
manifest records the training tasks and `overlap()` checks an eval run against it;
the rule of thumb is *train on seeds ≥ 10000, evaluate on seeds 0–99*.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from envforge.core.evaluate import filter_trajectories
from envforge.core.registry import get_pack
from envforge.core.trajectory import Trajectory, TrajectoryStore, load_run
from envforge.train.filter import dataset_lines, tool_schemas

TaskKey = tuple[str, int, int]  # (pack, tier, seed)


def task_key(t: Trajectory) -> TaskKey:
    return (t.task.pack, t.task.tier, t.task.seed)


def _better(a: Trajectory, b: Trajectory) -> Trajectory:
    """When the same task was solved more than once, keep the higher reward, then the
    shorter episode (fewer steps = a more efficient demonstration)."""
    return a if (a.reward, -len(a.steps)) >= (b.reward, -len(b.steps)) else b


def _is_val(key: TaskKey, val_frac: float) -> bool:
    """Stable train/val split by task: the same task always lands on the same side."""
    h = int(hashlib.sha256(repr(key).encode()).hexdigest()[:8], 16)
    return (h % 10_000) / 10_000 < val_frac


def _ranges(seeds: list[int]) -> list[list[int]]:
    """[1,2,3,7,8] -> [[1,3],[7,8]]: compact seed lists for the manifest."""
    out: list[list[int]] = []
    for s in sorted(set(seeds)):
        if out and s == out[-1][1] + 1:
            out[-1][1] = s
        else:
            out.append([s, s])
    return out


@dataclass
class DatasetReport:
    """What `build_dataset` did, for the CLI to print."""

    out_dir: Path
    total: int = 0
    kept: int = 0
    duplicates: int = 0
    n_train: int = 0
    n_val: int = 0
    by_pack_tier: Counter = field(default_factory=Counter)
    models: Counter = field(default_factory=Counter)


def build_dataset(
    run_dirs: list[Path],
    out_dir: Path,
    *,
    min_reward: float = 1.0,
    val_frac: float = 0.05,
) -> DatasetReport:
    """Filter runs to their best episodes, de-duplicate by task, split by task, write files."""
    report = DatasetReport(out_dir=out_dir)
    best: dict[TaskKey, Trajectory] = {}
    sources = []
    for rd in run_dirs:
        trajs = load_run(rd)
        report.total += len(trajs)
        sources.append({"run_dir": str(rd), "meta": TrajectoryStore(Path(rd)).read_meta()})
        for t in filter_trajectories(trajs, min_reward=min_reward):
            k = task_key(t)
            if k in best:
                report.duplicates += 1
                best[k] = _better(best[k], t)
            else:
                best[k] = t
    if not best:
        raise ValueError(f"no episodes with reward >= {min_reward} in {len(run_dirs)} run(s)")

    train = [t for k, t in sorted(best.items()) if not _is_val(k, val_frac)]
    val = [t for k, t in sorted(best.items()) if _is_val(k, val_frac)]
    if not train:  # tiny datasets: keep at least one training example
        train, val = val[:1], val[1:]

    packs = sorted({k[0] for k in best})
    tools = tool_schemas(get_pack(packs[0])) if len(packs) == 1 else None
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in (("train.jsonl", train), ("val.jsonl", val)):
        with (out_dir / name).open("w", encoding="utf-8") as f:
            f.writelines(dataset_lines(rows, fmt="chat", tools=tools))

    report.kept = len(best)
    report.n_train, report.n_val = len(train), len(val)
    for k, t in best.items():
        report.by_pack_tier[(k[0], k[1])] += 1
        report.models[t.model] += 1
    seeds: dict[str, list[int]] = {}
    for p, _tier, s in best:
        seeds.setdefault(p, []).append(s)
    manifest = {
        "format_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "min_reward": min_reward,
        "val_frac": val_frac,
        "sources": sources,
        "counts": {
            "episodes_seen": report.total,
            "kept": report.kept,
            "duplicates_dropped": report.duplicates,
            "train": report.n_train,
            "val": report.n_val,
        },  # fmt: skip
        "models": dict(report.models),
        "by_pack_tier": {f"{p}/tier{t}": n for (p, t), n in sorted(report.by_pack_tier.items())},
        # every task the model will have seen (train AND val), as compact seed ranges
        "tasks": {p: _ranges(s) for p, s in seeds.items()},
        "task_keys": sorted([list(k) for k in best]),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return report


def load_manifest(dataset_dir: Path) -> dict[str, Any]:
    return json.loads((Path(dataset_dir) / "manifest.json").read_text())


def overlap(dataset_dir: Path, run_dir: Path) -> list[TaskKey]:
    """Tasks in an evaluation run that the dataset already contains (contamination).

    An empty list means the eval is clean. Any overlap means those episodes measure
    memory, not skill — re-run the eval on other seeds (e.g. `--seed-start 0`).
    """
    seen = {tuple(k) for k in load_manifest(dataset_dir)["task_keys"]}
    return sorted({task_key(t) for t in load_run(run_dir)} & seen)  # type: ignore[arg-type]

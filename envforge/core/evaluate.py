"""Evaluation: turn a run directory of trajectories into numbers.

Metrics are deliberately simple and are the same for every pack:

- solve_rate        fraction of episodes with verdict.success
- mean_reward       mean verdict.reward (packs define the scale; netops uses 0..1)
- mean_steps        mean number of model turns per episode
- mean_tool_calls   mean tool calls per episode
- tool_error_rate   fraction of tool calls that raised ToolError
- collateral_rate   fraction of episodes where verdict.details["collateral"] is truthy
- max_steps_rate    fraction of episodes that ran out of budget
- tokens            summed input/output tokens
- wall_time_s       total wall-clock

Per-tier breakdowns use the same fields.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from envforge.core.trajectory import Trajectory, TrajectoryStore


class Metrics(BaseModel):
    n: int = 0
    solve_rate: float = 0.0
    mean_reward: float = 0.0
    mean_steps: float = 0.0
    mean_tool_calls: float = 0.0
    tool_error_rate: float = 0.0
    collateral_rate: float = 0.0
    max_steps_rate: float = 0.0
    error_rate: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    wall_time_s: float = 0.0


def compute_metrics(trajs: list[Trajectory]) -> Metrics:
    n = len(trajs)
    if n == 0:
        return Metrics()
    total_calls = sum(t.n_tool_calls for t in trajs)
    total_errs = sum(t.n_tool_errors for t in trajs)
    return Metrics(
        n=n,
        solve_rate=sum(1 for t in trajs if t.success) / n,
        mean_reward=sum(t.reward for t in trajs) / n,
        mean_steps=sum(len(t.steps) for t in trajs) / n,
        mean_tool_calls=total_calls / n,
        tool_error_rate=(total_errs / total_calls) if total_calls else 0.0,
        collateral_rate=sum(1 for t in trajs if t.verdict and t.verdict.details.get("collateral"))
        / n,
        max_steps_rate=sum(1 for t in trajs if t.finished_reason == "max_steps") / n,
        error_rate=sum(1 for t in trajs if t.finished_reason == "error") / n,
        input_tokens=sum(t.usage.input_tokens for t in trajs),
        output_tokens=sum(t.usage.output_tokens for t in trajs),
        wall_time_s=sum(t.wall_time_s for t in trajs),
    )


class RunReport(BaseModel):
    run_dir: str
    meta: dict[str, Any]
    overall: Metrics
    by_tier: dict[int, Metrics]


def evaluate_run(run_dir: str | Path) -> RunReport:
    store = TrajectoryStore(Path(run_dir))
    trajs = list(store.read())
    by_tier: dict[int, list[Trajectory]] = defaultdict(list)
    for t in trajs:
        by_tier[t.task.tier].append(t)
    return RunReport(
        run_dir=str(run_dir),
        meta=store.read_meta(),
        overall=compute_metrics(trajs),
        by_tier={tier: compute_metrics(ts) for tier, ts in sorted(by_tier.items())},
    )


def filter_trajectories(
    trajs: list[Trajectory], *, min_reward: float = 1.0, require_success: bool = True
) -> list[Trajectory]:
    """Rejection sampling: keep only episodes good enough to learn from.

    This is the whole trick behind the first training phase — generate many
    episodes with a strong model, keep the ones the verifier likes, and
    fine-tune a smaller model to imitate exactly those.
    """
    keep = []
    for t in trajs:
        if t.verdict is None:
            continue
        if require_success and not t.verdict.success:
            continue
        if t.verdict.reward < min_reward:
            continue
        if t.finished_reason == "error":
            continue
        keep.append(t)
    return keep

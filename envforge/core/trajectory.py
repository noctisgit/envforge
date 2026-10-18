"""Trajectories: the record of one episode, and how they are stored.

A trajectory is the unit of data for everything downstream — evaluation reads
them, the filter keeps the good ones, the trainer learns from them. They are
written as JSON Lines (one episode per line) so they can be streamed, grepped,
and appended to from many workers at once.

Schema stability matters: bump SCHEMA_VERSION on any breaking change and note
the migration in docs/ARCHITECTURE.md.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from envforge.core.env import Task, ToolCall, Verdict

SCHEMA_VERSION = 1


class Usage(BaseModel):
    """Token accounting for one model call, or summed over an episode."""

    input_tokens: int = 0
    output_tokens: int = 0

    def __add__(self, other: Usage) -> Usage:
        return Usage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
        )


class Step(BaseModel):
    """One turn of the agent loop.

    `assistant_text` is any prose the model emitted alongside tool calls.
    `tool_calls` and `observations` are parallel lists: observations[i] is the
    environment's reply to tool_calls[i]. `errors[i]` is True if that call
    raised ToolError.
    """

    index: int
    assistant_text: str = ""
    tool_calls: list[ToolCall] = Field(default_factory=list)
    observations: list[str] = Field(default_factory=list)
    errors: list[bool] = Field(default_factory=list)
    usage: Usage = Field(default_factory=Usage)
    latency_s: float = 0.0


class Trajectory(BaseModel):
    """A complete episode: task, what the agent did, and the verdict."""

    schema_version: int = SCHEMA_VERSION
    run_id: str
    episode: int
    model: str
    task: Task
    system_prompt: str
    steps: list[Step] = Field(default_factory=list)
    verdict: Verdict | None = None
    finished_reason: str = ""  # "finish_tool" | "no_tool_calls" | "max_steps" | "error"
    error: str | None = None
    usage: Usage = Field(default_factory=Usage)
    wall_time_s: float = 0.0
    meta: dict[str, Any] = Field(default_factory=dict)

    # ---- convenience ----
    @property
    def n_tool_calls(self) -> int:
        return sum(len(s.tool_calls) for s in self.steps)

    @property
    def n_tool_errors(self) -> int:
        return sum(sum(1 for e in s.errors if e) for s in self.steps)

    @property
    def success(self) -> bool:
        return bool(self.verdict and self.verdict.success)

    @property
    def reward(self) -> float:
        return self.verdict.reward if self.verdict else 0.0


# ---- storage ---------------------------------------------------------------

TRAJ_FILE = "trajectories.jsonl"
META_FILE = "meta.json"


class TrajectoryStore:
    """Append-only JSONL store for one run directory."""

    def __init__(self, run_dir: Path) -> None:
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.run_dir / TRAJ_FILE

    def append(self, traj: Trajectory) -> None:
        with self.path.open("a", encoding="utf-8") as f:
            f.write(traj.model_dump_json() + "\n")

    def read(self) -> Iterator[Trajectory]:
        if not self.path.exists():
            return iter(())
        return self._iter()

    def _iter(self) -> Iterator[Trajectory]:
        with self.path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield Trajectory.model_validate_json(line)

    def write_meta(self, meta: dict[str, Any]) -> None:
        (self.run_dir / META_FILE).write_text(json.dumps(meta, indent=2), encoding="utf-8")

    def read_meta(self) -> dict[str, Any]:
        p = self.run_dir / META_FILE
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def load_run(run_dir: str | Path) -> list[Trajectory]:
    """Read every trajectory in a run directory."""
    return list(TrajectoryStore(Path(run_dir)).read())

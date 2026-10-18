"""The environment contract.

Every domain pack implements `Environment`. The runner, the trajectory store,
the evaluator and the trainer only ever talk to this interface, which is what
makes the pipeline domain-agnostic.

Vocabulary (RL-for-LLMs, mapped onto plain engineering words):

- **Task**      a seeded problem instance (the "episode spec"). Think: a ticket.
- **Observation** what the agent sees after an action. Think: command output.
- **Tool**      a verb the agent may call. Think: a CLI command with a schema.
- **Step**      one tool call + its observation. An episode is a list of steps.
- **Verdict**   the verifier's judgment at the end: success flag + scalar reward.
- **Reward**    a number the training loop will try to make larger. Here it is
                computed by code, not by a human or a model, so it is *verifiable*.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, ClassVar, Literal

from pydantic import BaseModel, Field

EnvKind = Literal["sim", "sandbox"]


class ToolSpec(BaseModel):
    """A tool the agent may call. `parameters` is a JSON Schema object."""

    name: str
    description: str
    parameters: dict[str, Any] = Field(default_factory=lambda: {"type": "object", "properties": {}})


class ToolCall(BaseModel):
    """A concrete invocation of a tool, as emitted by a model."""

    id: str
    name: str
    args: dict[str, Any] = Field(default_factory=dict)


class ToolError(Exception):
    """Raised by a pack when a tool is called badly (unknown host, bad args, etc.).

    The runner catches it and returns the message to the model as the observation,
    so the agent can recover. It is *not* a crash.
    """


class Task(BaseModel):
    """One problem instance. `goal` is shown to the agent; `hidden` is not."""

    id: str
    pack: str
    tier: int
    seed: int
    goal: str
    hidden: dict[str, Any] = Field(default_factory=dict)


class Observation(BaseModel):
    """What the agent sees after `reset()` or `step()`."""

    text: str
    data: dict[str, Any] = Field(default_factory=dict)


class Verdict(BaseModel):
    """The verifier's judgment on a finished episode."""

    success: bool
    reward: float
    details: dict[str, Any] = Field(default_factory=dict)


class Environment(ABC):
    """Base class for all packs.

    Lifecycle, as driven by `envforge.core.runner`:

        task = Env.generate_task(seed, tier)
        env  = Env(task)
        obs  = env.reset()
        for each tool call from the model:
            obs = env.step(call)          # may raise ToolError
        verdict = env.verify()

    Class attributes a pack must set:
        name        short id used in the CLI, e.g. "netops"
        kind        "sim" (in-process Python) or "sandbox" (container)
        max_steps   step budget per episode
        tiers       difficulty tiers the task generator supports
    """

    name: ClassVar[str]
    kind: ClassVar[EnvKind] = "sim"
    max_steps: ClassVar[int] = 20
    tiers: ClassVar[tuple[int, ...]] = (1, 2, 3)
    description: ClassVar[str] = ""

    def __init__(self, task: Task) -> None:
        self.task = task

    # ---- task generation -------------------------------------------------
    @classmethod
    @abstractmethod
    def generate_task(cls, seed: int, tier: int) -> Task:
        """Deterministically build a task from (seed, tier)."""

    # ---- episode lifecycle ----------------------------------------------
    @abstractmethod
    def reset(self) -> Observation:
        """Initialise state for `self.task` and return the first observation."""

    @abstractmethod
    def tools(self) -> list[ToolSpec]:
        """Tools available to the agent in this environment."""

    @abstractmethod
    def step(self, call: ToolCall) -> Observation:
        """Apply one tool call and return what the agent sees. Raise ToolError on misuse."""

    @abstractmethod
    def verify(self) -> Verdict:
        """Judge the final state. Must be pure code — no model, no human."""

    @abstractmethod
    def system_prompt(self) -> str:
        """Role + rules for the agent. Keep tool semantics in tool descriptions, not here."""

    # ---- for offline testing and demos ----------------------------------
    @abstractmethod
    def reference_solve(self) -> list[ToolCall]:
        """A correct sequence of tool calls for `self.task`, used by the scripted model.

        It does not need to be optimal; it needs to pass `verify()`.
        """

    # ---- optional hooks --------------------------------------------------
    def close(self) -> None:
        """Release resources (containers, temp dirs). Sims usually need nothing."""
        return None

    def snapshot(self) -> dict[str, Any]:
        """Serialisable view of current state, for debugging and `envforge show`."""
        return {}

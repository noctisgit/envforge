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
    """A tool the agent may call. `parameters` is a JSON Schema object.

    `category` is optional documentation for humans (the Environments page groups
    tools by it): "observe" for read-only tools, "act" for tools that change state.
    Models never see it.
    """

    name: str
    description: str
    parameters: dict[str, Any] = Field(default_factory=lambda: {"type": "object", "properties": {}})
    category: Literal["observe", "act"] | None = None


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


# ---- self-description (for humans: the web UI's Environments page) -----------
#
# A pack can describe its world in a domain-neutral shape so one generic UI can show
# any environment: a diagram is just nodes and edges with a few standard states, and
# everything else is tables of strings. Nothing here is sent to the model.


class DiagramNode(BaseModel):
    """One thing in the world: a router, a host, a PBX, a file, a service..."""

    id: str
    label: str
    kind: str = "node"  # the domain's own word, shown in legends and tooltips
    group: str | None = None  # a zone or cluster label, e.g. a site name
    # standard states the UI knows how to colour:
    #   ok | broken (part of the fault) | target (named in the goal) | inactive
    state: str = "ok"
    badges: list[str] = Field(default_factory=list)  # e.g. ["policy", "changed"]
    detail: str = ""  # tooltip / side-panel text; multi-line is fine
    x: float | None = None  # optional fixed position in [0, 1]; else auto-layout
    y: float | None = None
    parent: str | None = None  # tree layouts: id of the parent node


class DiagramEdge(BaseModel):
    """A connection: a link, a trunk, a dependency, a call path."""

    source: str
    target: str
    label: str = ""
    # ok | broken | path (on the goal's route) | inactive
    state: str = "ok"
    directed: bool = False


class Diagram(BaseModel):
    """A picture of the world as data. `layout` is "graph" (nodes + edges) or "tree"
    (nodes with `parent`, e.g. a file system)."""

    title: str = ""
    layout: Literal["graph", "tree"] = "graph"
    nodes: list[DiagramNode] = Field(default_factory=list)
    edges: list[DiagramEdge] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)  # short captions under the picture


class InfoTable(BaseModel):
    """A titled table of strings: fault catalogues, policy lists, corpus contents..."""

    title: str
    columns: list[str]
    rows: list[list[str]]
    note: str = ""


class RewardRule(BaseModel):
    outcome: str
    reward: float


class EnvDescription(BaseModel):
    """Everything the Environments page shows about a pack (see Environment.describe)."""

    name: str
    kind: str
    description: str
    overview: str = ""  # a paragraph: what the agent is and what it must achieve
    world: list[str] = Field(default_factory=list)  # bullet facts about the world
    observation: str = ""  # what the agent sees
    action: str = ""  # what the agent can do
    max_steps: int
    tiers: list[int]
    tier_notes: dict[int, str] = Field(default_factory=dict)
    reward_rules: list[RewardRule] = Field(default_factory=list)
    tables: list[InfoTable] = Field(default_factory=list)
    tools: list[ToolSpec] = Field(default_factory=list)


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

    # ---- optional: self-description for humans ----------------------------
    @classmethod
    def describe(cls) -> EnvDescription:
        """What this environment is, for the Environments page.

        The default is built from the class attributes and the tool list, so a new
        pack shows up with no extra work. Override it to add an overview, world facts,
        tier notes, reward rules and tables — start from `super().describe()` and
        `model_copy(update={...})` the fields you fill in.
        """
        env = cls(cls.generate_task(0, cls.tiers[0]))
        try:
            tools = env.tools()  # no reset(): sandbox packs must not start a container here
        finally:
            env.close()
        return EnvDescription(
            name=cls.name,
            kind=cls.kind,
            description=cls.description,
            observation="The text output of each tool call.",
            action=f"Call any of {len(tools)} tools (several per turn), then `finish`.",
            max_steps=cls.max_steps,
            tiers=list(cls.tiers),
            tools=tools,
        )

    def diagram(self) -> Diagram | None:
        """The current world as a diagram, or None if the pack doesn't draw one.

        Sim packs: called after `reset()` and after each replayed step, so it should
        reflect live state (mark faults `broken`, the goal's endpoints `target`).
        Sandbox packs: must work *without* `reset()` (from the task alone), because
        previews and replays never start a container.
        """
        return None

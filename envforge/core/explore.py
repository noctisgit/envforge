"""Looking inside environments without a model: descriptions, task previews, replays.

Everything here works for any pack through the contract in env.py — there is no
domain code. It powers the web UI's Environments page and episode replays:

- `pack_overview(env_cls)`   describe() + a sample world diagram + derived "dimensions"
- `preview_task(env_cls, seed, tier)`
                             the task an agent would get, its hidden faults, and the
                             reference solution applied step by step
- `replay_episode(env_cls, traj)`
                             re-run a recorded episode's tool calls on a fresh copy of
                             its task and capture the diagram after every call

Replays rely on the determinism rule (same seed + same actions = same outcome), so
they are only offered for `kind="sim"` packs; sandbox packs would need a container.
"""

from __future__ import annotations

from typing import Any

from envforge.core.env import Diagram, Environment, ToolCall, ToolError
from envforge.core.runner import FINISH_TOOL, _observation_text
from envforge.core.trajectory import Trajectory

OBSERVATION_LIMIT = 4000  # the runner truncates each tool output to this many characters


class NotReplayable(Exception):
    """The pack can't be replayed in-process (e.g. a sandbox pack)."""


def _diagram(env: Environment) -> dict[str, Any] | None:
    d: Diagram | None = env.diagram()
    return d.model_dump() if d is not None else None


def _call(c: ToolCall) -> dict[str, Any]:
    return {"name": c.name, "args": c.args}


def dimensions(env_cls: type[Environment], diagram: dict[str, Any] | None) -> list[list[str]]:
    """The environment's "spaces" as a table — the LLM-agent version of obs/action dims.

    There are no tensors: observations and actions are text. What can be counted is
    the tool surface, the step budget, the reward range and the size of the world.
    """
    desc = env_cls.describe()
    tools = desc.tools
    observe = [t.name for t in tools if t.category == "observe"]
    act = [t.name for t in tools if t.category == "act"]
    n_params = sum(len(t.parameters.get("properties", {})) for t in tools)
    rewards = [r.reward for r in desc.reward_rules]
    rows = [
        ["observation", f"text, up to {OBSERVATION_LIMIT:,} characters per tool result"],
        [
            "action",
            f"tool calls with JSON arguments: {len(tools)} tools + finish, {n_params} parameters"
            + (f" ({len(observe)} observe, {len(act)} act)" if observe or act else ""),
        ],
        ["episode budget", f"at most {desc.max_steps} model turns"],
        ["difficulty tiers", ", ".join(str(t) for t in desc.tiers)],
        [
            "reward",
            f"{min(rewards):g} – {max(rewards):g}, once, at the end (verified by code)"
            if rewards
            else "scalar, once, at the end (verified by code)",
        ],
        [
            "environment kind",
            "in-process simulation" if desc.kind == "sim" else "container sandbox",
        ],
    ]
    if diagram:
        kinds: dict[str, int] = {}
        for n in diagram["nodes"]:
            kinds[n["kind"]] = kinds.get(n["kind"], 0) + 1
        world = ", ".join(
            f"{v} {k}{'s' if v != 1 and not k.endswith('s') else ''}"
            for k, v in sorted(kinds.items())
        )
        if diagram["edges"]:
            world += f", {len(diagram['edges'])} connections"
        rows.append(["world size (sample task)", world])
    return rows


def pack_overview(env_cls: type[Environment]) -> dict[str, Any]:
    """describe() plus a sample world diagram (seed 0, lowest tier) and dimensions."""
    desc = env_cls.describe()
    sample = preview_task(env_cls, seed=0, tier=env_cls.tiers[0], with_solution=False)
    diagram = sample["diagram_before"]
    return {
        **desc.model_dump(),
        "finish_tool": FINISH_TOOL.model_dump(),
        "world_diagram": diagram,
        "dimensions": dimensions(env_cls, diagram),
        "replayable": env_cls.kind == "sim",
    }


def preview_task(
    env_cls: type[Environment], seed: int, tier: int, *, with_solution: bool = True
) -> dict[str, Any]:
    """The task for (seed, tier), what the agent would see first, and — for sims — the
    reference solution applied call by call with the diagram after each change."""
    task = env_cls.generate_task(seed, tier)
    env = env_cls(task)
    sim = env_cls.kind == "sim"
    out: dict[str, Any] = {
        "task": task.model_dump(),
        "system_prompt": env.system_prompt(),
        "first_observation": None,
        "diagram_before": None,
        "solution": [],
        "diagram_after": None,
        "verdict": None,
    }
    try:
        if sim:
            out["first_observation"] = env.reset().text
        out["diagram_before"] = _diagram(env)
        if not with_solution:
            return out
        try:
            calls = env.reference_solve()
        except Exception:  # noqa: BLE001 - some sandbox packs need reset() for this
            calls = []
        for c in calls:
            step: dict[str, Any] = {**_call(c), "observation": None}
            if sim:
                try:
                    step["observation"] = _observation_text(env.step(c).text)
                except ToolError as e:
                    step["observation"] = f"ToolError: {e}"
            out["solution"].append(step)
        if sim:
            out["diagram_after"] = _diagram(env)
            out["verdict"] = env.verify().model_dump()
    finally:
        env.close()
    return out


def replay_episode(env_cls: type[Environment], traj: Trajectory) -> dict[str, Any]:
    """Frames for an episode: the diagram before any action and after every tool call.

    `changed` marks frames whose diagram differs from the previous one, so a UI can
    skip read-only calls. Observations come from the recording, not the replay.
    """
    if env_cls.kind != "sim":
        raise NotReplayable(f"{env_cls.name} is a {env_cls.kind} pack; replays need a sim")
    env = env_cls(traj.task)
    try:
        env.reset()
        known = {t.name for t in env.tools()}
        first = _diagram(env)
        if first is None:
            raise NotReplayable(f"{env_cls.name} does not draw a diagram")
        frames = [{"step": None, "call": None, "observation": None, "error": False,
                   "diagram": first, "changed": False}]  # fmt: skip
        for step in traj.steps:
            for call, obs, err in zip(
                step.tool_calls, step.observations, step.errors, strict=False
            ):
                if call.name == "finish" or call.name not in known:
                    continue
                try:
                    env.step(call)
                except ToolError:
                    pass  # recorded as an error already; state is unchanged
                d = _diagram(env)
                frames.append(
                    {
                        "step": step.index,
                        "call": _call(call),
                        "observation": obs,
                        "error": err,
                        "diagram": d,
                        "changed": d != frames[-1]["diagram"],
                    }
                )
        return {"frames": frames, "verdict": env.verify().model_dump()}
    finally:
        env.close()

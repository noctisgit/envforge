"""Rejection sampling → training data.

Phase 3 trains a small model to imitate the episodes the verifier liked. This module
turns kept trajectories into files a trainer can read:

- format="trajectory": the raw envforge Trajectory, one per line (lossless; re-filter later)
- format="chat": one OpenAI-style conversation per line — the exact messages the model saw
  and the assistant turns it produced, with tool definitions — which is what TRL's
  SFTTrainer (and most fine-tuning APIs) take.

    {"messages": [{"role": "system", ...}, {"role": "user", ...},
                  {"role": "assistant", "content": "...", "tool_calls": [...]},
                  {"role": "tool", "tool_call_id": "...", "content": "..."}, ...],
     "tools": [...], "meta": {"run_id": ..., "episode": ..., "task_id": ..., "task_record": ..., "reward": ...}}

The selection itself (`filter_trajectories`) lives in core/evaluate.py because the
evaluator and the CLI share it.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from typing import Any, Literal

from envforge.core.evaluate import filter_trajectories
from envforge.core.trajectory import Trajectory

Format = Literal["trajectory", "chat"]

__all__ = ["Format", "filter_trajectories", "to_chat_example", "dataset_lines"]


def _first_user_message(t: Trajectory) -> str:
    """Rebuild the opening user turn exactly as the runner sent it."""
    first_obs = t.meta.get("first_observation")
    if first_obs is None:
        # Trajectories written before the runner stored the first observation.
        return f"TASK\n{t.task.goal}"
    return f"TASK\n{t.task.goal}\n\nINITIAL STATE\n{first_obs}"


def to_chat_example(t: Trajectory, tools: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """One trajectory as an OpenAI-format conversation, ready for supervised fine-tuning."""
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": t.system_prompt},
        {"role": "user", "content": _first_user_message(t)},
    ]
    for step in t.steps:
        assistant: dict[str, Any] = {"role": "assistant", "content": step.assistant_text or ""}
        if step.tool_calls:
            assistant["tool_calls"] = [
                {
                    "id": c.id,
                    "type": "function",
                    "function": {"name": c.name, "arguments": json.dumps(c.args)},
                }
                for c in step.tool_calls
            ]
        messages.append(assistant)
        for call, obs in zip(step.tool_calls, step.observations, strict=False):
            messages.append({"role": "tool", "tool_call_id": call.id, "content": obs})
    example: dict[str, Any] = {
        "messages": messages,
        "meta": {
            "run_id": t.run_id,
            "episode": t.episode,
            "model": t.model,
            "task_id": t.task.id,
            # declarative packs: the task record (one chat, whatever seed picked it)
            "task_record": t.task.hidden.get("task_record"),
            "tier": t.task.tier,
            "reward": t.reward,
        },
    }
    if tools is not None:
        example["tools"] = tools
    return example


def dataset_lines(
    trajs: Iterable[Trajectory],
    *,
    fmt: Format = "trajectory",
    tools: list[dict[str, Any]] | None = None,
) -> Iterator[str]:
    """JSONL lines (with trailing newline) for already-filtered trajectories."""
    for t in trajs:
        if fmt == "chat":
            yield json.dumps(to_chat_example(t, tools)) + "\n"
        else:
            yield t.model_dump_json() + "\n"


def tool_schemas(env_cls: Any) -> list[dict[str, Any]]:
    """OpenAI-style tool definitions for a pack, including the core `finish` tool.

    Tools are an instance method, so we build an environment for a throwaway task.
    No `reset()` happens, so sandbox packs don't start a container here.
    """
    from envforge.core.runner import FINISH_TOOL

    env = env_cls(env_cls.generate_task(0, env_cls.tiers[0]))
    try:
        specs = [*env.tools(), FINISH_TOOL]
    finally:
        env.close()
    return [
        {
            "type": "function",
            "function": {"name": s.name, "description": s.description, "parameters": s.parameters},
        }
        for s in specs
    ]

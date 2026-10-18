"""Scripted model: replays a pack's `reference_solve()` with no network.

Why it exists:
- tests run offline and deterministically
- `envforge run <pack> --model scripted` proves a pack's verifier accepts its own
  reference solution (a pack that fails this is broken)
- it shows what a perfect trajectory looks like in the JSONL

Optionally it can be made imperfect (`error_every`) so evaluation code gets
exercised on failures too.
"""

from __future__ import annotations

import uuid

from envforge.core.env import Environment, ToolCall, ToolSpec
from envforge.models.base import AssistantTurn, ChatModel, Message


class ScriptedModel(ChatModel):
    def __init__(self, *, error_every: int = 0, calls_per_turn: int = 1) -> None:
        self.model_id = "scripted"
        self._plan: list[ToolCall] = []
        self._i = 0
        self._error_every = error_every
        self._calls_per_turn = max(1, calls_per_turn)
        self._episode = 0

    @property
    def is_offline(self) -> bool:
        return True

    def on_reset(self, env: Environment) -> None:
        self._plan = list(env.reference_solve())
        self._i = 0
        self._episode += 1
        # Deliberately sabotage some episodes so eval sees failures.
        if self._error_every and self._episode % self._error_every == 0:
            self._plan = self._plan[: max(0, len(self._plan) // 2)]

    def complete(self, messages: list[Message], tools: list[ToolSpec]) -> AssistantTurn:
        if self._i >= len(self._plan):
            return AssistantTurn(
                text="Done.",
                tool_calls=[
                    ToolCall(
                        id=f"call_{uuid.uuid4().hex[:8]}",
                        name="finish",
                        args={"summary": "Applied the reference fix."},
                    )
                ],
            )
        batch = self._plan[self._i : self._i + self._calls_per_turn]
        self._i += len(batch)
        calls = [
            ToolCall(id=f"call_{uuid.uuid4().hex[:8]}", name=c.name, args=c.args) for c in batch
        ]
        return AssistantTurn(text="", tool_calls=calls)

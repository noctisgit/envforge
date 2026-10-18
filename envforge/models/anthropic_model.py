"""Native Anthropic Messages API adapter (tool use)."""

from __future__ import annotations

import json
from typing import Any

from envforge.config import env
from envforge.core.env import ToolCall, ToolSpec
from envforge.core.trajectory import Usage
from envforge.models.base import AssistantTurn, ChatModel, Message


class AnthropicModel(ChatModel):
    def __init__(
        self,
        model: str,
        *,
        max_tokens: int = 2048,
        temperature: float = 0.0,
        api_key: str | None = None,
    ) -> None:
        from anthropic import Anthropic  # imported lazily so offline use needs no key

        api_key = api_key or env("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY is not set (see .env.example)")
        self._client = Anthropic(api_key=api_key)
        self._model = model
        self._max_tokens = max_tokens
        self._temperature = temperature
        self.model_id = f"anthropic:{model}"

    # ---- format conversion ---------------------------------------------
    @staticmethod
    def _to_anthropic(messages: list[Message]) -> tuple[str, list[dict[str, Any]]]:
        """Split out the system prompt and convert the rest to Anthropic blocks."""
        system = ""
        out: list[dict[str, Any]] = []
        for m in messages:
            if m.role == "system":
                system += m.content
            elif m.role == "user":
                out.append({"role": "user", "content": m.content})
            elif m.role == "assistant":
                blocks: list[dict[str, Any]] = []
                if m.content:
                    blocks.append({"type": "text", "text": m.content})
                for tc in m.tool_calls:
                    blocks.append(
                        {"type": "tool_use", "id": tc.id, "name": tc.name, "input": tc.args}
                    )
                out.append(
                    {"role": "assistant", "content": blocks or [{"type": "text", "text": ""}]}
                )
            elif m.role == "tool":
                # Anthropic wants tool results as a user turn; merge consecutive ones.
                block = {
                    "type": "tool_result",
                    "tool_use_id": m.tool_call_id,
                    "content": m.content,
                }
                if out and out[-1]["role"] == "user" and isinstance(out[-1]["content"], list):
                    out[-1]["content"].append(block)
                else:
                    out.append({"role": "user", "content": [block]})
        return system, out

    @staticmethod
    def _tools(tools: list[ToolSpec]) -> list[dict[str, Any]]:
        return [
            {"name": t.name, "description": t.description, "input_schema": t.parameters}
            for t in tools
        ]

    # ---- call ------------------------------------------------------------
    def complete(self, messages: list[Message], tools: list[ToolSpec]) -> AssistantTurn:
        system, msgs = self._to_anthropic(messages)
        resp = self._client.messages.create(
            model=self._model,
            max_tokens=self._max_tokens,
            temperature=self._temperature,
            system=system,
            messages=msgs,
            tools=self._tools(tools),
        )
        text_parts: list[str] = []
        calls: list[ToolCall] = []
        for block in resp.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "tool_use":
                args = block.input if isinstance(block.input, dict) else json.loads(block.input)
                calls.append(ToolCall(id=block.id, name=block.name, args=args))
        usage = Usage(
            input_tokens=getattr(resp.usage, "input_tokens", 0) or 0,
            output_tokens=getattr(resp.usage, "output_tokens", 0) or 0,
        )
        return AssistantTurn(text="".join(text_parts), tool_calls=calls, usage=usage)

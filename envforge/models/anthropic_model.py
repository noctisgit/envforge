"""Native Anthropic Messages API adapter (tool use)."""

from __future__ import annotations

import functools
import json
from typing import Any

from envforge.config import env
from envforge.core.env import ToolCall, ToolSpec
from envforge.core.trajectory import Usage
from envforge.models.base import AssistantTurn, ChatModel, Message, retry_unreachable


def _unreachable(e: Exception) -> bool:
    """True when the error means the API is out of reach (worth waiting for)."""
    from anthropic import APIConnectionError, APIStatusError

    return isinstance(e, APIConnectionError) or (
        isinstance(e, APIStatusError) and e.status_code in (502, 503, 504)
    )


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
        self._send_temperature = True  # cleared if the model rejects it
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
    def _create(self, system: str, msgs: list[dict[str, Any]], tools: list[ToolSpec]) -> Any:
        """Call the Messages API, sending `temperature` only where the model takes it.

        The 1.x SDK no longer has a `temperature` argument, but the API still honours it
        on older models (Sonnet/Haiku 4.5, the 4.6 line), so we send it as raw JSON via
        `extra_body` for reproducible evals. Newer models (Opus 4.7+, Sonnet 5.x) reject
        it with a 400; we learn that from the first refusal and stop sending it.

        Connection errors and gateway errors are retried for a while (the SDK's own
        retries give up within seconds), then raise `ModelUnreachable`.
        """
        from anthropic import BadRequestError

        while True:
            kwargs: dict[str, Any] = {
                "model": self._model,
                "max_tokens": self._max_tokens,
                "messages": msgs,
                "tools": self._tools(tools),
                # Prompt caching. Render order is tools -> system -> messages, so:
                #  1. an explicit breakpoint on the system block caches tools + system —
                #     the static prefix shared by every call of an episode / builder turn
                #     (and by every episode of the same environment);
                #  2. top-level automatic caching moves a breakpoint to the end of the
                #     conversation on each call, so the growing history is read back at
                #     ~0.1x instead of being paid in full every step.
                # Both use the default 5-minute TTL (calls in a loop are seconds apart).
                "cache_control": {"type": "ephemeral"},
            }
            if system:
                kwargs["system"] = [
                    {"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}
                ]
            if self._send_temperature:
                kwargs["extra_body"] = {"temperature": self._temperature}
            try:
                return retry_unreachable(
                    functools.partial(self._client.messages.create, **kwargs),
                    _unreachable,
                    model_id=self.model_id,
                )
            except BadRequestError as e:
                if self._send_temperature and "temperature" in str(e).lower():
                    self._send_temperature = False
                    continue
                raise

    def complete(self, messages: list[Message], tools: list[ToolSpec]) -> AssistantTurn:
        system, msgs = self._to_anthropic(messages)
        resp = self._create(system, msgs, tools)
        text_parts: list[str] = []
        calls: list[ToolCall] = []
        for block in resp.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "tool_use":
                args = block.input if isinstance(block.input, dict) else json.loads(block.input)
                calls.append(ToolCall(id=block.id, name=block.name, args=args))
        u = resp.usage
        # Anthropic's input_tokens is only the uncached remainder; the prompt's real size
        # is input + cache writes + cache reads. Normalise to "input = whole prompt".
        cache_read = getattr(u, "cache_read_input_tokens", 0) or 0
        cache_write = getattr(u, "cache_creation_input_tokens", 0) or 0
        usage = Usage(
            input_tokens=(getattr(u, "input_tokens", 0) or 0) + cache_read + cache_write,
            output_tokens=getattr(u, "output_tokens", 0) or 0,
            cache_read_tokens=cache_read,
            cache_write_tokens=cache_write,
        )
        return AssistantTurn(
            text="".join(text_parts),
            tool_calls=calls,
            usage=usage,
            raw={"stop_reason": getattr(resp, "stop_reason", None)},
        )

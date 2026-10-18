"""OpenAI-compatible Chat Completions adapter.

One class serves three providers, because they speak the same wire protocol:

- openai   https://api.openai.com/v1        OPENAI_API_KEY
- vercel   https://ai-gateway.vercel.sh/v1   AI_GATEWAY_API_KEY   model ids "provider/model"
- vllm     your GPU box, e.g. http://100.x.y.z:8000/v1   (VLLM_BASE_URL / VLLM_API_KEY)
"""

from __future__ import annotations

import json
from typing import Any

from envforge import config
from envforge.core.env import ToolCall, ToolSpec
from envforge.core.trajectory import Usage
from envforge.models.base import AssistantTurn, ChatModel, Message

PROVIDERS: dict[str, dict[str, str | None]] = {
    "openai": {"base_url": None, "key_env": "OPENAI_API_KEY"},
    "vercel": {"base_url": config.VERCEL_BASE_URL, "key_env": "AI_GATEWAY_API_KEY"},
    "vllm": {"base_url": config.VLLM_BASE_URL, "key_env": "VLLM_API_KEY"},
}


class OpenAICompatModel(ChatModel):
    def __init__(
        self,
        provider: str,
        model: str,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        max_tokens: int = 2048,
        temperature: float = 0.0,
    ) -> None:
        from openai import OpenAI  # lazy import

        if provider not in PROVIDERS:
            raise ValueError(f"Unknown OpenAI-compatible provider '{provider}'")
        spec = PROVIDERS[provider]
        key = api_key or config.env(spec["key_env"] or "")
        if provider == "vllm" and not key:
            key = "not-needed"  # vLLM ignores the key unless configured with --api-key
        if not key:
            raise RuntimeError(f"{spec['key_env']} is not set (see .env.example)")
        self._client = OpenAI(api_key=key, base_url=base_url or spec["base_url"])
        self._model = model
        self._max_tokens = max_tokens
        self._temperature = temperature
        self.model_id = f"{provider}:{model}"

    @staticmethod
    def _to_openai(messages: list[Message]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for m in messages:
            if m.role in ("system", "user"):
                out.append({"role": m.role, "content": m.content})
            elif m.role == "assistant":
                msg: dict[str, Any] = {"role": "assistant", "content": m.content or None}
                if m.tool_calls:
                    msg["tool_calls"] = [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {"name": tc.name, "arguments": json.dumps(tc.args)},
                        }
                        for tc in m.tool_calls
                    ]
                out.append(msg)
            elif m.role == "tool":
                out.append({"role": "tool", "tool_call_id": m.tool_call_id, "content": m.content})
        return out

    @staticmethod
    def _tools(tools: list[ToolSpec]) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.parameters,
                },
            }
            for t in tools
        ]

    def complete(self, messages: list[Message], tools: list[ToolSpec]) -> AssistantTurn:
        resp = self._client.chat.completions.create(
            model=self._model,
            messages=self._to_openai(messages),
            tools=self._tools(tools),
            max_tokens=self._max_tokens,
            temperature=self._temperature,
        )
        choice = resp.choices[0].message
        calls: list[ToolCall] = []
        for tc in choice.tool_calls or []:
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {"_raw": tc.function.arguments}
            calls.append(ToolCall(id=tc.id, name=tc.function.name, args=args))
        usage = Usage(
            input_tokens=getattr(resp.usage, "prompt_tokens", 0) or 0,
            output_tokens=getattr(resp.usage, "completion_tokens", 0) or 0,
        )
        return AssistantTurn(text=choice.content or "", tool_calls=calls, usage=usage)

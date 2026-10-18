"""Model loading. A model string is `<provider>:<model-id>`.

anthropic:claude-sonnet-4-5
openai:gpt-4o-mini
vercel:anthropic/claude-opus-5       (Vercel AI Gateway; note the provider/ prefix in the id)
vllm:Qwen/Qwen2.5-3B-Instruct        (your GPU box)
scripted                              (offline)
"""

from __future__ import annotations

from envforge.models.base import AssistantTurn, ChatModel, Message

PROVIDER_HELP = (
    "anthropic:<model> | openai:<model> | vercel:<provider/model> | vllm:<model> | scripted"
)


def load_model(spec: str, *, temperature: float = 0.0, max_tokens: int = 2048) -> ChatModel:
    spec = spec.strip()
    if spec in ("scripted", "scripted:"):
        from envforge.models.scripted import ScriptedModel

        return ScriptedModel()
    if spec.startswith("scripted:"):
        # scripted:error_every=3
        from envforge.models.scripted import ScriptedModel

        kv = dict(p.split("=", 1) for p in spec.split(":", 1)[1].split(",") if "=" in p)
        return ScriptedModel(
            error_every=int(kv.get("error_every", 0)),
            calls_per_turn=int(kv.get("calls_per_turn", 1)),
        )
    if ":" not in spec:
        raise ValueError(f"Model spec must be provider:model-id. Got '{spec}'. {PROVIDER_HELP}")
    provider, model = spec.split(":", 1)
    if provider == "anthropic":
        from envforge.models.anthropic_model import AnthropicModel

        return AnthropicModel(model, temperature=temperature, max_tokens=max_tokens)
    if provider in ("openai", "vercel", "vllm"):
        from envforge.models.openai_compat import OpenAICompatModel

        return OpenAICompatModel(provider, model, temperature=temperature, max_tokens=max_tokens)
    raise ValueError(f"Unknown provider '{provider}'. {PROVIDER_HELP}")


__all__ = ["AssistantTurn", "ChatModel", "Message", "load_model", "PROVIDER_HELP"]

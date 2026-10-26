"""Model adapters. Offline tests use a fake client; `-m network` tests hit real APIs.

    uv run pytest -m network tests/test_models_adapters.py     # needs keys in .env

Each network test is skipped when its key isn't set. They make one short call each.
"""

from __future__ import annotations

import json
import os
from types import SimpleNamespace

import httpx
import httpx2
import openai
import pytest

from envforge.core.env import ToolSpec
from envforge.models import load_model
from envforge.models.base import Message

ECHO = ToolSpec(
    name="echo",
    description="Echo a word back.",
    parameters={"type": "object", "properties": {"word": {"type": "string"}}, "required": ["word"]},
)
MESSAGES = [
    Message(role="system", content="You call tools. Never answer in prose."),
    Message(role="user", content="Call the echo tool with the word 'pineapple'."),
]


def _bad_request(msg: str) -> openai.BadRequestError:
    req = httpx.Request("POST", "https://example.invalid/v1/chat/completions")
    return openai.BadRequestError(msg, response=httpx.Response(400, request=req), body=None)


def _ok_response():
    call = SimpleNamespace(
        id="call_1", function=SimpleNamespace(name="echo", arguments='{"word": "pineapple"}')
    )
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="", tool_calls=[call]))],
        usage=SimpleNamespace(prompt_tokens=12, completion_tokens=3),
    )


def test_openai_compat_adapts_to_reasoning_model_params():
    model = load_model("openai:gpt-5", api_key="sk-test")
    seen: list[dict] = []

    def create(**kwargs):
        seen.append(kwargs)
        if "temperature" in kwargs:
            raise _bad_request("Unsupported parameter: 'temperature' is not supported")
        if "max_tokens" in kwargs:
            raise _bad_request("Unsupported parameter: 'max_tokens'; use max_completion_tokens")
        return _ok_response()

    model._client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )
    turn = model.complete(MESSAGES, [ECHO])
    assert turn.tool_calls[0].args == {"word": "pineapple"}
    assert turn.usage.input_tokens == 12 and turn.usage.output_tokens == 3
    assert len(seen) == 3 and "max_completion_tokens" in seen[-1]
    # remembered: the next call goes straight through
    model.complete(MESSAGES, [ECHO])
    assert len(seen) == 4


def test_openai_compat_other_errors_propagate():
    model = load_model("vercel:acme/model", api_key="vck-test")

    def create(**kwargs):
        raise _bad_request("model not found")

    model._client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )
    with pytest.raises(openai.BadRequestError):
        model.complete(MESSAGES, [ECHO])


def _connection_error() -> openai.APIConnectionError:
    req = httpx.Request("POST", "https://example.invalid/v1/chat/completions")
    return openai.APIConnectionError(request=req)


def test_openai_compat_waits_for_an_unreachable_server(monkeypatch):
    """A GPU box dropping off the network for a moment: the call retries, then succeeds."""
    monkeypatch.setattr("envforge.models.base.time.sleep", lambda s: None)
    model = load_model("vllm:qwen", api_key="k", base_url="http://127.0.0.1:9/v1")
    calls = []

    def create(**kwargs):
        calls.append(1)
        if len(calls) < 3:
            raise _connection_error()
        return _ok_response()

    model._client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )
    assert model.complete(MESSAGES, [ECHO]).tool_calls[0].name == "echo"
    assert len(calls) == 3


def test_openai_compat_gives_up_on_a_server_that_stays_down(monkeypatch):
    from envforge.models.base import ModelUnreachable

    monkeypatch.setattr("envforge.models.base.RECONNECT_WAIT_S", 0.0)
    model = load_model("vllm:qwen", api_key="k", base_url="http://127.0.0.1:9/v1")

    def create(**kwargs):
        raise _connection_error()

    model._client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )
    with pytest.raises(ModelUnreachable, match="vllm:qwen unreachable"):
        model.complete(MESSAGES, [ECHO])


def test_retry_unreachable_retries_only_unreachable_errors(monkeypatch):
    from envforge.models.base import retry_unreachable

    monkeypatch.setattr("envforge.models.base.time.sleep", lambda s: None)
    calls = []

    def boom():
        calls.append(1)
        raise ValueError("a real error")

    with pytest.raises(ValueError):
        retry_unreachable(boom, lambda e: isinstance(e, ConnectionError), model_id="m")
    assert len(calls) == 1  # not retried


def _anthropic_bad_request(msg: str):
    import anthropic

    req = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
    return anthropic.BadRequestError(msg, response=httpx2.Response(400, request=req), body=None)


def _anthropic_ok():
    block = SimpleNamespace(type="tool_use", id="tu_1", name="echo", input={"word": "pineapple"})
    return SimpleNamespace(content=[block], usage=SimpleNamespace(input_tokens=20, output_tokens=4))


def test_anthropic_sends_temperature_via_extra_body():
    model = load_model("anthropic:claude-haiku-4-5", api_key="sk-ant-test")
    seen: list[dict] = []

    def create(**kwargs):
        seen.append(kwargs)
        return _anthropic_ok()

    model._client = SimpleNamespace(messages=SimpleNamespace(create=create))
    turn = model.complete(MESSAGES, [ECHO])
    assert turn.tool_calls[0].args == {"word": "pineapple"}
    assert "temperature" not in seen[0]  # the 1.x SDK has no such argument
    assert seen[0]["extra_body"] == {"temperature": 0.0}


def test_anthropic_drops_temperature_when_model_rejects_it():
    model = load_model("anthropic:claude-opus-5-5", api_key="sk-ant-test")
    seen: list[dict] = []

    def create(**kwargs):
        seen.append(kwargs)
        if "extra_body" in kwargs:
            raise _anthropic_bad_request("temperature is not supported for this model")
        return _anthropic_ok()

    model._client = SimpleNamespace(messages=SimpleNamespace(create=create))
    model.complete(MESSAGES, [ECHO])
    model.complete(MESSAGES, [ECHO])
    assert len(seen) == 3 and "extra_body" not in seen[-1]  # learned once, remembered


# ---- real APIs (opt-in) ------------------------------------------------------------

LIVE = [
    ("ANTHROPIC_API_KEY", "anthropic:claude-haiku-4-5"),
    ("OPENAI_API_KEY", "openai:gpt-4o-mini"),
    ("AI_GATEWAY_API_KEY", "vercel:anthropic/claude-haiku-4.5"),
    ("AI_GATEWAY_API_KEY", "vercel:openai/gpt-5-mini"),
]


@pytest.mark.network
@pytest.mark.parametrize(("key_env", "spec"), LIVE)
def test_live_tool_call(key_env: str, spec: str):
    if not os.getenv(key_env):
        pytest.skip(f"{key_env} not set")
    model = load_model(os.getenv("ENVFORGE_LIVE_MODEL_" + key_env, spec), max_tokens=256)
    turn = model.complete(MESSAGES, [ECHO])
    assert turn.tool_calls, f"{spec} made no tool call: {turn.text!r}"
    assert turn.tool_calls[0].name == "echo"
    assert "pineapple" in str(turn.tool_calls[0].args).lower()
    assert turn.usage.input_tokens > 0 and turn.usage.output_tokens > 0


# ---- prompt caching -------------------------------------------------------------------


def test_anthropic_request_caches_static_prefix_and_conversation_tail():
    model = load_model("anthropic:claude-haiku-5-5", api_key="sk-ant-test")
    seen: list[dict] = []

    def create(**kwargs):
        seen.append(kwargs)
        u = SimpleNamespace(input_tokens=50, output_tokens=7, cache_read_input_tokens=900,
                            cache_creation_input_tokens=30)  # fmt: skip
        block = SimpleNamespace(
            type="tool_use", id="tu_1", name="echo", input={"word": "pineapple"}
        )
        return SimpleNamespace(content=[block], usage=u, stop_reason="tool_use")

    model._client = SimpleNamespace(messages=SimpleNamespace(create=create))
    turn = model.complete(MESSAGES, [ECHO])
    kw = seen[0]
    # static prefix (tools + system) ends in an explicit breakpoint …
    assert kw["system"] == [
        {"type": "text", "text": MESSAGES[0].content, "cache_control": {"type": "ephemeral"}}
    ]
    # … and automatic caching follows the growing conversation
    assert kw["cache_control"] == {"type": "ephemeral"}
    # the system prompt is not repeated in messages, so it never shifts the cached prefix
    assert all(m["role"] != "system" for m in kw["messages"])
    # usage normalised: input = whole prompt, cache parts are subsets
    assert turn.usage.input_tokens == 980
    assert turn.usage.cache_read_tokens == 900 and turn.usage.cache_write_tokens == 30
    assert turn.raw == {"stop_reason": "tool_use"}


def test_anthropic_requests_are_byte_identical_for_the_same_prefix():
    model = load_model("anthropic:claude-haiku-5-5", api_key="sk-ant-test")
    seen: list[str] = []

    def create(**kwargs):
        seen.append(json.dumps({k: kwargs[k] for k in ("system", "tools")}, sort_keys=False))
        return SimpleNamespace(content=[], usage=SimpleNamespace(input_tokens=1, output_tokens=1))

    model._client = SimpleNamespace(messages=SimpleNamespace(create=create))
    model.complete(MESSAGES, [ECHO])
    model.complete([*MESSAGES, Message(role="assistant", content="ok")], [ECHO])
    assert seen[0] == seen[1]  # no timestamps, no reordering: the prefix stays cacheable


def test_openai_compat_records_cached_prompt_tokens():
    model = load_model("openai:gpt-5-mini", api_key="sk-test")

    def create(**kwargs):
        r = _ok_response()
        r.usage = SimpleNamespace(prompt_tokens=2000, completion_tokens=10,
                                  prompt_tokens_details=SimpleNamespace(cached_tokens=1536))  # fmt: skip
        return r

    model._client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )
    turn = model.complete(MESSAGES, [ECHO])
    assert turn.usage.input_tokens == 2000 and turn.usage.cache_read_tokens == 1536


def test_usage_cost_prices_cache_reads_and_writes():
    from envforge.core.trajectory import Usage
    from envforge.models import catalog

    catalog.set_gateway_models([{
        "id": "anthropic/claude-x-1", "owned_by": "anthropic", "type": "language", "tags": ["tool-use"],
        "pricing": {"input": "0.000004", "output": "0.00002", "input_cache_read": "0.0000004",
                    "input_cache_write": "0.000005"},
    }])  # fmt: skip
    try:
        u = Usage(
            input_tokens=10_000,
            output_tokens=1_000,
            cache_read_tokens=8_000,
            cache_write_tokens=1_000,
        )
        cost = catalog.usage_cost("anthropic:claude-x-1", u)
        assert cost == pytest.approx(1_000 * 4e-6 + 8_000 * 4e-7 + 1_000 * 5e-6 + 1_000 * 2e-5)
        uncached = catalog.usage_cost(
            "anthropic:claude-x-1", Usage(input_tokens=10_000, output_tokens=1_000)
        )
        assert cost < uncached
    finally:
        catalog.set_gateway_models([])


def test_v1_trajectories_still_load():
    from envforge.core.trajectory import Trajectory

    v1 = {"schema_version": 1, "run_id": "r", "episode": 0, "model": "scripted",
          "task": {"id": "t", "pack": "netops", "tier": 1, "seed": 0, "goal": "g"},
          "system_prompt": "s", "usage": {"input_tokens": 5, "output_tokens": 2}}  # fmt: skip
    t = Trajectory.model_validate(v1)
    assert t.usage.cache_read_tokens == 0 and t.usage.input_tokens == 5

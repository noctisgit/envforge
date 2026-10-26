"""The model interface. Every provider adapts to this; the runner only sees this.

Messages use a small provider-neutral format so adapters stay thin:

    Message(role="system",    content="...")
    Message(role="user",      content="...")
    Message(role="assistant", content="...", tool_calls=[ToolCall(...)])
    Message(role="tool",      content="...", tool_call_id="...", name="...")
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, Field

from envforge.core.env import ToolCall, ToolSpec
from envforge.core.trajectory import Usage

if TYPE_CHECKING:  # avoid import cycle at runtime
    from envforge.core.env import Environment

Role = Literal["system", "user", "assistant", "tool"]

#: How long a model call keeps retrying a server it can't reach before giving up: long
#: enough to ride out a network blip or a GPU box's SSH tunnel being restarted.
RECONNECT_WAIT_S = 180.0


class ModelUnreachable(RuntimeError):
    """The model's server could not be reached (connection refused, reset or timed out, or
    a gateway error), even after waiting `RECONNECT_WAIT_S` for it to come back.

    This says nothing about the model, so the runner does not score the episode as a
    failure: it stops the run instead (see `run_episode`)."""


def retry_unreachable[T](
    call: Callable[[], T],
    is_unreachable: Callable[[Exception], bool],
    *,
    model_id: str,
    wait_s: float | None = None,
) -> T:
    """Run `call`, retrying with backoff (2 s, 4 s, ... up to 30 s apart) while
    `is_unreachable` says its error means the server is out of reach. Raises
    `ModelUnreachable` once `wait_s` (default `RECONNECT_WAIT_S`) has passed; any other
    error is raised at once."""
    wait_s = RECONNECT_WAIT_S if wait_s is None else wait_s
    deadline = time.monotonic() + wait_s
    delay = 2.0
    while True:
        try:
            return call()
        except Exception as e:
            if not is_unreachable(e):
                raise
            if time.monotonic() + delay > deadline:
                raise ModelUnreachable(
                    f"{model_id} unreachable for {wait_s:.0f}s ({type(e).__name__}: {e})"
                ) from e
            time.sleep(delay)
            delay = min(delay * 2, 30.0)


class Message(BaseModel):
    role: Role
    content: str = ""
    tool_calls: list[ToolCall] = Field(default_factory=list)  # assistant only
    tool_call_id: str | None = None  # tool only
    name: str | None = None  # tool only


class AssistantTurn(BaseModel):
    """What a model returns for one call: prose and/or tool calls, plus usage."""

    text: str = ""
    tool_calls: list[ToolCall] = Field(default_factory=list)
    usage: Usage = Field(default_factory=Usage)
    raw: dict | None = None


class ChatModel(ABC):
    """A chat model that can call tools."""

    #: human-readable id, e.g. "anthropic:claude-sonnet-4-5"
    model_id: str

    @abstractmethod
    def complete(self, messages: list[Message], tools: list[ToolSpec]) -> AssistantTurn:
        """One model call. Must return tool calls in `AssistantTurn.tool_calls`."""

    def on_reset(self, env: Environment) -> None:  # noqa: B027 - optional hook
        """Called once per episode before the first `complete`. Scripted models use it."""
        return None

    @property
    def is_offline(self) -> bool:
        """True if this model makes no network calls (scripted)."""
        return False

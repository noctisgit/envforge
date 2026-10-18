"""The model interface. Every provider adapts to this; the runner only sees this.

Messages use a small provider-neutral format so adapters stay thin:

    Message(role="system",    content="...")
    Message(role="user",      content="...")
    Message(role="assistant", content="...", tool_calls=[ToolCall(...)])
    Message(role="tool",      content="...", tool_call_id="...", name="...")
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, Field

from envforge.core.env import ToolCall, ToolSpec
from envforge.core.trajectory import Usage

if TYPE_CHECKING:  # avoid import cycle at runtime
    from envforge.core.env import Environment

Role = Literal["system", "user", "assistant", "tool"]


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

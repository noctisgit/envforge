"""The declarative environment spec: an environment written as data, not code.

The environment builder produces one of these from a user's intent and data; the
runtime (`envforge.declarative.runtime`) turns a spec plus a list of task records into
a normal `Environment` class, so runs, evaluation, the Environments page, replays and
training all work unchanged.

The first archetype is **support_desk**: an agent handles a customer's ticket using
tools over mock business records (customers, orders, subscriptions, devices, ...),
may ask the customer for facts, must take the right actions and record a resolution,
and is penalised for breaking policy. Everything is checked by code:

    collections   the record types in the mock systems (with a key and fields)
    tools         verbs the agent may call; each has a declarative *effect*
                  (get / find / set / add a record, ask the customer, resolve)
    policies      rules shown to the agent; each carries a *check* that must still
                  hold at the end — a failing policy check is collateral damage
    task records  one per real ticket: the customer's message, the starting
                  records, what the customer knows, the success checks, and a
                  reference solution (tool calls) that must pass those checks

Reward, as in every pack: 1.0 if all success checks pass and no policy check fails,
0.3 if the task is solved but a policy was broken, otherwise 0.0.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field

FORMAT_VERSION = 1

Scalar = str | int | float | bool | None


# ---- the mock systems ------------------------------------------------------------


class FieldDef(BaseModel):
    """One field of a record type."""

    name: str
    type: Literal["string", "number", "boolean", "enum", "datetime"] = "string"
    enum: list[str] | None = None  # allowed values when type == "enum"
    description: str = ""


class CollectionDef(BaseModel):
    """A record type, e.g. `orders` keyed by `order_id`, owned by a customer."""

    name: str
    key: str  # the field that identifies a record
    fields: list[FieldDef]
    description: str = ""
    # field holding the customer id the record belongs to; used to scope lookups and
    # to detect changes to *other* customers' records (collateral)
    owner_field: str | None = None


# ---- tools and their effects --------------------------------------------------------
#
# Values in `set` / `values` / `match` are either literals or "$param" references to
# the tool call's arguments, e.g. {"status": "$new_status", "updated_by": "agent"}.


class GetRecord(BaseModel):
    """Return one record by key (observe). Unknown key -> tool error."""

    op: Literal["get"] = "get"
    collection: str
    key_param: str


class FindRecords(BaseModel):
    """Return records whose fields equal the given values (observe, at most 20)."""

    op: Literal["find"] = "find"
    collection: str
    match: dict[str, str]  # field -> "$param" or literal


class SetFields(BaseModel):
    """Change fields on one record (act)."""

    op: Literal["set"] = "set"
    collection: str
    key_param: str
    set: dict[str, Any]  # field -> "$param" or literal


class AddRecord(BaseModel):
    """Create a record (act), e.g. a refund or an escalation. The key is generated."""

    op: Literal["add"] = "add"
    collection: str
    values: dict[str, Any]  # field -> "$param" or literal


class AskCustomer(BaseModel):
    """Ask the customer about a topic (observe). Answers come from the task's
    `customer_answers`; unknown topics get a polite "I'm not sure"."""

    op: Literal["ask_customer"] = "ask_customer"
    topic_param: str


class Resolve(BaseModel):
    """Record the agent's diagnosis/resolution category and a note (act)."""

    op: Literal["resolve"] = "resolve"
    category_param: str
    note_param: str | None = None


Effect = Annotated[
    GetRecord | FindRecords | SetFields | AddRecord | AskCustomer | Resolve,
    Field(discriminator="op"),
]


class ToolParam(BaseModel):
    name: str
    type: Literal["string", "number", "boolean", "enum"] = "string"
    enum: list[str] | None = None
    description: str = ""
    required: bool = True


class ToolDef(BaseModel):
    name: str
    description: str
    category: Literal["observe", "act"]
    params: list[ToolParam] = Field(default_factory=list)
    effect: Effect


# ---- checks (success conditions and policy rules) ----------------------------------
#
# `where` matches records field by field. A key may carry an operator suffix:
# "amount__gt": 100, "amount__lte": 50, "status__ne": "closed", "status__in": [...].


class FieldEquals(BaseModel):
    check: Literal["field_equals"] = "field_equals"
    collection: str
    key: str
    field: str
    value: Scalar


class RecordExists(BaseModel):
    check: Literal["record_exists"] = "record_exists"
    collection: str
    where: dict[str, Any]


class NoRecord(BaseModel):
    check: Literal["no_record"] = "no_record"
    collection: str
    where: dict[str, Any]


class ResolutionIs(BaseModel):
    check: Literal["resolution"] = "resolution"
    category: str


class ToolCalled(BaseModel):
    check: Literal["tool_called"] = "tool_called"
    tool: str
    args: dict[str, Any] = Field(default_factory=dict)  # these args must match (subset)


class ToolNotCalled(BaseModel):
    check: Literal["tool_not_called"] = "tool_not_called"
    tool: str
    args: dict[str, Any] = Field(default_factory=dict)


class OnlyTouchCustomer(BaseModel):
    """No record owned by a different customer was changed or created."""

    check: Literal["only_touch_customer"] = "only_touch_customer"


Check = Annotated[
    FieldEquals
    | RecordExists
    | NoRecord
    | ResolutionIs
    | ToolCalled
    | ToolNotCalled
    | OnlyTouchCustomer,
    Field(discriminator="check"),
]


class Policy(BaseModel):
    """A rule the agent is told about. `check` must hold at the end of the episode;
    if it fails, the episode has collateral damage (reward capped at 0.3)."""

    id: str
    text: str
    check: Check
    # optional condition: the policy only applies when this check holds at the end of
    # the episode, e.g. "verify identity before changing a plan" is
    # when=tool_called(change_plan), check=tool_called(ask_customer, topic=account_pin).
    # None (the default) means the policy always applies.
    when: Check | None = None


# ---- the environment ---------------------------------------------------------------


class EnvSpec(BaseModel):
    """Everything needed to build an environment except the task records."""

    format_version: int = FORMAT_VERSION
    archetype: Literal["support_desk"] = "support_desk"
    name: str  # slug, e.g. "isp-support"
    title: str
    description: str  # one line
    overview: str = ""  # a paragraph for the Environments page
    agent_role: str  # the core of the system prompt: who the agent is, how to work
    resolution_categories: list[str]  # allowed values for the resolve tool
    answer_topics: list[str] = Field(default_factory=list)  # what ask_customer may ask about
    collections: list[CollectionDef]
    tools: list[ToolDef]
    policies: list[Policy] = Field(default_factory=list)
    max_steps: int = 20
    tiers: list[int] = Field(default_factory=lambda: [1, 2, 3])
    tier_notes: dict[int, str] = Field(default_factory=dict)


class RefCall(BaseModel):
    """One step of a task's reference solution."""

    tool: str
    args: dict[str, Any] = Field(default_factory=dict)


class TaskRecord(BaseModel):
    """One ticket turned into a task (usually extracted from a real conversation)."""

    id: str
    tier: int = 1
    customer_id: str
    goal: str  # the customer's message as the agent sees it (already redacted)
    state: dict[str, list[dict[str, Any]]]  # collection -> starting records
    customer_answers: dict[str, str] = Field(default_factory=dict)  # topic -> answer
    success: list[Check]
    reference: list[RefCall]  # must make every success check pass without breaking policy
    source: dict[str, Any] = Field(default_factory=dict)  # e.g. {"file": ..., "conversation": ...}
    notes: str = ""

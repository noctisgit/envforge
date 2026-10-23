"""The rules engine: the live world of a declarative environment and how checks judge it.

Three small pieces, all pure and deterministic (no randomness, no clocks):

- `World`            the mutable state of one episode: records keyed by collection and
                     record key, the log of tool calls that took effect, the recorded
                     resolution, and which records the agent changed or created
- `matches(...)`     `where`-style matching with operator suffixes
                     ("amount__gt": 50, "status__in": ["open", "pending"],
                     "value__contains": "news", ...)
- `evaluate_check()` turns one `Check` from spec.py into a `CheckResult`
                     (passed? + a human sentence + which records it is about)

`resolve_value()` resolves "$param" references in effects against a tool call's
arguments; the runtime uses it when applying `set` / `add` / `find` effects.
"""

from __future__ import annotations

import copy
import re
from dataclasses import dataclass, field
from typing import Any

from envforge.declarative.spec import (
    Check,
    EnvSpec,
    FieldEquals,
    NoRecord,
    OnlyTouchCustomer,
    RecordExists,
    ResolutionIs,
    TaskRecord,
    ToolCalled,
    ToolNotCalled,
)

# operator suffixes allowed on `where` keys, e.g. "amount__lte"
OPERATORS = ("gt", "gte", "lt", "lte", "ne", "in", "contains")

RecordKey = tuple[str, str]  # (collection name, record key as a string)


# ---- matching ----------------------------------------------------------------------


def split_operator(key: str) -> tuple[str, str]:
    """Split "amount__gt" into ("amount", "gt"); a plain "amount" is ("amount", "eq").

    Only the known suffixes count, so a field that happens to contain "__" still works.
    """
    name, sep, op = key.rpartition("__")
    if sep and name and op in OPERATORS:
        return name, op
    return key, "eq"


def _is_number(v: Any) -> bool:
    return isinstance(v, int | float) and not isinstance(v, bool)


def values_equal(a: Any, b: Any) -> bool:
    """Equality that treats 15 and 15.0 as the same amount."""
    if _is_number(a) and _is_number(b):
        return float(a) == float(b)
    return a == b


def normalize_text(v: Any) -> str:
    """Lower case, with runs of punctuation, underscores and spaces collapsed to one space:
    "Credit_Card!" and "credit card" both become "credit card"."""
    return " ".join(re.sub(r"[\W_]+", " ", format_value(v).lower()).split())


def compare(actual: Any, op: str, expected: Any) -> bool:
    """Apply one operator. Ordering operators need two numbers or two strings
    (ISO dates compare correctly as strings); anything else simply doesn't match.

    `contains` is for free text the agent writes itself (a reason, a note): it matches when
    the expected words appear anywhere in the value, ignoring case and punctuation.
    """
    if op == "contains":
        return actual is not None and normalize_text(expected) in normalize_text(actual)
    if op == "eq":
        return values_equal(actual, expected)
    if op == "ne":
        return not values_equal(actual, expected)
    if op == "in":
        options = expected if isinstance(expected, list | tuple) else [expected]
        return any(values_equal(actual, e) for e in options)
    comparable = (_is_number(actual) and _is_number(expected)) or (
        isinstance(actual, str) and isinstance(expected, str)
    )
    if not comparable:
        return False
    if op == "gt":
        return actual > expected
    if op == "gte":
        return actual >= expected
    if op == "lt":
        return actual < expected
    if op == "lte":
        return actual <= expected
    raise ValueError(f"unknown operator '{op}'")


def matches(record: dict[str, Any], where: dict[str, Any]) -> bool:
    """True if every `where` entry holds for `record` (a missing field reads as None)."""
    for key, expected in where.items():
        name, op = split_operator(key)
        if not compare(record.get(name), op, expected):
            return False
    return True


def where_text(where: dict[str, Any]) -> str:
    """Render a `where` dict as "customer_id = C-1, amount > 50" for humans."""
    symbols = {"eq": "=", "ne": "≠", "gt": ">", "gte": "≥", "lt": "<", "lte": "≤", "in": "in",
               "contains": "contains"}  # fmt: skip
    parts = []
    for key, value in where.items():
        name, op = split_operator(key)
        parts.append(f"{name} {symbols[op]} {format_value(value)}")
    return ", ".join(parts) if parts else "any values"


def format_value(v: Any) -> str:
    """Show a value the way a support tool would: 15 not 15.0, (none) for null."""
    if v is None:
        return "(none)"
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, float) and v.is_integer():
        return str(int(v))
    if isinstance(v, list | tuple):
        return "[" + ", ".join(format_value(x) for x in v) + "]"
    return str(v)


# ---- $param references ---------------------------------------------------------------


def is_param_ref(value: Any) -> bool:
    """A string like "$amount" refers to the tool call's `amount` argument."""
    return isinstance(value, str) and value.startswith("$") and len(value) > 1


def resolve_value(value: Any, args: dict[str, Any]) -> Any:
    """Replace a "$param" reference with the argument's value (None if it was omitted);
    literals pass through unchanged."""
    if is_param_ref(value):
        return args.get(value[1:])
    return value


def resolve_values(mapping: dict[str, Any], args: dict[str, Any]) -> dict[str, Any]:
    """`resolve_value` over every entry of an effect's `set` / `values` / `match`."""
    return {k: resolve_value(v, args) for k, v in mapping.items()}


# ---- the world ----------------------------------------------------------------------


@dataclass
class World:
    """The live state of one episode.

    `records[collection][key]` is a plain dict per record (insertion-ordered, so output
    is deterministic). `calls` logs only tool calls that took effect — a call rejected
    with a ToolError changed nothing and does not count for tool_called checks.
    """

    customer_id: str
    records: dict[str, dict[str, dict[str, Any]]]
    calls: list[tuple[str, dict[str, Any]]] = field(default_factory=list)
    resolution: str | None = None
    resolution_note: str | None = None
    touched: dict[RecordKey, str] = field(default_factory=dict)  # -> "changed" | "new"
    original_owner: dict[RecordKey, Any] = field(default_factory=dict)
    changes: list[str] = field(default_factory=list)  # human-readable change log

    @classmethod
    def from_task(cls, spec: EnvSpec, task: TaskRecord) -> World:
        """A fresh world from a task's starting state (deep-copied, keyed by record key)."""
        keys = {c.name: c.key for c in spec.collections}
        records: dict[str, dict[str, dict[str, Any]]] = {c.name: {} for c in spec.collections}
        for coll, rows in task.state.items():
            key_field = keys.get(coll)
            if key_field is None:
                continue  # unknown collection: validate_task reports it
            for row in rows:
                records[coll][str(row.get(key_field))] = copy.deepcopy(row)
        return cls(customer_id=task.customer_id, records=records)

    def count(self, collection: str) -> int:
        return len(self.records.get(collection, {}))

    def all_records(self, collection: str) -> list[dict[str, Any]]:
        return list(self.records.get(collection, {}).values())

    def key_index(self) -> dict[str, list[RecordKey]]:
        """Every record key value -> the records with that key (used to spot references:
        a field or argument whose value is another record's key points at that record)."""
        index: dict[str, list[RecordKey]] = {}
        for coll, rows in self.records.items():
            for key in rows:
                index.setdefault(key, []).append((coll, key))
        return index

    def snapshot(self) -> dict[str, Any]:
        """A JSON-friendly copy for trajectories and debugging."""
        return {
            "customer_id": self.customer_id,
            "records": {c: list(rows.values()) for c, rows in self.records.items()},
            "resolution": self.resolution,
            "resolution_note": self.resolution_note,
            "calls": [{"tool": t, "args": a} for t, a in self.calls],
            "changes": list(self.changes),
        }


# ---- checks -------------------------------------------------------------------------


@dataclass
class CheckResult:
    """The outcome of one check, with enough context to explain it."""

    passed: bool
    description: str  # what the check requires, as a sentence
    detail: str = ""  # why it failed (empty when it passed)
    refs: set[RecordKey] = field(default_factory=set)  # records the check is about


def describe_check(check: Check) -> str:
    """One readable sentence for a check, for prompts, tables and diagram notes."""
    if isinstance(check, FieldEquals):
        return f"{check.collection} {check.key} has {check.field} = {format_value(check.value)}"
    if isinstance(check, RecordExists):
        return f"{check.collection} has a record with {where_text(check.where)}"
    if isinstance(check, NoRecord):
        return f"{check.collection} has no record with {where_text(check.where)}"
    if isinstance(check, ResolutionIs):
        return "ticket resolved as " + " or ".join(check.categories())
    if isinstance(check, ToolCalled):
        args = f" with {where_text(check.args)}" if check.args else ""
        return f"{check.tool} was called{args}"
    if isinstance(check, ToolNotCalled):
        args = f" with {where_text(check.args)}" if check.args else ""
        return f"{check.tool} was not called{args}"
    if isinstance(check, OnlyTouchCustomer):
        return "no other customer's records were changed or created"
    raise TypeError(f"unknown check {check!r}")


def _referenced(values: list[Any], index: dict[str, list[RecordKey]]) -> set[RecordKey]:
    """Records whose key equals one of these values (strings only)."""
    out: set[RecordKey] = set()
    for v in values:
        if isinstance(v, str):
            out.update(index.get(v, []))
        elif isinstance(v, list | tuple):
            out |= _referenced(list(v), index)
    return out


def _owner_offenders(world: World, spec: EnvSpec) -> list[RecordKey]:
    """Touched records owned (now or before the change) by someone else."""
    owners = {c.name: c.owner_field for c in spec.collections}
    bad: list[RecordKey] = []
    for coll, key in world.touched:
        owner_field = owners.get(coll)
        if owner_field is None:
            continue  # shared records (plans, outages ...) have no owner
        record = world.records.get(coll, {}).get(key, {})
        candidates = [record.get(owner_field), world.original_owner.get((coll, key))]
        if any(o is not None and str(o) != world.customer_id for o in candidates):
            bad.append((coll, key))
    return bad


def evaluate_check(check: Check, world: World, spec: EnvSpec) -> CheckResult:
    """Judge one check against the world. Pure: never changes `world`."""
    desc = describe_check(check)
    index = world.key_index()

    if isinstance(check, FieldEquals):
        record = world.records.get(check.collection, {}).get(str(check.key))
        ref = {(check.collection, str(check.key))}
        if record is None:
            return CheckResult(False, desc, f"{check.collection} {check.key} does not exist", ref)
        actual = record.get(check.field)
        ok = values_equal(actual, check.value)
        return CheckResult(ok, desc, "" if ok else f"{check.field} is {format_value(actual)}", ref)

    if isinstance(check, RecordExists | NoRecord):
        rows = world.records.get(check.collection, {})
        hits = [(check.collection, k) for k, r in rows.items() if matches(r, check.where)]
        refs = _referenced(list(check.where.values()), index)
        if isinstance(check, RecordExists):
            ok = bool(hits)
            return CheckResult(ok, desc, "" if ok else "no matching record", refs)
        ok = not hits
        detail = "" if ok else "found " + ", ".join(k for _, k in hits)
        return CheckResult(ok, desc, detail, refs | set(hits))

    if isinstance(check, ResolutionIs):
        ok = world.resolution in check.categories()
        detail = "" if ok else f"resolution is {world.resolution or '(not resolved)'}"
        return CheckResult(ok, desc, detail)

    if isinstance(check, ToolCalled | ToolNotCalled):
        hits = [a for t, a in world.calls if t == check.tool and matches(a, check.args)]
        refs = _referenced(list(check.args.values()), index)
        if isinstance(check, ToolCalled):
            ok = bool(hits)
            return CheckResult(ok, desc, "" if ok else "never called", refs)
        ok = not hits
        for a in hits:  # point at the records the offending calls were about
            refs |= _referenced(list(a.values()), index)
        return CheckResult(ok, desc, "" if ok else f"called {len(hits)} time(s)", refs)

    if isinstance(check, OnlyTouchCustomer):
        bad = _owner_offenders(world, spec)
        detail = "" if not bad else "touched " + ", ".join(f"{c} {k}" for c, k in bad)
        return CheckResult(not bad, desc, detail, set(bad))

    raise TypeError(f"unknown check {check!r}")

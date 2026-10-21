"""Checking specs and task records before anyone trains on them.

A declarative environment is only useful if its tasks are *solvable and verified*:
every task must start broken, and its reference solution must reach reward 1.0 in the
real runtime. These functions return lists of human-readable problems (empty = fine)
so the builder, the CLI and the web layer can show them as-is.

    validate_spec(spec)            problems with the spec alone
    validate_task(spec, task)      problems with one task (runs its reference solution)
    validate_dataset(spec, tasks)  a summary over everything
"""

from __future__ import annotations

from collections import Counter
from typing import Any

from envforge.core.env import ToolError
from envforge.declarative.rules import (
    OPERATORS,
    World,
    evaluate_check,
    is_param_ref,
    split_operator,
)
from envforge.declarative.runtime import (
    apply_effect,
    clean_args,
    collection_def,
    field_def,
    field_names,
    judge,
    resolve_tool,
)
from envforge.declarative.spec import (
    AddRecord,
    AskCustomer,
    Check,
    EnvSpec,
    FieldEquals,
    FindRecords,
    GetRecord,
    NoRecord,
    OnlyTouchCustomer,
    RecordExists,
    ResolutionIs,
    Resolve,
    SetFields,
    TaskRecord,
    ToolCalled,
    ToolNotCalled,
)

RESERVED_TOOLS = {"finish"}  # injected by the runner for every environment


def _dupes(names: list[str]) -> list[str]:
    return sorted(n for n, k in Counter(names).items() if k > 1)


def _value_fits(ftype: str, enum: list[str] | None, value: Any) -> bool:
    """Does a literal fit a field type? (None always fits: fields are nullable.)"""
    if value is None:
        return True
    if ftype == "number":
        return isinstance(value, int | float) and not isinstance(value, bool)
    if ftype == "boolean":
        return isinstance(value, bool)
    if ftype == "enum":
        return isinstance(value, str) and (not enum or value in enum)
    return isinstance(value, str)


def _where_problems(spec: EnvSpec, coll_name: str, where: dict[str, Any], label: str) -> list[str]:
    coll = collection_def(spec, coll_name)
    if coll is None:
        return [f"{label}: unknown collection '{coll_name}'"]
    problems = []
    for key, value in where.items():
        name, op = split_operator(key)
        if name not in field_names(coll):
            base, sep, suffix = key.rpartition("__")
            if sep and base in field_names(coll):
                problems.append(f"{label}: unknown operator '__{suffix}' (use {OPERATORS})")
            else:
                problems.append(f"{label}: {coll_name} has no field '{name}'")
            continue
        if op == "in" and not isinstance(value, list):
            problems.append(f"{label}: '{key}' needs a list of values")
    return problems


def check_problems(spec: EnvSpec, check: Check, label: str) -> list[str]:
    """Does a check refer only to things the spec defines?"""
    tools = {t.name: t for t in spec.tools}
    if isinstance(check, FieldEquals):
        coll = collection_def(spec, check.collection)
        if coll is None:
            return [f"{label}: unknown collection '{check.collection}'"]
        if field_def(coll, check.field) is None:
            return [f"{label}: {check.collection} has no field '{check.field}'"]
        return []
    if isinstance(check, RecordExists | NoRecord):
        return _where_problems(spec, check.collection, check.where, label)
    if isinstance(check, ResolutionIs):
        if check.category not in spec.resolution_categories:
            return [f"{label}: '{check.category}' is not a resolution category"]
        return []
    if isinstance(check, ToolCalled | ToolNotCalled):
        tool = tools.get(check.tool)
        if tool is None:
            return [f"{label}: unknown tool '{check.tool}'"]
        params = {p.name for p in tool.params}
        return [
            f"{label}: {check.tool} has no parameter '{split_operator(k)[0]}'"
            for k in check.args
            if split_operator(k)[0] not in params
        ]
    if isinstance(check, OnlyTouchCustomer):
        if not any(c.owner_field for c in spec.collections):
            return [f"{label}: only_touch_customer needs a collection with an owner_field"]
        return []
    return [f"{label}: unknown check type"]


def validate_spec(spec: EnvSpec) -> list[str]:
    """Problems with the spec itself (names, effects, params, policies)."""
    problems: list[str] = []
    for d in _dupes([c.name for c in spec.collections]):
        problems.append(f"duplicate collection '{d}'")
    for d in _dupes([t.name for t in spec.tools]):
        problems.append(f"duplicate tool '{d}'")
    for d in _dupes([p.id for p in spec.policies]):
        problems.append(f"duplicate policy id '{d}'")
    for d in _dupes(spec.resolution_categories):
        problems.append(f"duplicate resolution category '{d}'")
    if not spec.resolution_categories:
        problems.append("resolution_categories is empty")
    if not spec.tiers:
        problems.append("tiers is empty")
    if spec.max_steps < 1:
        problems.append("max_steps must be at least 1")

    for c in spec.collections:
        names = [f.name for f in c.fields]
        for d in _dupes(names):
            problems.append(f"collection {c.name}: duplicate field '{d}'")
        if c.key not in names:
            problems.append(f"collection {c.name}: key '{c.key}' is not a field")
        if c.owner_field and c.owner_field not in names:
            problems.append(f"collection {c.name}: owner_field '{c.owner_field}' is not a field")
        for f in c.fields:
            if f.type == "enum" and not f.enum:
                problems.append(f"collection {c.name}: enum field '{f.name}' has no enum values")

    for t in spec.tools:
        label = f"tool {t.name}"
        if t.name in RESERVED_TOOLS:
            problems.append(f"{label}: the name is reserved by the runner")
        params = {p.name: p for p in t.params}
        for d in _dupes([p.name for p in t.params]):
            problems.append(f"{label}: duplicate parameter '{d}'")
        for p in t.params:
            if p.type == "enum" and not p.enum:
                problems.append(f"{label}: enum parameter '{p.name}' has no enum values")
        problems += _effect_problems(spec, t.effect, params, label)
        expected = (
            "observe" if isinstance(t.effect, GetRecord | FindRecords | AskCustomer) else "act"
        )
        if t.category != expected:
            problems.append(f"{label}: category should be '{expected}' for a {t.effect.op} effect")

    if resolve_tool(spec) is None:
        problems.append("no tool has a resolve effect (the agent could never record an outcome)")

    for p in spec.policies:
        problems += check_problems(spec, p.check, f"policy {p.id}")
        if p.when is not None:
            problems += check_problems(spec, p.when, f"policy {p.id} (when)")
    return problems


def _effect_problems(spec: EnvSpec, eff: Any, params: dict, label: str) -> list[str]:
    problems: list[str] = []

    def need_param(name: str | None, what: str) -> None:
        if name is not None and name not in params:
            problems.append(f"{label}: {what} '{name}' is not a parameter")

    if isinstance(eff, AskCustomer):
        need_param(eff.topic_param, "topic_param")
        return problems
    if isinstance(eff, Resolve):
        need_param(eff.category_param, "category_param")
        need_param(eff.note_param, "note_param")
        return problems

    coll = collection_def(spec, eff.collection)
    if coll is None:
        return [f"{label}: unknown collection '{eff.collection}'"]
    if isinstance(eff, GetRecord | SetFields):
        need_param(eff.key_param, "key_param")
    mapping: dict[str, Any] = {}
    if isinstance(eff, FindRecords):
        mapping = eff.match
    elif isinstance(eff, SetFields):
        mapping = eff.set
        if coll.key in mapping:
            problems.append(f"{label}: must not change the key field '{coll.key}'")
    elif isinstance(eff, AddRecord):
        mapping = eff.values
        if coll.key in mapping:
            problems.append(f"{label}: the key '{coll.key}' is generated; don't set it")
    for field_name, value in mapping.items():
        fdef = field_def(coll, field_name)
        if fdef is None:
            problems.append(f"{label}: {coll.name} has no field '{field_name}'")
            continue
        if name := _template_param(value, params):
            # "{customer_id}" is stored as that literal text; only "$customer_id" is filled in
            problems.append(
                f"{label}: {value!r} is literal text, not a reference; "
                f'write "${name}" to fill {coll.name}.{field_name} from the argument'
            )
        elif is_param_ref(value):
            need_param(value[1:], "$-reference")
            p = params.get(value[1:])
            if p is not None and fdef.type == "number" and p.type != "number":
                problems.append(f"{label}: '{value}' feeds number field '{field_name}'")
        elif not _value_fits(fdef.type, fdef.enum, value):
            problems.append(f"{label}: {value!r} does not fit {coll.name}.{field_name}")
    return problems


def _template_param(value: Any, params: dict) -> str | None:
    """The parameter a literal was meant to reference ("{name}", "{{name}}", "<name>" or
    "${name}" where `name` is one of the tool's parameters), else None."""
    if not isinstance(value, str):
        return None
    v = value.strip().removeprefix("$")
    wrapped = (v.startswith("{") and v.endswith("}")) or (v.startswith("<") and v.endswith(">"))
    name = v.strip("{}<> ")
    return name if wrapped and name in params else None


def _customer_collection(spec: EnvSpec) -> str | None:
    """The collection whose records *are* customers: keyed by the owner field others use."""
    owner_fields = {c.owner_field for c in spec.collections if c.owner_field}
    for c in spec.collections:
        if c.key in owner_fields:
            return c.name
    return None


def validate_task(spec: EnvSpec, task: TaskRecord) -> list[str]:
    """Problems with one task record. The decisive test: the task must start unsolved,
    and replaying its reference solution in the runtime must earn reward 1.0."""
    problems: list[str] = []
    if task.tier not in spec.tiers:
        problems.append(f"tier {task.tier} is not one of the spec's tiers {spec.tiers}")

    for coll_name, rows in task.state.items():
        coll = collection_def(spec, coll_name)
        if coll is None:
            problems.append(f"state: unknown collection '{coll_name}'")
            continue
        keys = []
        for i, row in enumerate(rows):
            where = f"state.{coll_name}[{i}]"
            if row.get(coll.key) is None:
                problems.append(f"{where}: missing key '{coll.key}'")
            else:
                keys.append(str(row[coll.key]))
            for name, value in row.items():
                fdef = field_def(coll, name)
                if fdef is None:
                    problems.append(f"{where}: unknown field '{name}'")
                elif not _value_fits(fdef.type, fdef.enum, value):
                    problems.append(f"{where}: {value!r} does not fit field '{name}' ({fdef.type})")
        for d in _dupes(keys):
            problems.append(f"state.{coll_name}: duplicate key '{d}'")

    customers = _customer_collection(spec)
    if customers is not None:
        cust_coll = collection_def(spec, customers)
        assert cust_coll is not None
        ids = {str(r.get(cust_coll.key)) for r in task.state.get(customers, [])}
        if task.customer_id not in ids:
            problems.append(f"customer {task.customer_id} is not in state.{customers}")

    if spec.answer_topics:
        for topic in task.customer_answers:
            if topic not in spec.answer_topics:
                problems.append(f"customer_answers: '{topic}' is not one of the answer_topics")

    for i, check in enumerate(task.success):
        problems += check_problems(spec, check, f"success[{i}]")
    if not task.success:
        problems.append("no success checks (the task could never be verified)")
    if not task.reference:
        problems.append("no reference solution")
    if len(task.reference) + 1 > spec.max_steps:
        problems.append(
            f"reference needs {len(task.reference)} calls + finish, more than max_steps "
            f"{spec.max_steps}"
        )

    tools = {t.name: t for t in spec.tools}
    for i, call in enumerate(task.reference):
        if call.tool not in tools:
            problems.append(f"reference[{i}]: unknown tool '{call.tool}'")
            continue
        try:
            clean_args(tools[call.tool], call.args, spec)
        except ToolError as e:
            problems.append(f"reference[{i}]: {e}")
    if problems:
        return problems  # don't try to run a task that is malformed

    problems += _run_reference(spec, task)
    return problems


def _run_reference(spec: EnvSpec, task: TaskRecord) -> list[str]:
    """Replay the reference in a fresh world; report what stops it reaching 1.0."""
    world = World.from_task(spec, task)
    problems: list[str] = []
    if all(evaluate_check(c, world, spec).passed for c in task.success):
        problems.append("already solved before any action (every success check passes at start)")
    tools = {t.name: t for t in spec.tools}
    for i, call in enumerate(task.reference):
        tool = tools[call.tool]
        try:
            args = clean_args(tool, call.args, spec)
            apply_effect(spec, tool, args, world, task.customer_answers)
            world.calls.append((tool.name, args))
        except ToolError as e:
            problems.append(f"reference[{i}] {call.tool}: ToolError: {e}")
    verdict = judge(spec, task, world)
    if verdict.reward != 1.0:
        failing = [c for c in verdict.details["success_checks"] if not c["passed"]]
        for c in failing:
            problems.append(
                f"reference leaves a success check failing: {c['check']} ({c['detail']})"
            )
        for c in verdict.details["collateral"]:
            problems.append(f"reference breaks policy: {c}")
    return problems


def validate_dataset(spec: EnvSpec, tasks: list[TaskRecord]) -> dict[str, Any]:
    """Everything at once, as a JSON-friendly summary:

    {"spec_problems": [...], "n_tasks": 30, "ok": 29, "problems": {task_id: [...]},
     "tiers": {1: 10, 2: 10, 3: 10}, "duplicate_ids": [...]}
    """
    spec_problems = validate_spec(spec)
    problems: dict[str, list[str]] = {}
    for t in tasks:
        found = validate_task(spec, t)
        if found:
            problems[t.id] = found
    dup_ids = _dupes([t.id for t in tasks])
    return {
        "spec_problems": spec_problems,
        "n_tasks": len(tasks),
        "ok": sum(1 for t in tasks if t.id not in problems),
        "problems": problems,
        "tiers": dict(sorted(Counter(t.tier for t in tasks).items())),
        "duplicate_ids": dup_ids,
    }

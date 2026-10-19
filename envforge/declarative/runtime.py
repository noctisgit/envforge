"""The declarative runtime: an `EnvSpec` plus task records become an `Environment` class.

    env_cls = make_environment(spec, tasks)          # a real Environment subclass
    task    = env_cls.generate_task(seed=3, tier=2)  # picks one task record
    env     = env_cls(task); env.reset(); env.step(call); env.verify()

Everything the pipeline needs (runs, evaluation, previews, replays, the Environments
page) then works unchanged, because the class satisfies the contract in
`envforge/core/env.py`.

How a declarative episode works
-------------------------------
- `generate_task(seed, tier)` chooses a task record of that tier deterministically
  (records sorted by id, index = seed % count). The `Task` carries the record id in
  `hidden["task_record"]`; the full record stays in the class's task table.
- `reset()` deep-copies the record's starting state into a `World` (rules.py). The
  first observation is the ticket plus a list of the systems the agent can query —
  never the records themselves; the agent has to look things up.
- `step()` applies the tool's declarative effect (get / find / set / add /
  ask_customer / resolve) and returns records as `field: value` lines.
- `verify()` runs the task's success checks and the spec's policy checks:
  reward 1.0 (solved, no policy broken), 0.3 (solved, a policy broken), 0.0 otherwise.
"""

from __future__ import annotations

import re
from typing import Any, ClassVar

from envforge.core.env import (
    Diagram,
    DiagramEdge,
    DiagramNode,
    EnvDescription,
    Environment,
    InfoTable,
    Observation,
    RewardRule,
    Task,
    ToolCall,
    ToolError,
    ToolSpec,
    Verdict,
)
from envforge.declarative.rules import (
    CheckResult,
    RecordKey,
    World,
    describe_check,
    evaluate_check,
    format_value,
    matches,
    resolve_values,
    where_text,
)
from envforge.declarative.spec import (
    AddRecord,
    AskCustomer,
    CollectionDef,
    EnvSpec,
    FieldDef,
    FindRecords,
    GetRecord,
    Policy,
    ResolutionIs,
    Resolve,
    SetFields,
    TaskRecord,
    ToolDef,
    ToolParam,
)

FIND_LIMIT = 20  # find effects show at most this many records
DIAGRAM_NODE_LIMIT = 30  # keep pictures readable


# ---- tool schemas and argument checking ---------------------------------------------


def param_enum(param: ToolParam, tool: ToolDef, spec: EnvSpec) -> list[str] | None:
    """The allowed values for a parameter, if there is a fixed list.

    Explicit `enum`s win. Otherwise: the resolve tool's category takes the spec's
    resolution categories, ask_customer's topic takes the answer topics, and a parameter
    written straight into an enum field (e.g. an escalation's `team`) takes that field's
    values — so models see the choices in the JSON schema.
    """
    if param.enum:
        return list(param.enum)
    eff = tool.effect
    if isinstance(eff, Resolve) and param.name == eff.category_param:
        return list(spec.resolution_categories)
    if isinstance(eff, AskCustomer) and param.name == eff.topic_param and spec.answer_topics:
        return list(spec.answer_topics)
    mapping: dict[str, Any] = {}
    if isinstance(eff, SetFields):
        mapping = eff.set
    elif isinstance(eff, AddRecord):
        mapping = eff.values
    elif isinstance(eff, FindRecords):
        mapping = eff.match
    coll = collection_def(spec, getattr(eff, "collection", ""))
    if coll is not None:
        for field_name, value in mapping.items():
            fdef = field_def(coll, field_name)
            if value == f"${param.name}" and fdef is not None and fdef.enum:
                return list(fdef.enum)
    return None


def tool_spec(tool: ToolDef, spec: EnvSpec) -> ToolSpec:
    """Translate a ToolDef into the JSON-Schema ToolSpec models see."""
    props: dict[str, Any] = {}
    for p in tool.params:
        schema: dict[str, Any] = {
            "type": {"number": "number", "boolean": "boolean"}.get(p.type, "string")
        }
        enum = param_enum(p, tool, spec)
        if enum:
            schema["enum"] = enum
        if p.description:
            schema["description"] = p.description
        props[p.name] = schema
    parameters: dict[str, Any] = {"type": "object", "properties": props}
    required = [p.name for p in tool.params if p.required]
    if required:
        parameters["required"] = required
    return ToolSpec(
        name=tool.name, description=tool.description, parameters=parameters, category=tool.category
    )


def _is_number(v: Any) -> bool:
    return isinstance(v, int | float) and not isinstance(v, bool)


def clean_args(tool: ToolDef, args: Any, spec: EnvSpec) -> dict[str, Any]:
    """Check a call's arguments against the tool's params; return the accepted ones.

    Raises ToolError (which the agent sees) for unexpected or missing arguments, wrong
    types and values outside an enum. Omitted optional arguments are simply absent.
    """
    if not isinstance(args, dict):
        raise ToolError(f"{tool.name}: arguments must be an object")
    known = {p.name for p in tool.params}
    unexpected = sorted(set(args) - known)
    if unexpected:
        raise ToolError(
            f"{tool.name}: unexpected argument(s) {unexpected}; expected {sorted(known)}"
        )
    out: dict[str, Any] = {}
    for p in tool.params:
        value = args.get(p.name)
        if value is None:
            if p.required:
                raise ToolError(f"{tool.name}: missing required argument '{p.name}'")
            continue
        if p.type == "number" and not _is_number(value):
            raise ToolError(f"{tool.name}: '{p.name}' must be a number, got {value!r}")
        if p.type == "boolean" and not isinstance(value, bool):
            raise ToolError(f"{tool.name}: '{p.name}' must be true or false, got {value!r}")
        if p.type in ("string", "enum") and not isinstance(value, str):
            raise ToolError(f"{tool.name}: '{p.name}' must be a string, got {value!r}")
        enum = param_enum(p, tool, spec)
        if enum and value not in enum:
            raise ToolError(f"{tool.name}: '{p.name}' must be one of {enum}, got {value!r}")
        out[p.name] = value
    return out


# ---- small lookups ------------------------------------------------------------------


def collection_def(spec: EnvSpec, name: str) -> CollectionDef | None:
    return next((c for c in spec.collections if c.name == name), None)


def field_def(coll: CollectionDef, name: str) -> FieldDef | None:
    return next((f for f in coll.fields if f.name == name), None)


def resolve_tool(spec: EnvSpec) -> ToolDef | None:
    """The tool whose effect records the resolution (every spec needs one)."""
    return next((t for t in spec.tools if isinstance(t.effect, Resolve)), None)


def render_record(coll: CollectionDef, record: dict[str, Any]) -> str:
    """A record as an operator would see it in a support console:

    devices DEV-1042
      customer_id: C-1001
      status: offline
    """
    lines = [f"{coll.name} {format_value(record.get(coll.key))}"]
    order = [f.name for f in coll.fields] + [k for k in record if k not in field_names(coll)]
    for name in order:
        if name != coll.key:
            lines.append(f"  {name}: {format_value(record.get(name))}")
    return "\n".join(lines)


def field_names(coll: CollectionDef) -> set[str]:
    return {f.name for f in coll.fields}


def _check_field_value(coll: CollectionDef, name: str, value: Any) -> None:
    """Raise ToolError if `value` doesn't fit the field's declared type."""
    fdef = field_def(coll, name)
    if fdef is None:
        raise ToolError(f"{coll.name} has no field '{name}'")
    if value is None:
        return
    if fdef.type == "number" and not _is_number(value):
        raise ToolError(f"{coll.name}.{name} must be a number, got {value!r}")
    if fdef.type == "boolean" and not isinstance(value, bool):
        raise ToolError(f"{coll.name}.{name} must be true or false, got {value!r}")
    if fdef.type in ("string", "datetime", "enum") and not isinstance(value, str):
        raise ToolError(f"{coll.name}.{name} must be text, got {value!r}")
    if fdef.type == "enum" and fdef.enum and value not in fdef.enum:
        raise ToolError(f"{coll.name}.{name} must be one of {fdef.enum}, got {value!r}")


def new_key(coll: CollectionDef, existing: dict[str, Any]) -> str:
    """A fresh, deterministic key for an added record.

    Follows the pattern of the collection's existing keys ("CR-1007" -> "CR-1008");
    with no records to copy, uses the first three letters of the collection name.
    """
    numbered = [m for k in existing if (m := re.fullmatch(r"(.*?)(\d+)", k))]
    if numbered:
        prefix, width = numbered[0].group(1), len(numbered[0].group(2))
        n = max(int(m.group(2)) for m in numbered if m.group(1) == prefix) + 1
    else:
        prefix, width, n = coll.name[:3].upper() + "-", 4, 1001
    while f"{prefix}{n:0{width}d}" in existing:
        n += 1
    return f"{prefix}{n:0{width}d}"


# ---- effects ------------------------------------------------------------------------


def _record_or_error(world: World, coll: CollectionDef, key: Any) -> dict[str, Any]:
    record = world.records.get(coll.name, {}).get(str(key))
    if record is None:
        raise ToolError(f"no {coll.name} record with {coll.key} '{key}'")
    return record


def _mark_touched(world: World, coll: CollectionDef, key: str, how: str) -> None:
    if (coll.name, key) in world.touched:
        return  # a record created then changed stays "new"
    if coll.owner_field and how == "changed":
        world.original_owner[(coll.name, key)] = world.records[coll.name][key].get(coll.owner_field)
    world.touched[(coll.name, key)] = how


def apply_effect(spec: EnvSpec, tool: ToolDef, args: dict[str, Any], world: World,
                 answers: dict[str, str]) -> str:  # fmt: skip
    """Apply one tool's effect to the world and return the observation text.

    `args` must already have passed `clean_args`. Raises ToolError on unknown keys or
    values that don't fit a field.
    """
    eff = tool.effect

    if isinstance(eff, AskCustomer):
        topic = str(args.get(eff.topic_param, ""))
        normal = {k.strip().lower().replace(" ", "_"): v for k, v in answers.items()}
        answer = answers.get(topic) or normal.get(topic.strip().lower().replace(" ", "_"))
        return f'Customer: "{answer or "Sorry, I am not sure about that."}"'

    if isinstance(eff, Resolve):
        category = args.get(eff.category_param)
        if category not in spec.resolution_categories:
            raise ToolError(f"category must be one of {spec.resolution_categories}")
        world.resolution = category
        world.resolution_note = args.get(eff.note_param) if eff.note_param else None
        world.changes.append(f"{tool.name}: resolution = {category}")
        return f"Ticket resolved as '{category}'. Call finish to end the conversation."

    coll = collection_def(spec, eff.collection)
    if coll is None:  # validate_spec reports this; never let it crash an episode
        raise ToolError(f"{tool.name} is misconfigured: unknown collection '{eff.collection}'")
    rows = world.records.setdefault(coll.name, {})

    if isinstance(eff, GetRecord):
        return render_record(coll, _record_or_error(world, coll, args.get(eff.key_param)))

    if isinstance(eff, FindRecords):
        # an omitted optional argument drops that filter instead of matching null
        where = {k: v for k, v in resolve_values(eff.match, args).items() if v is not None}
        hits = [r for r in rows.values() if matches(r, where)]
        if not hits:
            return f"No {coll.name} records match {where_text(where)}."
        shown = hits[:FIND_LIMIT]
        head = f"{len(hits)} {coll.name} record(s) match {where_text(where)}"
        if len(hits) > FIND_LIMIT:
            head += f" (showing the first {FIND_LIMIT})"
        return head + ":\n\n" + "\n\n".join(render_record(coll, r) for r in shown)

    if isinstance(eff, SetFields):
        key = str(args.get(eff.key_param))
        record = _record_or_error(world, coll, key)
        values = {k: v for k, v in resolve_values(eff.set, args).items() if v is not None}
        for name, value in values.items():
            _check_field_value(coll, name, value)
        diffs = [(n, record.get(n), v) for n, v in values.items() if record.get(n) != v]
        if not diffs:
            return f"{coll.name} {key}: no change (already up to date)."
        _mark_touched(world, coll, key, "changed")
        parts = []
        for name, old, new in diffs:
            record[name] = new
            parts.append(f"{name}: {format_value(old)} -> {format_value(new)}")
        world.changes.append(f"{tool.name}: {coll.name} {key} " + ", ".join(parts))
        return f"Updated {coll.name} {key}:\n" + "\n".join("  " + x for x in parts)

    if isinstance(eff, AddRecord):
        values = {k: v for k, v in resolve_values(eff.values, args).items() if v is not None}
        if coll.owner_field and coll.owner_field not in values:
            values[coll.owner_field] = (
                world.customer_id
            )  # new records belong to the ticket's customer
        for name, value in values.items():
            _check_field_value(coll, name, value)
        key = new_key(coll, rows)
        record = {f.name: values.get(f.name) for f in coll.fields}
        record[coll.key] = key
        rows[key] = record
        _mark_touched(world, coll, key, "new")
        world.changes.append(f"{tool.name}: created {coll.name} {key}")
        return "Created " + render_record(coll, record)

    raise ToolError(f"{tool.name}: unsupported effect")


# ---- judging ------------------------------------------------------------------------


def broken_policies(spec: EnvSpec, world: World) -> list[tuple[Policy, CheckResult]]:
    """Policies that apply (their `when`, if any, holds) and whose check fails."""
    out = []
    for p in spec.policies:
        if p.when is not None and not evaluate_check(p.when, world, spec).passed:
            continue
        result = evaluate_check(p.check, world, spec)
        if not result.passed:
            out.append((p, result))
    return out


def judge(spec: EnvSpec, record: TaskRecord, world: World) -> Verdict:
    """Success checks + policy checks -> Verdict (1.0 / 0.3 / 0.0)."""
    results = [evaluate_check(c, world, spec) for c in record.success]
    solved = all(r.passed for r in results)
    broken = broken_policies(spec, world)
    collateral = [f"policy {p.id} broken: {r.detail or r.description}" for p, r in broken]
    reward = 1.0 if solved and not collateral else 0.3 if solved else 0.0
    return Verdict(
        success=solved and not collateral,
        reward=reward,
        details={
            "task_record": record.id,
            "success_checks": [
                {"check": r.description, "passed": r.passed, "detail": r.detail} for r in results
            ],
            "collateral": collateral,
            "policies_broken": [p.id for p, _ in broken],
            "resolution": world.resolution,
            "changes": list(world.changes),
        },
    )


# ---- the Environment ----------------------------------------------------------------


def _short(text: str, n: int = 90) -> str:
    text = " ".join(text.split())
    return text if len(text) <= n else text[: n - 1] + "…"


class DeclarativeEnvironment(Environment):
    """Base class for environments built by `make_environment`.

    Subclasses get these class attributes filled in: `spec`, `task_table` (record id ->
    TaskRecord) and `tasks_by_tier` (tier -> records sorted by id).
    """

    kind: ClassVar[str] = "sim"
    spec: ClassVar[EnvSpec]
    task_table: ClassVar[dict[str, TaskRecord]]
    tasks_by_tier: ClassVar[dict[int, list[TaskRecord]]]

    def __init__(self, task: Task) -> None:
        super().__init__(task)
        record_id = task.hidden.get("task_record")
        if record_id not in self.task_table:
            raise ValueError(f"{self.name}: task record '{record_id}' is not in this dataset")
        self.record: TaskRecord = self.task_table[record_id]
        self._tools = {t.name: t for t in self.spec.tools}
        self._world: World | None = None

    # ---- task generation ---------------------------------------------------------
    @classmethod
    def generate_task(cls, seed: int, tier: int) -> Task:
        if tier not in cls.tasks_by_tier:  # nearest tier that has tasks (lower on ties)
            tier = min(cls.tasks_by_tier, key=lambda t: (abs(t - tier), t))
        records = cls.tasks_by_tier[tier]
        rec = records[seed % len(records)]
        expected = [c.category for c in rec.success if isinstance(c, ResolutionIs)]
        return Task(
            id=f"{cls.name}-t{rec.tier}-{rec.id}-s{seed}",
            pack=cls.name,
            tier=rec.tier,
            seed=seed,
            goal=rec.goal,
            hidden={
                "task_record": rec.id,
                "customer_id": rec.customer_id,
                "expected_resolution": expected[0] if expected else None,
                "success_checks": [describe_check(c) for c in rec.success],
                "reference": [c.model_dump() for c in rec.reference],
                "notes": rec.notes,
                "source": rec.source,
            },
        )

    # ---- lifecycle ---------------------------------------------------------------
    @property
    def world(self) -> World:
        if self._world is None:
            self._world = World.from_task(self.spec, self.record)
        return self._world

    def reset(self) -> Observation:
        self._world = World.from_task(self.spec, self.record)
        counts = {c.name: self._world.count(c.name) for c in self.spec.collections}
        systems = "\n".join(
            f"  {c.name}: {counts[c.name]} record{'' if counts[c.name] == 1 else 's'} "
            f"(key: {c.key})"
            for c in self.spec.collections
        )
        text = (
            f"TICKET from customer {self.record.customer_id}\n"
            f"{self.record.goal}\n\n"
            f"AVAILABLE SYSTEMS\n{systems}\n\n"
            "Records are not shown until you look them up with the tools."
        )
        return Observation(text=text, data={"customer_id": self.record.customer_id})

    def tools(self) -> list[ToolSpec]:
        return [tool_spec(t, self.spec) for t in self.spec.tools]

    def step(self, call: ToolCall) -> Observation:
        tool = self._tools.get(call.name)
        if tool is None:
            raise ToolError(f"unknown tool '{call.name}'. Available: {sorted(self._tools)}")
        args = clean_args(tool, call.args, self.spec)
        text = apply_effect(self.spec, tool, args, self.world, self.record.customer_answers)
        self.world.calls.append((tool.name, args))  # only calls that took effect
        return Observation(text=text)

    def system_prompt(self) -> str:
        spec = self.spec
        lines = [spec.agent_role.strip(), ""]
        if spec.policies:
            lines.append(
                "POLICIES (checked after the conversation; breaking one counts against you)"
            )
            lines += [f"- [{p.id}] {p.text}" for p in spec.policies]
            lines.append("")
        res = resolve_tool(spec)
        res_name = res.name if res else "resolve"
        lines += [
            "HOW TO FINISH",
            "Investigate with the read-only tools before changing anything, and only change "
            "records of the customer on this ticket. When the ticket is handled, call "
            f"`{res_name}` with the category that best describes the outcome (one of: "
            f"{', '.join(spec.resolution_categories)}), then call `finish`.",
        ]
        return "\n".join(lines)

    def verify(self) -> Verdict:
        return judge(self.spec, self.record, self.world)

    def reference_solve(self) -> list[ToolCall]:
        return [
            ToolCall(id=f"ref{i + 1}", name=c.tool, args=dict(c.args))
            for i, c in enumerate(self.record.reference)
        ]

    def snapshot(self) -> dict[str, Any]:
        return self.world.snapshot()

    # ---- self-description ------------------------------------------------------------
    @classmethod
    def describe(cls) -> EnvDescription:
        spec = cls.spec
        records = list(cls.task_table.values())
        counts = []
        for c in spec.collections:
            per_task = [len(r.state.get(c.name, [])) for r in records]
            lo, hi = min(per_task), max(per_task)
            counts.append(f"{c.name} {lo}" if lo == hi else f"{c.name} {lo}–{hi}")
        tiers_text = ", ".join(f"tier {t}: {len(cls.tasks_by_tier[t])}" for t in cls.tasks_by_tier)
        world = [
            f"{len(spec.collections)} record collections: "
            + ", ".join(c.name for c in spec.collections),
            "records per task: " + ", ".join(counts),
            f"{len(records)} tasks ({tiers_text}), each a real-style ticket with its own records",
            f"{len(spec.policies)} policies checked by code at the end of every episode",
            f"{len(spec.resolution_categories)} resolution categories",
        ]
        if spec.answer_topics:
            world.append("the customer can be asked about: " + ", ".join(spec.answer_topics))
        observe = [t.name for t in spec.tools if t.category == "observe"]
        act = [t.name for t in spec.tools if t.category == "act"]
        res = resolve_tool(spec)

        policy_rows = []
        for p in spec.policies:
            checked = describe_check(p.check)
            if p.when is not None:
                checked = f"if {describe_check(p.when)}: {checked}"
            policy_rows.append([p.id, p.text, checked])
        res_counts = {cat: 0 for cat in spec.resolution_categories}
        for r in records:
            for c in r.success:
                if isinstance(c, ResolutionIs) and c.category in res_counts:
                    res_counts[c.category] += 1
        dataset_rows = [
            [str(t), str(len(recs)), " | ".join(_short(r.goal, 70) for r in recs[:2])]
            for t, recs in cls.tasks_by_tier.items()
        ]
        tables = [
            InfoTable(
                title="Collections",
                columns=["collection", "key", "fields", "owner field"],
                rows=[
                    [c.name, c.key, ", ".join(f.name for f in c.fields), c.owner_field or "—"]
                    for c in spec.collections
                ],
            ),
            InfoTable(
                title="Policies",
                columns=["id", "rule shown to the agent", "what is checked"],
                rows=policy_rows,
                note="A broken policy caps the reward at 0.3 even if the ticket is solved.",
            ),
            InfoTable(
                title="Resolution categories",
                columns=["category", "expected in tasks"],
                rows=[[cat, str(n)] for cat, n in res_counts.items()],
            ),
            InfoTable(
                title="Dataset",
                columns=["tier", "tasks", "example tickets"],
                rows=dataset_rows,
            ),
        ]
        return (
            super()
            .describe()
            .model_copy(
                update={
                    "overview": spec.overview or spec.description,
                    "world": world,
                    "observation": (
                        "First the ticket (customer id and message) and a list of the systems "
                        "with record counts; then each tool's output as plain `field: value` "
                        "lines. Records are only shown when the agent looks them up."
                    ),
                    "action": (
                        f"{len(observe)} read-only tools ({', '.join(observe)}) and "
                        f"{len(act)} tools that change records ({', '.join(act)}); "
                        f"record the outcome with `{res.name if res else 'resolve'}`, "
                        "then `finish`."
                    ),
                    "tier_notes": {t: n for t, n in spec.tier_notes.items() if t in cls.tiers},
                    "reward_rules": [
                        RewardRule(
                            outcome="every success check passes and no policy is broken",
                            reward=1.0,
                        ),
                        RewardRule(outcome="solved, but a policy was broken", reward=0.3),
                        RewardRule(outcome="a success check fails", reward=0.0),
                    ],
                    "tables": tables,
                }
            )
        )

    def diagram(self) -> Diagram | None:
        """The ticket's customer and the records around them, drawn live.

        The customer is the `target`; records a failing success check is about are
        `broken`; records the agent changed or created carry "changed" / "new" badges,
        and records a broken policy is about carry "policy".
        """
        spec, world = self.spec, self.world
        cid = world.customer_id
        colls = {c.name: c for c in spec.collections}

        success = [evaluate_check(c, world, spec) for c in self.record.success]
        broken_refs: set[RecordKey] = set()
        for r in success:
            if not r.passed:
                broken_refs |= r.refs
        policy_fails = broken_policies(spec, world)
        policy_refs: set[RecordKey] = set()
        for _, r in policy_fails:
            policy_refs |= r.refs

        # which records to draw, most relevant first
        customer_node = next(
            ((c.name, cid) for c in spec.collections if cid in world.records.get(c.name, {})),
            None,
        )
        chosen: list[RecordKey] = [customer_node] if customer_node else []
        owned = [
            (c.name, k)
            for c in spec.collections
            if c.owner_field
            for k, rec in world.records.get(c.name, {}).items()
            if str(rec.get(c.owner_field)) == cid
        ]
        index = world.key_index()
        linked: list[RecordKey] = []  # records the owned ones point at (plans, outages ...)
        for coll, key in owned:
            for field_name, value in world.records[coll][key].items():
                if field_name != colls[coll].key and isinstance(value, str):
                    linked += index.get(value, [])
        everything = [(c, k) for c, rows in world.records.items() for k in rows]
        groups = [
            owned,
            sorted(world.touched),
            sorted(broken_refs | policy_refs),
            linked,
            everything,
        ]
        for group in groups:
            for ref in group:
                if ref not in chosen and ref[1] in world.records.get(ref[0], {}):
                    chosen.append(ref)
        chosen = chosen[:DIAGRAM_NODE_LIMIT]

        def node_id(ref: RecordKey) -> str:
            return f"{ref[0]}:{ref[1]}"

        nodes: list[DiagramNode] = []
        if customer_node is None:  # no customer record: draw a stand-in
            nodes.append(DiagramNode(id=f"customer:{cid}", label=cid, kind="customer",
                                     state="target", detail=f"customer {cid}"))  # fmt: skip
        for ref in chosen:
            coll, key = ref
            rec = world.records[coll][key]
            label_field = next((f for f in ("name", "status", "title") if f in rec), None)
            label = key + (f" · {format_value(rec[label_field])}" if label_field else "")
            badges = []
            if ref in world.touched:
                badges.append(world.touched[ref])
            if ref in policy_refs:
                badges.append("policy")
            if ref == customer_node:
                state = "target"
            elif ref in broken_refs:
                state = "broken"
            else:
                state = "ok"
            nodes.append(
                DiagramNode(
                    id=node_id(ref),
                    label=label,
                    kind=coll,
                    group=coll,
                    state=state,
                    badges=badges,
                    detail=render_record(colls[coll], rec),
                )
            )

        drawn = set(chosen)
        edges: list[DiagramEdge] = []
        seen: set[tuple[str, str]] = set()
        for ref in chosen:
            coll, key = ref
            for field_name, value in world.records[coll][key].items():
                if field_name == colls[coll].key or not isinstance(value, str):
                    continue
                for other in index.get(value, []):
                    pair = (node_id(ref), node_id(other))
                    if other in drawn and other != ref and pair not in seen:
                        seen.add(pair)
                        state = "broken" if ref in broken_refs else "ok"
                        edges.append(DiagramEdge(source=pair[0], target=pair[1],
                                                 label=field_name, state=state, directed=True))  # fmt: skip
            if customer_node is None and ref in owned:
                edges.append(DiagramEdge(source=node_id(ref), target=f"customer:{cid}",
                                         label=colls[coll].owner_field or "", directed=True))  # fmt: skip

        notes = [("✓ " if r.passed else "✗ ") + r.description for r in success]
        notes += [f"policy broken: {p.id} — {r.detail or p.text}" for p, r in policy_fails]
        if world.resolution:
            notes.append(f"resolution recorded: {world.resolution}")
        return Diagram(title=f"{spec.title}: customer {cid}", nodes=nodes, edges=edges, notes=notes)


def _class_name(name: str) -> str:
    parts = re.split(r"[^0-9A-Za-z]+", name)
    return "".join(p[:1].upper() + p[1:] for p in parts if p) + "Environment"


def make_environment(
    spec: EnvSpec, tasks: list[TaskRecord], *, name: str | None = None
) -> type[Environment]:
    """Build an Environment subclass from a spec and its task records.

    `name` overrides the pack name (default `spec.name`), e.g. "isp-support@v3" for a
    versioned dataset stored in a database. Tiers are the spec's tiers that have tasks
    (or, if none overlap, the tiers the tasks actually use).
    """
    if not tasks:
        raise ValueError(f"{spec.name}: an environment needs at least one task record")
    table: dict[str, TaskRecord] = {}
    for t in tasks:
        if t.id in table:
            raise ValueError(f"{spec.name}: duplicate task record id '{t.id}'")
        table[t.id] = t
    present = sorted({t.tier for t in tasks})
    tiers = tuple(sorted(t for t in set(spec.tiers) if t in present)) or tuple(present)
    by_tier = {
        tier: sorted((t for t in tasks if t.tier == tier), key=lambda r: r.id) for tier in tiers
    }
    env_name = name or spec.name
    attrs: dict[str, Any] = {
        "name": env_name,
        "kind": "sim",
        "max_steps": spec.max_steps,
        "tiers": tiers,
        "description": spec.description,
        "spec": spec,
        "task_table": table,
        "tasks_by_tier": by_tier,
        "__module__": __name__,
        "__doc__": f"{spec.title}: {spec.description} (declarative, {len(tasks)} tasks)",
    }
    return type(_class_name(env_name), (DeclarativeEnvironment,), attrs)

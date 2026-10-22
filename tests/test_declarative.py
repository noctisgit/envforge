"""Declarative environments: rules, effects, runtime, validation, store, registry, example."""

from __future__ import annotations

from pathlib import Path

import pytest

from envforge.core.env import TRAIN_SEED_MIN, Diagram, ToolCall, ToolError
from envforge.core.explore import pack_overview, preview_task, replay_episode
from envforge.core.registry import get_pack
from envforge.core.runner import run_episode
from envforge.declarative.rules import World, compare, evaluate_check, matches, resolve_value
from envforge.declarative.runtime import is_eval_record, make_environment
from envforge.declarative.spec import (
    AddRecord,
    AskCustomer,
    CollectionDef,
    EnvSpec,
    FieldDef,
    FieldEquals,
    FindRecords,
    GetRecord,
    NoRecord,
    OnlyTouchCustomer,
    Policy,
    RecordExists,
    RefCall,
    ResolutionIs,
    Resolve,
    SetFields,
    TaskRecord,
    ToolCalled,
    ToolDef,
    ToolNotCalled,
    ToolParam,
)
from envforge.declarative.store import load_env_dir, load_spec_and_tasks, save_env_dir
from envforge.declarative.validate import validate_dataset, validate_spec, validate_task
from envforge.models.scripted import ScriptedModel

EXAMPLE = Path(__file__).parent.parent / "envforge/declarative/examples/isp_support"
EX_SPEC, EX_TASKS = load_spec_and_tasks(EXAMPLE)


# ---- a tiny shop spec for unit tests ---------------------------------------------------


def shop_spec(**overrides) -> EnvSpec:
    spec = EnvSpec(
        name="shop",
        title="Shop",
        description="toy shop desk",
        agent_role="You help shop customers.",
        resolution_categories=["refunded", "info"],
        answer_topics=["order_number", "address"],
        collections=[
            CollectionDef(name="customers", key="customer_id", owner_field="customer_id",
                          fields=[FieldDef(name="customer_id"), FieldDef(name="name")]),
            CollectionDef(name="orders", key="order_id", owner_field="customer_id", fields=[
                FieldDef(name="order_id"), FieldDef(name="customer_id"),
                FieldDef(name="status", type="enum", enum=["open", "shipped", "cancelled"]),
                FieldDef(name="total", type="number")]),
            CollectionDef(name="refunds", key="refund_id", owner_field="customer_id", fields=[
                FieldDef(name="refund_id"), FieldDef(name="customer_id"),
                FieldDef(name="order_id"), FieldDef(name="amount", type="number")]),
        ],
        tools=[
            ToolDef(name="get_order", description="get", category="observe",
                    params=[ToolParam(name="order_id")],
                    effect=GetRecord(collection="orders", key_param="order_id")),
            ToolDef(name="find_orders", description="find", category="observe",
                    params=[ToolParam(name="customer_id"),
                            ToolParam(name="status", required=False)],
                    effect=FindRecords(collection="orders",
                                       match={"customer_id": "$customer_id", "status": "$status"})),
            ToolDef(name="cancel_order", description="cancel", category="act",
                    params=[ToolParam(name="order_id")],
                    effect=SetFields(collection="orders", key_param="order_id",
                                     set={"status": "cancelled"})),
            ToolDef(name="set_status", description="set", category="act",
                    params=[ToolParam(name="order_id"), ToolParam(name="status")],
                    effect=SetFields(collection="orders", key_param="order_id",
                                     set={"status": "$status"})),
            ToolDef(name="refund", description="refund", category="act",
                    params=[ToolParam(name="customer_id"), ToolParam(name="order_id"),
                            ToolParam(name="amount", type="number")],
                    effect=AddRecord(collection="refunds", values={
                        "customer_id": "$customer_id", "order_id": "$order_id",
                        "amount": "$amount"})),
            ToolDef(name="ask", description="ask", category="observe",
                    params=[ToolParam(name="topic")], effect=AskCustomer(topic_param="topic")),
            ToolDef(name="resolve", description="resolve", category="act",
                    params=[ToolParam(name="category"), ToolParam(name="note", required=False)],
                    effect=Resolve(category_param="category", note_param="note")),
        ],
        policies=[
            Policy(id="refund-limit", text="Refunds up to $30.",
                   check=NoRecord(collection="refunds", where={"amount__gt": 30})),
            Policy(id="own", text="Only this customer.", check=OnlyTouchCustomer()),
            Policy(id="ask-first", text="Ask for the order number before refunding.",
                   when=ToolCalled(tool="refund"),
                   check=ToolCalled(tool="ask", args={"topic": "order_number"})),
        ],
    )  # fmt: skip
    return spec.model_copy(update=overrides)


def shop_task(**overrides) -> TaskRecord:
    t = TaskRecord(
        id="t1",
        tier=1,
        customer_id="C1",
        goal="My order O1 is late, please cancel it and refund me.",
        state={
            "customers": [
                {"customer_id": "C1", "name": "Ann"},
                {"customer_id": "C2", "name": "Bo"},
            ],
            "orders": [
                {"order_id": "O1", "customer_id": "C1", "status": "open", "total": 25},
                {"order_id": "O2", "customer_id": "C2", "status": "open", "total": 80},
                {"order_id": "O3", "customer_id": "C1", "status": "shipped", "total": 12.5},
            ],
            "refunds": [{"refund_id": "R-007", "customer_id": "C2", "order_id": "O9", "amount": 5}],
        },
        customer_answers={"order_number": "It's O1."},
        success=[
            FieldEquals(collection="orders", key="O1", field="status", value="cancelled"),
            RecordExists(collection="refunds", where={"order_id": "O1", "amount": 25}),
            ResolutionIs(category="refunded"),
        ],
        reference=[
            RefCall(tool="ask", args={"topic": "order_number"}),
            RefCall(tool="cancel_order", args={"order_id": "O1"}),
            RefCall(tool="refund", args={"customer_id": "C1", "order_id": "O1", "amount": 25}),
            RefCall(tool="resolve", args={"category": "refunded"}),
        ],
    )
    return t.model_copy(update=overrides)


def shop_env(task: TaskRecord | None = None, spec: EnvSpec | None = None):
    cls = make_environment(spec or shop_spec(), [task or shop_task()])
    env = cls(cls.generate_task(0, 1))
    env.reset()
    return env


def call(name: str, **args) -> ToolCall:
    return ToolCall(id="x", name=name, args=args)


def run_calls(env, calls: list[ToolCall]):
    for c in calls:
        env.step(c)
    return env.verify()


# ---- rules -------------------------------------------------------------------------


def test_where_operators():
    r = {"amount": 50, "status": "open", "day": "2026-10-02", "flag": None}
    assert matches(r, {"amount": 50.0, "status": "open"})
    assert matches(r, {"amount__gt": 49, "amount__gte": 50, "amount__lt": 51, "amount__lte": 50})
    assert not matches(r, {"amount__gt": 50})
    assert matches(r, {"status__ne": "closed", "status__in": ["open", "pending"]})
    assert not matches(r, {"status__in": ["closed"]})
    assert matches(r, {"day__lt": "2026-10-15"})  # ISO dates compare as strings
    assert matches(r, {"flag": None}) and not matches(r, {"flag__ne": None})
    assert not compare("50", "gt", 10)  # mixed types never order-match
    assert matches({"a__b": 1}, {"a__b": 1})  # unknown suffix is part of the field name


def test_resolve_value():
    assert resolve_value("$amount", {"amount": 5}) == 5
    assert resolve_value("$missing", {}) is None
    assert resolve_value("literal", {"literal": 1}) == "literal"
    assert resolve_value(7, {}) == 7


def test_each_check_type():
    spec, task = shop_spec(), shop_task()
    w = World.from_task(spec, task)

    def ok(check):
        return evaluate_check(check, w, spec).passed

    assert ok(FieldEquals(collection="orders", key="O1", field="status", value="open"))
    assert not ok(FieldEquals(collection="orders", key="nope", field="status", value="open"))
    assert ok(RecordExists(collection="orders", where={"total__gt": 50}))
    assert not ok(NoRecord(collection="orders", where={"total__gt": 50}))
    assert not ok(ResolutionIs(category="refunded"))
    assert not ok(ToolCalled(tool="ask"))
    assert ok(ToolNotCalled(tool="ask"))
    assert ok(OnlyTouchCustomer())
    w.calls.append(("ask", {"topic": "order_number"}))
    w.resolution = "refunded"
    assert ok(ToolCalled(tool="ask", args={"topic": "order_number"}))
    assert not ok(ToolCalled(tool="ask", args={"topic": "other"}))
    assert not ok(ToolNotCalled(tool="ask"))
    assert ok(ResolutionIs(category="refunded"))
    w.touched[("orders", "O2")] = "changed"  # someone else's order
    assert not ok(OnlyTouchCustomer())


# ---- effects -----------------------------------------------------------------------


def test_get_and_find():
    env = shop_env()
    out = env.step(call("get_order", order_id="O1")).text
    assert out.splitlines()[0] == "orders O1" and "  status: open" in out
    found = env.step(call("find_orders", customer_id="C1")).text
    assert "2 orders record(s)" in found and "O3" in found and "O2" not in found
    found = env.step(call("find_orders", customer_id="C1", status="shipped")).text
    assert "1 orders record(s)" in found and "orders O3" in found
    assert "No orders records" in env.step(call("find_orders", customer_id="C9")).text
    with pytest.raises(ToolError, match="no orders record"):
        env.step(call("get_order", order_id="O404"))


def test_find_is_capped():
    task = shop_task()
    task.state["orders"] = [
        {"order_id": f"O{i}", "customer_id": "C1", "status": "open", "total": 1} for i in range(30)
    ]
    env = shop_env(task.model_copy(update={"success": [ResolutionIs(category="info")]}))
    out = env.step(call("find_orders", customer_id="C1")).text
    assert "30 orders record(s)" in out and "showing the first 20" in out
    assert out.count("\norders O") == 20


def test_set_add_ask_resolve():
    env = shop_env()
    assert "status: open -> cancelled" in env.step(call("cancel_order", order_id="O1")).text
    assert "no change" in env.step(call("cancel_order", order_id="O1")).text
    out = env.step(call("refund", customer_id="C1", order_id="O1", amount=25)).text
    assert out.startswith("Created refunds R-008")  # follows the existing key pattern
    assert env.world.records["refunds"]["R-008"]["amount"] == 25
    assert env.world.touched == {("orders", "O1"): "changed", ("refunds", "R-008"): "new"}
    assert "It's O1." in env.step(call("ask", topic="order_number")).text
    assert "not sure" in env.step(call("ask", topic="address")).text  # no answer in the task
    with pytest.raises(ToolError, match="must be one of"):
        env.step(call("ask", topic="weather"))  # not an answer topic
    assert "resolved as 'refunded'" in env.step(call("resolve", category="refunded")).text
    assert env.world.resolution == "refunded"


def test_add_defaults_owner_to_ticket_customer():
    spec = shop_spec()
    spec.tools[4] = spec.tools[4].model_copy(
        update={"effect": AddRecord(collection="refunds", values={"amount": "$amount"})}
    )
    env = shop_env(spec=spec)
    env.step(call("refund", customer_id="C1", order_id="O1", amount=3))
    assert env.world.records["refunds"]["R-008"]["customer_id"] == "C1"


def test_tool_errors():
    env = shop_env()
    with pytest.raises(ToolError, match="unknown tool"):
        env.step(call("nope"))
    with pytest.raises(ToolError, match="missing required"):
        env.step(call("get_order"))
    with pytest.raises(ToolError, match="unexpected argument"):
        env.step(call("get_order", order_id="O1", extra=1))
    with pytest.raises(ToolError, match="must be a number"):
        env.step(call("refund", customer_id="C1", order_id="O1", amount="25"))
    with pytest.raises(ToolError, match="must be a string"):
        env.step(call("get_order", order_id=1))
    with pytest.raises(ToolError, match="must be one of"):
        env.step(call("resolve", category="made_up"))
    with pytest.raises(ToolError, match="must be one of"):  # enum taken from the field
        env.step(call("set_status", order_id="O1", status="lost"))
    assert env.world.calls == []  # failed calls don't count for tool_called checks
    assert env.world.records["orders"]["O1"]["status"] == "open"


def test_schemas_carry_enums():
    cls = make_environment(shop_spec(), [shop_task()])
    tools = {t.name: t for t in cls(cls.generate_task(0, 1)).tools()}
    props = tools["resolve"].parameters["properties"]
    assert props["category"]["enum"] == ["refunded", "info"]
    assert tools["ask"].parameters["properties"]["topic"]["enum"] == ["order_number", "address"]
    assert tools["set_status"].parameters["properties"]["status"]["enum"] == [
        "open", "shipped", "cancelled"]  # fmt: skip
    assert tools["refund"].parameters["properties"]["amount"]["type"] == "number"
    assert tools["find_orders"].parameters["required"] == ["customer_id"]
    assert tools["cancel_order"].category == "act"


# ---- runtime -----------------------------------------------------------------------


def test_reward_levels_and_policies():
    env = shop_env()
    assert env.verify().reward == 0.0
    v = run_calls(env, env.reference_solve())
    assert (v.success, v.reward, v.details["collateral"]) == (True, 1.0, [])
    assert all(c["passed"] for c in v.details["success_checks"])

    env = shop_env()  # skips the `when`-conditional identity question -> collateral
    v = run_calls(env, env.reference_solve()[1:])
    assert v.reward == 0.3 and v.details["policies_broken"] == ["ask-first"]

    env = shop_env()  # over-limit refund on top of the reference
    v = run_calls(env, [call("refund", customer_id="C1", order_id="O1", amount=40),
                        *env.reference_solve()])  # fmt: skip
    assert v.reward == 0.3 and "refund-limit" in v.details["policies_broken"]


def test_only_touch_customer_catches_other_customers_records():
    env = shop_env()
    v = run_calls(env, [call("cancel_order", order_id="O2"), *env.reference_solve()])
    assert v.reward == 0.3 and v.details["policies_broken"] == ["own"]
    env = shop_env()  # creating a record owned by someone else also counts
    v = run_calls(env, [call("refund", customer_id="C2", order_id="O2", amount=1),
                        *env.reference_solve()])  # fmt: skip
    assert v.details["policies_broken"] == ["own"]


def test_generate_task_selection_and_names():
    tasks = [shop_task(id=f"t{i}", tier=1 + (i % 2)) for i in range(5)]  # tiers 1 and 2
    cls = make_environment(shop_spec(), tasks, name="shop@v3")
    assert cls.name == "shop@v3" and cls.tiers == (1, 2) and cls.kind == "sim"
    t = cls.generate_task(seed=4, tier=1)  # tier 1: t0, t2, t4; only t2 is held out for eval
    assert t.hidden["task_record"] == "t2" and t.id == "shop@v3-t1-t2-s4"
    assert t.goal == tasks[2].goal and t.hidden["expected_resolution"] == "refunded"
    assert cls.generate_task(seed=0, tier=3).tier == 2  # nearest tier with tasks
    assert cls.generate_task(4, 1) == cls.generate_task(4, 1)
    with pytest.raises(ValueError):
        make_environment(shop_spec(), [])
    with pytest.raises(ValueError):
        make_environment(shop_spec(), [shop_task(), shop_task()])


def test_evaluation_and_training_seeds_use_different_records():
    tasks = [shop_task(id=f"t{i}", tier=1 + (i % 2)) for i in range(40)]
    cls = make_environment(shop_spec(), tasks, name="shop@v1")

    def records(env, seeds):
        return {env.generate_task(s, tier).hidden["task_record"] for s in seeds for tier in (1, 2)}

    held_out = records(cls, range(200))
    training = records(cls, range(TRAIN_SEED_MIN, TRAIN_SEED_MIN + 200))
    assert not held_out & training  # never the same ticket on both sides
    assert held_out | training == {t.id for t in tasks}
    assert held_out == {t.id for t in tasks if is_eval_record(t.id)}
    assert 5 <= len(held_out) <= 15  # about a quarter
    assert cls.generate_task(TRAIN_SEED_MIN, 1).hidden["task_record"] == cls.train_by_tier[1][0].id
    assert any("held out for evaluation" in w for w in cls.describe().world)

    # a new version with more tasks keeps every old record on its side
    grown = make_environment(
        shop_spec(), tasks + [shop_task(id=f"new{i}") for i in range(20)], name="shop@v2"
    )
    assert held_out <= records(grown, range(400))
    assert not held_out & records(grown, range(TRAIN_SEED_MIN, TRAIN_SEED_MIN + 400))


def test_a_run_over_every_tier_reaches_every_record():
    # tier 3 has 6 held-out and 24 training records: seed % count with the run's stride of 3
    # would reach only a third of them
    tasks = [shop_task(id=f"t{i}", tier=1 + (i % 3)) for i in range(90)]
    spec = shop_spec(tiers=[1, 2, 3])
    cls = make_environment(spec, tasks)
    for start, side in ((0, cls.eval_by_tier), (TRAIN_SEED_MIN, cls.train_by_tier)):
        for i, tier in enumerate(cls.tiers):
            n = len(side[tier])
            seen = {
                cls.generate_task(start + i + 3 * k, tier).hidden["task_record"] for k in range(n)
            }
            assert seen == {r.id for r in side[tier]}


def test_a_task_set_too_small_to_split_uses_every_record():
    cls = make_environment(shop_spec(), [shop_task(id="t2")])  # t2 is a held-out record
    assert cls.generate_task(TRAIN_SEED_MIN, 1).hidden["task_record"] == "t2"
    assert cls.generate_task(0, 1).hidden["task_record"] == "t2"
    assert any("too few tasks" in w for w in cls.describe().world)


def test_first_observation_hides_records():
    env = make_environment(shop_spec(), [shop_task()])
    e = env(env.generate_task(0, 1))
    text = e.reset().text
    assert "TICKET from customer C1" in text and "orders: 3 records" in text
    assert "O2" not in text and "Bo" not in text
    assert "refund-limit" in e.system_prompt() and "`resolve`" in e.system_prompt()


def test_determinism_same_actions_same_outcome():
    a, b = shop_env(), shop_env()
    calls = [call("cancel_order", order_id="O1"), call("refund", customer_id="C1",
             order_id="O1", amount=25), call("resolve", category="info")]  # fmt: skip
    assert run_calls(a, calls) == run_calls(b, calls)
    assert a.snapshot() == b.snapshot()


def test_reset_restores_starting_state():
    env = shop_env()
    env.step(call("cancel_order", order_id="O1"))
    env.reset()
    assert env.world.records["orders"]["O1"]["status"] == "open" and not env.world.touched
    assert env.record.state["orders"][0]["status"] == "open"  # the record itself is untouched


def test_diagram_live():
    env = shop_env()
    d = env.diagram()
    nodes = {n.id: n for n in d.nodes}
    assert nodes["customers:C1"].state == "target"
    assert nodes["orders:O1"].state == "broken"
    assert any(e.source == "orders:O1" and e.target == "customers:C1" for e in d.edges)
    assert any(n.startswith("✗") for n in d.notes)
    env.step(call("cancel_order", order_id="O2"))  # someone else's order
    nodes = {n.id: n for n in env.diagram().nodes}
    assert {"changed", "policy"} <= set(nodes["orders:O2"].badges)
    for c in env.reference_solve():
        env.step(c)
    d = env.diagram()
    assert not [n for n in d.nodes if n.state == "broken"]
    assert any("new" in n.badges for n in d.nodes)
    assert all(n.startswith("✓") for n in d.notes[:3])


# ---- validation --------------------------------------------------------------------


def test_validate_spec_ok_and_planted_errors():
    assert validate_spec(shop_spec()) == []
    spec = shop_spec()
    spec.tools.append(spec.tools[0])  # duplicate tool
    spec.tools.append(ToolDef(name="bad_set", description="", category="act",
        params=[ToolParam(name="order_id")],
        effect=SetFields(collection="orders", key_param="order_id",
                         set={"colour": "red", "status": "$nope"})))  # fmt: skip
    spec.tools.append(ToolDef(name="bad_get", description="", category="observe",
        params=[ToolParam(name="kind", type="enum")],
        effect=GetRecord(collection="ghosts", key_param="id")))  # fmt: skip
    spec.collections[1] = spec.collections[1].model_copy(update={"owner_field": "owner"})
    spec.policies.append(Policy(id="p", text="", check=ToolCalled(tool="teleport")))
    spec.policies.append(
        Policy(id="q", text="", check=NoRecord(collection="orders", where={"total__foo": 1}))
    )
    spec.tools = [t for t in spec.tools if t.name != "resolve"]
    problems = "\n".join(validate_spec(spec))
    for expected in ["duplicate tool 'get_order'", "orders has no field 'colour'",
                     "'nope' is not a parameter", "unknown collection 'ghosts'",
                     "enum parameter 'kind' has no enum", "owner_field 'owner' is not a field",
                     "unknown tool 'teleport'", "unknown operator '__foo'",
                     "no tool has a resolve effect"]:  # fmt: skip
        assert expected in problems, expected


def test_validate_spec_rejects_template_style_references():
    """ "{order_id}" would be stored as that literal text; only "$order_id" is filled in."""
    spec = shop_spec()
    for i, value in enumerate(["{order_id}", "{{order_id}}", "<order_id>", "${order_id}"]):
        spec.tools.append(ToolDef(name=f"tmpl_{i}", description="", category="act",
            params=[ToolParam(name="order_id")],
            effect=SetFields(collection="orders", key_param="order_id", set={"status": value})))  # fmt: skip
    problems = validate_spec(spec)
    assert sum("is literal text, not a reference" in p for p in problems) == 4
    assert all('write "$order_id"' in p for p in problems if "literal text" in p)
    # braces around a word that isn't a parameter are an ordinary literal
    ok = shop_spec()
    ok.tools.append(ToolDef(name="lit", description="", category="act",
        params=[ToolParam(name="order_id")],
        effect=SetFields(collection="orders", key_param="order_id", set={"status": "{shipped}"})))  # fmt: skip
    assert not any("literal text" in p for p in validate_spec(ok))


def test_validate_task_ok_and_planted_errors():
    spec = shop_spec()
    assert validate_task(spec, shop_task()) == []

    t = shop_task()
    t.state["orders"].append({"order_id": "O1", "customer_id": "C1", "status": "weird", "x": 1})
    t.state["boxes"] = []
    problems = "\n".join(validate_task(spec, t.model_copy(update={"customer_id": "C7"})))
    for expected in ["duplicate key 'O1'", "unknown field 'x'", "'weird' does not fit",
                     "unknown collection 'boxes'", "customer C7 is not in state.customers"]:  # fmt: skip
        assert expected in problems, expected

    bad_ref = shop_task(reference=[RefCall(tool="teleport"), RefCall(tool="get_order")])
    problems = "\n".join(validate_task(spec, bad_ref))
    assert "unknown tool 'teleport'" in problems and "missing required" in problems

    # well-formed, but the reference doesn't solve it: say which check fails
    weak = shop_task(id="t2", reference=shop_task().reference[:2] + shop_task().reference[3:])
    problems = validate_task(spec, weak)
    assert any("refunds has a record with order_id = O1" in p for p in problems)

    # solves it but breaks a policy
    rude = shop_task(reference=shop_task().reference[1:])
    assert any("ask-first" in p for p in validate_task(spec, rude))

    solved = shop_task(success=[ResolutionIs(category="refunded")], reference=[])
    assert any("no reference" in p for p in validate_task(spec, solved))
    already = shop_task(success=[FieldEquals(collection="orders", key="O3", field="status",
                                             value="shipped")])  # fmt: skip
    assert any("already solved" in p for p in validate_task(spec, already))

    summary = validate_dataset(spec, [shop_task(), weak])
    assert summary["ok"] == 1 and list(summary["problems"]) == ["t2"]


# ---- store and registry ------------------------------------------------------------


def test_store_round_trip(tmp_path):
    save_env_dir(tmp_path / "shop", shop_spec(), [shop_task()])
    spec, tasks = load_spec_and_tasks(tmp_path / "shop")
    assert spec == shop_spec() and tasks == [shop_task()]
    cls = get_pack(str(tmp_path / "shop"))
    assert cls.name == "shop" and load_env_dir(tmp_path / "shop", name="x@v1").name == "x@v1"


def test_registry_loads_example_dir():
    cls = get_pack(str(EXAMPLE))
    assert cls.name == "isp-support" and cls.tiers == (1, 2, 3)
    with pytest.raises(KeyError):
        get_pack("definitely-not-a-pack")


# ---- the isp_support example -------------------------------------------------------


def test_example_is_valid_and_sized():
    summary = validate_dataset(EX_SPEC, EX_TASKS)
    assert summary["spec_problems"] == [] and summary["problems"] == {}
    assert summary["ok"] == len(EX_TASKS) >= 30
    assert set(summary["tiers"]) == {1, 2, 3} and min(summary["tiers"].values()) >= 8
    assert 8 <= len(EX_SPEC.resolution_categories) <= 12


ISP = make_environment(EX_SPEC, EX_TASKS)


@pytest.mark.parametrize("record", EX_TASKS, ids=lambda r: r.id)
def test_example_task_broken_then_reference_passes(record):
    cls = make_environment(EX_SPEC, [record])
    env = cls(cls.generate_task(0, record.tier))
    env.reset()
    assert env.verify().reward == 0.0
    v = run_calls(env, env.reference_solve())
    assert (v.success, v.reward) == (True, 1.0), v.details
    assert not [n for n in env.diagram().nodes if n.state == "broken"]


def _isp_env(task_id: str):
    rec = next(r for r in EX_TASKS if r.id == task_id)
    cls = make_environment(EX_SPEC, [rec])
    env = cls(cls.generate_task(0, rec.tier))
    env.reset()
    return env


def test_example_policy_traps():
    env = _isp_env("21-duplicate-charge-over-limit")  # credit $90 yourself, then reference
    v = run_calls(env, [call("apply_credit", customer_id="C-1121", invoice_id="INV-5213",
                             amount=90, reason="dup"), *env.reference_solve()])  # fmt: skip
    assert v.reward == 0.0  # the task also forbids agent credits; policy flags it too
    assert "credit-limit" in v.details["policies_broken"]

    env = _isp_env("27-credit-at-limit")  # an over-limit credit plus the reference -> 0.3
    v = run_calls(env, [call("apply_credit", customer_id="C-1127", invoice_id="INV-5272",
                             amount=60, reason="oops"), *env.reference_solve()])  # fmt: skip
    assert v.reward == 0.3 and v.details["policies_broken"] == ["credit-limit"]

    env = _isp_env("22-reboot-demand-in-outage")
    v = run_calls(env, [call("reboot_device", device_id="D-3122"), *env.reference_solve()])
    assert v.reward == 0.3 and v.details["policies_broken"] == ["no-restart-in-outage"]

    env = _isp_env("23-flatmate-account")
    v = run_calls(env, [call("reboot_device", device_id="D-9002"), *env.reference_solve()])
    assert v.reward == 0.3 and v.details["policies_broken"] == ["own-account-only"]

    env = _isp_env("24-rushed-upgrade")
    v = run_calls(env, env.reference_solve()[1:])
    assert v.reward == 0.3 and v.details["policies_broken"] == ["verify-before-plan-change"]


def test_example_describe_and_explore():
    desc = ISP.describe()
    assert desc.name == "isp-support" and len(desc.tools) == len(EX_SPEC.tools)
    titles = [t.title for t in desc.tables]
    assert titles == ["Collections", "Policies", "Resolution categories", "Dataset"]
    assert [r.reward for r in desc.reward_rules] == [1.0, 0.3, 0.0]
    o = pack_overview(ISP)
    Diagram.model_validate(o["world_diagram"])
    assert len(o["world_diagram"]["nodes"]) <= 30
    for tier in ISP.tiers:
        p = preview_task(ISP, seed=7, tier=tier)
        assert p["task"]["tier"] == tier and p["first_observation"]
        assert p["verdict"]["reward"] == 1.0
        assert not [n for n in p["diagram_after"]["nodes"] if n["state"] == "broken"]


def test_example_replay_with_scripted_model():
    for tier in ISP.tiers:
        task = ISP.generate_task(seed=11, tier=tier)
        traj = run_episode(ISP, task, ScriptedModel())
        assert traj.verdict.reward == 1.0 and traj.finished_reason == "finish_tool"
        r = replay_episode(ISP, traj)
        n_calls = sum(1 for s in traj.steps for c in s.tool_calls if c.name != "finish")
        assert len(r["frames"]) == n_calls + 1
        assert r["verdict"]["reward"] == 1.0
        assert any(f["changed"] for f in r["frames"][1:])

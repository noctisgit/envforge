"""Pack-level tests: determinism, the contract, the verifier, the reference solution."""

import pytest

from envforge.core.env import ToolCall, ToolError
from envforge.packs.netops import PACK

SEEDS = range(25)


def test_task_generation_is_deterministic():
    a = PACK.generate_task(seed=7, tier=2)
    b = PACK.generate_task(seed=7, tier=2)
    assert a == b


def test_task_is_broken_initially():
    for seed in SEEDS:
        for tier in PACK.tiers:
            env = PACK(PACK.generate_task(seed, tier))
            env.reset()
            assert not env.verify().success, f"seed={seed} tier={tier} not broken"


@pytest.mark.parametrize("tier", PACK.tiers)
def test_reference_solution_passes_cleanly(tier):
    for seed in SEEDS:
        env = PACK(PACK.generate_task(seed, tier))
        env.reset()
        for call in env.reference_solve():
            env.step(call)
        v = env.verify()
        assert v.success and v.reward == 1.0, f"seed={seed}: {v.details}"


def test_partial_fix_fails():
    env = PACK(PACK.generate_task(seed=3, tier=3))
    env.reset()
    calls = env.reference_solve()
    for call in calls[:-1]:
        env.step(call)
    assert not env.verify().success


def test_removing_policy_rule_is_collateral():
    for seed in SEEDS:
        task = PACK.generate_task(seed, 2)
        if not task.hidden["decoy_rules"]:
            continue
        env = PACK(task)
        env.reset()
        for call in env.reference_solve():
            env.step(call)
        router, rid = task.hidden["decoy_rules"][0]
        env.step(ToolCall(id="x", name="remove_acl_rule", args={"router": router, "rule_id": rid}))
        v = env.verify()
        assert not v.success and v.reward == 0.3 and v.details["collateral"]
        return
    pytest.fail("no tier-2 task with a decoy rule found in seed range")


def test_breaking_bystanders_is_collateral():
    # pick a siteA -> DC task so the siteB -> DC path is a pure bystander
    task = next(
        t
        for t in (PACK.generate_task(s, 1) for s in SEEDS)
        if t.hidden["src_host"].endswith("a1") or t.hidden["src_host"].endswith("a2")
        if t.hidden["dst_host"].endswith("d1")
    )
    env = PACK(task)
    env.reset()
    for call in env.reference_solve():
        env.step(call)
    assert env.verify().success
    # now "fix" something unrelated badly: cut siteB off from the DC
    env.step(ToolCall(id="x", name="delete_route", args={"router": "R3", "prefix": "10.4.0.0/24"}))
    v = env.verify()
    assert v.details["target_reachable"]
    assert v.details["collateral"] and v.reward == 0.3, v.details


def test_tool_errors_are_recoverable():
    env = PACK(PACK.generate_task(seed=0, tier=1))
    env.reset()
    with pytest.raises(ToolError):
        env.step(ToolCall(id="1", name="show_routes", args={"router": "R9"}))
    with pytest.raises(ToolError):
        env.step(
            ToolCall(
                id="2", name="set_route", args={"router": "R1", "prefix": "nope", "next_hop": "R2"}
            )
        )
    with pytest.raises(ToolError):
        env.step(ToolCall(id="3", name="remove_acl_rule", args={"router": "R1", "rule_id": 999}))
    # environment still usable afterwards
    assert "R1" in env.step(ToolCall(id="4", name="show_routes", args={"router": "R1"})).text


def test_tool_specs_are_valid_json_schema_shapes():
    env = PACK(PACK.generate_task(seed=0, tier=1))
    env.reset()
    names = set()
    for t in env.tools():
        assert t.name not in names
        names.add(t.name)
        assert t.parameters["type"] == "object"
        assert t.description

"""Pack-level tests: determinism, the contract, the verifier, the reference solution."""

import pytest

from envforge.core.env import ToolCall, ToolError
from envforge.packs.sipops import PACK

SEEDS = range(25)


def _solved(task):
    """An environment for `task` with the reference fix applied."""
    env = PACK(task)
    env.reset()
    for call in env.reference_solve():
        env.step(call)
    return env


def _call(tool, **args):
    return ToolCall(id="x", name=tool, args=args)


def _find(tier, pred):
    for seed in range(200):
        task = PACK.generate_task(seed, tier)
        if pred(task):
            return task
    pytest.fail(f"no tier-{tier} task matching predicate in seed range")


def _has_fault(prefix):
    return lambda t: len(t.hidden["faults"]) == 1 and t.hidden["faults"][0].startswith(prefix)


def test_task_generation_is_deterministic():
    a = PACK.generate_task(seed=7, tier=3)
    b = PACK.generate_task(seed=7, tier=3)
    assert a == b


def test_same_seed_same_outcome():
    task = PACK.generate_task(seed=11, tier=3)
    first, second = _solved(task), _solved(task)
    assert first.verify() == second.verify()
    assert first.snapshot() == second.snapshot()


def test_task_is_broken_initially():
    for seed in SEEDS:
        for tier in PACK.tiers:
            env = PACK(PACK.generate_task(seed, tier))
            env.reset()
            assert not env.verify().success, f"seed={seed} tier={tier} not broken"


@pytest.mark.parametrize("tier", PACK.tiers)
def test_reference_solution_passes_cleanly(tier):
    for seed in SEEDS:
        v = _solved(PACK.generate_task(seed, tier)).verify()
        assert v.success and v.reward == 1.0, f"seed={seed}: {v.details}"


@pytest.mark.parametrize("tier", [2, 3])
def test_partial_fix_fails(tier):
    """Every fault matters: skipping any one fix leaves the target call broken."""
    for seed in SEEDS:
        task = PACK.generate_task(seed, tier)
        env = PACK(task)
        env.reset()
        calls = env.reference_solve()
        for skip in range(len(calls)):
            env = PACK(task)
            env.reset()
            for i, call in enumerate(calls):
                if i != skip:
                    env.step(call)
            v = env.verify()
            assert not v.success and v.reward == 0.0, f"seed={seed} skip={skip}"


def test_tier3_always_has_a_return_path_fault():
    for seed in SEEDS:
        faults = PACK.generate_task(seed, 3).hidden["faults"]
        assert len(faults) == 3 and any("RETURN PATH" in f for f in faults)


@pytest.mark.parametrize("tier", [2, 3])
def test_removing_decoy_policy_rule_is_collateral(tier):
    for seed in SEEDS:
        task = PACK.generate_task(seed, tier)
        for table, rid in task.hidden["decoy_rules"]:
            env = _solved(task)
            env.step(_call("remove_rule", table=table, rule_id=rid))
            v = env.verify()
            assert not v.success and v.reward == 0.3, v.details
            assert any("policy violated" in c for c in v.details["collateral"])
            assert any("removed POLICY" in c for c in v.details["collateral"])


def test_weakening_policy_rule_is_collateral():
    env = _solved(PACK.generate_task(seed=0, tier=1))
    env.step(_call("set_rule", table="pbx-hq", rule_id=3, pattern=r"91900\d{8}"))
    v = env.verify()
    assert v.reward == 0.3 and any("modified POLICY rule 3" in c for c in v.details["collateral"])


def test_reordering_past_policy_is_collateral():
    """Fixing a shadowed rule by jumping it above everything bypasses the premium block."""
    task = _find(2, lambda t: any("catch-all" in f for f in t.hidden["faults"]))
    env = PACK(task)
    env.reset()
    fixes = env.reference_solve()
    shadow_fix = next(c for c in fixes if "priority" in c.args)
    for call in fixes:
        if call is not shadow_fix:
            env.step(call)
    # the national/international rule of the caller's PBX, moved to the very top
    table = shadow_fix.args["table"]
    dialed = env._scn.forward.dialed
    rule = next(r for r in env.platform.rules(table) if r.matches(dialed) and r.action == "route")
    env.step(_call("set_rule", table=table, rule_id=rule.id, priority=1))
    v = env.verify()
    assert v.details["target_completes"]
    assert v.reward == 0.3 and any("policy violated" in c for c in v.details["collateral"]), (
        v.details
    )


def test_breaking_bystanders_is_collateral():
    # tier-1 targets are always outside numbers, so HQ <-> branch calls are bystanders
    env = _solved(PACK.generate_task(seed=0, tier=1))
    assert env.verify().success
    env.step(_call("set_trunk", trunk="intersite", enabled=False))
    v = env.verify()
    assert v.details["target_completes"]
    assert v.reward == 0.3 and any("broke" in c for c in v.details["collateral"]), v.details


def test_breaking_emergency_calls_is_collateral():
    env = _solved(PACK.generate_task(seed=1, tier=1))
    env.step(_call("set_trunk", trunk="emergency", enabled=False))
    v = env.verify()
    assert v.reward == 0.3 and any("emergency" in c for c in v.details["collateral"])


def test_alternative_codec_fix_is_accepted():
    task = _find(1, _has_fault("trunk carrier-a codecs"))
    env = PACK(task)
    env.reset()
    env.step(_call("set_trunk", trunk="carrier-a", codecs=["PCMA", "PCMU", "G729"]))
    assert env.verify().reward == 1.0


def test_alternative_dns_fix_is_accepted():
    """An SRV pointing at a dead name can be fixed by pointing it back, or by giving the name an A record."""
    task = _find(1, _has_fault("DNS SRV"))
    env = PACK(task)
    env.reset()
    old_host = task.hidden["faults"][0].split(" points to ")[1].split()[0]
    good_ip = env.platform.dns_a[env.reference_solve()[0].args["value"].split(":")[0]]
    env.step(_call("set_dns", record_type="A", name=old_host, value=good_ip))
    assert env.verify().reward == 1.0


def test_codec_fix_that_strands_a_phone_is_collateral():
    """PCMA-only on carrier-a still fixes most calls but breaks the PCMU-only door phone inbound."""
    task = _find(1, _has_fault("trunk carrier-a codecs"))
    env = PACK(task)
    env.reset()
    env.step(_call("set_trunk", trunk="carrier-a", codecs=["PCMA"]))
    v = env.verify()
    assert v.details["target_completes"] and v.reward == 0.3
    assert any("+14155550190" in c for c in v.details["collateral"]), v.details


def test_goal_does_not_leak_hidden_faults():
    for seed in SEEDS:
        task = PACK.generate_task(seed, 3)
        for fault in task.hidden["faults"]:
            assert fault not in task.goal


def test_tool_errors_are_recoverable():
    env = PACK(PACK.generate_task(seed=0, tier=1))
    env.reset()
    bad_calls = [
        _call("show_dialplan", table="pbx-xx"),
        _call("sip_trace", src="nobody", dst="101"),
        _call("set_rule", table="pbx-hq", rule_id=7, pattern="91[2-9"),
        _call("set_rule", table="pbx-hq", rule_id=7, target="carrier-z"),
        _call("set_rule", table="pbx-hq", rule_id=7),
        _call("remove_rule", table="pbx-hq", rule_id=999),
        _call("add_rule", table="pbx-hq", rule_id=7, priority=1, pattern="1", action="reject"),
        _call("set_trunk", trunk="carrier-a", auth_ref="cred-nope"),
        _call("set_trunk", trunk="carrier-a", codecs=["AMR"]),
        _call("set_dns", record_type="A", name="x", value="not-an-ip"),
        _call("reboot_phone", extension="999"),
        _call("set_trunk"),
        _call("no_such_tool"),
    ]
    for call in bad_calls:
        with pytest.raises(ToolError):
            env.step(call)
    # environment still usable afterwards
    assert "pbx-hq" in env.step(_call("show_dialplan", table="pbx-hq")).text


def test_observation_tools_return_text():
    env = PACK(PACK.generate_task(seed=4, tier=3))
    assert "trunks" in env.reset().text
    for call in [
        _call("show_trunks"),
        _call("show_dns"),
        _call("show_credentials"),
        _call("show_extensions", pbx="pbx-br"),
        _call("show_registrations", pbx="pbx-hq"),
        _call("show_dialplan", table="sbc1"),
        _call("sip_trace", src="+12125550147", dst="+14155550190"),
    ]:
        assert env.step(call).text.strip()


def test_tool_specs_are_valid_json_schema_shapes():
    env = PACK(PACK.generate_task(seed=0, tier=1))
    env.reset()
    names = set()
    for t in env.tools():
        assert t.name not in names
        names.add(t.name)
        assert t.parameters["type"] == "object"
        assert t.description

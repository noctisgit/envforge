"""sipops self-description: describe(), the live diagram, previews and replays."""

import pytest

from envforge.core.env import Diagram, EnvDescription
from envforge.core.explore import preview_task, replay_episode
from envforge.core.runner import run_episode
from envforge.models.scripted import ScriptedModel
from envforge.packs.sipops import PACK

SEEDS = range(12)


def _broken(d: Diagram) -> list[str]:
    return [n.id for n in d.nodes if n.state == "broken"] + [
        f"{e.source}-{e.target}" for e in d.edges if e.state == "broken"
    ]


def _targets(task) -> set[str]:
    """The goal's endpoints: the caller's extension and the far end (number or extension)."""
    return {task.hidden["forward"].split(" -> ")[0], task.hidden["back"].split(" -> ")[0]}


def test_describe_validates_and_every_tool_has_a_category():
    desc = PACK.describe()
    EnvDescription.model_validate(desc.model_dump())
    assert desc.overview and desc.world and desc.observation and desc.action
    assert set(desc.tier_notes) == set(PACK.tiers)
    assert sorted(r.reward for r in desc.reward_rules) == [0.0, 0.3, 1.0]
    titles = [t.title for t in desc.tables]
    assert "Fault catalogue" in titles and "Protected policy" in titles
    for table in desc.tables:
        assert all(len(row) == len(table.columns) for row in table.rows), table.title
    assert all(t.category in ("observe", "act") for t in desc.tools)
    cats = {t.name: t.category for t in desc.tools}
    assert cats["sip_trace"] == "observe" and cats["show_dns"] == "observe"
    assert cats["set_rule"] == "act" and cats["reboot_phone"] == "act"


def test_no_diagram_before_reset():
    assert PACK(PACK.generate_task(0, 1)).diagram() is None


@pytest.mark.parametrize("tier", PACK.tiers)
def test_diagram_shows_the_fault_then_clears_after_the_fix(tier):
    for seed in SEEDS:
        task = PACK.generate_task(seed, tier)
        env = PACK(task)
        env.reset()
        d = env.diagram()
        assert d is not None and d.layout == "graph"
        assert _broken(d), f"seed={seed}: nothing broken before the fix"
        states = {n.id: n.state for n in d.nodes}
        assert all(states.get(t) == "target" for t in _targets(task)), (seed, states)
        assert all(0 <= n.x <= 1 and 0 <= n.y <= 1 for n in d.nodes)
        ids = {n.id for n in d.nodes}
        assert {"190", "299", "sbc1", "carrier-a", "carrier-b", "emergency"} <= ids
        assert all(e.source in ids and e.target in ids for e in d.edges)
        assert len(d.notes) == 2 and not all(n.endswith(": OK") for n in d.notes)

        for call in env.reference_solve():
            env.step(call)
        after = env.diagram()
        assert not _broken(after), f"seed={seed}: still broken {_broken(after)}"
        assert all(n.endswith(": OK") for n in after.notes), after.notes
        assert any(e.state == "path" for e in after.edges)
        assert any("changed" in n.badges for n in after.nodes)


def test_diagram_is_side_effect_free():
    for tier in PACK.tiers:
        env = PACK(PACK.generate_task(3, tier))
        env.reset()
        before, snap = env.verify(), env.snapshot()
        first = env.diagram()
        assert env.verify() == before and env.snapshot() == snap
        assert env.diagram() == first  # deterministic


def test_diagram_follows_state_changes():
    env = PACK(PACK.generate_task(3, 1))  # stale carrier-a A record: forward 408, inbound 403
    env.reset()
    assert {"dns", "carrier-a"} <= set(_broken(env.diagram()))
    for call in env.reference_solve():
        env.step(call)
    d = env.diagram()
    dns = next(n for n in d.nodes if n.id == "dns")
    assert dns.state == "ok" and "changed" in dns.badges


def test_policy_badges():
    env = PACK(PACK.generate_task(0, 3))
    env.reset()
    badges = {n.id: n.badges for n in env.diagram().nodes}
    assert "policy" in badges["pbx-hq"] and "policy" in badges["pbx-br"]
    assert "policy" in badges["sbc1"]  # tier 3: the door-phone rule lives on sbc1
    assert "policy" in badges["emergency"]


def test_preview_task():
    p = preview_task(PACK, 5, 3)
    Diagram.model_validate(p["diagram_before"])
    assert p["verdict"]["success"] is True
    assert all(n.endswith(": OK") for n in p["diagram_after"]["notes"])


@pytest.mark.parametrize("tier", PACK.tiers)
def test_replay_matches_the_episode(tier):
    for seed in (1, 4):
        task = PACK.generate_task(seed, tier)
        traj = run_episode(PACK, task, ScriptedModel())
        r = replay_episode(PACK, traj)
        assert r["verdict"] == traj.verdict.model_dump()
        assert _broken(Diagram.model_validate(r["frames"][0]["diagram"]))
        last = Diagram.model_validate(r["frames"][-1]["diagram"])
        assert not _broken(last) and all(n.endswith(": OK") for n in last.notes)

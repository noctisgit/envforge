"""Generic checks for envforge.core.explore, run against every installed pack.

New packs get these for free: if a pack's describe()/diagram() hooks are wrong (or a
sim isn't deterministic enough to replay), these fail.
"""

from __future__ import annotations

import pytest

from envforge.core.env import Diagram, EnvDescription
from envforge.core.explore import NotReplayable, pack_overview, preview_task, replay_episode
from envforge.core.registry import discover_packs
from envforge.core.runner import run_episode
from envforge.core.trajectory import Trajectory
from envforge.models.scripted import ScriptedModel

PACKS = sorted(discover_packs().items())


@pytest.mark.parametrize(("name", "env_cls"), PACKS)
def test_describe_and_overview(name, env_cls):
    desc = env_cls.describe()
    assert isinstance(desc, EnvDescription) and desc.name == name and desc.tools
    o = pack_overview(env_cls)
    assert o["dimensions"][0][0] == "observation"
    if o["world_diagram"] is not None:
        Diagram.model_validate(o["world_diagram"])


@pytest.mark.parametrize(("name", "env_cls"), PACKS)
def test_preview_every_tier(name, env_cls):
    for tier in env_cls.tiers:
        p = preview_task(env_cls, seed=7, tier=tier)
        assert p["task"]["tier"] == tier and p["system_prompt"]
        if env_cls.kind == "sim":
            assert p["first_observation"] and p["solution"]
            assert p["verdict"]["success"] is True  # reference solution solves it


@pytest.mark.parametrize(("name", "env_cls"), [p for p in PACKS if p[1].kind == "sim"])
def test_replay_reproduces_the_recorded_outcome(name, env_cls):
    for tier in env_cls.tiers:
        task = env_cls.generate_task(seed=11, tier=tier)
        traj = run_episode(env_cls, task, ScriptedModel())
        try:
            r = replay_episode(env_cls, traj)
        except NotReplayable:
            pytest.skip(f"{name} draws no diagram")
        n_calls = sum(1 for s in traj.steps for c in s.tool_calls if c.name != "finish")
        assert len(r["frames"]) == n_calls + 1
        assert r["verdict"]["reward"] == traj.verdict.reward
        assert any(f["changed"] for f in r["frames"][1:])  # the fix shows up in the picture


def test_sandbox_packs_are_not_replayed():
    sandbox = [c for _, c in PACKS if c.kind == "sandbox"]
    if not sandbox:
        pytest.skip("no sandbox pack installed")
    env_cls = sandbox[0]
    task = env_cls.generate_task(0, env_cls.tiers[0])
    traj = Trajectory(run_id="x", episode=0, model="scripted", task=task, system_prompt="")
    with pytest.raises(NotReplayable):
        replay_episode(env_cls, traj)

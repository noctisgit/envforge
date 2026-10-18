"""Core-level tests: runner, trajectory round-trip, evaluation, filtering."""

from envforge.core.evaluate import compute_metrics, evaluate_run, filter_trajectories
from envforge.core.registry import discover_packs, get_pack
from envforge.core.runner import run_episode, run_many
from envforge.core.trajectory import Trajectory, load_run
from envforge.models import load_model
from envforge.packs.netops import PACK


def test_registry_finds_netops():
    assert "netops" in discover_packs()
    assert get_pack("netops") is PACK


def test_run_episode_with_scripted_model():
    task = PACK.generate_task(seed=5, tier=2)
    traj = run_episode(PACK, task, load_model("scripted"))
    assert traj.success and traj.reward == 1.0
    assert traj.finished_reason == "finish_tool"
    assert traj.n_tool_errors == 0
    assert traj.steps[-1].tool_calls[-1].name == "finish"
    assert traj.system_prompt


def test_unknown_tool_is_reported_not_raised():
    from envforge.core.env import ToolCall
    from envforge.models.base import AssistantTurn, ChatModel

    class Bad(ChatModel):
        model_id = "bad"

        def complete(self, messages, tools):
            return AssistantTurn(tool_calls=[ToolCall(id="1", name="rm_rf", args={})])

    traj = run_episode(PACK, PACK.generate_task(0, 1), Bad(), max_steps=2)
    assert traj.finished_reason == "max_steps"
    assert traj.n_tool_errors == 2
    assert traj.verdict is not None and not traj.success


def test_run_many_roundtrip_and_eval(tmp_path):
    run_dir = run_many(
        PACK, load_model("scripted:error_every=3"), runs_dir=tmp_path, episodes=9, tiers=[1, 2, 3]
    )
    trajs = load_run(run_dir)
    assert len(trajs) == 9
    assert all(isinstance(t, Trajectory) for t in trajs)
    # JSON round-trip preserved everything
    assert trajs[0].task.seed == 0 and trajs[8].task.seed == 8

    report = evaluate_run(run_dir)
    assert report.overall.n == 9
    assert 0 < report.overall.solve_rate < 1  # sabotaged episodes failed
    assert set(report.by_tier) == {1, 2, 3}
    assert report.meta["pack"] == "netops"


def test_filter_keeps_only_clean_successes(tmp_path):
    run_dir = run_many(
        PACK, load_model("scripted:error_every=2"), runs_dir=tmp_path, episodes=8, tiers=[1]
    )
    trajs = load_run(run_dir)
    kept = filter_trajectories(trajs, min_reward=1.0)
    assert 0 < len(kept) < len(trajs)
    assert all(t.reward == 1.0 for t in kept)


def test_metrics_empty():
    m = compute_metrics([])
    assert m.n == 0 and m.solve_rate == 0.0

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


def test_run_many_repeats_give_each_task_a_group_of_attempts(tmp_path):
    run_dir = run_many(PACK, load_model("scripted"), runs_dir=tmp_path, episodes=6,
                       tiers=[1, 2], seed_start=10_000, repeats=3)  # fmt: skip
    trajs = load_run(run_dir)
    assert [t.episode for t in trajs] == list(range(6))
    assert [(t.task.seed, t.task.tier) for t in trajs] == [(10_000, 1)] * 3 + [(10_001, 2)] * 3


def test_run_many_concurrency_keeps_episode_order(tmp_path):
    import random
    import threading
    import time

    from envforge.models.base import ChatModel

    local = threading.local()  # the scripted model keeps per-episode state: one per thread
    seen_threads: set[str] = set()

    class Threaded(ChatModel):
        """Scripted answers with random delays, so episodes finish out of order."""

        model_id = "threaded"

        def _inner(self):
            if not hasattr(local, "model"):
                local.model = load_model("scripted")
            return local.model

        def on_reset(self, env):
            self._inner().on_reset(env)

        def complete(self, messages, tools):
            seen_threads.add(threading.current_thread().name)
            time.sleep(random.random() * 0.02)
            return self._inner().complete(messages, tools)

    order: list[int] = []
    run_dir = run_many(PACK, Threaded(), runs_dir=tmp_path, episodes=8, tiers=[1, 2, 3],
                       concurrency=4, progress=lambda t: order.append(t.episode))  # fmt: skip
    trajs = load_run(run_dir)
    assert [t.episode for t in trajs] == order == list(range(8))
    assert all(t.success for t in trajs) and len(seen_threads) > 1


def test_run_many_stops_when_progress_raises(tmp_path):
    import pytest

    class Stop(Exception):
        pass

    def progress(t):
        if t.episode == 2:
            raise Stop

    with pytest.raises(Stop):
        run_many(PACK, load_model("scripted"), runs_dir=tmp_path, episodes=20, tiers=[1],
                 concurrency=4, progress=progress, run_id="stopped")  # fmt: skip
    assert len(load_run(tmp_path / "stopped")) == 3  # nothing written after the stop


def test_a_stopped_run_leaves_no_episode_calling_the_model(tmp_path):
    """Cancelling a concurrent run must stop the episodes already in flight, not only
    the queued ones (audit finding C2): after run_many returns, no more model calls."""
    import threading
    import time

    import pytest

    from envforge.core.env import ToolCall
    from envforge.models.base import AssistantTurn, ChatModel

    calls: list[float] = []
    first_goal = PACK.generate_task(seed=0, tier=1).goal

    class Slow(ChatModel):
        """Episode 0 finishes at once (so the run stops); the others keep working."""

        model_id = "slow"

        def complete(self, messages, tools):
            if first_goal in messages[1].content:
                return AssistantTurn(
                    tool_calls=[ToolCall(id="f", name="finish", args={"summary": "done"})]
                )
            calls.append(time.monotonic())
            time.sleep(0.05)
            return AssistantTurn(tool_calls=[ToolCall(id="1", name="list_devices", args={})])

    class Stop(Exception):
        pass

    def progress(t):
        raise Stop

    with pytest.raises(Stop):
        run_many(PACK, Slow(), runs_dir=tmp_path, episodes=16, tiers=[1], concurrency=8,
                 progress=progress, run_id="cancelled")  # fmt: skip
    returned, n = time.monotonic(), len(calls)
    time.sleep(0.3)
    assert len(calls) == n  # nothing kept calling the model after the run stopped
    assert not [t for t in threading.enumerate() if t.name.startswith("episode")]
    assert all(c <= returned for c in calls)

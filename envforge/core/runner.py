"""The agent loop: model <-> environment through tool calls.

One `run_episode()` call produces one `Trajectory`. `run_many()` produces a run
directory full of them. This is the "rollout" in RL vocabulary: the policy
(the model) acts in the environment until it finishes or runs out of budget,
then the verifier scores the outcome.

The loop injects one universal tool, `finish`, so every pack ends episodes the
same way. The model may also end an episode by replying with no tool calls.
"""

from __future__ import annotations

import re
import threading
import time
import uuid
from collections.abc import Callable, Iterable
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from envforge.core.env import Environment, Task, ToolError, ToolSpec
from envforge.core.trajectory import Step, Trajectory, TrajectoryStore, Usage
from envforge.models.base import ChatModel, Message, ModelUnreachable

FINISH_TOOL = ToolSpec(
    name="finish",
    description=(
        "Call this when you believe the task is complete (or cannot be completed). "
        "Give a one-sentence summary of what you changed and why."
    ),
    parameters={
        "type": "object",
        "properties": {"summary": {"type": "string"}},
        "required": ["summary"],
    },
)


def _observation_text(obs_text: str, limit: int = 4000) -> str:
    """Keep tool output bounded so long observations don't blow the context."""
    if len(obs_text) <= limit:
        return obs_text
    return obs_text[:limit] + f"\n... [truncated {len(obs_text) - limit} chars]"


def run_episode(
    env_cls: type[Environment],
    task: Task,
    model: ChatModel,
    *,
    run_id: str = "adhoc",
    episode: int = 0,
    max_steps: int | None = None,
    on_step: Callable[[Step], None] | None = None,
    stop: threading.Event | None = None,
) -> Trajectory:
    """Run one episode and return its trajectory. Never raises for model/tool errors;
    they are recorded in the trajectory instead, so a bad episode doesn't kill a run.

    The one exception is `ModelUnreachable`: when the model's server can't be reached,
    the episode measured nothing, so it is not recorded (it would count as a failure);
    the error propagates and stops the run.

    `stop` (set by run_many when a run is cancelled) ends the episode before its next
    model call, with finished_reason "stopped"."""
    env = env_cls(task)
    budget = max_steps or env_cls.max_steps
    t0 = time.perf_counter()

    traj = Trajectory(
        run_id=run_id,
        episode=episode,
        model=model.model_id,
        task=task,
        system_prompt="",
    )
    total_usage = Usage()  # outside the try: tokens spent before an error still count

    try:
        first_obs = env.reset()
        # kept so the exact opening prompt can be rebuilt for training data (train/filter.py)
        traj.meta["first_observation"] = first_obs.text
        model.on_reset(env)
        system_prompt = env.system_prompt()
        traj.system_prompt = system_prompt
        tools = env.tools() + [FINISH_TOOL]
        tool_names = {t.name for t in tools}

        messages: list[Message] = [
            Message(role="system", content=system_prompt),
            Message(
                role="user",
                content=f"TASK\n{task.goal}\n\nINITIAL STATE\n{first_obs.text}",
            ),
        ]

        for i in range(budget):
            if stop is not None and stop.is_set():
                traj.finished_reason = "stopped"
                break
            t_call = time.perf_counter()
            turn = model.complete(messages, tools)
            latency = time.perf_counter() - t_call
            total_usage = total_usage + turn.usage

            step = Step(index=i, assistant_text=turn.text, usage=turn.usage, latency_s=latency)
            messages.append(
                Message(role="assistant", content=turn.text, tool_calls=turn.tool_calls)
            )

            if not turn.tool_calls:
                traj.steps.append(step)
                traj.finished_reason = "no_tool_calls"
                break

            finished = False
            for call in turn.tool_calls:
                if call.name == "finish":
                    obs_text = "Episode finished."
                    is_err = False
                    finished = True
                elif call.name not in tool_names:
                    obs_text = (
                        f"ToolError: unknown tool '{call.name}'. Available: {sorted(tool_names)}"
                    )
                    is_err = True
                else:
                    try:
                        obs_text = env.step(call).text
                        is_err = False
                    except ToolError as e:
                        obs_text = f"ToolError: {e}"
                        is_err = True
                obs_text = _observation_text(obs_text)
                step.tool_calls.append(call)
                step.observations.append(obs_text)
                step.errors.append(is_err)
                messages.append(
                    Message(role="tool", content=obs_text, tool_call_id=call.id, name=call.name)
                )

            traj.steps.append(step)
            if on_step:
                on_step(step)
            if finished:
                traj.finished_reason = "finish_tool"
                break
        else:
            traj.finished_reason = "max_steps"

        traj.verdict = env.verify()
        traj.usage = total_usage
        traj.meta["final_snapshot"] = env.snapshot()

    except ModelUnreachable:
        raise  # not the model's fault: stop the run rather than score this episode
    except Exception as e:  # noqa: BLE001 - record and continue the run
        traj.finished_reason = "error"
        traj.error = f"{type(e).__name__}: {e}"
        traj.usage = total_usage
        try:
            traj.verdict = env.verify()
        except Exception:  # noqa: BLE001
            traj.verdict = None
    finally:
        env.close()

    traj.wall_time_s = time.perf_counter() - t0
    return traj


def new_run_id(pack: str, model_id: str) -> str:
    """`<stamp>-<pack>-<model>-<6 hex>`; it names the run directory, so the model part is
    reduced to letters, digits, '.', '_' and '-' (a user-supplied model string may contain
    anything)."""
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    safe_model = re.sub(r"[^A-Za-z0-9._-]+", "-", model_id.replace(":", "_"))
    return f"{stamp}-{pack}-{safe_model}-{uuid.uuid4().hex[:6]}"


def run_many(
    env_cls: type[Environment],
    model: ChatModel,
    *,
    runs_dir: Path,
    episodes: int,
    tiers: Iterable[int],
    seed_start: int = 0,
    max_steps: int | None = None,
    run_id: str | None = None,
    progress: Callable[[Trajectory], None] | None = None,
    extra_meta: dict[str, Any] | None = None,
    repeats: int = 1,
    concurrency: int = 1,
) -> Path:
    """Run `episodes` episodes, cycling through `tiers`, and write them to a run dir.

    Seeds are `seed_start + episode`, so a run is fully reproducible for a
    deterministic model (and the *tasks* are reproducible for any model).

    `repeats=k` gives every task k attempts in a row (episodes 0..k-1 share a seed, then
    the next task): the *groups* GRPO compares, sampled at temperature > 0.
    `concurrency=n` runs up to n episodes at once in threads, for servers that batch
    requests (vLLM). Trajectories are still written, and `progress` still called, in
    episode order, from this thread.

    If the model's server stays unreachable, `ModelUnreachable` is raised; the episodes
    written so far stay in the run dir.
    """
    if repeats < 1 or concurrency < 1:
        raise ValueError("repeats and concurrency must be at least 1")
    tiers = list(tiers)
    rid = run_id or new_run_id(env_cls.name, model.model_id)
    run_dir = runs_dir / rid
    store = TrajectoryStore(run_dir)
    store.write_meta(
        {
            "run_id": rid,
            "pack": env_cls.name,
            "model": model.model_id,
            "episodes": episodes,
            "tiers": tiers,
            "seed_start": seed_start,
            "max_steps": max_steps or env_cls.max_steps,
            "started_at": datetime.now(UTC).isoformat(),
            **(extra_meta or {}),
        }
    )
    stop = threading.Event()  # set when the run stops early: episodes in flight end too

    def one(ep: int) -> Trajectory:
        i = ep // repeats  # the task index: `repeats` episodes in a row share it
        task = env_cls.generate_task(seed=seed_start + i, tier=tiers[i % len(tiers)])
        return run_episode(env_cls, task, model, run_id=rid, episode=ep, max_steps=max_steps,
                           stop=stop)  # fmt: skip

    def record(traj: Trajectory) -> None:
        store.append(traj)
        if progress:
            progress(traj)  # may raise (e.g. a cancelled run): stops the run here

    if concurrency == 1:
        for ep in range(episodes):
            record(one(ep))
        return run_dir

    # Up to `concurrency` episodes in flight; finished ones wait in `done` until every
    # earlier episode is written, so the file stays in episode order.
    pool = ThreadPoolExecutor(max_workers=concurrency, thread_name_prefix="episode")
    try:
        futures: dict[int, Future[Trajectory]] = {}
        next_submit = next_write = 0
        while next_write < episodes:
            while next_submit < episodes and len(futures) < concurrency:
                futures[next_submit] = pool.submit(one, next_submit)
                next_submit += 1
            record(futures.pop(next_write).result())  # waits for the oldest in flight
            next_write += 1
    finally:
        # On an error (a cancelled run, an unreachable model): drop what's queued, tell the episodes in flight
        # to stop before their next model call, and wait for them, so nothing keeps
        # calling the model after this returns.
        stop.set()
        pool.shutdown(wait=True, cancel_futures=True)
    return run_dir

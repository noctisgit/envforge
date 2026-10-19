"""/api/workspaces/{ws}/runs — enqueue, list, inspect, cancel; read and export trajectories."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse
from sqlalchemy import select
from sqlalchemy.orm import Session

from envforge.core.trajectory import TrajectoryStore
from envforge.train.filter import dataset_lines, filter_trajectories, tool_schemas
from envforge.web import services as svc
from envforge.web.auth import current_user
from envforge.web.builder import resolve_pack
from envforge.web.db import get_db
from envforge.web.models import Run, User
from envforge.web.schemas import RunIn, RunOut

router = APIRouter(prefix="/api/workspaces/{ws_id}/runs", tags=["runs"])


def _out(r: Run) -> RunOut:
    return RunOut.model_validate(r, from_attributes=True)


def _get_run(db: Session, ws_id: str, run_id: str) -> Run:
    r = db.get(Run, run_id)
    if not r or r.workspace_id != ws_id or r.deleted_at is not None:
        raise HTTPException(404, "run not found")
    return r


@router.post("", response_model=RunOut, status_code=202)
def create_run(
    ws_id: str, body: RunIn, user: User = Depends(current_user), db: Session = Depends(get_db)
):
    ws = svc.require_workspace(db, user, ws_id)
    run = svc.enqueue_run(
        db,
        ws,
        user,
        pack=body.pack or ws.default_pack,
        model=body.model or ws.default_model,
        episodes=body.episodes or ws.default_episodes,
        tiers=body.tiers or [],
        seed_start=body.seed_start,
        max_steps=body.max_steps,
        temperature=body.temperature,
        label=body.label,
    )
    return _out(run)


@router.get("", response_model=list[RunOut])
def list_runs(
    ws_id: str,
    limit: int = Query(50, le=500),
    user: User = Depends(current_user),
    db: Session = Depends(get_db),
):
    svc.require_workspace(db, user, ws_id)
    rows = db.scalars(
        select(Run)
        .where(Run.workspace_id == ws_id, Run.deleted_at.is_(None))
        .order_by(Run.created_at.desc())
        .limit(limit)
    )
    return [_out(r) for r in rows]


@router.get("/export")
def export_dataset(
    ws_id: str,
    runs: str = Query(..., description="comma-separated run ids from this workspace"),
    min_reward: float = Query(1.0, ge=0, le=1),
    fmt: Literal["chat", "trajectory"] = Query("chat", alias="format"),
    user: User = Depends(current_user),
    db: Session = Depends(get_db),
):
    """Rejection sampling as a download: the episodes from these runs that scored at
    least `min_reward`, as JSONL — chat format for fine-tuning, or raw trajectories."""
    ws = svc.require_workspace(db, user, ws_id)
    run_ids = [r for r in runs.split(",") if r][:50]
    rows = [_get_run(db, ws_id, rid) for rid in run_ids]  # 404s on anything not ours
    kept, tools_by_pack = [], {}
    for r in rows:
        if not r.run_dir:
            continue
        trajs = list(TrajectoryStore(Path(r.run_dir)).read())
        kept.extend(filter_trajectories(trajs, min_reward=min_reward))
        if fmt == "chat" and r.pack not in tools_by_pack:
            tools_by_pack[r.pack] = tool_schemas(resolve_pack(db, ws, r.pack))
    # one pack → embed its tool schemas in every example; mixed packs → per-example
    packs = {r.pack for r in rows}
    tools = tools_by_pack.get(next(iter(packs))) if len(packs) == 1 else None
    name = f"envforge-{fmt}-{len(kept)}ep.jsonl"
    return StreamingResponse(
        dataset_lines(kept, fmt=fmt, tools=tools),
        media_type="application/x-ndjson",
        headers={
            "Content-Disposition": f'attachment; filename="{name}"',
            "X-Episodes": str(len(kept)),
        },
    )


@router.get("/{run_id}", response_model=RunOut)
def get_run(
    ws_id: str, run_id: str, user: User = Depends(current_user), db: Session = Depends(get_db)
):
    svc.require_workspace(db, user, ws_id)
    return _out(_get_run(db, ws_id, run_id))


@router.delete("/{run_id}", status_code=204)
def delete_run(
    ws_id: str, run_id: str, user: User = Depends(current_user), db: Session = Depends(get_db)
) -> None:
    """Delete a finished run's trajectories. Usage history keeps its tokens and cost."""
    ws = svc.require_workspace(db, user, ws_id)
    svc.delete_run(db, ws, user, _get_run(db, ws_id, run_id))


@router.post("/{run_id}/cancel", response_model=RunOut)
def cancel_run(
    ws_id: str, run_id: str, user: User = Depends(current_user), db: Session = Depends(get_db)
):
    svc.require_workspace(db, user, ws_id)
    r = _get_run(db, ws_id, run_id)
    if r.status in ("queued", "running"):
        r.status = "cancelled"
        db.commit()
    return _out(r)


@router.get("/{run_id}/episodes")
def list_episodes(
    ws_id: str,
    run_id: str,
    user: User = Depends(current_user),
    db: Session = Depends(get_db),
):
    """Compact per-episode summary (no transcripts)."""
    svc.require_workspace(db, user, ws_id)
    r = _get_run(db, ws_id, run_id)
    if not r.run_dir:
        return []
    out = []
    for t in TrajectoryStore(Path(r.run_dir)).read():
        out.append(
            {
                "episode": t.episode,
                "task_id": t.task.id,
                "tier": t.task.tier,
                "seed": t.task.seed,
                "success": t.success,
                "reward": t.reward,
                "steps": len(t.steps),
                "tool_calls": t.n_tool_calls,
                "tool_errors": t.n_tool_errors,
                "finished_reason": t.finished_reason,
                "collateral": (t.verdict.details.get("collateral") if t.verdict else None) or [],
                "tokens": t.usage.input_tokens + t.usage.output_tokens,
            }
        )
    return out


@router.get("/{run_id}/episodes/{episode}")
def get_episode(
    ws_id: str,
    run_id: str,
    episode: int,
    user: User = Depends(current_user),
    db: Session = Depends(get_db),
):
    """Full trajectory for one episode (transcript + verdict)."""
    svc.require_workspace(db, user, ws_id)
    r = _get_run(db, ws_id, run_id)
    if not r.run_dir:
        raise HTTPException(404, "no trajectories yet")
    for t in TrajectoryStore(Path(r.run_dir)).read():
        if t.episode == episode:
            return t.model_dump()
    raise HTTPException(404, "episode not found")


@router.get("/{run_id}/episodes/{episode}/replay")
def replay_episode(
    ws_id: str,
    run_id: str,
    episode: int,
    user: User = Depends(current_user),
    db: Session = Depends(get_db),
):
    """The episode re-run on a fresh copy of its task: one diagram per tool call."""
    from envforge.core.explore import NotReplayable
    from envforge.core.explore import replay_episode as replay

    ws = svc.require_workspace(db, user, ws_id)
    r = _get_run(db, ws_id, run_id)
    if not r.run_dir:
        raise HTTPException(404, "no trajectories yet")
    for t in TrajectoryStore(Path(r.run_dir)).read():
        if t.episode == episode:
            try:
                return replay(resolve_pack(db, ws, r.pack), t)
            except NotReplayable as e:
                raise HTTPException(409, str(e)) from e
    raise HTTPException(404, "episode not found")

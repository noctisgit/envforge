"""Run worker: claims queued runs and executes them with the workspace's own keys.

    uv run envforge-worker            # loop forever
    uv run envforge-worker --once     # process one run and exit (tests, cron)

Claiming is a single UPDATE ... WHERE status='queued' so several workers can share
one database safely. Progress is written back after every episode; the UI polls.
"""

from __future__ import annotations

import time
import traceback
from datetime import UTC, datetime

from sqlalchemy import select, update
from sqlalchemy.orm import Session

from envforge import config
from envforge.core.evaluate import evaluate_run
from envforge.core.registry import get_pack
from envforge.core.runner import run_many
from envforge.models import load_model
from envforge.web import services as svc
from envforge.web.db import session_factory
from envforge.web.models import Run, Workspace


def claim_next(db: Session) -> Run | None:
    row = db.scalar(select(Run).where(Run.status == "queued").order_by(Run.created_at).limit(1))
    if not row:
        return None
    res = db.execute(
        update(Run)
        .where(Run.id == row.id, Run.status == "queued")
        .values(status="running", started_at=datetime.now(UTC))
    )
    db.commit()
    if res.rowcount != 1:  # someone else got it
        return None
    db.refresh(row)
    return row


def execute(db: Session, run: Run) -> None:
    ws = db.get(Workspace, run.workspace_id)
    assert ws is not None
    provider = svc.provider_of(run.model)
    api_key, base_url = (
        (None, None) if provider == "scripted" else svc.resolve_key(db, ws, provider)
    )
    try:
        model = load_model(
            run.model, temperature=run.temperature, api_key=api_key, base_url=base_url
        )
        env_cls = get_pack(run.pack)
    except Exception as e:  # noqa: BLE001
        run.status, run.error, run.finished_at = (
            "failed",
            f"{type(e).__name__}: {e}",
            datetime.now(UTC),
        )
        db.commit()
        return

    def progress(traj):
        # cheap per-episode heartbeat; also lets `cancel` stop a run between episodes
        db.refresh(run)
        if run.status == "cancelled":
            raise _Cancelled()
        run.progress = traj.episode + 1
        db.commit()

    try:
        run_dir = run_many(
            env_cls,
            model,
            runs_dir=config.runs_dir(ws.slug),
            episodes=run.episodes,
            tiers=run.tiers,
            seed_start=run.seed_start,
            max_steps=run.max_steps,
            run_id=run.id,
            progress=progress,
            extra_meta={"workspace": ws.slug, "label": run.label},
        )
        run.run_dir = str(run_dir)
        report = evaluate_run(run_dir)
        run.metrics = report.model_dump()
        if report.overall.n and report.overall.error_rate == 1.0:
            # every episode blew up before verification (bad key, dead endpoint, ...)
            from envforge.core.trajectory import load_run

            first = next((t for t in load_run(run_dir) if t.error), None)
            run.status = "failed"
            run.error = first.error if first else "all episodes errored"
        else:
            run.status = "completed"
    except _Cancelled:
        run.run_dir = str(config.runs_dir(ws.slug) / run.id)
        run.metrics = evaluate_run(run.run_dir).model_dump()
        run.status = "cancelled"
    except Exception:  # noqa: BLE001
        run.status = "failed"
        run.error = traceback.format_exc()[-4000:]
    finally:
        run.finished_at = datetime.now(UTC)
        db.commit()


class _Cancelled(Exception):
    pass


def run_forever(poll_s: float = 2.0, once: bool = False) -> int:
    from envforge.web.settings import get_settings

    if get_settings().DATABASE_URL.startswith("sqlite"):
        from envforge.web.db import create_all

        create_all()  # dev convenience; Postgres runs `alembic upgrade head` first
    Session_ = session_factory()
    handled = 0
    while True:
        try:
            with Session_() as db:
                run = claim_next(db)
                if run:
                    execute(db, run)
                    handled += 1
                    if once:
                        return handled
                    continue
        except Exception as e:  # noqa: BLE001 - keep the worker alive on transient DB errors
            print(f"[worker] {type(e).__name__}: {e}", flush=True)
            if once:
                raise
        if once:
            return handled
        time.sleep(poll_s)


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="envforge run worker")
    ap.add_argument("--once", action="store_true", help="process one queued run and exit")
    ap.add_argument("--poll", type=float, default=2.0)
    args = ap.parse_args()
    n = run_forever(poll_s=args.poll, once=args.once)
    if args.once:
        print(f"processed {n} run(s)")


if __name__ == "__main__":
    main()

"""Run worker: claims queued runs and executes them with the workspace's own keys.

    uv run envforge-worker            # loop forever
    uv run envforge-worker --once     # process one run and exit (tests, cron)

Claiming is a single UPDATE ... WHERE status='queued' so several workers can share
one database safely (scale with `docker compose up -d --scale worker=3`). Progress,
tokens and cost are written back after every episode; the UI polls. A side thread
heartbeats the run, and any worker fails runs whose heartbeat went stale.
"""

from __future__ import annotations

import json
import os
import threading
import time
import traceback
from datetime import UTC, datetime, timedelta
from pathlib import Path

from sqlalchemy import select, update
from sqlalchemy.orm import Session

from envforge import config
from envforge.core.evaluate import evaluate_run, trajectory_cost
from envforge.core.runner import run_many
from envforge.models import load_model
from envforge.web import services as svc
from envforge.web.db import session_factory
from envforge.web.models import Run, Workspace
from envforge.web.settings import get_settings


def claim_next(db: Session) -> Run | None:
    row = db.scalar(select(Run).where(Run.status == "queued").order_by(Run.created_at).limit(1))
    if not row:
        return None
    res = db.execute(
        update(Run)
        .where(Run.id == row.id, Run.status == "queued")
        .values(status="running", started_at=datetime.now(UTC), heartbeat_at=datetime.now(UTC))
    )
    db.commit()
    if res.rowcount != 1:  # someone else got it
        return None
    db.refresh(row)
    return row


def _user_error(e: BaseException) -> str:
    """What the user sees: the exception type and message, never a traceback."""
    return f"{type(e).__name__}: {e}"[:2000]


def _finish(db: Session, run: Run, status: str, error: str | None = None) -> None:
    """Set the final status, unless someone cancelled the run while it was finishing."""
    db.commit()  # flush metrics/usage first; status is set conditionally below
    final = status
    if status != "cancelled":
        res = db.execute(
            update(Run)
            .where(Run.id == run.id, Run.status == "running")
            .values(status=status, error=error, finished_at=datetime.now(UTC))
        )
        if res.rowcount != 1:
            final = "cancelled"  # the API flipped it to cancelled after our last check
    if final == "cancelled":
        db.execute(update(Run).where(Run.id == run.id).values(finished_at=datetime.now(UTC)))
    db.commit()
    db.refresh(run)


class _Heartbeat:
    """Touches runs.heartbeat_at every few seconds from a side thread while a run executes,
    so a crashed or killed worker is noticed (see reap_stale) even mid-episode."""

    def __init__(self, run_id: str, every_s: float = 15.0) -> None:
        self.run_id, self.every_s = run_id, every_s
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def _loop(self) -> None:
        Session_ = session_factory()
        while not self._stop.is_set():
            try:
                with Session_() as db:
                    db.execute(
                        update(Run)
                        .where(Run.id == self.run_id)
                        .values(heartbeat_at=datetime.now(UTC))
                    )
                    db.commit()
            except Exception as e:  # noqa: BLE001 - a missed beat is not fatal
                print(f"[worker] heartbeat: {type(e).__name__}: {e}", flush=True)
            self._stop.wait(self.every_s)

    def __enter__(self) -> _Heartbeat:
        # SQLite (dev/tests) shares one connection across threads; a second writer
        # there is unsafe, and single-process dev doesn't need stale-run detection.
        if not get_settings().DATABASE_URL.startswith("sqlite"):
            self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=5)


def reap_stale(db: Session) -> int:
    """Fail runs stuck in 'running' whose worker stopped sending heartbeats."""
    limit = timedelta(seconds=get_settings().STALE_RUN_SECONDS)
    now = datetime.now(UTC)
    reaped = 0
    for run in db.scalars(select(Run).where(Run.status == "running")):
        last = run.heartbeat_at or run.started_at or run.created_at
        if last.tzinfo is None:  # SQLite drops tzinfo
            last = last.replace(tzinfo=UTC)
        if now - last > limit:
            run.status = "failed"
            run.error = "worker stopped while running this run (no heartbeat); queue it again"
            run.finished_at = now
            reaped += 1
    db.commit()
    return reaped


def execute(db: Session, run: Run) -> None:
    ws = db.get(Workspace, run.workspace_id)
    assert ws is not None
    provider = svc.provider_of(run.model)
    try:
        api_key, base_url = (
            (None, None)
            if provider == "scripted"
            else svc.resolve_key(db, run.created_by, provider)
        )
        model = load_model(
            run.model, temperature=run.temperature, api_key=api_key, base_url=base_url
        )
        from envforge.web.builder import resolve_pack

        env_cls = resolve_pack(db, ws, run.pack)
        if env_cls.kind == "sandbox" and os.environ.get("ENVFORGE_SANDBOX", "docker") != "docker":
            # the local backend runs model-written code on this machine, unisolated
            raise RuntimeError("sandbox packs need ENVFORGE_SANDBOX=docker on the worker")
    except Exception as e:  # noqa: BLE001
        _finish(db, run, "failed", _user_error(e))
        return

    run.input_tokens, run.output_tokens, run.cost_usd = 0, 0, 0.0
    run.cache_read_tokens = 0
    db.commit()

    def progress(traj):
        # per-episode bookkeeping; also lets `cancel` stop a run between episodes
        run.progress = traj.episode + 1
        run.input_tokens += traj.usage.input_tokens
        run.output_tokens += traj.usage.output_tokens
        run.cache_read_tokens += traj.usage.cache_read_tokens
        cost = trajectory_cost(traj)
        run.cost_usd = None if cost is None or run.cost_usd is None else run.cost_usd + cost
        db.commit()
        db.refresh(run)
        if run.status == "cancelled":
            raise _Cancelled()

    with _Heartbeat(run.id):
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
                _finish(db, run, "failed", first.error if first else "all episodes errored")
            else:
                _finish(db, run, "completed")
        except _Cancelled:
            run.run_dir = str(config.runs_dir(ws.slug) / run.id)
            run.metrics = evaluate_run(run.run_dir).model_dump()
            _finish(db, run, "cancelled")
        except Exception as e:  # noqa: BLE001
            traceback.print_exc()  # full detail goes to the worker log, not to the user
            _finish(db, run, "failed", _user_error(e))


class _Cancelled(Exception):
    pass


def _reap_sandboxes() -> None:
    """Remove sandbox containers a previous (crashed) worker left behind."""
    if os.environ.get("ENVFORGE_SANDBOX", "docker") != "docker":
        return
    try:
        from envforge.sandbox.docker import reap_sandboxes

        if n := reap_sandboxes(older_than_s=7200):
            print(f"[worker] removed {n} leftover sandbox container(s)", flush=True)
    except Exception as e:  # noqa: BLE001 - no Docker here just means no sandbox packs
        print(f"[worker] sandbox cleanup skipped: {type(e).__name__}: {e}", flush=True)


def process_upload(db: Session, job) -> None:
    """Parse an uploaded log file, redact personal data, profile it, and delete the raw file.

    Only the redacted conversations are kept; nothing downstream (the builder model,
    the UI) ever sees the original text.
    """
    from dataclasses import asdict

    from envforge.builder.ingest import Mapping, parse_file, profile
    from envforge.builder.redact import redact_conversation
    from envforge.web.models import EnvUpload

    up = db.get(EnvUpload, (job.payload or {}).get("upload_id"))
    assert up is not None
    raw = Path(up.path or "")
    up.status = "processing"
    db.commit()
    try:
        m = (job.payload or {}).get("mapping")
        parsed = parse_file(raw, mapping=Mapping(**m) if m else None)
        counts: dict[str, int] = {}
        redacted = []
        for conv in parsed.conversations:
            clean, c = redact_conversation(conv)
            for k, v in c.items():
                counts[k] = counts.get(k, 0) + v
            redacted.append(clean)
        out = raw.with_suffix(".jsonl")
        with out.open("w", encoding="utf-8") as f:
            for conv in redacted:
                f.write(json.dumps(conv.to_dict()) + "\n")
        guess = asdict(parsed.format)
        up.detected_format = {
            "format": guess.get("format"),
            "confidence": guess.get("confidence"),
            "reasons": guess.get("reasons"),
            "mapping": asdict(parsed.mapping),
            "warnings": parsed.warnings[:20],
            "skipped_rows": parsed.skipped_rows,
            "truncated": parsed.truncated,
        }
        up.n_conversations = len(redacted)
        up.redactions = counts
        up.profile = profile(redacted)
        up.path = str(out)
        up.status = "ready" if redacted else "failed"
        up.error = None if redacted else "no conversations found — check the format"
    except Exception as e:  # noqa: BLE001
        up.status, up.error = "failed", _user_error(e)
        raise
    finally:
        raw.unlink(missing_ok=True)  # never keep the unredacted original
        db.commit()


def process_job(db: Session, job) -> None:
    """Run one builder job and record how it ended."""
    from envforge.web import builder as B
    from envforge.web import builder_agent
    from envforge.web.models import EnvProject

    try:
        if job.kind == "builder_turn":
            builder_agent.run_turn(db, job)
        elif job.kind == "ingest":
            process_upload(db, job)
        else:
            raise ValueError(f"unknown job kind {job.kind}")
        job.status = "done"
    except Exception as e:  # noqa: BLE001
        traceback.print_exc()
        db.rollback()
        job.status, job.error = "failed", _user_error(e)
        if job.kind == "builder_turn" and (project := db.get(EnvProject, job.project_id)):
            B.add_message(db, project, "assistant", f"⚠️ The builder hit an error: {job.error}")
    finally:
        job.progress = ""
        job.finished_at = datetime.now(UTC)
        db.commit()


def reap_stale_jobs(db: Session, minutes: int = 20) -> int:
    from envforge.web.models import Job

    cutoff = datetime.now(UTC) - timedelta(minutes=minutes)
    n = 0
    for job in db.scalars(select(Job).where(Job.status == "running")):
        last = job.heartbeat_at or job.started_at or job.created_at
        if last.tzinfo is None:
            last = last.replace(tzinfo=UTC)
        if last < cutoff:
            job.status, job.error, job.finished_at = (
                "failed",
                "worker stopped (no heartbeat)",
                datetime.now(UTC),
            )
            n += 1
    db.commit()
    return n


def run_forever(poll_s: float = 2.0, once: bool = False, kind: str = "all") -> int:
    """Process work until stopped. `kind`: "runs" (evaluation runs), "builder" (builder
    chat turns + upload processing) or "all". Production runs one of each, so a long
    evaluation never blocks the builder chat."""
    from envforge.web import builder as B

    if get_settings().DATABASE_URL.startswith("sqlite"):
        from envforge.web.db import create_all

        create_all()  # dev convenience; Postgres runs `alembic upgrade head` first
    if kind in ("runs", "all"):
        _reap_sandboxes()
    Session_ = session_factory()
    handled = 0
    last_reap = 0.0
    while True:
        try:
            with Session_() as db:
                if time.monotonic() - last_reap > 30:
                    last_reap = time.monotonic()
                    if kind in ("runs", "all") and (n := reap_stale(db)):
                        print(f"[worker] failed {n} stale run(s)", flush=True)
                    if kind in ("builder", "all") and (n := reap_stale_jobs(db)):
                        print(f"[worker] failed {n} stale job(s)", flush=True)
                if kind in ("builder", "all") and (
                    job := B.claim_job(db, ("builder_turn", "ingest"))
                ):
                    process_job(db, job)
                    handled += 1
                    if once:
                        return handled
                    continue
                if kind in ("runs", "all") and (run := claim_next(db)):
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

    ap = argparse.ArgumentParser(description="envforge worker")
    ap.add_argument("--once", action="store_true", help="process one queued item and exit")
    ap.add_argument("--poll", type=float, default=2.0)
    ap.add_argument("--kind", choices=("runs", "builder", "all"), default="all")
    args = ap.parse_args()
    n = run_forever(poll_s=args.poll, once=args.once, kind=args.kind)
    if args.once:
        print(f"processed {n} item(s)")


if __name__ == "__main__":
    main()

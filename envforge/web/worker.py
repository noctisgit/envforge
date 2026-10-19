"""Run worker: claims queued runs and executes them with the workspace's own keys.

    uv run envforge-worker            # loop forever
    uv run envforge-worker --once     # process one run and exit (tests, cron)

Claiming is a single UPDATE ... WHERE status='queued' so several workers can share
one database safely (scale with `docker compose up -d --scale worker=3`). Progress,
tokens and cost are written back after every episode; the UI polls. A side thread
heartbeats the run, and any worker fails runs whose heartbeat went stale.
"""

from __future__ import annotations

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


def download_to(url: str, dest: Path, *, max_bytes: int, on_progress=None) -> int:
    """Stream an https URL to `dest`. Every redirect hop is re-checked against the same
    public-hosts-only rule as the original URL, and the size is capped while reading
    (a server can lie about Content-Length)."""
    import urllib.request
    from urllib.parse import urlparse

    from envforge.web.security import check_base_url

    class _CheckedRedirects(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: N802
            if urlparse(newurl).scheme != "https":
                raise ValueError("refusing a redirect to a non-https URL")
            check_base_url(newurl)
            return super().redirect_request(req, fp, code, msg, headers, newurl)

    check_base_url(url)
    opener = urllib.request.build_opener(_CheckedRedirects())
    got, last = 0, 0.0
    with opener.open(
        urllib.request.Request(url, headers={"User-Agent": "envforge"}), timeout=60
    ) as r:
        total = int(r.headers.get("Content-Length") or 0)
        if total > max_bytes:
            raise ValueError(
                f"file is {total / 2**30:.1f} GiB; the limit is {max_bytes / 2**30:.0f} GiB"
            )
        with dest.open("wb") as f:
            while chunk := r.read(1024 * 1024):
                got += len(chunk)
                if got > max_bytes:
                    raise ValueError(f"download exceeded the {max_bytes / 2**30:.0f} GiB limit")
                f.write(chunk)
                if on_progress and time.monotonic() - last > 2:
                    last = time.monotonic()
                    on_progress({"stage": "downloading", "bytes": got, "total": total or None})
    return got


def _gunzip(src: Path, dest: Path, *, max_bytes: int) -> None:
    """Decompress a .gz upload, capping the output size (a tiny file can inflate hugely)."""
    import gzip
    import shutil as sh

    written = 0
    with gzip.open(src, "rb") as fin, dest.open("wb") as fout:
        while chunk := fin.read(1024 * 1024):
            written += len(chunk)
            if written > max_bytes:
                raise ValueError(f"decompressed size exceeds {max_bytes / 2**30:.0f} GiB")
            fout.write(chunk)
    sh.copystat(src, dest)


def process_upload(db: Session, job) -> None:
    """Bring one upload into the project's conversation store.

    download (URL imports) -> decompress (.gz) -> stream, parse and REDACT every
    conversation with the project's privacy settings before storing it -> deduplicate and
    cluster the whole project corpus -> a redaction report on a stratified sample. The raw
    file is always deleted at the end; only redacted text is ever stored.
    """
    from envforge.builder import corpus as C
    from envforge.builder.ingest import Mapping
    from envforge.builder.redact import build_redaction_report
    from envforge.web import builder as B
    from envforge.web.models import EnvProject, EnvUpload

    up = db.get(EnvUpload, (job.payload or {}).get("upload_id"))
    assert up is not None
    project = db.get(EnvProject, up.project_id)
    ws = db.get(Workspace, project.workspace_id)
    raw = Path(up.path or "")
    src = raw
    up.status, up.progress = "processing", {"stage": "starting"}
    db.commit()
    last = [0.0]

    def progress(info: dict, force: bool = False) -> None:
        if force or time.monotonic() - last[0] > 2:
            last[0] = time.monotonic()
            up.progress = info
            job.progress = f"{info.get('stage', 'processing')} {up.filename}"
            job.heartbeat_at = datetime.now(UTC)
            db.commit()

    limit = get_settings().MAX_CORPUS_BYTES
    try:
        if up.source == "url":
            progress({"stage": "downloading"}, force=True)
            up.size_bytes = download_to(up.source_url, raw, max_bytes=limit, on_progress=progress)
        if up.filename.lower().endswith(".gz"):
            progress({"stage": "decompressing"}, force=True)
            inner = Path(up.filename[:-3]).suffix or ".jsonl"
            src = raw.with_name(f"{up.id}{inner}")
            _gunzip(raw, src, max_bytes=limit * 4)
        m = (job.payload or {}).get("mapping")
        # NER / the PII span model load a model per process: keep the pool small then
        heavy = project.privacy_ner or project.privacy_tier2 in ("auto", "all")
        workers = 1 if heavy else max(1, min(4, (os.cpu_count() or 2) - 1))
        with C.CorpusStore(B.corpus_path(ws, project)) as store:
            rep = C.ingest_file_to_store(
                src, store, up.id, mapping=Mapping(**m) if m else None, redact=B.make_redactor(project),
                workers=workers,
                on_progress=lambda p: progress({"stage": "redacting", "bytes": p.bytes_read,
                                                "total": p.bytes_total, "conversations": p.conversations}),
            )  # fmt: skip
            progress({"stage": "deduplicating", "conversations": rep.conversations}, force=True)
            dedupe = C.dedupe_near(store)
            progress({"stage": "clustering", "conversations": rep.conversations}, force=True)
            clusters = C.cluster(store)
            sample = store.get(C.stratified_sample(store, 200, seed=0))
            report = build_redaction_report(
                sample, profile=project.privacy_profile if project.privacy_profile != "off" else "standard",
                counts=rep.pii_counts, ner_requested=project.privacy_ner, n_input=rep.conversations,
            )  # fmt: skip
        up.detected_format = {"format": rep.format, "confidence": rep.confidence, "reasons": rep.reasons,
                              "mapping": rep.mapping, "skipped_rows": rep.skipped_rows,
                              "truncated": rep.truncated, "warnings": []}  # fmt: skip
        up.n_conversations = rep.conversations
        up.redactions = rep.pii_counts
        up.profile = {"ingest": rep.to_dict(), "dedupe": dedupe.to_dict(),
                      "clusters": len(clusters), "redaction_report": report}  # fmt: skip
        up.path = None  # the data now lives (redacted) in the corpus store
        up.status = "ready" if rep.conversations else "failed"
        up.error = None if rep.conversations else "no conversations found — check the format"
        up.progress = None
        # new data: the last human review no longer covers it
        if project.privacy_review:
            project.privacy_review = {**project.privacy_review, "status": "outdated"}
        project.privacy_signed_off_at = project.privacy_signed_off_by = None
    except Exception as e:  # noqa: BLE001
        up.status, up.error, up.progress = "failed", _user_error(e), None
        raise
    finally:
        raw.unlink(missing_ok=True)  # never keep the unredacted original
        if src != raw:
            src.unlink(missing_ok=True)
        db.commit()


def recluster(db: Session, job) -> None:
    """After an upload is deleted: refresh duplicates and clusters for what's left."""
    from envforge.builder import corpus as C
    from envforge.web import builder as B
    from envforge.web.models import EnvProject

    project = db.get(EnvProject, job.project_id)
    ws = db.get(Workspace, project.workspace_id)
    path = B.corpus_path(ws, project)
    if not path.exists():
        return
    with C.CorpusStore(path) as store:
        if uid := (job.payload or {}).get("delete_upload"):
            store.delete_upload(uid)
            store.db.commit()
        if store.count():
            C.dedupe_near(store)
            C.cluster(store)


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
        elif job.kind == "recluster":
            recluster(db, job)
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
                    job := B.claim_job(db, ("builder_turn", "ingest", "recluster"))
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

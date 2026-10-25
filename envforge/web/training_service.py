"""Datasets, LoRA training jobs and adapters in the browser (Phase 3), reusing
`envforge.train` unchanged:

    datasets       train.dataset.build_dataset + overlap   (rejection sampling, manifest)
    training jobs  train.remote.start_training + poll_once + fetch_adapter
    adapters       served with vLLM --lora-modules via compute_service.serve_on_box

Datasets are built on the API side (local files only, no GPU). Training starts as a
`train_start` job on the compute worker (upload + detached start, minutes); after
that the compute worker *polls* every running job every TRAIN_POLL_SECONDS (a few
short ssh calls), so one multi-hour training never blocks other box jobs. A job is
stopped and failed when it exceeds TRAIN_MAX_HOURS or the box stays unreachable for
TRAIN_MAX_POLL_FAILURES polls in a row.

The seed rule from docs/TRAINING.md is enforced: datasets only take runs on training
seeds (seed_start >= 10000), and the preview checks the workspace's evaluation runs
(seeds below 10000) for overlap with `train.dataset.overlap`.
"""

from __future__ import annotations

import contextlib
import json
import shutil
import tarfile
import tempfile
from dataclasses import asdict, fields
from datetime import UTC, datetime, timedelta
from pathlib import Path
from statistics import mean
from typing import Any

from fastapi import HTTPException
from sqlalchemy import func, or_, select, update
from sqlalchemy.orm import Session

from envforge import config
from envforge.compute.base import ComputeError
from envforge.core.env import TRAIN_SEED_MIN
from envforge.train import dataset as ds
from envforge.train import remote
from envforge.web import compute_service as C
from envforge.web import crypto
from envforge.web.models import Adapter, Dataset, GpuBox, Job, Run, TrainingJob, User, Workspace
from envforge.web.settings import get_settings

ACTIVE_TRAINING = ("queued", "starting", "running", "fetching")


def _aware(dt: datetime | None) -> datetime | None:
    """SQLite drops tzinfo; treat naive datetimes as UTC."""
    if dt is not None and dt.tzinfo is None:
        return dt.replace(tzinfo=UTC)
    return dt


# ---- datasets ------------------------------------------------------------------------


def _ws_run(db: Session, ws: Workspace, run_id: str) -> Run:
    r = db.get(Run, run_id)
    if r is None or r.workspace_id != ws.id or r.deleted_at is not None:
        raise HTTPException(404, f"run {run_id} not found")
    return r


def ineligible_reason(run: Run) -> str | None:
    """Why a run can't feed a dataset, or None if it can."""
    if run.status != "completed":
        return f"not completed ({run.status})"
    if run.seed_start < TRAIN_SEED_MIN:
        return f"uses evaluation seeds ({run.seed_start}+); training data must come from seeds ≥ {TRAIN_SEED_MIN}"
    if not run.run_dir or not Path(run.run_dir).is_dir():
        return "no trajectories on disk"
    return None


def dataset_sources(db: Session, ws: Workspace) -> list[dict[str, Any]]:
    """Completed runs of the workspace, each marked eligible or with the reason it isn't."""
    rows = db.scalars(
        select(Run).where(Run.workspace_id == ws.id, Run.deleted_at.is_(None), Run.status == "completed")
        .order_by(Run.created_at.desc()).limit(500)
    )  # fmt: skip
    out = []
    for r in rows:
        reason = ineligible_reason(r)
        out.append({
            "id": r.id, "label": r.label, "pack": r.pack, "model": r.model, "episodes": r.episodes,
            "seed_start": r.seed_start, "created_at": r.created_at,
            "solve_rate": (r.metrics or {}).get("overall", {}).get("solve_rate"),
            "eligible": reason is None, "reason": reason,
        })  # fmt: skip
    return out


def _validated_runs(db: Session, ws: Workspace, run_ids: list[str]) -> list[Run]:
    if not run_ids:
        raise HTTPException(400, "pick at least one run")
    if len(run_ids) > 50:
        raise HTTPException(400, "at most 50 runs per dataset")
    runs = [_ws_run(db, ws, rid) for rid in dict.fromkeys(run_ids)]  # 404 on anything not ours
    for r in runs:
        if reason := ineligible_reason(r):
            raise HTTPException(400, f"run {r.label or r.id} can't be used: {reason}")
    return runs


def _pack_resolver(db: Session, ws: Workspace):
    from envforge.web.builder import resolve_pack

    return lambda name: resolve_pack(db, ws, name, include_deleted=True)


def _check_filters(min_reward: float, val_frac: float) -> None:
    if not 0 <= min_reward <= 1:
        raise HTTPException(400, "min reward must be between 0 and 1")
    if not 0 <= val_frac <= 0.5:
        raise HTTPException(400, "validation split must be between 0 and 0.5")


def _summary(out_dir: Path, rep: ds.DatasetReport) -> dict[str, Any]:
    """Counts the UI shows: split, examples per task, per pack/tier, models."""
    per_task: dict[str, int] = {}
    for split in ("train.jsonl", "val.jsonl"):
        p = out_dir / split
        if p.exists():
            for line in p.read_text(encoding="utf-8").splitlines():
                meta = json.loads(line).get("meta", {})
                # a declarative task is its record, not the seed that picked it
                tid = meta.get("task_record") or meta.get("task_id", "?")
                per_task[tid] = per_task.get(tid, 0) + 1
    counts = list(per_task.values()) or [0]
    return {
        "episodes_seen": rep.total,
        "kept": rep.kept,
        "duplicates_dropped": rep.duplicates,
        "n_train": rep.n_train,
        "n_val": rep.n_val,
        "tasks": len(per_task),
        "examples_per_task": {
            "min": min(counts),
            "max": max(counts),
            "mean": round(mean(counts), 2),
        },
        "by_pack_tier": {f"{p} · tier {t}": n for (p, t), n in sorted(rep.by_pack_tier.items())},
        "models": dict(rep.models),
    }


def _overlaps(db: Session, ws: Workspace, out_dir: Path, packs: set[str]) -> list[dict[str, Any]]:
    """Evaluation runs (seeds < TRAIN_SEED_MIN) of the same packs, or of any version of
    the same published environment, that share tasks with the dataset, via
    `train.dataset.overlap`."""
    hits = []
    bases = {p.split("@", 1)[0] for p in packs}
    same_env = or_(Run.pack.in_(packs), *(Run.pack.like(f"{b}@%") for b in bases))
    evals = db.scalars(
        select(Run).where(Run.workspace_id == ws.id, Run.deleted_at.is_(None), Run.status == "completed",
                          Run.seed_start < TRAIN_SEED_MIN, same_env)
        .order_by(Run.created_at.desc()).limit(100)
    )  # fmt: skip
    for r in evals:
        if not r.run_dir or not Path(r.run_dir).is_dir():
            continue
        found = ds.overlap(out_dir, Path(r.run_dir))
        if found:
            hits.append({"run_id": r.id, "label": r.label, "model": r.model, "tasks": len(found),
                         "sample": [ds.describe_key(k) for k in found[:5]]})  # fmt: skip
    return hits


def _build(db: Session, ws: Workspace, runs: list[Run], out_dir: Path, *, min_reward: float,
           val_frac: float, best_per_task: bool) -> ds.DatasetReport:  # fmt: skip
    return ds.build_dataset(
        [Path(r.run_dir) for r in runs if r.run_dir],
        out_dir,
        min_reward=min_reward,
        val_frac=val_frac,
        best_per_task=best_per_task,
        pack_resolver=_pack_resolver(db, ws),
    )


def preview_dataset(db: Session, ws: Workspace, *, run_ids: list[str], min_reward: float = 1.0,
                    val_frac: float = 0.05, best_per_task: bool = True) -> dict[str, Any]:  # fmt: skip
    """What saving would produce, built in a scratch directory and thrown away."""
    _check_filters(min_reward, val_frac)
    runs = _validated_runs(db, ws, run_ids)
    packs = {r.pack for r in runs}
    scratch = config.workspace_dir(ws.slug) / "tmp"
    scratch.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=scratch) as tmp:
        out = Path(tmp) / "preview"
        try:
            rep = _build(
                db,
                ws,
                runs,
                out,
                min_reward=min_reward,
                val_frac=val_frac,
                best_per_task=best_per_task,
            )
        except ValueError as e:
            return {"kept": 0, "n_train": 0, "n_val": 0, "tasks": 0, "packs": sorted(packs),
                    "overlap": [], "warnings": [str(e)]}  # fmt: skip
        summary = _summary(out, rep)
        overlap = _overlaps(db, ws, out, packs)
    warnings = []
    if overlap:
        warnings.append(
            f"{len(overlap)} evaluation run(s) share tasks with this dataset; evaluate on other seeds."
        )
    if summary["kept"] < 200:
        warnings.append(
            "A few hundred to a few thousand examples is a typical SFT dataset; this is fine for a smoke test."
        )
    if len(packs) > 1:
        warnings.append("Mixed environments: each example carries its own tool schemas.")
    return {**summary, "packs": sorted(packs), "overlap": overlap, "warnings": warnings}


def _sanitize_files(db: Session, ws: Workspace, out_dir: Path, packs: set[str]) -> None:
    """Environments built from customer data are re-sanitised on the way out, exactly
    like the JSONL export (builder.export_sanitizer)."""
    from envforge.web.builder import export_sanitizer

    clean = export_sanitizer(db, ws, packs)
    if clean is None:
        return
    for split in ("train.jsonl", "val.jsonl"):
        p = out_dir / split
        if p.exists():
            lines = p.read_text(encoding="utf-8").splitlines(keepends=True)
            p.write_text("".join(clean(line) for line in lines), encoding="utf-8")


def save_dataset(db: Session, ws: Workspace, user: User, *, name: str, run_ids: list[str],
                 min_reward: float = 1.0, val_frac: float = 0.05, best_per_task: bool = True) -> Dataset:  # fmt: skip
    """Build and keep a named dataset under data/workspaces/<slug>/sft/<name>/."""
    try:
        name = remote.check_name(name.strip())
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    _check_filters(min_reward, val_frac)
    if db.scalar(select(Dataset).where(Dataset.workspace_id == ws.id, Dataset.name == name)):
        raise HTTPException(409, f"a dataset named {name} already exists")
    runs = _validated_runs(db, ws, run_ids)
    packs = {r.pack for r in runs}
    root = config.sft_dir(ws.slug).resolve()
    out = (root / name).resolve()
    if not out.is_relative_to(root) or out == root:
        raise HTTPException(400, "invalid dataset name")
    if out.exists():  # left over from a CLI build or a deleted row: not ours to merge into
        shutil.rmtree(out)
    try:
        rep = _build(
            db, ws, runs, out, min_reward=min_reward, val_frac=val_frac, best_per_task=best_per_task
        )
    except ValueError as e:
        shutil.rmtree(out, ignore_errors=True)
        raise HTTPException(400, str(e)) from e
    _sanitize_files(db, ws, out, packs)
    summary = _summary(out, rep)
    row = Dataset(
        workspace_id=ws.id,
        name=name,
        source_runs=[r.id for r in runs],
        filters={"min_reward": min_reward, "val_frac": val_frac, "best_per_task": best_per_task},
        packs=sorted(packs),
        counts=summary,
        n_train=rep.n_train,
        n_val=rep.n_val,
        path=str(out),
        manifest=ds.load_manifest(out),
        created_by=user.id,
    )
    db.add(row)
    db.commit()
    return row


def list_datasets(db: Session, ws: Workspace) -> list[Dataset]:
    return list(
        db.scalars(
            select(Dataset).where(Dataset.workspace_id == ws.id).order_by(Dataset.created_at.desc())
        )
    )


def get_dataset(db: Session, ws: Workspace, dataset_id: str) -> Dataset:
    d = db.get(Dataset, dataset_id)
    if d is None or d.workspace_id != ws.id:
        raise HTTPException(404, "dataset not found")
    return d


def dataset_out(d: Dataset, *, full: bool = False) -> dict[str, Any]:
    out = {
        "id": d.id, "name": d.name, "source_runs": d.source_runs, "filters": d.filters,
        "packs": d.packs, "counts": d.counts, "n_train": d.n_train, "n_val": d.n_val,
        "created_at": d.created_at,
        "tasks": (d.manifest or {}).get("tasks"),
    }  # fmt: skip
    if full:
        out["manifest"] = d.manifest
    return out


def dataset_file(ws: Workspace, d: Dataset, split: str) -> Path:
    """train.jsonl / val.jsonl of a dataset, only from inside the workspace's sft dir."""
    if split not in ("train", "val"):
        raise HTTPException(400, "split must be train or val")
    root = config.sft_dir(ws.slug).resolve()
    p = (Path(d.path) / f"{split}.jsonl").resolve()
    if not p.is_relative_to(root) or not p.is_file():
        raise HTTPException(404, "dataset file not found")
    return p


def delete_dataset(db: Session, ws: Workspace, user: User, d: Dataset) -> None:
    """Remove the files and the row. Adapters trained on it keep their lineage (by name)."""
    busy = db.scalar(
        select(func.count()).select_from(TrainingJob).where(
            TrainingJob.dataset_id == d.id, TrainingJob.status.in_(ACTIVE_TRAINING))
    )  # fmt: skip
    if busy:
        raise HTTPException(409, "a training job is using this dataset; cancel it first")
    db.execute(update(TrainingJob).where(TrainingJob.dataset_id == d.id).values(dataset_id=None))
    db.execute(update(Adapter).where(Adapter.dataset_id == d.id).values(dataset_id=None))
    root = config.sft_dir(ws.slug).resolve()
    target = Path(d.path).resolve()
    db.delete(d)
    db.commit()
    if target.is_relative_to(root) and target != root:
        shutil.rmtree(target, ignore_errors=True)


# ---- training jobs ---------------------------------------------------------------------

#: The CLI's defaults (`envforge train sft`), shown in the form.
TRAIN_DEFAULTS = asdict(remote.TrainArgs())
LIMITS = {
    "epochs": (0.05, 20), "lr": (1e-6, 1e-2), "lora_r": (1, 256), "lora_alpha": (1, 1024),
    "max_len": (256, 65536), "batch_size": (1, 64), "grad_accum": (1, 512), "max_steps": (1, 1_000_000),
}  # fmt: skip


def _train_args(params: dict[str, Any]) -> remote.TrainArgs:
    names = {f.name for f in fields(remote.TrainArgs)}
    clean: dict[str, Any] = {}
    for k, v in (params or {}).items():
        if k not in names or v is None:
            continue
        lo, hi = LIMITS[k]
        if not lo <= float(v) <= hi:
            raise HTTPException(400, f"{k} must be between {lo:g} and {hi:g}")
        clean[k] = float(v) if k in ("epochs", "lr") else int(v)
    return remote.TrainArgs(**{**TRAIN_DEFAULTS, **clean})


def start_training(db: Session, ws: Workspace, user: User, *, dataset_id: str, box_id: str,
                   base_model: str, adapter_name: str | None = None,
                   hyperparams: dict[str, Any] | None = None) -> TrainingJob:  # fmt: skip
    """Queue a LoRA SFT job: upload the dataset to the box and train there."""
    C.require_gpu_admin(user)
    d = get_dataset(db, ws, dataset_id)
    box = C.get_box(db, ws, box_id)
    try:
        base = C.check_model_id(base_model)
        name = remote.check_name((adapter_name or d.name).strip())
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    args = _train_args(hyperparams or {})
    if db.scalar(select(Adapter).where(Adapter.workspace_id == ws.id, Adapter.name == name)):
        raise HTTPException(409, f"an adapter named {name} already exists; pick another name")
    same = list(db.scalars(select(TrainingJob).where(TrainingJob.workspace_id == ws.id,
                                                     TrainingJob.adapter_name == name)))  # fmt: skip
    if any(t.status in (*ACTIVE_TRAINING, "completed") for t in same):
        raise HTTPException(409, f"a training job for {name} already exists; pick another name")
    on_box = db.scalar(
        select(func.count()).select_from(TrainingJob).where(
            TrainingJob.box_id == box.id, TrainingJob.status.in_(ACTIVE_TRAINING))
    )  # fmt: skip
    if on_box:
        raise HTTPException(409, f"{box.name} is already training; one job per box at a time")
    tj = TrainingJob(
        workspace_id=ws.id,
        dataset_id=d.id,
        box_id=box.id,
        adapter_name=name,
        base_model=base,
        # a retry after a failed/cancelled job of the same name may overwrite its leftovers
        hyperparams={**asdict(args), "dataset_name": d.name, "replace": bool(same)},
        hourly_usd=box.hourly_usd,
        status="queued",
        created_by=user.id,
    )
    db.add(tj)
    db.flush()
    job = C.enqueue_box_job(db, box, user, "train_start", {"training_job_id": tj.id})
    tj.job_id = job.id
    db.commit()
    return tj


def run_train_start(db: Session, job: Job, box: GpuBox) -> None:
    """Worker: upload the dataset + sft.py and start training detached (train.remote)."""
    tj = db.get(TrainingJob, (job.payload or {}).get("training_job_id"))
    if tj is None or tj.workspace_id != job.workspace_id:
        raise ValueError("training job not found")
    if tj.cancel_requested or tj.status == "cancelled":
        tj.status, tj.finished_at = "cancelled", datetime.now(UTC)
        db.commit()
        return
    d = db.get(Dataset, tj.dataset_id) if tj.dataset_id else None
    if d is None:
        raise ValueError("the dataset was deleted")
    tj.status = "starting"
    job.progress = f"uploading {d.name} to {box.name}"
    db.commit()
    params = dict(tj.hyperparams)
    args = remote.TrainArgs(**{k: params[k] for k in TRAIN_DEFAULTS if k in params})
    hf_token = crypto.decrypt(box.hf_token_ciphertext) if box.hf_token_ciphertext else None
    with C.open_box(db, box) as cb:
        runtime = C.resolve_box_runtime(db, box, cb)
        remote.start_training(
            cb, Path(d.path), base=tj.base_model, name=tj.adapter_name, args=args,
            hf_token=hf_token, replace=bool(params.get("replace")), runtime=runtime,  # type: ignore[arg-type]
        )  # fmt: skip
    tj.runtime = runtime
    tj.status, tj.started_at = "running", datetime.now(UTC)
    db.commit()


def request_cancel(db: Session, ws: Workspace, user: User, tj: TrainingJob) -> TrainingJob:
    """Queued: cancelled at once. Starting/running: the next poll stops it on the box."""
    C.require_gpu_admin(user)
    if tj.status == "queued":
        job = db.get(Job, tj.job_id) if tj.job_id else None
        if job is not None and job.status == "queued":
            job.status, job.finished_at = "cancelled", datetime.now(UTC)
        tj.status, tj.finished_at = "cancelled", datetime.now(UTC)
    elif tj.status in ("starting", "running", "fetching"):
        tj.cancel_requested = True
    db.commit()
    return tj


def get_training(db: Session, ws: Workspace, tj_id: str) -> TrainingJob:
    tj = db.get(TrainingJob, tj_id)
    if tj is None or tj.workspace_id != ws.id:
        raise HTTPException(404, "training job not found")
    return tj


def list_trainings(db: Session, ws: Workspace) -> list[TrainingJob]:
    return list(db.scalars(select(TrainingJob).where(TrainingJob.workspace_id == ws.id)
                           .order_by(TrainingJob.created_at.desc()).limit(200)))  # fmt: skip


def training_out(db: Session, tj: TrainingJob, *, full: bool = False) -> dict[str, Any]:
    box = db.get(GpuBox, tj.box_id) if tj.box_id else None
    started, finished = _aware(tj.started_at), _aware(tj.finished_at)
    elapsed = ((finished or datetime.now(UTC)) - started).total_seconds() if started else 0.0
    adapter = db.scalar(select(Adapter).where(Adapter.training_job_id == tj.id))
    out = {
        "id": tj.id, "status": tj.status, "adapter_name": tj.adapter_name, "base_model": tj.base_model,
        "dataset_id": tj.dataset_id, "dataset_name": tj.hyperparams.get("dataset_name"),
        "box_id": tj.box_id, "box_name": box.name if box else None,
        "hyperparams": {k: v for k, v in tj.hyperparams.items() if k in TRAIN_DEFAULTS},
        "step": tj.step, "total_steps": tj.total_steps, "epoch": tj.epoch,
        "last_loss": next((p["loss"] for p in reversed(tj.losses or []) if "loss" in p), None),
        "cancel_requested": tj.cancel_requested, "error": tj.error,
        "elapsed_s": round(elapsed), "hourly_usd": tj.hourly_usd,
        "est_cost_usd": round(elapsed / 3600 * tj.hourly_usd, 2) if tj.hourly_usd is not None else None,
        "adapter_id": adapter.id if adapter else None,
        "created_at": tj.created_at, "started_at": tj.started_at, "finished_at": tj.finished_at,
        "polled_at": tj.polled_at,
    }  # fmt: skip
    if full:
        out["losses"] = tj.losses or []
        out["log_tail"] = tj.log_tail or ""
    return out


def _apply_events(tj: TrainingJob, events: list[dict[str, Any]]) -> str | None:
    """Fold sft.py's progress events into the row; returns an error message if one came."""
    losses = list(tj.losses or [])
    error = None
    for ev in events:
        kind = ev.get("event")
        if kind == "start":
            tj.total_steps = ev.get("total_steps") or tj.total_steps
        elif kind == "log":
            tj.step = int(ev.get("step") or tj.step)
            tj.epoch = ev.get("epoch", tj.epoch)
            if isinstance(ev.get("loss"), int | float):
                losses.append({"step": tj.step, "loss": round(float(ev["loss"]), 5)})
        elif kind == "eval":
            if isinstance(ev.get("loss"), int | float):
                losses.append(
                    {
                        "step": int(ev.get("step") or tj.step),
                        "eval_loss": round(float(ev["loss"]), 5),
                    }
                )
        elif kind == "error":
            error = str(ev.get("message") or "training failed")[:2000]
    tj.losses = losses[-2000:]
    return error


def _finish(db: Session, tj: TrainingJob, status: str, error: str | None = None) -> None:
    tj.status, tj.error, tj.finished_at = status, error, datetime.now(UTC)
    db.commit()


def _create_adapter(db: Session, tj: TrainingJob, local: Path) -> Adapter:
    d = db.get(Dataset, tj.dataset_id) if tj.dataset_id else None
    manifest: dict[str, Any] = {}
    with contextlib.suppress(OSError, ValueError):
        manifest = json.loads((local / "envforge_manifest.json").read_text())
    rank = None
    with contextlib.suppress(OSError, ValueError):
        rank = int(json.loads((local / "adapter_config.json").read_text()).get("r") or 0) or None
    keep = ("n_train", "n_val", "global_step", "final_train_loss", "mean_train_loss", "final_eval_loss",
            "trainable_params", "dataset_sha256", "started_at", "finished_at")  # fmt: skip
    metrics = {k: manifest[k] for k in keep if k in manifest}
    metrics["lineage"] = {"dataset": tj.hyperparams.get("dataset_name"), "packs": d.packs if d else None,
                          "source_runs": d.source_runs if d else None}  # fmt: skip
    a = Adapter(
        workspace_id=tj.workspace_id, name=tj.adapter_name, base_model=tj.base_model,
        pack=(d.packs[0] if d and d.packs else None), dataset_id=tj.dataset_id,
        training_job_id=tj.id, box_id=tj.box_id, path=str(local), lora_rank=rank,
        metrics=metrics, created_by=tj.created_by,
    )  # fmt: skip
    db.add(a)
    db.commit()
    return a


def poll_training(db: Session, tj: TrainingJob) -> None:
    """One poll of a running job: progress, logs, and the end (fetch the adapter)."""
    s = get_settings()
    box = db.get(GpuBox, tj.box_id) if tj.box_id else None
    if box is None:
        _finish(db, tj, "failed", "the GPU box was removed while training")
        return
    ws = db.get(Workspace, tj.workspace_id)
    runtime = tj.runtime or "docker"
    tj.polled_at = datetime.now(UTC)
    started = _aware(tj.started_at) or datetime.now(UTC)
    too_long = datetime.now(UTC) - started > timedelta(hours=s.TRAIN_MAX_HOURS)
    try:
        with C.open_box(db, box) as cb:
            if tj.cancel_requested or too_long:
                remote.remove_container(cb, tj.adapter_name, runtime=runtime)  # type: ignore[arg-type]
                if too_long:
                    _finish(
                        db,
                        tj,
                        "failed",
                        f"stopped: ran longer than TRAIN_MAX_HOURS={s.TRAIN_MAX_HOURS:g}",
                    )
                else:
                    _finish(db, tj, "cancelled")
                return
            events, cursor, st = remote.poll_once(
                cb, tj.adapter_name, tj.log_cursor, runtime=runtime
            )  # type: ignore[arg-type]
            tj.log_cursor = cursor
            error = _apply_events(tj, events)
            tj.log_tail = remote.logs(cb, tj.adapter_name, 60, runtime=runtime)[-8000:]  # type: ignore[arg-type]
            tj.poll_failures = 0
            db.commit()
            if st.running:
                return
            if not st.exists:
                _finish(
                    db, tj, "failed", "the training job disappeared from the box (box restarted?)"
                )
                return
            if st.exit_code != 0:
                _finish(
                    db,
                    tj,
                    "failed",
                    error or f"training exited with code {st.exit_code}; see the log",
                )
                return
            tj.status = "fetching"
            db.commit()
            local = remote.fetch_adapter(
                cb, tj.adapter_name, config.workspace_dir(ws.slug) / "adapters"
            )
            remote.remove_container(cb, tj.adapter_name, runtime=runtime)  # type: ignore[arg-type]
    except (ComputeError, OSError, ValueError) as e:
        db.rollback()
        tj.poll_failures += 1
        tj.polled_at = datetime.now(UTC)
        if tj.poll_failures >= s.TRAIN_MAX_POLL_FAILURES:
            _finish(db, tj, "failed", f"box unreachable for {tj.poll_failures} polls: {e}"[:2000])
        else:
            db.commit()
        return
    _create_adapter(db, tj, local)
    if tj.total_steps and tj.step < tj.total_steps:
        tj.step = tj.total_steps
    _finish(db, tj, "completed")


def poll_trainings(db: Session) -> int:
    """Advance every running training job whose last poll is older than TRAIN_POLL_SECONDS."""
    cutoff = datetime.now(UTC) - timedelta(seconds=get_settings().TRAIN_POLL_SECONDS)
    n = 0
    for tj in list(
        db.scalars(select(TrainingJob).where(TrainingJob.status.in_(("running", "fetching"))))
    ):
        last = _aware(tj.polled_at)
        if last is not None and last > cutoff:
            continue
        try:
            poll_training(db, tj)
            n += 1
        except Exception as e:  # noqa: BLE001 - one bad job must not stop the others
            db.rollback()
            print(f"[compute] poll {tj.id}: {type(e).__name__}: {e}", flush=True)
    return n


# ---- adapters ------------------------------------------------------------------------


def list_adapters(db: Session, ws: Workspace) -> list[Adapter]:
    return list(
        db.scalars(
            select(Adapter).where(Adapter.workspace_id == ws.id).order_by(Adapter.created_at.desc())
        )
    )


def get_adapter(db: Session, ws: Workspace, adapter_id: str) -> Adapter:
    a = db.get(Adapter, adapter_id)
    if a is None or a.workspace_id != ws.id:
        raise HTTPException(404, "adapter not found")
    return a


def adapter_out(db: Session, a: Adapter) -> dict[str, Any]:
    d = db.get(Dataset, a.dataset_id) if a.dataset_id else None
    served = [
        {"box_id": b.id, "box_name": b.name, "endpoint_url": b.endpoint_url}
        for b in C.list_boxes(db, db.get(Workspace, a.workspace_id))
        if b.serve_state == "serving" and a.name in (b.served_adapters or [])
    ]
    lineage = (a.metrics or {}).get("lineage") or {}
    return {
        "id": a.id, "name": a.name, "model_spec": f"vllm:{a.name}", "base_model": a.base_model,
        "pack": a.pack, "dataset_id": a.dataset_id, "dataset_name": d.name if d else lineage.get("dataset"),
        "dataset_deleted": a.dataset_id is None, "training_job_id": a.training_job_id,
        "lora_rank": a.lora_rank, "metrics": {k: v for k, v in (a.metrics or {}).items() if k != "lineage"},
        "source_runs": lineage.get("source_runs") or [], "test_run_id": a.test_run_id,
        "served_on": served, "created_at": a.created_at,
    }  # fmt: skip


def serve_adapter(db: Session, ws: Workspace, user: User, a: Adapter, box_id: str) -> Job:
    """Load the adapter on a box (with its base model). Keeps adapters already served
    there on the same base, so several can be compared side by side."""
    box = C.get_box(db, ws, box_id)
    same_base = box.serve_state == "serving" and box.served_model == a.base_model
    keep = (box.served_adapters or []) if same_base else []
    names = [*[n for n in keep if n != a.name], a.name]
    return C.request_serve(db, ws, user, box, model=a.base_model, adapters=names)


def test_adapter(
    db: Session, ws: Workspace, user: User, a: Adapter, box_id: str, episodes: int = 50
) -> Job:
    """Queue `adapter_test`: serve on the box if needed, then an eval run on seeds 0+."""
    C.require_gpu_admin(user)
    if not a.pack:
        raise HTTPException(
            400, "this adapter has no environment to test on (its dataset was deleted)"
        )
    if not 1 <= episodes <= 1000:
        raise HTTPException(400, "episodes must be 1..1000")
    box = C.get_box(db, ws, box_id)
    return C.enqueue_box_job(
        db, box, user, "adapter_test", {"adapter_id": a.id, "episodes": episodes}
    )


def run_adapter_test(db: Session, job: Job, box: GpuBox) -> None:
    """Worker: make sure the box serves the adapter, then queue the evaluation run."""
    from envforge.web.services import enqueue_run

    a = db.get(Adapter, (job.payload or {}).get("adapter_id"))
    if a is None or a.workspace_id != job.workspace_id:
        raise ValueError("adapter not found")
    ws = db.get(Workspace, a.workspace_id)
    user = db.get(User, job.created_by)
    if not (
        box.serve_state == "serving"
        and box.served_model == a.base_model
        and a.name in (box.served_adapters or [])
    ):
        same_base = box.serve_state == "serving" and box.served_model == a.base_model
        keep = (box.served_adapters or []) if same_base else []
        names = [*[n for n in keep if n != a.name], a.name]
        ranks = [x.lora_rank for x in db.scalars(select(Adapter).where(Adapter.workspace_id == ws.id,
                                                                     Adapter.name.in_(names))) if x.lora_rank]  # fmt: skip
        C.serve_on_box(
            db,
            job,
            box,
            model=a.base_model,
            adapters=names,
            max_lora_rank=C._lora_rank_bucket(ranks),
        )
    try:
        run = enqueue_run(
            db, ws, user, pack=a.pack or ws.default_pack, model=f"vllm:{a.name}",
            episodes=int((job.payload or {}).get("episodes") or 50), tiers=[], seed_start=0,
            label=f"{a.name} · test (eval seeds)",
        )  # fmt: skip
    except HTTPException as e:
        raise ValueError(str(e.detail)) from e
    a.test_run_id = run.id
    job.progress = f"queued run {run.id}"
    db.commit()


def adapter_archive(ws: Workspace, a: Adapter) -> Path:
    """A temporary .tar.gz of the local adapter directory (caller deletes it)."""
    root = (config.workspace_dir(ws.slug) / "adapters").resolve()
    src = Path(a.path).resolve()
    if not src.is_relative_to(root) or not src.is_dir():
        raise HTTPException(404, "the adapter files are not on this server")
    fd, tmp = tempfile.mkstemp(prefix=f"{a.name}-", suffix=".tar.gz")
    with open(fd, "wb") as f, tarfile.open(fileobj=f, mode="w:gz") as tar:
        tar.add(src, arcname=a.name)
    return Path(tmp)


def delete_adapter(db: Session, ws: Workspace, user: User, a: Adapter) -> None:
    """Remove the local copy and the record (a copy may remain on the GPU box)."""
    C.require_gpu_admin(user)
    root = (config.workspace_dir(ws.slug) / "adapters").resolve()
    target = Path(a.path).resolve()
    db.delete(a)
    db.commit()
    if target.is_relative_to(root) and target != root:
        shutil.rmtree(target, ignore_errors=True)


# ---- workspace deletion ----------------------------------------------------------------


def workspace_busy(db: Session, ws: Workspace) -> bool:
    """Queued/running compute jobs or training jobs in the workspace."""
    jobs = db.scalar(
        select(func.count())
        .select_from(Job)
        .where(Job.workspace_id == ws.id, Job.status.in_(C.ACTIVE))
    )
    trains = db.scalar(select(func.count()).select_from(TrainingJob).where(
        TrainingJob.workspace_id == ws.id, TrainingJob.status.in_(ACTIVE_TRAINING)))  # fmt: skip
    return bool(jobs or trains)


def delete_workspace_rows(db: Session, ws: Workspace) -> None:
    """Delete the workspace's boxes, datasets, training jobs, adapters and compute jobs
    (leaves first). Files go with the workspace directory."""
    from sqlalchemy import delete as sql_delete

    db.execute(sql_delete(Adapter).where(Adapter.workspace_id == ws.id))
    db.execute(sql_delete(TrainingJob).where(TrainingJob.workspace_id == ws.id))
    db.execute(sql_delete(Dataset).where(Dataset.workspace_id == ws.id))
    for box in C.list_boxes(db, ws):
        C.known_hosts_path(box.id).unlink(missing_ok=True)
    db.execute(sql_delete(GpuBox).where(GpuBox.workspace_id == ws.id))
    db.execute(sql_delete(Job).where(Job.workspace_id == ws.id))

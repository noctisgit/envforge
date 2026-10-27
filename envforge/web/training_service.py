"""Datasets, LoRA training jobs and adapters in the browser (Phase 3), reusing
`envforge.train` unchanged:

    datasets       train.dataset.build_dataset + overlap   (rejection sampling, manifest)
                   train.preference.build_preference_dataset (kind="preference": DPO pairs)
    training jobs  train.remote.start_training + poll_once + fetch_adapter
                   method sft | dpo (one run of a script on the box) | grpo (a loop, below)
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
from dataclasses import asdict, dataclass, fields
from datetime import UTC, datetime, timedelta
from pathlib import Path
from statistics import mean
from typing import Any

from fastapi import HTTPException
from sqlalchemy import func, or_, select, update
from sqlalchemy.orm import Session

from envforge import config
from envforge.compute.base import ComputeError
from envforge.compute.vllm import stop_vllm
from envforge.core.env import ALL_TASKS_TIER, TRAIN_SEED_MIN
from envforge.train import dataset as ds
from envforge.train import preference as pref
from envforge.train import remote
from envforge.web import compute_service as C
from envforge.web.models import (
    Adapter,
    Dataset,
    EnvVersion,
    GpuBox,
    Job,
    Run,
    TrainingJob,
    User,
    Workspace,
)
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


def _check_kind(kind: str, max_pairs_per_task: int) -> None:
    if kind not in ("sft", "preference"):
        raise HTTPException(400, "kind must be sft or preference")
    if not 1 <= max_pairs_per_task <= 32:
        raise HTTPException(400, "pairs per task must be between 1 and 32")


def _check_filters(min_reward: float, val_frac: float) -> None:
    if not 0 <= min_reward <= 1:
        raise HTTPException(400, "min reward must be between 0 and 1")
    if not 0 <= val_frac <= 0.5:
        raise HTTPException(400, "validation split must be between 0 and 0.5")


def _pref_summary(out_dir: Path, rep: pref.PreferenceReport) -> dict[str, Any]:
    """Counts the UI shows for preference pairs: pairs, split, how many tasks could pair."""
    models: dict[str, int] = {}
    for split in ("train.jsonl", "val.jsonl"):
        p = out_dir / split
        if p.exists():
            for line in p.read_text(encoding="utf-8").splitlines():
                meta = json.loads(line).get("meta", {})
                for side in ("chosen", "rejected"):
                    key = f"{side}: {meta.get(side, {}).get('model', '?')}"
                    models[key] = models.get(key, 0) + 1
    return {
        "kind": "preference",
        "episodes_seen": rep.episodes_seen,
        "kept": rep.pairs,
        "n_train": rep.n_train,
        "n_val": rep.n_val,
        "tasks": rep.tasks_with_pairs,
        "chosen_episodes": rep.chosen_episodes,
        "rejected_episodes": rep.rejected_episodes,
        "tasks_without_rejected": rep.tasks_without_rejected,
        "tasks_without_chosen": rep.tasks_without_chosen,
        "by_pack_tier": {f"{p} · tier {t}": n for (p, t), n in sorted(rep.by_pack_tier.items())},
        "models": models,
    }


def _summary(out_dir: Path, rep: ds.DatasetReport | pref.PreferenceReport) -> dict[str, Any]:
    """Counts the UI shows: split, examples per task, per pack/tier, models."""
    if isinstance(rep, pref.PreferenceReport):
        return _pref_summary(out_dir, rep)
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
           val_frac: float, best_per_task: bool, kind: str = "sft",
           max_pairs_per_task: int = 4) -> ds.DatasetReport | pref.PreferenceReport:  # fmt: skip
    if kind == "preference":
        return pref.build_preference_dataset(
            [Path(r.run_dir) for r in runs if r.run_dir],
            out_dir,
            chosen_min_reward=min_reward,
            max_pairs_per_task=max_pairs_per_task,
            val_frac=val_frac,
            pack_resolver=_pack_resolver(db, ws),
        )
    return ds.build_dataset(
        [Path(r.run_dir) for r in runs if r.run_dir],
        out_dir,
        min_reward=min_reward,
        val_frac=val_frac,
        best_per_task=best_per_task,
        pack_resolver=_pack_resolver(db, ws),
    )


def preview_dataset(db: Session, ws: Workspace, *, run_ids: list[str], min_reward: float = 1.0,
                    val_frac: float = 0.05, best_per_task: bool = True, kind: str = "sft",
                    max_pairs_per_task: int = 4) -> dict[str, Any]:  # fmt: skip
    """What saving would produce, built in a scratch directory and thrown away."""
    _check_filters(min_reward, val_frac)
    _check_kind(kind, max_pairs_per_task)
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
                kind=kind,
                max_pairs_per_task=max_pairs_per_task,
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
    if kind == "preference":
        if summary["kept"] < 50:
            warnings.append(
                "DPO works best with a hundred or more pairs; this is fine for a smoke test."
            )
        if summary["tasks_without_chosen"]:
            warnings.append(
                f"{summary['tasks_without_chosen']} task(s) were never solved, so they have no chosen "
                "side: a teacher run on the same training seeds would supply one."
            )
    elif summary["kept"] < 200:
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
                 min_reward: float = 1.0, val_frac: float = 0.05, best_per_task: bool = True,
                 kind: str = "sft", max_pairs_per_task: int = 4) -> Dataset:  # fmt: skip
    """Build and keep a named dataset under data/workspaces/<slug>/sft/<name>/."""
    try:
        name = remote.check_name(name.strip())
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    _check_filters(min_reward, val_frac)
    _check_kind(kind, max_pairs_per_task)
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
        rep = _build(db, ws, runs, out, min_reward=min_reward, val_frac=val_frac,
                     best_per_task=best_per_task, kind=kind, max_pairs_per_task=max_pairs_per_task)  # fmt: skip
    except ValueError as e:
        shutil.rmtree(out, ignore_errors=True)
        raise HTTPException(400, str(e)) from e
    _sanitize_files(db, ws, out, packs)
    summary = _summary(out, rep)
    filters: dict[str, Any] = {"min_reward": min_reward, "val_frac": val_frac}
    if kind == "preference":
        filters["max_pairs_per_task"] = max_pairs_per_task
    else:
        filters["best_per_task"] = best_per_task
    row = Dataset(
        workspace_id=ws.id,
        name=name,
        kind=kind,
        source_runs=[r.id for r in runs],
        filters=filters,
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
        "id": d.id, "name": d.name, "kind": d.kind, "source_runs": d.source_runs, "filters": d.filters,
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


@dataclass(frozen=True)
class GRPOLoop:
    """The GRPO loop's own settings (the per-round update's are remote.GRPOArgs)."""

    rounds: int = 4  # serve -> rollouts -> update, this many times
    tasks_per_round: int = 16  # different training tasks attempted per round
    group_size: int = 8  # attempts per task; advantages compare an attempt with its group
    temperature: float = 1.0  # sampling temperature for the attempts (0 would make them identical)


#: Defaults per method: what the form starts from and what an omitted field means.
METHOD_DEFAULTS: dict[str, dict[str, Any]] = {
    "sft": TRAIN_DEFAULTS,
    "dpo": asdict(remote.DPOArgs()),
    "grpo": {**asdict(remote.GRPOArgs()), **asdict(GRPOLoop())},
}
LIMITS = {
    "epochs": (0.05, 20), "lr": (1e-7, 1e-2), "lora_r": (1, 256), "lora_alpha": (1, 1024),
    "max_len": (256, 65536), "batch_size": (1, 64), "grad_accum": (1, 512), "max_steps": (1, 1_000_000),
    "beta": (0.001, 10), "kl_beta": (0, 10), "clip_eps": (0.01, 0.9), "rounds": (1, 100),
    "tasks_per_round": (1, 500), "group_size": (2, 32), "temperature": (0.1, 2),
}  # fmt: skip
INT_FIELDS = {"lora_r", "lora_alpha", "max_len", "batch_size", "grad_accum", "max_steps", "rounds",
              "tasks_per_round", "group_size"}  # fmt: skip


def _hyperparams(method: str, params: dict[str, Any]) -> dict[str, Any]:
    """The method's defaults with the given (range-checked) values on top."""
    defaults = METHOD_DEFAULTS[method]
    clean: dict[str, Any] = {}
    for k, v in (params or {}).items():
        if k not in defaults or v is None:
            continue
        lo, hi = LIMITS[k]
        if not lo <= float(v) <= hi:
            raise HTTPException(400, f"{k} must be between {lo:g} and {hi:g}")
        clean[k] = int(v) if k in INT_FIELDS else float(v)
    return {**defaults, **clean}


def _train_args(params: dict[str, Any]) -> remote.TrainArgs:
    return remote.TrainArgs(**_hyperparams("sft", params))


def _script_args(method: str, hp: dict[str, Any]) -> remote.TrainingArgs:
    """The dataclass whose argv the method's script takes, from stored hyperparams."""
    cls = {"sft": remote.TrainArgs, "dpo": remote.DPOArgs, "grpo": remote.GRPOArgs}[method]
    names = {f.name for f in fields(cls)}
    return cls(**{k: v for k, v in hp.items() if k in names})


def _init_adapter(db: Session, ws: Workspace, name: str | None, base: str) -> Adapter:
    """The workspace adapter DPO / GRPO start from; it must sit on the same base model."""
    if not name:
        raise HTTPException(400, "pick the adapter to start from (usually your SFT adapter)")
    a = db.scalar(select(Adapter).where(Adapter.workspace_id == ws.id, Adapter.name == name))
    if a is None:
        raise HTTPException(404, f"adapter {name} not found")
    if a.base_model != base:
        raise HTTPException(
            400, f"{name} was trained on {a.base_model}; an adapter only works on that base model"
        )
    return a


def start_training(db: Session, ws: Workspace, user: User, *, dataset_id: str | None = None,
                   box_id: str, base_model: str, adapter_name: str | None = None,
                   hyperparams: dict[str, Any] | None = None, method: str = "sft",
                   init_adapter: str | None = None, pack: str | None = None) -> TrainingJob:  # fmt: skip
    """Queue a training job on a box.

    sft: imitate a dataset's solved episodes. dpo: prefer the chosen side of a
    preference dataset, continuing from `init_adapter`. grpo: reinforcement learning on
    environment `pack` from `init_adapter`, no dataset (it makes its own data each round).
    """
    C.require_gpu_admin(user)
    if method not in METHOD_DEFAULTS:
        raise HTTPException(400, "method must be sft, dpo or grpo")
    box = C.get_box(db, ws, box_id)
    C.refuse_while_runs_use(db, box, "training")
    d = get_dataset(db, ws, dataset_id) if dataset_id else None
    if method in ("sft", "dpo"):
        if d is None:
            raise HTTPException(400, "pick a dataset")
        want = "preference" if method == "dpo" else "sft"
        if d.kind != want:
            raise HTTPException(
                400, f"{method.upper()} needs a{'' if want == 'sft' else ' preference'} dataset "
                f"(kind {want}); {d.name} is kind {d.kind}",
            )  # fmt: skip
    try:
        base = C.check_model_id(base_model)
        name = remote.check_name((adapter_name or (d.name if d else "")).strip())
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    hp = _hyperparams(method, hyperparams or {})
    init = _init_adapter(db, ws, init_adapter, base) if method in ("dpo", "grpo") else None
    env_name = None
    if method == "grpo":
        from envforge.web.builder import resolve_pack

        env_name = pack or (init.pack if init else None)
        if not env_name:
            raise HTTPException(400, "pick the environment to train on")
        resolve_pack(db, ws, env_name)  # 404 unless built in or published in this workspace
        if len(name) > 58:  # rounds are saved as <name>-r<k> on the box
            raise HTTPException(400, "GRPO adapter names can be at most 58 characters")
        if hp["tasks_per_round"] * hp["group_size"] > 1000:  # a run's episode cap
            raise HTTPException(400, "tasks per round x attempts per task must be at most 1000")
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
        dataset_id=d.id if d else None,
        box_id=box.id,
        adapter_name=name,
        base_model=base,
        method=method,
        init_adapter=init.name if init else None,
        # a retry after a failed/cancelled job of the same name may overwrite its leftovers
        hyperparams={
            **hp,
            "dataset_name": d.name if d else None,
            "pack": env_name,
            "replace": bool(same),
        },  # fmt: skip
        hourly_usd=box.hourly_usd,
        status="queued",
        created_by=user.id,
    )
    db.add(tj)
    db.flush()
    if method == "grpo":
        tj.state = {"round": 0, "phase": "rollout", "current": init.name, "rounds": [],
                    "steps_done": 0}  # fmt: skip
        tj.total_steps = hp["rounds"]
        job = C.enqueue_box_job(db, box, user, "grpo_rollout", {"training_job_id": tj.id})
    else:
        job = C.enqueue_box_job(db, box, user, "train_start", {"training_job_id": tj.id})
    tj.job_id = job.id
    db.commit()
    return tj


def run_train_start(db: Session, job: Job, box: GpuBox) -> None:
    """Worker: stop vLLM if it's running, upload the dataset + sft.py and start training
    detached (train.remote)."""
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
    args = _script_args(tj.method or "sft", params)
    hf_token = C.hf_token_for(db, box, tj.created_by)
    with C.open_box(db, box) as cb:
        runtime = C.resolve_box_runtime(db, box, cb)
        if box.serve_state != "stopped":
            # vLLM reserves ~90% of the GPU's memory for its cache, so training can't share
            # the card: stop it (serving the trained adapter starts it again)
            job.progress = f"stopping vLLM on {box.name} to free the GPU for training"
            db.commit()
            stop_vllm(cb, box.vllm_port, runtime=runtime)  # type: ignore[arg-type]
            box.serve_state, box.serve_error = "stopped", None
            box.served_model = box.served_adapters = box.endpoint_url = None
            db.commit()
        if tj.init_adapter:  # DPO continues from this adapter: it must be on the box
            C.ensure_adapter_on_box(db, cb, box, tj.init_adapter, job)
        remote.start_training(
            cb, Path(d.path), base=tj.base_model, name=tj.adapter_name, args=args,
            hf_token=hf_token, replace=bool(params.get("replace")), runtime=runtime,  # type: ignore[arg-type]
            method=tj.method or "sft", init_adapter=tj.init_adapter,
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
        "method": tj.method or "sft", "init_adapter": tj.init_adapter, "pack": tj.hyperparams.get("pack"),
        "dataset_id": tj.dataset_id, "dataset_name": tj.hyperparams.get("dataset_name"),
        "box_id": tj.box_id, "box_name": box.name if box else None,
        "hyperparams": {k: v for k, v in tj.hyperparams.items() if k in METHOD_DEFAULTS[tj.method or "sft"]},
        "grpo": _grpo_out(tj) if tj.method == "grpo" else None,
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


def _grpo_out(tj: TrainingJob) -> dict[str, Any]:
    """Where a GRPO run is: round, phase, the rollout run, and each finished round's numbers."""
    st = tj.state or {}
    return {"round": st.get("round", 0) + 1, "rounds_total": tj.hyperparams.get("rounds"),
            "phase": st.get("phase"), "run_id": st.get("run_id"), "rounds": st.get("rounds", [])}  # fmt: skip


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
                point = {"step": tj.step, "loss": round(float(ev["loss"]), 5)}
                if isinstance(ev.get("accuracy"), int | float):  # DPO: chosen preferred
                    point["accuracy"] = round(float(ev["accuracy"]), 4)
                losses.append(point)
        elif kind == "eval":
            if isinstance(ev.get("loss"), int | float):
                point = {
                    "step": int(ev.get("step") or tj.step),
                    "eval_loss": round(float(ev["loss"]), 5),
                }
                if isinstance(ev.get("accuracy"), int | float):
                    point["eval_accuracy"] = round(float(ev["accuracy"]), 4)
                losses.append(point)
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
            "final_train_accuracy", "final_eval_accuracy", "trainable_params", "dataset_sha256",
            "started_at", "finished_at")  # fmt: skip
    metrics = {k: manifest[k] for k in keep if k in manifest}
    metrics["method"] = tj.method or "sft"
    if tj.method == "grpo":  # the reward curve is the result that matters
        metrics["rounds"] = (tj.state or {}).get("rounds", [])
    metrics["lineage"] = {"dataset": tj.hyperparams.get("dataset_name"), "packs": d.packs if d else None,
                          "source_runs": d.source_runs if d else None, "method": tj.method or "sft",
                          "init_adapter": tj.init_adapter}  # fmt: skip
    pack = d.packs[0] if d and d.packs else tj.hyperparams.get("pack")
    if pack is None and tj.init_adapter:  # DPO on a mixed dataset: test where the parent was
        parent = db.scalar(select(Adapter).where(Adapter.workspace_id == tj.workspace_id,
                                                 Adapter.name == tj.init_adapter))  # fmt: skip
        pack = parent.pack if parent else None
    a = Adapter(
        workspace_id=tj.workspace_id, name=tj.adapter_name, base_model=tj.base_model,
        pack=pack, dataset_id=tj.dataset_id,
        training_job_id=tj.id, box_id=tj.box_id, path=str(local), lora_rank=rank,
        metrics=metrics, created_by=tj.created_by,
    )  # fmt: skip
    db.add(a)
    db.commit()
    return a


def poll_training(db: Session, tj: TrainingJob) -> None:
    """One poll of a running job: progress, logs, and the end (fetch the adapter)."""
    if tj.method == "grpo":
        poll_grpo(db, tj)
        return
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


# ---- GRPO: reinforcement learning against the environment's verifier --------------------
#
# One training job, many rounds. Round r (0-based) runs four phases, kept in tj.state:
#
#   rollout   compute job grpo_rollout: serve base + the current adapter, then queue a
#             rollout *run*: tasks_per_round training tasks x group_size attempts each
#             (Run.repeats), sampled at `temperature`, on seeds 10000 + r*tasks_per_round
#   rollouts  the runs worker plays them (8 at a time on vLLM); polls wait for the run
#   train     compute job grpo_train: score each group -> advantages, drop groups whose
#             attempts all scored the same (no signal), stop vLLM, upload, start
#             grpo_step.py: current adapter -> <name>-r<r+1> (the last round: <name>)
#   training  polls follow grpo_step.py like an SFT job; then round r+1 starts
#
# The KL anchor of every update is the adapter the run started from (init_adapter).
# Each round's rollout run is an ordinary run in the Runs list, so the reward curve and
# every attempt's replay are there to look at.


def _grpo_job(db: Session, job: Job) -> TrainingJob:
    tj = db.get(TrainingJob, (job.payload or {}).get("training_job_id"))
    if tj is None or tj.workspace_id != job.workspace_id or tj.method != "grpo":
        raise ValueError("GRPO training job not found")
    return tj


def _round_out(tj: TrainingJob, r: int) -> str:
    """The adapter round r's update writes: <name>-r<k>, or <name> for the last round."""
    return tj.adapter_name if r + 1 >= tj.hyperparams["rounds"] else f"{tj.adapter_name}-r{r + 1}"


def grpo_rollouts(
    run_dir: Path, tools: list[dict[str, Any]] | None
) -> tuple[list[dict], dict[str, Any]]:
    """Score a rollout run's attempts against their group: `(rows, round stats)`.

    A group is every attempt at the same task. advantage = (reward - mean) / std within
    the group; groups whose attempts all scored the same teach nothing and are dropped.
    Attempts that crashed on an infrastructure error don't count at all.
    """
    from envforge.core.trajectory import load_run
    from envforge.train.filter import to_chat_example

    groups: dict[tuple, list] = {}
    for t in load_run(run_dir):
        if not t.error:
            groups.setdefault((t.task.pack, t.task.tier, t.task.seed), []).append(t)
    rows, rewards, informative = [], [], 0
    for key, attempts in sorted(groups.items()):
        rs = [t.reward for t in attempts]
        rewards += rs
        mu = sum(rs) / len(rs)
        std = (sum((x - mu) ** 2 for x in rs) / len(rs)) ** 0.5
        if len(rs) < 2 or std < 1e-6:
            continue
        informative += 1
        for t, x in zip(attempts, rs, strict=True):
            ex = to_chat_example(t, tools)
            rows.append({"messages": ex["messages"], "tools": tools, "advantage": (x - mu) / std,
                         "reward": x, "group": "/".join(map(str, key)), "meta": ex["meta"]})  # fmt: skip
    stats = {
        "attempts": len(rewards),
        "mean_reward": round(sum(rewards) / len(rewards), 4) if rewards else 0.0,
        "solve_rate": round(sum(x >= 1.0 for x in rewards) / len(rewards), 4) if rewards else 0.0,
        "groups": len(groups),
        "informative_groups": informative,
        "trained_on": len(rows),
    }
    return rows, stats


def run_grpo_rollout(db: Session, job: Job, box: GpuBox) -> None:
    """Worker, phase `rollout`: serve the current adapter, queue the round's rollout run."""
    from envforge.web.services import enqueue_run

    tj = _grpo_job(db, job)
    if tj.cancel_requested or tj.status == "cancelled":
        _finish(db, tj, "cancelled")
        return
    st, hp = dict(tj.state or {}), tj.hyperparams
    r = st["round"]
    if tj.status == "queued":
        tj.status, tj.started_at = "running", datetime.now(UTC)
    tj.polled_at = datetime.now(UTC)
    db.commit()
    current = st["current"]
    serving = (box.serve_state == "serving" and box.served_model == tj.base_model
               and current in (box.served_adapters or []))  # fmt: skip
    if not serving:  # (a retry after the run limit was hit finds it already serving)
        ranks = [a.lora_rank for a in db.scalars(select(Adapter).where(
            Adapter.workspace_id == tj.workspace_id, Adapter.name == tj.init_adapter)) if a.lora_rank]  # fmt: skip
        C.serve_on_box(db, job, box, model=tj.base_model, adapters=[current],
                       max_lora_rank=C._lora_rank_bucket(ranks))  # fmt: skip
    ws = db.get(Workspace, tj.workspace_id)
    user = db.get(User, tj.created_by)
    try:
        run = enqueue_run(
            db, ws, user, pack=hp["pack"], model=f"vllm:{current}",
            episodes=hp["tasks_per_round"] * hp["group_size"], tiers=list(hp.get("tiers") or []),
            seed_start=TRAIN_SEED_MIN + r * hp["tasks_per_round"], temperature=hp["temperature"],
            label=f"{tj.adapter_name} · GRPO round {r + 1}/{hp['rounds']} rollouts",
            repeats=hp["group_size"],
        )  # fmt: skip
    except HTTPException as e:
        if e.status_code == 429:  # the workspace's active-run limit: wait for a free slot
            tj.state = {**st, "phase": "queue_rollout"}
            job.progress = f"waiting for a free run slot: {e.detail}"
            db.commit()
            return
        raise ValueError(str(e.detail)) from e
    tj.state = {**st, "phase": "rollouts", "run_id": run.id}
    job.progress = f"queued rollout run {run.id}"
    db.commit()


def run_grpo_train(db: Session, job: Job, box: GpuBox) -> None:
    """Worker, phase `train`: score the rollouts, free the GPU, start this round's update."""
    from envforge.web.builder import resolve_pack

    tj = _grpo_job(db, job)
    if tj.cancel_requested or tj.status == "cancelled":
        _finish(db, tj, "cancelled")
        return
    st, hp = dict(tj.state or {}), tj.hyperparams
    r = st["round"]
    run = db.get(Run, st.get("run_id"))
    if run is None or not run.run_dir:
        raise ValueError("the round's rollout run is gone")
    ws = db.get(Workspace, tj.workspace_id)
    tools = ds.tool_schemas(resolve_pack(db, ws, hp["pack"], include_deleted=True))
    rows, stats = grpo_rollouts(Path(run.run_dir), tools)
    rounds = [*st.get("rounds", []), {"round": r + 1, "run_id": run.id, **stats}]
    out = _round_out(tj, r)
    if not rows:
        # every group was all-solved or all-failed: nothing to learn from this round
        tj.state = {**st, "rounds": rounds}
        db.commit()
        _next_round(db, tj, skipped=True)
        return
    # its own namespace (not sft/<name>, which datasets use), removed once uploaded
    data_dir = config.workspace_dir(ws.slug) / "grpo" / tj.id / f"round-{r + 1}"
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "train.jsonl").write_text("".join(json.dumps(x) + "\n" for x in rows))
    job.progress = f"round {r + 1}: training on {len(rows)} attempts"
    db.commit()
    hf_token = C.hf_token_for(db, box, tj.created_by)
    try:
        with C.open_box(db, box) as cb:
            runtime = C.resolve_box_runtime(db, box, cb)
            if box.serve_state != "stopped":  # vLLM holds the GPU's memory; training needs it
                stop_vllm(cb, box.vllm_port, runtime=runtime)  # type: ignore[arg-type]
                box.serve_state, box.serve_error = "stopped", None
                box.served_model = box.served_adapters = box.endpoint_url = None
                db.commit()
            C.ensure_adapter_on_box(db, cb, box, tj.init_adapter or st["current"], job)
            remote.start_training(
                cb, data_dir, base=tj.base_model, name=out, args=_script_args("grpo", hp),
                hf_token=hf_token, replace=True, runtime=runtime,  # type: ignore[arg-type]
                method="grpo", init_adapter=st["current"], ref_adapter=tj.init_adapter,
            )  # fmt: skip
    finally:
        shutil.rmtree(data_dir, ignore_errors=True)  # on the box now (or the round failed)
    tj.runtime = runtime
    tj.log_cursor = 0
    tj.state = {**st, "rounds": rounds, "phase": "training", "out": out, "round_steps": 0}
    db.commit()


def _next_round(db: Session, tj: TrainingJob, *, skipped: bool = False) -> None:
    """Move to the next round's rollouts, or finish after the last one."""
    st, hp = dict(tj.state or {}), tj.hyperparams
    r = st["round"]
    if not skipped:
        st["current"] = st["out"]
    st["steps_done"] = st.get("steps_done", 0) + st.get("round_steps", 0)
    tj.step = r + 1
    if r + 1 >= hp["rounds"]:
        tj.state = {**st, "phase": "fetching"}
        db.commit()
        return  # the next poll fetches the final adapter
    # the next poll queues the round's grpo_rollout job (this may run inside a job, and a
    # box takes one job at a time)
    tj.state = {**st, "round": r + 1, "phase": "queue_rollout", "run_id": None, "out": None}
    db.commit()


def _enqueue_phase(db: Session, tj: TrainingJob, box: GpuBox, kind: str, phase: str) -> None:
    """Queue the compute job for the next phase; if the box is busy, the next poll retries."""
    try:
        job = C.enqueue_box_job(
            db, box, db.get(User, tj.created_by), kind, {"training_job_id": tj.id}
        )
    except HTTPException:
        db.rollback()
        return
    tj.job_id = job.id
    tj.state = {**(tj.state or {}), "phase": phase}
    db.commit()


def _apply_grpo_events(tj: TrainingJob, events: list[dict[str, Any]]) -> str | None:
    """Fold grpo_step.py's events into the row, with steps counted across rounds."""
    st = dict(tj.state or {})
    losses = list(tj.losses or [])
    error = None
    for ev in events:
        if ev.get("event") == "log" and isinstance(ev.get("loss"), int | float):
            st["round_steps"] = int(ev.get("step") or 0)
            losses.append({"step": st.get("steps_done", 0) + st["round_steps"], "round": st["round"] + 1,
                           "loss": round(float(ev["loss"]), 5), "kl": round(float(ev.get("kl") or 0), 6)})  # fmt: skip
        elif ev.get("event") == "error":
            error = str(ev.get("message") or "training failed")[:2000]
    tj.losses, tj.state = losses[-2000:], st
    return error


def _grpo_cancel(db: Session, tj: TrainingJob, cb: Any | None) -> None:
    st = tj.state or {}
    run = db.get(Run, st.get("run_id")) if st.get("run_id") else None
    if run is not None and run.status in ("queued", "running"):
        run.status = "cancelled"  # the runs worker stops it between episodes
    if cb is not None and st.get("phase") == "training" and st.get("out"):
        remote.remove_container(cb, st["out"], runtime=tj.runtime or "docker")  # type: ignore[arg-type]
    job = db.get(Job, tj.job_id) if tj.job_id else None
    if job is not None and job.status == "queued":
        job.status, job.finished_at = "cancelled", datetime.now(UTC)


def poll_grpo(db: Session, tj: TrainingJob) -> None:
    """One poll of a GRPO job: advance whichever phase it is in."""
    s = get_settings()
    st = dict(tj.state or {})
    phase = st.get("phase")
    box = db.get(GpuBox, tj.box_id) if tj.box_id else None
    if box is None:
        _finish(db, tj, "failed", "the GPU box was removed while training")
        return
    tj.polled_at = datetime.now(UTC)
    started = _aware(tj.started_at) or datetime.now(UTC)
    too_long = datetime.now(UTC) - started > timedelta(hours=s.TRAIN_MAX_HOURS)
    if (tj.cancel_requested or too_long) and phase != "training":
        _grpo_cancel(db, tj, None)
        if too_long:
            _finish(
                db, tj, "failed", f"stopped: ran longer than TRAIN_MAX_HOURS={s.TRAIN_MAX_HOURS:g}"
            )
        else:
            _finish(db, tj, "cancelled")
        return
    if phase in ("rollout", "train"):  # a compute job is doing this phase
        db.commit()
        return
    if phase == "queue_rollout":
        _enqueue_phase(db, tj, box, "grpo_rollout", "rollout")
        return
    if phase == "rollouts":
        run = db.get(Run, st.get("run_id"))
        if run is None or run.status in ("failed", "cancelled"):
            why = (run.error if run else None) or (run.status if run else "the run is gone")
            _finish(db, tj, "failed", f"round {st['round'] + 1} rollouts: {why}"[:2000])
        elif run.status == "completed":
            _enqueue_phase(db, tj, box, "grpo_train", "train")
        else:
            db.commit()
        return
    try:
        with C.open_box(db, box) as cb:
            if phase == "training":
                if tj.cancel_requested or too_long:
                    _grpo_cancel(db, tj, cb)
                    _finish(db, tj, "cancelled" if not too_long else "failed",
                            None if not too_long else f"stopped: ran longer than TRAIN_MAX_HOURS={s.TRAIN_MAX_HOURS:g}")  # fmt: skip
                    return
                runtime = tj.runtime or "docker"
                events, cursor, status = remote.poll_once(
                    cb, st["out"], tj.log_cursor, runtime=runtime
                )  # type: ignore[arg-type]
                tj.log_cursor = cursor
                error = _apply_grpo_events(tj, events)
                tj.log_tail = remote.logs(cb, st["out"], 60, runtime=runtime)[-8000:]  # type: ignore[arg-type]
                tj.poll_failures = 0
                db.commit()
                if status.running:
                    return
                if not status.exists or status.exit_code != 0:
                    _finish(db, tj, "failed", error or f"round {st['round'] + 1} update exited with code "
                            f"{status.exit_code}; see the log")  # fmt: skip
                    return
                remote.remove_container(cb, st["out"], runtime=runtime)  # type: ignore[arg-type]
                _next_round(db, tj)
                return
            if phase == "fetching":
                ws = db.get(Workspace, tj.workspace_id)
                final = (tj.state or {}).get("current")
                if final == tj.init_adapter:
                    _finish(db, tj, "failed", "no round had a learning signal (every group of attempts "
                            "scored the same); try more tasks per round or a higher temperature")  # fmt: skip
                    return
                local = remote.fetch_adapter(cb, final, config.workspace_dir(ws.slug) / "adapters")
    except (ComputeError, OSError, ValueError) as e:
        db.rollback()
        tj.poll_failures += 1
        tj.polled_at = datetime.now(UTC)
        if tj.poll_failures >= s.TRAIN_MAX_POLL_FAILURES:
            _finish(db, tj, "failed", f"box unreachable for {tj.poll_failures} polls: {e}"[:2000])
        else:
            db.commit()
        return
    if phase == "fetching":
        if final != tj.adapter_name:  # the last round was skipped: keep the job's name
            target = local.with_name(tj.adapter_name)
            shutil.rmtree(target, ignore_errors=True)
            local = local.rename(target)
        _create_adapter(db, tj, local)
        _finish(db, tj, "completed")


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
        "method": lineage.get("method") or "sft", "init_adapter": lineage.get("init_adapter"),
        "served_on": served, "created_at": a.created_at, "test_plan": test_plan(db, a),
    }  # fmt: skip


def test_plan(db: Session, a: Adapter) -> dict[str, Any] | None:
    """What "Test" scores the adapter on: {"pack", "every_task_once", "episodes"}.

    For an environment built in the builder that is its newest published version, because
    a task stays on the same side of the held-out split in every version: newer versions
    only add or fix evaluation tasks, never move training tasks into the test. Such an
    environment has a fixed task list, so the test covers every held-out task once, the
    same tasks other models are scored on with "every task once". A generated pack (netops)
    is tested on its own name for the number of episodes the caller picks (episodes None)."""
    if not a.pack:
        return None
    from envforge.web.builder import resolve_pack

    ws = db.get(Workspace, a.workspace_id)
    pack = a.pack
    current = db.scalar(
        select(EnvVersion).where(EnvVersion.workspace_id == ws.id, EnvVersion.pack_name == pack)
    )
    if current is not None:
        newest = db.scalar(
            select(EnvVersion)
            .where(EnvVersion.project_id == current.project_id, EnvVersion.deleted_at.is_(None))
            .order_by(EnvVersion.version.desc())
            .limit(1)
        )
        pack = newest.pack_name if newest is not None else pack
    try:
        counts = resolve_pack(db, ws, pack).task_counts()
    except Exception:  # noqa: BLE001 - a deleted environment: the run will say so
        counts = None
    return {"pack": pack, "every_task_once": counts is not None,
            "episodes": counts["eval"] if counts else None}  # fmt: skip


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
    """Queue `adapter_test`: serve on the box if needed, then an eval run on seeds 0+
    (see `test_plan` for which environment version and how many episodes)."""
    C.require_gpu_admin(user)
    if not a.pack:
        raise HTTPException(
            400, "this adapter has no environment to test on (its dataset was deleted)"
        )
    if not 1 <= episodes <= 1000:
        raise HTTPException(400, "episodes must be 1..1000")
    box = C.get_box(db, ws, box_id)
    C.refuse_while_training(db, box)
    served = (
        box.serve_state == "serving"
        and box.served_model == a.base_model
        and a.name in (box.served_adapters or [])
    )
    if not served:  # testing it means restarting vLLM with it
        C.refuse_while_runs_use(db, box, "serving this adapter")
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
    plan = test_plan(db, a) or {"pack": ws.default_pack, "every_task_once": False}
    if plan["every_task_once"]:
        episodes, tiers, what = plan["episodes"], [ALL_TASKS_TIER], "every task once"
    else:
        episodes, tiers, what = int((job.payload or {}).get("episodes") or 50), [], "eval seeds"
    try:
        run = enqueue_run(
            db, ws, user, pack=plan["pack"], model=f"vllm:{a.name}", episodes=episodes,
            tiers=tiers, seed_start=0, label=f"{a.name} · test ({what})",
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

"""/api/workspaces/{ws}/datasets, /training and /adapters — Phase 3 in the browser.

Thin: the logic (seed rules, overlap checks, admin-only, file locations) lives in
envforge/web/training_service.py.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Query
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session
from starlette.background import BackgroundTask

from envforge.web import compute_service as C
from envforge.web import services as svc
from envforge.web import training_service as T
from envforge.web.auth import current_user
from envforge.web.compute_schemas import AdapterBoxIn, DatasetFilterIn, DatasetIn, TrainIn
from envforge.web.db import get_db
from envforge.web.models import User

router = APIRouter(prefix="/api/workspaces/{ws_id}", tags=["training"])


# ---- datasets ----


@router.get("/datasets/sources")
def dataset_sources(
    ws_id: str, user: User = Depends(current_user), db: Session = Depends(get_db)
) -> list[dict]:
    """Completed runs, each marked eligible (training seeds) or with the reason it isn't."""
    ws = svc.require_workspace(db, user, ws_id)
    return T.dataset_sources(db, ws)


@router.post("/datasets/preview")
def preview_dataset(ws_id: str, body: DatasetFilterIn, user: User = Depends(current_user),
                    db: Session = Depends(get_db)) -> dict:  # fmt: skip
    """Counts, split, examples per task and eval-seed overlap, without saving anything."""
    ws = svc.require_workspace(db, user, ws_id)
    return T.preview_dataset(db, ws, **body.model_dump())


@router.get("/datasets")
def list_datasets(
    ws_id: str, user: User = Depends(current_user), db: Session = Depends(get_db)
) -> list[dict]:
    ws = svc.require_workspace(db, user, ws_id)
    return [T.dataset_out(d) for d in T.list_datasets(db, ws)]


@router.post("/datasets", status_code=201)
def save_dataset(
    ws_id: str, body: DatasetIn, user: User = Depends(current_user), db: Session = Depends(get_db)
) -> dict:
    ws = svc.require_workspace(db, user, ws_id)
    return T.dataset_out(T.save_dataset(db, ws, user, **body.model_dump()), full=True)


@router.get("/datasets/{dataset_id}")
def get_dataset(
    ws_id: str, dataset_id: str, user: User = Depends(current_user), db: Session = Depends(get_db)
) -> dict:
    ws = svc.require_workspace(db, user, ws_id)
    return T.dataset_out(T.get_dataset(db, ws, dataset_id), full=True)


@router.get("/datasets/{dataset_id}/download")
def download_dataset(ws_id: str, dataset_id: str, split: str = Query("train", pattern="^(train|val)$"),
                     user: User = Depends(current_user), db: Session = Depends(get_db)) -> FileResponse:  # fmt: skip
    """train.jsonl or val.jsonl (chat format, the file sft.py reads)."""
    ws = svc.require_workspace(db, user, ws_id)
    d = T.get_dataset(db, ws, dataset_id)
    return FileResponse(T.dataset_file(ws, d, split), media_type="application/x-ndjson",
                        filename=f"{d.name}-{split}.jsonl")  # fmt: skip


@router.delete("/datasets/{dataset_id}", status_code=204)
def delete_dataset(
    ws_id: str, dataset_id: str, user: User = Depends(current_user), db: Session = Depends(get_db)
) -> None:
    ws = svc.require_workspace(db, user, ws_id)
    T.delete_dataset(db, ws, user, T.get_dataset(db, ws, dataset_id))


# ---- training jobs ----


@router.get("/training")
def list_training(
    ws_id: str, user: User = Depends(current_user), db: Session = Depends(get_db)
) -> dict:
    ws = svc.require_workspace(db, user, ws_id)
    return {
        "jobs": [T.training_out(db, tj) for tj in T.list_trainings(db, ws)],
        "defaults": {**T.TRAIN_DEFAULTS, "base_model": ws.train_base_model},
        "method_defaults": T.METHOD_DEFAULTS,  # sft / dpo / grpo: what the form starts from
        "can_manage": C.can_manage(user),
    }


@router.post("/training", status_code=202)
def start_training(
    ws_id: str, body: TrainIn, user: User = Depends(current_user), db: Session = Depends(get_db)
) -> dict:
    ws = svc.require_workspace(db, user, ws_id)
    params = body.model_dump(exclude={"dataset_id", "box_id", "base_model", "adapter_name", "method",
                                      "init_adapter", "pack"})  # fmt: skip
    tj = T.start_training(db, ws, user, dataset_id=body.dataset_id, box_id=body.box_id,
                          base_model=body.base_model, adapter_name=body.adapter_name, hyperparams=params,
                          method=body.method, init_adapter=body.init_adapter, pack=body.pack)  # fmt: skip
    return T.training_out(db, tj, full=True)


@router.get("/training/{tj_id}")
def get_training(
    ws_id: str, tj_id: str, user: User = Depends(current_user), db: Session = Depends(get_db)
) -> dict:
    """Status, step/epoch, the loss curve and the tail of the log."""
    ws = svc.require_workspace(db, user, ws_id)
    return T.training_out(db, T.get_training(db, ws, tj_id), full=True)


@router.post("/training/{tj_id}/cancel")
def cancel_training(
    ws_id: str, tj_id: str, user: User = Depends(current_user), db: Session = Depends(get_db)
) -> dict:
    ws = svc.require_workspace(db, user, ws_id)
    return T.training_out(
        db, T.request_cancel(db, ws, user, T.get_training(db, ws, tj_id)), full=True
    )


# ---- adapters ----


@router.get("/adapters")
def list_adapters(
    ws_id: str, user: User = Depends(current_user), db: Session = Depends(get_db)
) -> list[dict]:
    ws = svc.require_workspace(db, user, ws_id)
    return [T.adapter_out(db, a) for a in T.list_adapters(db, ws)]


@router.get("/adapters/{adapter_id}")
def get_adapter(
    ws_id: str, adapter_id: str, user: User = Depends(current_user), db: Session = Depends(get_db)
) -> dict:
    ws = svc.require_workspace(db, user, ws_id)
    return T.adapter_out(db, T.get_adapter(db, ws, adapter_id))


@router.post("/adapters/{adapter_id}/serve", status_code=202)
def serve_adapter(ws_id: str, adapter_id: str, body: AdapterBoxIn, user: User = Depends(current_user),
                  db: Session = Depends(get_db)) -> dict:  # fmt: skip
    """Load the adapter (with its base model) on a box via vLLM --lora-modules."""
    ws = svc.require_workspace(db, user, ws_id)
    job = T.serve_adapter(db, ws, user, T.get_adapter(db, ws, adapter_id), body.box_id)
    return {"job_id": job.id}


@router.post("/adapters/{adapter_id}/test", status_code=202)
def test_adapter(ws_id: str, adapter_id: str, body: AdapterBoxIn, user: User = Depends(current_user),
                 db: Session = Depends(get_db)) -> dict:  # fmt: skip
    """Serve it if needed, then queue an evaluation run on seeds 0+ for its environment."""
    ws = svc.require_workspace(db, user, ws_id)
    job = T.test_adapter(
        db, ws, user, T.get_adapter(db, ws, adapter_id), body.box_id, body.episodes
    )
    return {"job_id": job.id}


@router.get("/adapters/{adapter_id}/download")
def download_adapter(ws_id: str, adapter_id: str, user: User = Depends(current_user),
                     db: Session = Depends(get_db)) -> FileResponse:  # fmt: skip
    """The adapter directory as a .tar.gz (PEFT format: load it on top of its base model)."""
    ws = svc.require_workspace(db, user, ws_id)
    a = T.get_adapter(db, ws, adapter_id)
    path = T.adapter_archive(ws, a)
    return FileResponse(path, media_type="application/gzip", filename=f"{a.name}.tar.gz",
                        background=BackgroundTask(path.unlink, missing_ok=True))  # fmt: skip


@router.delete("/adapters/{adapter_id}", status_code=204)
def delete_adapter(
    ws_id: str, adapter_id: str, user: User = Depends(current_user), db: Session = Depends(get_db)
) -> None:
    ws = svc.require_workspace(db, user, ws_id)
    T.delete_adapter(db, ws, user, T.get_adapter(db, ws, adapter_id))

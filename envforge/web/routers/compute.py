"""/api/workspaces/{ws}/gpu-boxes — add, list, delete, check, serve and stop GPU boxes.

Thin: every rule (workspace scoping, admin-only, validation, key handling) lives in
envforge/web/compute_service.py.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Response
from sqlalchemy.orm import Session

from envforge.web import compute_service as C
from envforge.web import services as svc
from envforge.web.auth import current_user
from envforge.web.compute_schemas import BoxIn, ServeIn
from envforge.web.db import get_db
from envforge.web.models import User
from envforge.web.settings import get_settings

router = APIRouter(prefix="/api/workspaces/{ws_id}/gpu-boxes", tags=["gpu-boxes"])


@router.get("")
def list_boxes(
    ws_id: str, user: User = Depends(current_user), db: Session = Depends(get_db)
) -> dict:
    """The workspace's boxes (keys masked) and whether you may manage them."""
    ws = svc.require_workspace(db, user, ws_id)
    return {
        "boxes": [C.box_out(db, b) for b in C.list_boxes(db, ws)],
        "can_manage": C.can_manage(user),
        "admin_only": get_settings().GPU_BOXES_ADMIN_ONLY,
    }


@router.post("", status_code=201)
def create_box(
    ws_id: str, body: BoxIn, user: User = Depends(current_user), db: Session = Depends(get_db)
) -> dict:
    ws = svc.require_workspace(db, user, ws_id)
    return C.box_out(db, C.create_box(db, ws, user, body.model_dump()))


@router.delete("/{box_id}")
def delete_box(ws_id: str, box_id: str, force: bool = False, user: User = Depends(current_user),
               db: Session = Depends(get_db)) -> Response:  # fmt: skip
    """Stop vLLM on the box, then forget it (202: a job does it). `force=true` forgets
    the box now without touching the machine (204)."""
    ws = svc.require_workspace(db, user, ws_id)
    job = C.delete_box(db, ws, user, C.get_box(db, ws, box_id), force=force)
    return Response(status_code=202 if job else 204)


@router.post("/{box_id}/check", status_code=202)
def check_box(
    ws_id: str, box_id: str, user: User = Depends(current_user), db: Session = Depends(get_db)
) -> dict:
    """Queue a readiness check (ssh, GPU, driver/CUDA, disk, vLLM)."""
    ws = svc.require_workspace(db, user, ws_id)
    box = C.get_box(db, ws, box_id)
    C.request_check(db, ws, user, box)
    return C.box_out(db, box)


@router.post("/{box_id}/serve", status_code=202)
def serve(ws_id: str, box_id: str, body: ServeIn, user: User = Depends(current_user),
          db: Session = Depends(get_db)) -> dict:  # fmt: skip
    """Queue vLLM serving an open model (plus optional LoRA adapters) on the box."""
    ws = svc.require_workspace(db, user, ws_id)
    box = C.get_box(db, ws, box_id)
    C.request_serve(db, ws, user, box, model=body.model, adapters=body.adapters,
                    max_model_len=body.max_model_len, gpu_memory_utilization=body.gpu_memory_utilization)  # fmt: skip
    return C.box_out(db, box)


@router.post("/{box_id}/stop", status_code=202)
def stop(
    ws_id: str, box_id: str, user: User = Depends(current_user), db: Session = Depends(get_db)
) -> dict:
    ws = svc.require_workspace(db, user, ws_id)
    box = C.get_box(db, ws, box_id)
    C.request_stop(db, ws, user, box)
    return C.box_out(db, box)

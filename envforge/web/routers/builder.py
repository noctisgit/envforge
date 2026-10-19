"""/api/workspaces/{ws}/builder — environment builder projects; /api/workspaces/{ws}/packs.

Everything is scoped to a workspace the caller belongs to (svc.require_workspace),
so projects, uploads and published environments are as private as the workspace.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from envforge.web import builder as B
from envforge.web import services as svc
from envforge.web.auth import current_user
from envforge.web.db import get_db
from envforge.web.models import EnvProject, EnvRevision, EnvUpload, EnvVersion, User

router = APIRouter(prefix="/api/workspaces/{ws_id}", tags=["builder"])


class ProjectIn(BaseModel):
    title: str = Field(min_length=1, max_length=200)
    purpose: str = Field(default="", max_length=4000)


class ProjectPatch(BaseModel):
    title: str | None = Field(default=None, min_length=1, max_length=200)
    purpose: str | None = Field(default=None, max_length=4000)


class MessageIn(BaseModel):
    content: str = Field(min_length=1, max_length=20000)


class SpecIn(BaseModel):
    spec: dict[str, Any]
    message: str = Field(default="Edited the spec by hand", max_length=500)


class PublishIn(BaseModel):
    revision_id: str | None = None  # default: head


# ---- shaping ---------------------------------------------------------------------------


def _rev_out(
    r: EnvRevision, head_id: str | None, published: dict[str, int], numbers: dict[str, int]
) -> dict:
    v = r.validation or {}
    return {
        "id": r.id,
        "number": r.number,
        "parent_number": numbers.get(r.parent_id or ""),
        "message": r.message,
        "author": r.author,
        "n_tasks": r.n_tasks,
        "spec_ok": v.get("spec_ok", False),
        "tasks_ok": v.get("tasks_ok", 0),
        "task_problem_count": len(v.get("task_problems") or {}),
        "is_head": r.id == head_id,
        "published_version": published.get(r.id),
        "created_at": r.created_at,
    }


def _versions(db: Session, p: EnvProject) -> list[EnvVersion]:
    from sqlalchemy import select

    return list(
        db.scalars(
            select(EnvVersion).where(EnvVersion.project_id == p.id).order_by(EnvVersion.version)
        )
    )


def _upload_out(u: EnvUpload) -> dict:
    return {
        "id": u.id,
        "filename": u.filename,
        "size_bytes": u.size_bytes,
        "status": u.status,
        "format": u.detected_format,
        "n_conversations": u.n_conversations,
        "redactions": u.redactions,
        "profile": u.profile,
        "error": u.error,
        "created_at": u.created_at,
    }


def _project_out(db: Session, ws, p: EnvProject, user: User) -> dict:
    from envforge.web.builder_agent import Ctx, readiness

    h = B.head(db, p)
    job = B.active_job(db, p)
    versions = _versions(db, p)
    return {
        "id": p.id,
        "name": p.name,
        "title": p.title,
        "purpose": p.purpose,
        "created_at": p.created_at,
        "head": {
            "id": h.id,
            "number": h.number,
            "spec": h.spec,
            "n_tasks": h.n_tasks,
            "validation": h.validation,
        }
        if h
        else None,  # fmt: skip
        "readiness": readiness(Ctx(db, ws, p, user, model=None)),
        "versions": [
            {
                "version": v.version,
                "pack_name": v.pack_name,
                "revision_id": v.revision_id,
                "created_at": v.created_at,
            }
            for v in versions
        ],  # fmt: skip
        "job": {"id": job.id, "kind": job.kind, "status": job.status, "progress": job.progress}
        if job
        else None,
        "uploads": [_upload_out(u) for u in B.list_uploads(db, p)],
        "builder_model": ws.builder_model,
    }


# ---- projects ----------------------------------------------------------------------------


@router.get("/builder/projects")
def list_projects(ws_id: str, user: User = Depends(current_user), db: Session = Depends(get_db)):
    ws = svc.require_workspace(db, user, ws_id)
    out = []
    for p in B.list_projects(db, ws):
        h = B.head(db, p)
        out.append({
            "id": p.id, "name": p.name, "title": p.title, "purpose": p.purpose[:300],
            "created_at": p.created_at, "head_number": h.number if h else None,
            "n_tasks": h.n_tasks if h else 0, "versions": len(_versions(db, p)),
        })  # fmt: skip
    return out


@router.post("/builder/projects", status_code=201)
def create_project(
    ws_id: str, body: ProjectIn, user: User = Depends(current_user), db: Session = Depends(get_db)
):
    ws = svc.require_workspace(db, user, ws_id)
    p = B.create_project(db, ws, user, body.title, body.purpose)
    return _project_out(db, ws, p, user)


@router.get("/builder/projects/{pid}")
def get_project(
    ws_id: str, pid: str, user: User = Depends(current_user), db: Session = Depends(get_db)
):
    ws = svc.require_workspace(db, user, ws_id)
    return _project_out(db, ws, B.get_project(db, ws, pid), user)


@router.patch("/builder/projects/{pid}")
def patch_project(
    ws_id: str,
    pid: str,
    body: ProjectPatch,
    user: User = Depends(current_user),
    db: Session = Depends(get_db),
):
    ws = svc.require_workspace(db, user, ws_id)
    p = B.get_project(db, ws, pid)
    if body.title is not None:
        p.title = body.title
    if body.purpose is not None:
        p.purpose = body.purpose
    db.commit()
    return _project_out(db, ws, p, user)


@router.delete("/builder/projects/{pid}", status_code=204)
def delete_project(
    ws_id: str, pid: str, user: User = Depends(current_user), db: Session = Depends(get_db)
) -> None:
    """Hide the project. Published versions stay runnable (runs may reference them)."""
    from datetime import UTC, datetime

    ws = svc.require_workspace(db, user, ws_id)
    p = B.get_project(db, ws, pid)
    if B.active_job(db, p):
        raise HTTPException(409, "the builder is busy with this project; try again in a moment")
    p.deleted_at = datetime.now(UTC)
    db.commit()


# ---- chat -------------------------------------------------------------------------------


def _message_out(m) -> dict:
    out = {
        "id": m.id,
        "seq": m.seq,
        "role": m.role,
        "content": m.content,
        "created_at": m.created_at,
    }
    if m.role == "assistant" and m.tool_calls:
        out["tool_calls"] = [
            {"name": c.get("name"), "args_preview": str(c.get("args"))[:200]} for c in m.tool_calls
        ]
    if m.role == "tool":
        out["name"] = m.name
        out["meta"] = m.meta
        out["content"] = m.content[:600]
    return out


@router.get("/builder/projects/{pid}/messages")
def get_messages(
    ws_id: str,
    pid: str,
    after: int = 0,
    user: User = Depends(current_user),
    db: Session = Depends(get_db),
):
    ws = svc.require_workspace(db, user, ws_id)
    p = B.get_project(db, ws, pid)
    job = B.active_job(db, p)
    return {
        "messages": [_message_out(m) for m in B.messages(db, p) if m.seq > after],
        "job": {"status": job.status, "progress": job.progress} if job else None,
    }


@router.post("/builder/projects/{pid}/messages", status_code=202)
def post_message(
    ws_id: str,
    pid: str,
    body: MessageIn,
    user: User = Depends(current_user),
    db: Session = Depends(get_db),
):
    ws = svc.require_workspace(db, user, ws_id)
    p = B.get_project(db, ws, pid)
    provider = svc.provider_of(ws.builder_model)
    if provider == "scripted" or not (
        svc.user_has_key(db, user.id, provider) or svc.get_settings().ALLOW_SERVER_KEYS
    ):
        raise HTTPException(
            400, f"the builder model is {ws.builder_model}: add a {provider} key on your Profile or pick "
            "another builder model in Workspace settings"
        )  # fmt: skip
    if not p.purpose.strip():
        p.purpose = body.content[:4000]
    B.add_message(db, p, "user", body.content)
    job = B.enqueue_job(db, "builder_turn", p, user)
    return {"job_id": job.id}


# ---- revisions -----------------------------------------------------------------------------


@router.get("/builder/projects/{pid}/revisions")
def list_revisions(
    ws_id: str, pid: str, user: User = Depends(current_user), db: Session = Depends(get_db)
):
    ws = svc.require_workspace(db, user, ws_id)
    p = B.get_project(db, ws, pid)
    revs = B.list_revisions(db, p)
    numbers = {r.id: r.number for r in revs}
    published = {v.revision_id: v.version for v in _versions(db, p)}
    return [_rev_out(r, p.head_revision_id, published, numbers) for r in revs]


@router.get("/builder/projects/{pid}/revisions/{rid}")
def get_revision(
    ws_id: str,
    pid: str,
    rid: str,
    user: User = Depends(current_user),
    db: Session = Depends(get_db),
):
    ws = svc.require_workspace(db, user, ws_id)
    p = B.get_project(db, ws, pid)
    r = B.get_revision(db, p, rid)
    return {"id": r.id, "number": r.number, "message": r.message, "spec": r.spec, "validation": r.validation,
            "n_tasks": r.n_tasks}  # fmt: skip


@router.get("/builder/projects/{pid}/revisions/{rid}/diff")
def diff_revision(
    ws_id: str, pid: str, rid: str, against: str | None = None,
    user: User = Depends(current_user), db: Session = Depends(get_db),
):  # fmt: skip
    ws = svc.require_workspace(db, user, ws_id)
    p = B.get_project(db, ws, pid)
    b = B.get_revision(db, p, rid)
    a = B.get_revision(db, p, against) if against else None
    return B.diff(db, ws, p, a, b)


@router.post("/builder/projects/{pid}/revisions/{rid}/restore")
def restore_revision(
    ws_id: str,
    pid: str,
    rid: str,
    user: User = Depends(current_user),
    db: Session = Depends(get_db),
):
    ws = svc.require_workspace(db, user, ws_id)
    p = B.get_project(db, ws, pid)
    rev = B.restore(db, ws, p, B.get_revision(db, p, rid), user)
    B.add_message(
        db,
        p,
        "user",
        f"(Restored revision {rev.number} from revision {B.get_revision(db, p, rid).number}.)",
    )
    return {"revision": rev.number}


@router.post("/builder/projects/{pid}/revisions/{rid}/checkout")
def checkout_revision(
    ws_id: str,
    pid: str,
    rid: str,
    user: User = Depends(current_user),
    db: Session = Depends(get_db),
):
    ws = svc.require_workspace(db, user, ws_id)
    p = B.get_project(db, ws, pid)
    r = B.get_revision(db, p, rid)
    B.checkout(db, p, r)
    B.add_message(
        db, p, "user", f"(Checked out revision {r.number}; new changes branch from here.)"
    )
    return {"head": r.number}


@router.put("/builder/projects/{pid}/spec")
def edit_spec(
    ws_id: str,
    pid: str,
    body: SpecIn,
    user: User = Depends(current_user),
    db: Session = Depends(get_db),
):
    from pydantic import ValidationError

    from envforge.declarative.spec import EnvSpec

    ws = svc.require_workspace(db, user, ws_id)
    p = B.get_project(db, ws, pid)
    try:
        EnvSpec.model_validate(body.spec)
    except ValidationError as e:
        raise HTTPException(
            400, "; ".join(f"{'.'.join(map(str, x['loc']))}: {x['msg']}" for x in e.errors()[:5])
        ) from e
    rev = B.create_revision(db, ws, p, spec=body.spec, message=body.message, author="user")
    return {"revision": rev.number, "validation": rev.validation}


@router.get("/builder/projects/{pid}/revisions/{rid}/tasks")
def revision_tasks(
    ws_id: str, pid: str, rid: str, limit: int = Query(200, le=2000),
    user: User = Depends(current_user), db: Session = Depends(get_db),
):  # fmt: skip
    ws = svc.require_workspace(db, user, ws_id)
    p = B.get_project(db, ws, pid)
    r = B.get_revision(db, p, rid)
    problems = (r.validation or {}).get("task_problems") or {}
    out = []
    for t in B.load_tasks(ws, p, r.tasks_sha)[:limit]:
        tid = str(t.get("id"))
        out.append({"id": tid, "tier": t.get("tier"), "goal": str(t.get("goal", ""))[:400],
                    "n_success": len(t.get("success") or []), "n_reference": len(t.get("reference") or []),
                    "source": t.get("source"), "problems": problems.get(tid, [])})  # fmt: skip
    return out


# ---- uploads ---------------------------------------------------------------------------------


@router.post("/builder/projects/{pid}/uploads", status_code=202)
async def upload(
    ws_id: str, pid: str, request: Request, filename: str = Query(..., max_length=255),
    user: User = Depends(current_user), db: Session = Depends(get_db),
):  # fmt: skip
    """Raw request body = the file (streamed to disk). It is parsed and redacted by the
    builder worker, then the original is deleted."""
    ws = svc.require_workspace(db, user, ws_id)
    p = B.get_project(db, ws, pid)
    if not filename.lower().endswith((".csv", ".txt", ".json", ".jsonl", ".log")):
        raise HTTPException(400, "supported files: .csv, .txt, .json, .jsonl, .log")
    up, raw = B.new_upload(db, ws, p, user, filename)
    size = 0
    try:
        with raw.open("wb") as f:
            async for chunk in request.stream():
                size += len(chunk)
                if size > B.MAX_UPLOAD_BYTES:
                    raise HTTPException(
                        413, f"file too large (limit {B.MAX_UPLOAD_BYTES // 2**20} MB); split it"
                    )
                f.write(chunk)
    except HTTPException:
        raw.unlink(missing_ok=True)
        db.rollback()
        raise
    if size == 0:
        raw.unlink(missing_ok=True)
        db.rollback()
        raise HTTPException(400, "empty file")
    up.size_bytes = size
    up.path = str(raw)
    db.commit()
    B.enqueue_job(db, "ingest", p, user, {"upload_id": up.id})
    return _upload_out(up)


@router.delete("/builder/projects/{pid}/uploads/{uid}", status_code=204)
def delete_upload(
    ws_id: str,
    pid: str,
    uid: str,
    user: User = Depends(current_user),
    db: Session = Depends(get_db),
) -> None:
    from pathlib import Path

    ws = svc.require_workspace(db, user, ws_id)
    p = B.get_project(db, ws, pid)
    up = db.get(EnvUpload, uid)
    if not up or up.project_id != p.id:
        raise HTTPException(404, "upload not found")
    if up.status == "processing":
        raise HTTPException(409, "still processing")
    if up.path:
        Path(up.path).unlink(missing_ok=True)
    db.delete(up)
    db.commit()


# ---- preview + publish ------------------------------------------------------------------------


@router.get("/builder/projects/{pid}/preview")
def preview(
    ws_id: str, pid: str, seed: int = Query(0, ge=0, le=2_000_000_000), tier: int | None = None,
    revision_id: str | None = None, user: User = Depends(current_user), db: Session = Depends(get_db),
):  # fmt: skip
    from envforge.core.explore import pack_overview, preview_task
    from envforge.web.builder_agent import head_env_class

    ws = svc.require_workspace(db, user, ws_id)
    p = B.get_project(db, ws, pid)
    r = B.get_revision(db, p, revision_id) if revision_id else B.head(db, p)
    if not r or not (r.validation or {}).get("spec_ok"):
        raise HTTPException(400, "needs a valid spec and at least one valid task")
    try:
        env_cls = head_env_class(ws, p, r)
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    t = tier if tier in env_cls.tiers else env_cls.tiers[0]
    return {"overview": pack_overview(env_cls), "preview": preview_task(env_cls, seed, t)}


@router.post("/builder/projects/{pid}/publish", status_code=201)
def publish(
    ws_id: str,
    pid: str,
    body: PublishIn,
    user: User = Depends(current_user),
    db: Session = Depends(get_db),
):
    ws = svc.require_workspace(db, user, ws_id)
    p = B.get_project(db, ws, pid)
    r = B.get_revision(db, p, body.revision_id) if body.revision_id else B.head(db, p)
    if r is None:
        raise HTTPException(400, "nothing to publish yet")
    v = B.publish(db, ws, p, r, user)
    B.add_message(db, p, "user", f"(Published revision {r.number} as {v.pack_name}.)")
    return {"version": v.version, "pack_name": v.pack_name}


# ---- workspace environments (built-in + published) ------------------------------------------


@router.get("/packs")
def workspace_packs(ws_id: str, user: User = Depends(current_user), db: Session = Depends(get_db)):
    ws = svc.require_workspace(db, user, ws_id)
    return [
        {
            "name": n,
            "kind": c.kind,
            "tiers": list(c.tiers),
            "max_steps": c.max_steps,
            "description": c.description,
            "custom": v is not None,
            "project_id": v.project_id if v else None,
        }  # fmt: skip
        for n, c, v in B.workspace_packs(db, ws)
    ]


@router.get("/packs/{name}")
def workspace_pack_detail(
    ws_id: str, name: str, user: User = Depends(current_user), db: Session = Depends(get_db)
):
    from envforge.core.explore import pack_overview

    ws = svc.require_workspace(db, user, ws_id)
    return pack_overview(B.resolve_pack(db, ws, name))


@router.get("/packs/{name}/preview")
def workspace_pack_preview(
    ws_id: str, name: str, seed: int = Query(0, ge=0, le=2_000_000_000), tier: int = 1,
    user: User = Depends(current_user), db: Session = Depends(get_db),
):  # fmt: skip
    from envforge.core.explore import preview_task

    ws = svc.require_workspace(db, user, ws_id)
    env_cls = B.resolve_pack(db, ws, name)
    if tier not in env_cls.tiers:
        raise HTTPException(400, f"tier must be one of {list(env_cls.tiers)}")
    return preview_task(env_cls, seed, tier)

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


class ApproveIn(BaseModel):
    quote_id: str = Field(min_length=1, max_length=64)


class SpecIn(BaseModel):
    spec: dict[str, Any]
    message: str = Field(default="Edited the spec by hand", max_length=500)


class PublishIn(BaseModel):
    revision_id: str | None = None  # default: head


class UploadInit(BaseModel):
    filename: str = Field(min_length=1, max_length=255)
    size_bytes: int = Field(gt=0)


class ImportIn(BaseModel):
    url: str = Field(min_length=8, max_length=2000)
    filename: str | None = Field(default=None, max_length=255)


class PrivacyIn(BaseModel):
    profile: str | None = Field(default=None, pattern="^(off|standard|strict)$")
    confirm_off: bool = False  # "off" is only for public or synthetic data
    ner: bool | None = None
    surrogates: bool | None = None
    tier2: str | None = Field(default=None, pattern="^(off|auto|all)$")
    threshold: float | None = Field(default=None, ge=0.005, le=0.5)


class ReviewIn(BaseModel):
    n: int | None = Field(default=None, ge=5, le=400)


class ReviewSubmit(BaseModel):
    review_id: str = Field(max_length=32)
    flagged: list[str] = Field(default_factory=list, max_length=400)
    notes: dict[str, str] | None = None


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


def _versions(db: Session, p: EnvProject, *, include_deleted: bool = False) -> list[EnvVersion]:
    from sqlalchemy import select

    query = select(EnvVersion).where(EnvVersion.project_id == p.id)
    if not include_deleted:
        query = query.where(EnvVersion.deleted_at.is_(None))
    return list(db.scalars(query.order_by(EnvVersion.version)))


def _upload_out(u: EnvUpload) -> dict:
    return {
        "id": u.id,
        "filename": u.filename,
        "size_bytes": u.size_bytes,
        "bytes_received": u.bytes_received,
        "source": u.source,
        "progress": u.progress,
        "chunk_bytes": B.CHUNK_BYTES,
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
        "privacy": _privacy_out(p),
        # a cost estimate waiting for the user's click, and conversations approved so far
        "extraction": {"quote": p.extraction_quote, "approved": p.extraction_allowance or 0},
        "spend_left_usd": svc.spend_left(db, ws),
    }


def _privacy_out(p: EnvProject) -> dict[str, Any]:
    from envforge.builder.review import sample_size_for

    review = B.review_summary(p)
    return {
        "profile": p.privacy_profile,
        "ner": p.privacy_ner,
        "surrogates": p.privacy_surrogates,
        "tier2": p.privacy_tier2,
        "threshold": p.privacy_threshold,
        "review_size": sample_size_for(p.privacy_threshold),
        "signed_off_at": p.privacy_signed_off_at,
        "review": review,
        # strict: extraction waits for a passed review; any profile: a failed one blocks
        "needs_signoff": p.privacy_profile == "strict" and p.privacy_signed_off_at is None,
        "blocked": bool(review and review.get("status") == "failed"),
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
    """Delete the project and the environments published from it. Runs already made on
    those environments keep their results and replays."""
    from datetime import UTC, datetime

    ws = svc.require_workspace(db, user, ws_id)
    p = B.get_project(db, ws, pid)
    if B.active_job(db, p):
        raise HTTPException(409, "the builder is busy with this project; try again in a moment")
    B.delete_versions(db, ws, user, _versions(db, p))
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
    return _start_turn(db, ws, p, user, body.content)


def _start_turn(db: Session, ws, p: EnvProject, user: User, content: str) -> dict:
    """Post a user message and queue the builder's turn on it."""
    provider = svc.provider_of(ws.builder_model)
    if provider == "scripted" or not (
        svc.user_has_key(db, user.id, provider) or svc.get_settings().ALLOW_SERVER_KEYS
    ):
        raise HTTPException(
            400, f"the builder model is {ws.builder_model}: add a {provider} key on your Profile or pick "
            "another builder model in Workspace settings"
        )  # fmt: skip
    svc.require_budget(db, ws)
    if not p.purpose.strip():
        p.purpose = content[:4000]
    B.add_message(db, p, "user", content)
    job = B.enqueue_job(db, "builder_turn", p, user)
    return {"job_id": job.id}


@router.post("/builder/projects/{pid}/extraction/approve", status_code=202)
def approve_extraction(ws_id: str, pid: str, body: ApproveIn, user: User = Depends(current_user),
                       db: Session = Depends(get_db)):  # fmt: skip
    """The user approves the builder's cost estimate (only a click does: never the model);
    the builder then continues with the approved conversations."""
    ws = svc.require_workspace(db, user, ws_id)
    p = B.get_project(db, ws, pid)
    if B.active_job(db, p):
        raise HTTPException(409, "the builder is still working; approve once it has finished")
    q = B.approve_extraction(db, ws, p, body.quote_id)
    cost = q.get("cost_usd")
    note = f" (estimated ${cost:.2f})" if isinstance(cost, (int, float)) else ""
    return _start_turn(db, ws, p, user, f"Approved: extract up to {q['conversations']} "
                       f"conversations{note}. Please continue.")  # fmt: skip


@router.delete("/builder/projects/{pid}/extraction/quote", status_code=204)
def decline_extraction(
    ws_id: str, pid: str, user: User = Depends(current_user), db: Session = Depends(get_db)
) -> None:
    ws = svc.require_workspace(db, user, ws_id)
    B.decline_extraction(db, B.get_project(db, ws, pid))


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
    _check_name(filename)
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
    except Exception:  # too large, or the client went away mid-upload: keep no partial file
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


def _check_name(filename: str) -> None:
    name = filename.lower().removesuffix(".gz")  # gzip of any supported file is fine
    if not name.endswith(B.SUPPORTED_SUFFIXES):
        raise HTTPException(
            400, "supported files: " + ", ".join(B.SUPPORTED_SUFFIXES) + " (optionally .gz)"
        )


@router.post("/builder/projects/{pid}/uploads/init", status_code=201)
def init_upload(
    ws_id: str,
    pid: str,
    body: UploadInit,
    user: User = Depends(current_user),
    db: Session = Depends(get_db),
):
    """Start a chunked upload: the browser then PUTs CHUNK_BYTES pieces in order. Files of
    any size (up to MAX_CORPUS_BYTES) get through proxies that cap request sizes, and an
    interrupted upload resumes from `bytes_received`."""
    ws = svc.require_workspace(db, user, ws_id)
    p = B.get_project(db, ws, pid)
    _check_name(body.filename)
    limit = svc.get_settings().MAX_CORPUS_BYTES
    if body.size_bytes > limit:
        raise HTTPException(
            413,
            f"file too large (limit {limit / 2**30:.0f} GiB) — import it from a URL or split it",
        )
    B.check_disk_for(ws, body.size_bytes)
    up, raw = B.new_upload(db, ws, p, user, body.filename)
    up.size_bytes, up.status, up.path = body.size_bytes, "receiving", str(raw)
    raw.write_bytes(b"")
    db.commit()
    return _upload_out(up)


@router.put("/builder/projects/{pid}/uploads/{uid}/chunk")
async def upload_chunk(
    ws_id: str, pid: str, uid: str, request: Request, offset: int = Query(..., ge=0),
    user: User = Depends(current_user), db: Session = Depends(get_db),
):  # fmt: skip
    """Append one piece at `offset`. A wrong offset returns 409 with the server's
    bytes_received, so the client can resume exactly where the server is."""
    ws = svc.require_workspace(db, user, ws_id)
    p = B.get_project(db, ws, pid)
    up = db.get(EnvUpload, uid)
    if not up or up.project_id != p.id:
        raise HTTPException(404, "upload not found")
    if up.status != "receiving" or not up.path:
        raise HTTPException(409, f"upload is {up.status}")
    if offset != up.bytes_received:
        raise HTTPException(
            409, {"message": "offset mismatch", "bytes_received": up.bytes_received}
        )
    from pathlib import Path

    written = 0
    with Path(up.path).open("ab") as f:
        async for chunk in request.stream():
            written += len(chunk)
            if written > B.CHUNK_BYTES + 1024 or up.bytes_received + written > up.size_bytes:
                f.truncate(up.bytes_received)
                raise HTTPException(413, "chunk too large or past the declared size")
            f.write(chunk)
    up.bytes_received += written
    db.commit()
    return {"bytes_received": up.bytes_received, "size_bytes": up.size_bytes}


@router.post("/builder/projects/{pid}/uploads/{uid}/complete", status_code=202)
def complete_upload(
    ws_id: str,
    pid: str,
    uid: str,
    user: User = Depends(current_user),
    db: Session = Depends(get_db),
):
    ws = svc.require_workspace(db, user, ws_id)
    p = B.get_project(db, ws, pid)
    up = db.get(EnvUpload, uid)
    if not up or up.project_id != p.id:
        raise HTTPException(404, "upload not found")
    if up.status != "receiving":
        raise HTTPException(409, f"upload is {up.status}")
    if up.bytes_received != up.size_bytes:
        raise HTTPException(409, {"message": "incomplete", "bytes_received": up.bytes_received})
    up.status = "uploaded"
    db.commit()
    B.enqueue_job(db, "ingest", p, user, {"upload_id": up.id})
    return _upload_out(up)


@router.post("/builder/projects/{pid}/uploads/import", status_code=202)
def import_url(
    ws_id: str,
    pid: str,
    body: ImportIn,
    user: User = Depends(current_user),
    db: Session = Depends(get_db),
):
    """Have the builder worker download a file from an https URL (e.g. a presigned S3/R2
    link) — the way to bring in multi-GB archives without going through the browser."""
    from urllib.parse import urlparse

    from envforge.web.security import check_base_url

    ws = svc.require_workspace(db, user, ws_id)
    p = B.get_project(db, ws, pid)
    url = body.url.strip()
    if urlparse(url).scheme != "https":
        raise HTTPException(400, "only https URLs can be imported")
    try:
        check_base_url(url)  # public hosts only: never the server itself or private networks
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    name = body.filename or (urlparse(url).path.rsplit("/", 1)[-1] or "import.jsonl")
    _check_name(name)
    up, raw = B.new_upload(db, ws, p, user, name)
    up.source, up.source_url, up.status, up.path = "url", url, "uploaded", str(raw)
    db.commit()
    B.enqueue_job(db, "ingest", p, user, {"upload_id": up.id})
    return _upload_out(up)


@router.patch("/builder/projects/{pid}/privacy")
def set_privacy(
    ws_id: str,
    pid: str,
    body: PrivacyIn,
    user: User = Depends(current_user),
    db: Session = Depends(get_db),
):
    """Privacy settings. Redaction settings apply to future uploads (already-processed
    uploads keep the redaction they got; re-upload to change) and reset the review."""
    ws = svc.require_workspace(db, user, ws_id)
    p = B.get_project(db, ws, pid)
    B.privacy_settings(p, body.model_dump())
    db.commit()
    return _project_out(db, ws, p, user)


@router.get("/builder/projects/{pid}/corpus")
def corpus(ws_id: str, pid: str, user: User = Depends(current_user), db: Session = Depends(get_db)):
    """Corpus statistics and topic clusters (redacted data only)."""
    ws = svc.require_workspace(db, user, ws_id)
    return {"corpus": B.corpus_overview(ws, B.get_project(db, ws, pid))}


@router.get("/builder/projects/{pid}/corpus/clusters/{cluster}")
def cluster_examples(
    ws_id: str, pid: str, cluster: int, n: int = Query(3, ge=1, le=10),
    user: User = Depends(current_user), db: Session = Depends(get_db),
):  # fmt: skip
    ws = svc.require_workspace(db, user, ws_id)
    return {"conversations": B.cluster_examples(ws, B.get_project(db, ws, pid), cluster, n)}


@router.post("/builder/projects/{pid}/privacy/review", status_code=201)
def start_review(
    ws_id: str, pid: str, body: ReviewIn,
    user: User = Depends(current_user), db: Session = Depends(get_db),
):  # fmt: skip
    """Draw a fresh random sample of redacted conversations for a person to check."""
    ws = svc.require_workspace(db, user, ws_id)
    review, convs = B.start_review(db, ws, B.get_project(db, ws, pid), user, body.n)
    return {"review": {k: v for k, v in review.items() if k != "ids"}, "conversations": convs}


@router.get("/builder/projects/{pid}/privacy/review")
def get_review(
    ws_id: str, pid: str, user: User = Depends(current_user), db: Session = Depends(get_db)
):
    ws = svc.require_workspace(db, user, ws_id)
    p = B.get_project(db, ws, pid)
    return {"review": B.review_summary(p), "conversations": B.open_review_conversations(ws, p)}


@router.post("/builder/projects/{pid}/privacy/review/submit")
def submit_review(
    ws_id: str, pid: str, body: ReviewSubmit,
    user: User = Depends(current_user), db: Session = Depends(get_db),
):  # fmt: skip
    """Which sampled conversations still contain personal data -> leak rate, pass/fail."""
    ws = svc.require_workspace(db, user, ws_id)
    p = B.get_project(db, ws, pid)
    B.submit_review(db, p, user, body.review_id, body.flagged, body.notes)
    return _project_out(db, ws, p, user)


@router.delete("/builder/projects/{pid}/uploads/{uid}", status_code=204)
def delete_upload(
    ws_id: str,
    pid: str,
    uid: str,
    user: User = Depends(current_user),
    db: Session = Depends(get_db),
) -> None:
    ws = svc.require_workspace(db, user, ws_id)
    p = B.get_project(db, ws, pid)
    up = db.get(EnvUpload, uid)
    if not up or up.project_id != p.id:
        raise HTTPException(404, "upload not found")
    if up.status == "processing":
        raise HTTPException(409, "still processing")
    B.delete_upload(db, p, up, user)


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
    rep = getattr(v, "privacy_report", {}) or {}
    dropped = rep.get("dropped") or {}
    note = f" Privacy re-scan ({rep.get('profile')}): {sum((rep.get('redactions') or {}).values())} redaction(s)"
    note += f", {len(dropped)} task(s) left out because sanitising broke them." if dropped else "."
    B.add_message(db, p, "user", f"(Published revision {r.number} as {v.pack_name}.{note})")
    return {"version": v.version, "pack_name": v.pack_name, "privacy": rep}


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
    return pack_overview(B.resolve_pack(db, ws, name, include_deleted=True))


@router.delete("/packs/{name}", status_code=204)
def delete_workspace_pack(
    ws_id: str, name: str, user: User = Depends(current_user), db: Session = Depends(get_db)
) -> None:
    """Delete a published environment (`<project>@v<n>`). Built-in packs can't be deleted."""
    from sqlalchemy import select

    ws = svc.require_workspace(db, user, ws_id)
    ver = db.scalar(
        select(EnvVersion).where(
            EnvVersion.workspace_id == ws.id,
            EnvVersion.pack_name == name,
            EnvVersion.deleted_at.is_(None),
        )
    )
    if ver is None:
        raise HTTPException(404, f"no published environment '{name}' in this workspace")
    B.delete_versions(db, ws, user, [ver])


@router.get("/packs/{name}/preview")
def workspace_pack_preview(
    ws_id: str, name: str, seed: int = Query(0, ge=0, le=2_000_000_000), tier: int = 1,
    user: User = Depends(current_user), db: Session = Depends(get_db),
):  # fmt: skip
    from envforge.core.explore import preview_task

    ws = svc.require_workspace(db, user, ws_id)
    env_cls = B.resolve_pack(db, ws, name, include_deleted=True)
    if tier not in env_cls.tiers:
        raise HTTPException(400, f"tier must be one of {list(env_cls.tiers)}")
    return preview_task(env_cls, seed, tier)

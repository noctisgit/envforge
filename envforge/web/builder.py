"""Environment builder: projects, revisions, uploads, publishing, jobs. Routers stay thin.

On disk, per project:

    data/workspaces/<ws>/builder/<project>/
        uploads/<upload-id>.raw            the file as uploaded — deleted after processing
        uploads/<upload-id>.jsonl          redacted conversations, one per line
        tasks/<sha256>.jsonl               task records (content-addressed, shared by revisions)
    data/workspaces/<ws>/environments/<project>/v<n>/   published: spec.json + tasks.jsonl

Revisions form a DAG (see models.py). Publishing copies a revision into an immutable
environment directory and registers it as the pack "<project>@v<n>" for that workspace
only; runs record that name, so results always point at the exact spec and tasks.
"""

from __future__ import annotations

import difflib
import hashlib
import json
import shutil
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path
from typing import Any

from fastapi import HTTPException
from sqlalchemy import func, select, update
from sqlalchemy.orm import Session

from envforge import config
from envforge.web.models import (
    BuilderMessage,
    EnvProject,
    EnvRevision,
    EnvUpload,
    EnvVersion,
    Job,
    User,
    Workspace,
)
from envforge.web.services import slugify

MAX_UPLOAD_BYTES = 95 * 1024 * 1024  # Cloudflare's free plan caps requests at 100 MB


# ---- paths -------------------------------------------------------------------------


def project_dir(ws: Workspace, project: EnvProject) -> Path:
    return config.workspace_dir(ws.slug) / "builder" / project.name


def environments_dir(ws: Workspace) -> Path:
    return config.workspace_dir(ws.slug) / "environments"


# ---- projects ----------------------------------------------------------------------


def create_project(
    db: Session, ws: Workspace, user: User, title: str, purpose: str = ""
) -> EnvProject:
    base = slugify(title)[:40] or "environment"
    name, i = base, 1
    while db.scalar(
        select(EnvProject).where(EnvProject.workspace_id == ws.id, EnvProject.name == name)
    ):
        i += 1
        name = f"{base}-{i}"
    p = EnvProject(workspace_id=ws.id, name=name, title=title, purpose=purpose, created_by=user.id)
    db.add(p)
    db.commit()
    return p


def get_project(db: Session, ws: Workspace, project_id: str) -> EnvProject:
    p = db.get(EnvProject, project_id)
    if not p or p.workspace_id != ws.id or p.deleted_at is not None:
        raise HTTPException(404, "project not found")
    return p


def list_projects(db: Session, ws: Workspace) -> list[EnvProject]:
    return list(
        db.scalars(
            select(EnvProject)
            .where(EnvProject.workspace_id == ws.id, EnvProject.deleted_at.is_(None))
            .order_by(EnvProject.created_at.desc())
        )
    )


# ---- task sets (content-addressed) ----------------------------------------------------


def save_tasks(ws: Workspace, project: EnvProject, tasks: list[dict[str, Any]]) -> str:
    """Write task records once per distinct content; returns the sha256 naming the file."""
    lines = "".join(json.dumps(t, sort_keys=True) + "\n" for t in tasks)
    sha = hashlib.sha256(lines.encode()).hexdigest()
    path = project_dir(ws, project) / "tasks" / f"{sha}.jsonl"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(lines)
    return sha


def load_tasks(ws: Workspace, project: EnvProject, sha: str | None) -> list[dict[str, Any]]:
    if not sha:
        return []
    path = project_dir(ws, project) / "tasks" / f"{sha}.jsonl"
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


# ---- revisions ---------------------------------------------------------------------


def head(db: Session, project: EnvProject) -> EnvRevision | None:
    return db.get(EnvRevision, project.head_revision_id) if project.head_revision_id else None


def get_revision(db: Session, project: EnvProject, revision_id: str) -> EnvRevision:
    r = db.get(EnvRevision, revision_id)
    if not r or r.project_id != project.id:
        raise HTTPException(404, "revision not found")
    return r


def list_revisions(db: Session, project: EnvProject) -> list[EnvRevision]:
    return list(
        db.scalars(
            select(EnvRevision)
            .where(EnvRevision.project_id == project.id)
            .order_by(EnvRevision.number)
        )
    )


def validate(spec: dict[str, Any] | None, tasks: list[dict[str, Any]]) -> dict[str, Any]:
    """Spec problems + per-task problems (including: does the reference solution earn 1.0?)."""
    from pydantic import ValidationError

    from envforge.declarative.spec import EnvSpec, TaskRecord
    from envforge.declarative.validate import validate_spec, validate_task

    out: dict[str, Any] = {"spec_ok": False, "spec_problems": [], "tasks_ok": 0,
                           "task_problems": {}, "tiers": {}}  # fmt: skip
    if spec is None:
        out["spec_problems"] = ["no spec yet"]
        return out
    try:
        es = EnvSpec.model_validate(spec)
    except ValidationError as e:
        out["spec_problems"] = [
            f"{'.'.join(map(str, err['loc']))}: {err['msg']}" for err in e.errors()
        ][:30]
        return out
    out["spec_problems"] = validate_spec(es)
    out["spec_ok"] = not out["spec_problems"]
    for t in tasks:
        tid = str(t.get("id", "?"))
        try:
            tr = TaskRecord.model_validate(t)
        except ValidationError as e:
            out["task_problems"][tid] = [
                f"{'.'.join(map(str, err['loc']))}: {err['msg']}" for err in e.errors()
            ][:10]
            continue
        problems = validate_task(es, tr) if out["spec_ok"] else ["spec has problems"]
        if problems:
            out["task_problems"][tid] = problems[:10]
        else:
            out["tasks_ok"] += 1
            out["tiers"][str(tr.tier)] = out["tiers"].get(str(tr.tier), 0) + 1
    return out


def create_revision(
    db: Session,
    ws: Workspace,
    project: EnvProject,
    *,
    spec: dict[str, Any] | None,
    tasks: list[dict[str, Any]] | None = None,
    tasks_sha: str | None = None,
    message: str,
    author: str = "builder",
    parent: EnvRevision | None = None,
) -> EnvRevision:
    """New immutable revision as a child of `parent` (default: the head), and make it the head.

    Pass `tasks` to store a new task set, or `tasks_sha` to reuse an existing one; with
    neither, the parent's tasks carry over.
    """
    parent = parent if parent is not None else head(db, project)
    if tasks is not None:
        tasks_sha = save_tasks(ws, project, tasks)
    elif tasks_sha is None and parent is not None:
        tasks_sha = parent.tasks_sha
    task_rows = tasks if tasks is not None else load_tasks(ws, project, tasks_sha)
    number = (
        db.scalar(select(func.max(EnvRevision.number)).where(EnvRevision.project_id == project.id))
        or 0
    ) + 1
    rev = EnvRevision(
        project_id=project.id,
        parent_id=parent.id if parent else None,
        number=number,
        message=message[:500],
        author=author,
        spec=spec,
        tasks_sha=tasks_sha,
        n_tasks=len(task_rows),
        validation=validate(spec, task_rows),
    )
    db.add(rev)
    db.flush()
    project.head_revision_id = rev.id
    db.commit()
    return rev


def restore(
    db: Session, ws: Workspace, project: EnvProject, revision: EnvRevision, user: User
) -> EnvRevision:
    """Rollback without losing history: a new head revision that copies an old one."""
    return create_revision(
        db, ws, project, spec=revision.spec, tasks_sha=revision.tasks_sha,
        message=f"Restored revision {revision.number}", author="user",
    )  # fmt: skip


def checkout(db: Session, project: EnvProject, revision: EnvRevision) -> None:
    """Move the head. The next edit becomes a child of this revision (a branch, if it
    isn't the latest)."""
    project.head_revision_id = revision.id
    db.commit()


def diff(
    db: Session, ws: Workspace, project: EnvProject, a: EnvRevision | None, b: EnvRevision
) -> dict[str, Any]:
    """What changed from revision `a` (default: b's parent) to `b`: a unified diff of the
    spec and a summary of task changes."""
    if a is None and b.parent_id:
        a = db.get(EnvRevision, b.parent_id)
    a_spec = json.dumps(a.spec, indent=2, sort_keys=True).splitlines() if a and a.spec else []
    b_spec = json.dumps(b.spec, indent=2, sort_keys=True).splitlines() if b.spec else []
    spec_diff = "\n".join(
        difflib.unified_diff(
            a_spec, b_spec, f"r{a.number}" if a else "empty", f"r{b.number}", lineterm="", n=2
        )
    )
    a_ids = {t.get("id") for t in load_tasks(ws, project, a.tasks_sha)} if a else set()
    b_ids = {t.get("id") for t in load_tasks(ws, project, b.tasks_sha)}
    return {
        "from": a.number if a else None,
        "to": b.number,
        "spec_diff": spec_diff,
        "tasks_added": sorted(map(str, b_ids - a_ids))[:200],
        "tasks_removed": sorted(map(str, a_ids - b_ids))[:200],
        "tasks_changed_file": (a.tasks_sha if a else None) != b.tasks_sha,
    }


# ---- publishing and resolving workspace environments ------------------------------------


def publish(
    db: Session, ws: Workspace, project: EnvProject, revision: EnvRevision, user: User
) -> EnvVersion:
    """Freeze a valid revision as `<project>@v<n>`, runnable in this workspace."""
    from envforge.declarative.spec import EnvSpec, TaskRecord
    from envforge.declarative.store import save_env_dir

    v = revision.validation or {}
    if not v.get("spec_ok") or not v.get("tasks_ok"):
        raise HTTPException(
            400, "only a revision with a valid spec and at least one valid task can be published"
        )
    problems = v.get("task_problems") or {}
    tasks = [TaskRecord.model_validate(t) for t in load_tasks(ws, project, revision.tasks_sha)
             if str(t.get("id")) not in problems]  # fmt: skip
    version = (
        db.scalar(select(func.max(EnvVersion.version)).where(EnvVersion.project_id == project.id))
        or 0
    ) + 1
    pack_name = f"{project.name}@v{version}"
    out = environments_dir(ws) / project.name / f"v{version}"
    if out.exists():
        shutil.rmtree(out)
    spec = EnvSpec.model_validate(revision.spec).model_copy(update={"name": pack_name})
    save_env_dir(out, spec, tasks)
    ver = EnvVersion(
        project_id=project.id, workspace_id=ws.id, version=version, revision_id=revision.id,
        pack_name=pack_name, env_dir=str(out), created_by=user.id,
    )  # fmt: skip
    db.add(ver)
    db.commit()
    return ver


@lru_cache(maxsize=64)
def _load_env_dir(path: str):
    from envforge.declarative.store import load_env_dir

    return load_env_dir(Path(path))


def resolve_pack(db: Session, ws: Workspace, name: str):
    """A built-in pack, or one of this workspace's published environments. Never another
    workspace's: published environments are private to their workspace."""
    from envforge.core.registry import discover_packs

    packs = discover_packs()
    if name in packs:
        return packs[name]
    ver = db.scalar(
        select(EnvVersion).where(EnvVersion.workspace_id == ws.id, EnvVersion.pack_name == name)
    )
    if ver is None:
        raise HTTPException(400, f"unknown environment '{name}'")
    return _load_env_dir(ver.env_dir)


def workspace_packs(db: Session, ws: Workspace) -> list[tuple[str, Any, EnvVersion | None]]:
    """(name, env class, version row or None for built-ins) for everything runnable here."""
    from envforge.core.registry import discover_packs

    out: list[tuple[str, Any, EnvVersion | None]] = [
        (n, c, None) for n, c in sorted(discover_packs().items())
    ]
    for ver in db.scalars(
        select(EnvVersion).where(EnvVersion.workspace_id == ws.id).order_by(EnvVersion.pack_name)
    ):
        try:
            out.append((ver.pack_name, _load_env_dir(ver.env_dir), ver))
        except Exception:  # noqa: BLE001 - a broken directory shouldn't hide the rest
            continue
    return out


# ---- chat messages ------------------------------------------------------------------


def add_message(
    db: Session, project: EnvProject, role: str, content: str = "", **kw: Any
) -> BuilderMessage:
    seq = (
        db.scalar(
            select(func.max(BuilderMessage.seq)).where(BuilderMessage.project_id == project.id)
        )
        or 0
    ) + 1
    m = BuilderMessage(project_id=project.id, seq=seq, role=role, content=content, **kw)
    db.add(m)
    db.commit()
    return m


def messages(db: Session, project: EnvProject) -> list[BuilderMessage]:
    return list(
        db.scalars(
            select(BuilderMessage)
            .where(BuilderMessage.project_id == project.id)
            .order_by(BuilderMessage.seq)
        )
    )


# ---- uploads ------------------------------------------------------------------------


def new_upload(
    db: Session, ws: Workspace, project: EnvProject, user: User, filename: str
) -> tuple[EnvUpload, Path]:
    safe = Path(filename).name[:200] or "upload"
    up = EnvUpload(project_id=project.id, filename=safe, created_by=user.id)
    db.add(up)
    db.flush()
    raw = project_dir(ws, project) / "uploads" / f"{up.id}.raw"
    raw.parent.mkdir(parents=True, exist_ok=True)
    return up, raw


def list_uploads(db: Session, project: EnvProject) -> list[EnvUpload]:
    return list(
        db.scalars(
            select(EnvUpload)
            .where(EnvUpload.project_id == project.id)
            .order_by(EnvUpload.created_at)
        )
    )


# ---- jobs (run by `envforge-worker --kind builder`) ------------------------------------


def enqueue_job(
    db: Session, kind: str, project: EnvProject, user: User, payload: dict | None = None
) -> Job:
    busy = db.scalar(
        select(func.count()).select_from(Job).where(
            Job.project_id == project.id, Job.kind == kind, Job.status.in_(("queued", "running"))
        )
    )  # fmt: skip
    if kind == "builder_turn" and busy:
        raise HTTPException(409, "the builder is still working on the previous message")
    job = Job(kind=kind, project_id=project.id, created_by=user.id, payload=payload or {})
    db.add(job)
    db.commit()
    return job


def claim_job(db: Session, kinds: tuple[str, ...]) -> Job | None:
    row = db.scalar(
        select(Job)
        .where(Job.status == "queued", Job.kind.in_(kinds))
        .order_by(Job.created_at)
        .limit(1)
    )
    if not row:
        return None
    now = datetime.now(UTC)
    res = db.execute(
        update(Job).where(Job.id == row.id, Job.status == "queued")
        .values(status="running", started_at=now, heartbeat_at=now)
    )  # fmt: skip
    db.commit()
    if res.rowcount != 1:
        return None
    db.refresh(row)
    return row


def active_job(db: Session, project: EnvProject) -> Job | None:
    return db.scalar(
        select(Job).where(Job.project_id == project.id, Job.status.in_(("queued", "running")))
        .order_by(Job.created_at.desc()).limit(1)
    )  # fmt: skip

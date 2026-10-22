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
    Run,
    User,
    Workspace,
    WorkspaceMember,
)
from envforge.web.services import slugify

MAX_UPLOAD_BYTES = 95 * 1024 * 1024  # one request: Cloudflare's free plan caps them at 100 MB
CHUNK_BYTES = 32 * 1024 * 1024  # chunked uploads: each piece is one request well under that
SUPPORTED_SUFFIXES = (".csv", ".txt", ".json", ".jsonl", ".log")


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
    """Freeze a valid revision as `<project>@v<n>`, runnable in this workspace.

    The task records pass a privacy re-scan with the project's profile on the way out
    (`sanitize_tasks`); the report is attached to the returned row as `privacy_report`.
    """
    from envforge.declarative.spec import EnvSpec
    from envforge.declarative.store import save_env_dir

    v = revision.validation or {}
    if not v.get("spec_ok") or not v.get("tasks_ok"):
        raise HTTPException(
            400, "only a revision with a valid spec and at least one valid task can be published"
        )
    problems = v.get("task_problems") or {}
    spec_obj = EnvSpec.model_validate(revision.spec)
    tasks, report = sanitize_tasks(
        spec_obj,
        [
            t
            for t in load_tasks(ws, project, revision.tasks_sha)
            if str(t.get("id")) not in problems
        ],
        profile=project.privacy_profile,
        ner=project.privacy_ner,
    )
    if not tasks:
        raise HTTPException(
            400, "no task survived the privacy re-scan: " + json.dumps(report["dropped"])[:500]
        )
    version = (
        db.scalar(select(func.max(EnvVersion.version)).where(EnvVersion.project_id == project.id))
        or 0
    ) + 1
    pack_name = f"{project.name}@v{version}"
    out = environments_dir(ws) / project.name / f"v{version}"
    if out.exists():
        shutil.rmtree(out)
    save_env_dir(out, spec_obj.model_copy(update={"name": pack_name}), tasks)
    ver = EnvVersion(
        project_id=project.id, workspace_id=ws.id, version=version, revision_id=revision.id,
        pack_name=pack_name, env_dir=str(out), created_by=user.id,
    )  # fmt: skip
    db.add(ver)
    db.commit()
    ver.privacy_report = report  # transient: returned to the caller, not stored
    return ver


# Fields that are structure, not content: never touched by the re-scan (record keys like
# "C-1001" are additionally protected by redact.py's keep patterns).
STRUCTURAL_KEYS = ("id", "tier", "customer_id", "check", "collection", "key", "field", "tool",
                   "op", "category", "source")  # fmt: skip


def sanitize_tasks(spec, tasks: list[dict[str, Any]], *, profile: str, ner: bool):
    """The exit check for publishing: re-redact every task's text with the project's
    privacy profile, then re-validate. A task whose reference solution stops earning
    1.0 after sanitising (e.g. a check depended on a redacted value) is left out and
    reported, so privacy never silently produces broken tasks."""
    from envforge.builder.redact import sanitize_json_strings
    from envforge.declarative.spec import TaskRecord
    from envforge.declarative.validate import validate_task

    kept: list[TaskRecord] = []
    report: dict[str, Any] = {"profile": profile, "redactions": {}, "dropped": {}}
    for t in tasks:
        clean, counts = sanitize_json_strings(t, profile, skip_keys=STRUCTURAL_KEYS, ner=ner)
        for k, n in counts.items():
            report["redactions"][k] = report["redactions"].get(k, 0) + n
        try:
            rec = TaskRecord.model_validate(clean)
            problems = validate_task(spec, rec)
        except Exception as e:  # noqa: BLE001
            problems = [str(e)]
        if problems:
            report["dropped"][str(t.get("id"))] = problems[:3]
        else:
            kept.append(rec)
    return kept, report


def export_sanitizer(db: Session, ws: Workspace, pack_names: set[str]):
    """For training-data exports: if any run comes from a builder-published environment,
    return a function that re-sanitises each exported example with that project's
    profile (the strictest one wins when runs are mixed); None for built-in packs."""
    from envforge.builder.redact import sanitize_json_strings

    projects = list(
        db.scalars(
            select(EnvProject)
            .join(EnvVersion, EnvVersion.project_id == EnvProject.id)
            .where(EnvVersion.workspace_id == ws.id, EnvVersion.pack_name.in_(pack_names))
        )
    )
    if not projects:
        return None
    profile = "strict" if any(p.privacy_profile == "strict" for p in projects) else "standard"
    ner = any(p.privacy_ner for p in projects)

    def clean(line: str) -> str:
        obj, _ = sanitize_json_strings(
            json.loads(line), profile, skip_keys=("meta", "tools"), ner=ner
        )
        return json.dumps(obj) + "\n"

    return clean


@lru_cache(maxsize=64)
def _load_env_dir(path: str):
    from envforge.declarative.store import load_env_dir

    return load_env_dir(Path(path))


def resolve_pack(db: Session, ws: Workspace, name: str, *, include_deleted: bool = False):
    """A built-in pack, or one of this workspace's published environments. Never another
    workspace's: published environments are private to their workspace.

    A deleted environment can't start new runs; pass `include_deleted=True` to look at
    runs it already produced (replays, exports, its overview page).
    """
    from envforge.core.registry import discover_packs

    packs = discover_packs()
    if name in packs:
        return packs[name]
    query = select(EnvVersion).where(EnvVersion.workspace_id == ws.id, EnvVersion.pack_name == name)
    if not include_deleted:
        query = query.where(EnvVersion.deleted_at.is_(None))
    ver = db.scalar(query)
    if ver is None:
        raise HTTPException(400, f"unknown environment '{name}'")
    return _load_env_dir(ver.env_dir)


def delete_versions(db: Session, ws: Workspace, user: User, versions: list[EnvVersion]) -> None:
    """Delete published environments: they leave the Environments list and can't start new
    runs. The files stay, so runs made on them still open, replay and export.

    Only the workspace owner or whoever published a version may delete it, and never while
    a run on it is queued or running.
    """
    live = [v for v in versions if v.deleted_at is None]
    if not live:
        return
    m = db.scalar(
        select(WorkspaceMember).where(
            WorkspaceMember.workspace_id == ws.id, WorkspaceMember.user_id == user.id
        )
    )
    is_owner = m is not None and m.role == "owner"
    for v in live:
        if v.created_by != user.id and not is_owner:
            raise HTTPException(
                403, f"only the workspace owner or its publisher can delete {v.pack_name}"
            )
    busy = db.scalar(
        select(Run.pack).where(
            Run.workspace_id == ws.id,
            Run.pack.in_([v.pack_name for v in live]),
            Run.status.in_(("queued", "running")),
        )
    )
    if busy:
        raise HTTPException(409, f"a run on {busy} is still going; cancel it or let it finish")
    now = datetime.now(UTC)
    for v in live:
        v.deleted_at = now
    db.commit()


def workspace_packs(db: Session, ws: Workspace) -> list[tuple[str, Any, EnvVersion | None]]:
    """(name, env class, version row or None for built-ins) for everything runnable here."""
    from envforge.core.registry import discover_packs

    out: list[tuple[str, Any, EnvVersion | None]] = [
        (n, c, None) for n, c in sorted(discover_packs().items())
    ]
    for ver in db.scalars(
        select(EnvVersion)
        .where(EnvVersion.workspace_id == ws.id, EnvVersion.deleted_at.is_(None))
        .order_by(EnvVersion.pack_name)
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


def check_disk_for(ws: Workspace, size_bytes: int) -> None:
    """Refuse an upload that the disk can't hold: the raw file, its redacted copy and the
    index all exist for a while, so ask for ~3x the size (plus 1 GiB headroom)."""
    root = config.workspace_dir(ws.slug)
    root.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(root).free
    if free < size_bytes * 3 + 2**30:
        raise HTTPException(507, f"not enough disk space for a {size_bytes / 2**30:.1f} GiB upload "
                                 f"({free / 2**30:.1f} GiB free)")  # fmt: skip


def corpus_path(ws: Workspace, project: EnvProject) -> Path:
    """The project's conversation store (SQLite): every upload, redacted, deduplicated,
    clustered. Builder tools sample from here instead of reading files."""
    return project_dir(ws, project) / "corpus.sqlite"


def make_redactor(project: EnvProject):
    """The per-conversation redaction function for this project's privacy settings, as
    a picklable partial (the corpus ingest may run it in a process pool). Options the
    installed redactor doesn't know yet are left out rather than failing."""
    import functools
    import inspect

    from envforge.builder.redact import redact_conversation

    wanted = {
        "profile": project.privacy_profile,
        "ner": project.privacy_ner,
        "surrogates": project.privacy_surrogates,
        "tier2": project.privacy_tier2,
    }
    accepted = inspect.signature(redact_conversation).parameters
    return functools.partial(
        redact_conversation, **{k: v for k, v in wanted.items() if k in accepted}
    )


def list_uploads(db: Session, project: EnvProject) -> list[EnvUpload]:
    return list(
        db.scalars(
            select(EnvUpload)
            .where(EnvUpload.project_id == project.id)
            .order_by(EnvUpload.created_at)
        )
    )


def delete_upload(db: Session, project: EnvProject, up: EnvUpload, user: User) -> None:
    """Delete an upload: its row and any file now; its conversations leave the corpus
    store in the builder worker (which owns the store), followed by a re-cluster."""
    if up.path:  # a raw file still waiting to be processed, or a pre-store redacted file
        Path(up.path).unlink(missing_ok=True)
    had_data = up.status == "ready"
    uid = up.id
    db.delete(up)
    db.commit()
    if had_data:
        enqueue_job(db, "recluster", project, user, {"delete_upload": uid})


# ---- corpus + privacy review -----------------------------------------------------------------


def _conv_out(conv: Any) -> dict[str, Any]:
    """A redacted conversation for the UI (redacted text only ever reaches this point)."""
    return {
        "id": conv.id,
        "source_id": conv.meta.get("ext_id"),
        "upload_id": conv.meta.get("upload_id"),
        "cluster": conv.meta.get("cluster"),
        "turns": [{"role": t.role, "text": t.text} for t in conv.turns],
    }


def corpus_overview(ws: Workspace, project: EnvProject) -> dict[str, Any] | None:
    """Stats + topic clusters of the project's redacted corpus, or None before any data."""
    from envforge.builder import corpus as C

    path = corpus_path(ws, project)
    if not path.exists():
        return None
    with C.CorpusStore(path) as store:
        if not store.count():
            return None
        ov = C.overview(store, max_clusters=200, n_terms=5)
        ov["stale"] = store.stale()
    return ov


def cluster_examples(ws: Workspace, project: EnvProject, cluster: int, n: int = 3) -> list[dict]:
    from envforge.builder import corpus as C

    path = corpus_path(ws, project)
    if not path.exists():
        return []
    with C.CorpusStore(path) as store:
        return [_conv_out(c) for c in store.examples(cluster, n)]


def privacy_settings(project: EnvProject, changes: dict[str, Any]) -> None:
    """Apply privacy settings. Redaction settings apply to future uploads; any change to
    them, or to the threshold, means the last review no longer stands."""
    if changes.get("profile") == "off" and not changes.get("confirm_off"):
        raise HTTPException(
            400, "turning redaction off needs confirm_off: only for public or synthetic data"
        )
    redaction = ("profile", "ner", "surrogates", "tier2")
    changed = False
    for key in (*redaction, "threshold"):
        value = changes.get(key)
        if value is not None and value != getattr(project, f"privacy_{key}"):
            setattr(project, f"privacy_{key}", value)
            changed = True
    if changed:
        project.privacy_signed_off_at = project.privacy_signed_off_by = None
        if project.privacy_review and project.privacy_review.get("status") != "open":
            project.privacy_review = {**project.privacy_review, "status": "outdated"}


def start_review(
    db: Session, ws: Workspace, project: EnvProject, user: User, n: int | None = None
) -> tuple[dict[str, Any], list[dict]]:
    """Draw a random, topic-stratified sample of redacted conversations for a person to
    check. Returns (review, conversations). Starting a new review replaces an open one."""
    from envforge.builder import corpus as C
    from envforge.builder import review as R

    path = corpus_path(ws, project)
    if not path.exists():
        raise HTTPException(400, "nothing to review yet: no processed uploads")
    n = n or R.sample_size_for(project.privacy_threshold)
    with C.CorpusStore(path) as store:
        population = store.count(unique_only=True)
        if not population:
            raise HTTPException(400, "nothing to review yet: no processed uploads")
        ids = C.stratified_sample(
            store, min(n, population), seed=int(datetime.now(UTC).timestamp())
        )
        convs = [_conv_out(c) for c in store.get(ids)]
    review = {
        "id": hashlib.sha1(f"{project.id}{datetime.now(UTC).isoformat()}".encode()).hexdigest()[
            :12
        ],
        "status": "open",
        "ids": ids,
        "population": population,
        "threshold": project.privacy_threshold,
        "profile": project.privacy_profile,
        "created_at": datetime.now(UTC).isoformat(),
        "created_by": user.id,
    }
    project.privacy_review = review
    db.commit()
    return review, convs


def open_review_conversations(ws: Workspace, project: EnvProject) -> list[dict]:
    from envforge.builder import corpus as C

    rv = project.privacy_review or {}
    if rv.get("status") != "open":
        return []
    with C.CorpusStore(corpus_path(ws, project)) as store:
        return [_conv_out(c) for c in store.get(rv["ids"])]


def submit_review(
    db: Session, project: EnvProject, user: User, review_id: str, flagged: list[str],
    notes: dict[str, str] | None = None,
) -> dict[str, Any]:  # fmt: skip
    """Record which sampled conversations still contain personal data; compute the leak
    rate and pass/fail. A pass counts as the privacy sign-off."""
    from envforge.builder import review as R

    rv = project.privacy_review or {}
    if rv.get("id") != review_id or rv.get("status") != "open":
        raise HTTPException(409, "this review is no longer open — start a new one")
    sampled = set(rv["ids"])
    bad = sorted(set(flagged) - sampled)
    if bad:
        raise HTTPException(400, f"not in this review's sample: {bad[:3]}")
    result = R.evaluate(
        len(set(flagged)), len(sampled), threshold=rv["threshold"], population=rv["population"]
    )
    now = datetime.now(UTC)
    project.privacy_review = {
        **rv,
        "status": "passed" if result.passed else "failed",
        "flagged": sorted(set(flagged)),
        "notes": {k: v[:500] for k, v in (notes or {}).items() if k in sampled},
        "result": result.to_dict(),
        "reviewed_by": user.id,
        "reviewed_at": now.isoformat(),
    }
    if result.passed:
        project.privacy_signed_off_at, project.privacy_signed_off_by = now, user.id
    else:
        project.privacy_signed_off_at = project.privacy_signed_off_by = None
    pct = lambda x: f"{x * 100:.1f}%"  # noqa: E731
    add_message(db, project, "user",
        f"(Privacy review: {result.flagged} of {result.reviewed} sampled conversations still had "
        f"personal data — leak rate {pct(result.leak_rate)}, 95% CI up to {pct(result.ci_high)}; "
        f"threshold {pct(result.threshold)}: {'passed' if result.passed else 'FAILED'}.)")  # fmt: skip
    db.commit()
    return project.privacy_review


def review_summary(project: EnvProject) -> dict[str, Any] | None:
    """The review without the sampled ids (for listing in the project)."""
    rv = project.privacy_review
    if not rv:
        return None
    return {k: v for k, v in rv.items() if k not in ("ids", "notes")} | {
        "n": len(rv.get("ids", []))
    }


# ---- extraction approvals ---------------------------------------------------------------


def approve_extraction(db: Session, ws: Workspace, project: EnvProject, quote_id: str) -> dict:
    """The user's click on the builder's cost estimate: adds its conversations to what the
    builder may extract. Refused when the estimate is stale or would pass the workspace's
    monthly spending limit. Returns the approved quote."""
    from envforge.web.services import spend_left

    quote = project.extraction_quote
    if not quote or quote.get("id") != quote_id:
        raise HTTPException(
            409, "that estimate is no longer current; ask the builder for a new one"
        )
    cost = quote.get("cost_usd_with_retries") or quote.get("cost_usd") or 0.0
    left = spend_left(db, ws)
    if left is not None and cost > left:
        raise HTTPException(
            409,
            f"approving would pass this workspace's monthly spending limit: about ${cost:.2f} "
            f"needed, ${max(left, 0):.2f} left this month (Workspace settings)",
        )
    project.extraction_allowance = (project.extraction_allowance or 0) + int(quote["conversations"])
    project.extraction_quote = None
    db.commit()
    return quote


def decline_extraction(db: Session, project: EnvProject) -> None:
    """Dismiss the pending estimate; nothing is approved."""
    project.extraction_quote = None
    db.commit()


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

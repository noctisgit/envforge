"""Business logic shared by routers and the worker. Routers stay thin."""

from __future__ import annotations

import json
import os
import re
import shutil
from collections import defaultdict
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

from fastapi import HTTPException, status
from sqlalchemy import delete as sql_delete
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from envforge.web import crypto
from envforge.web.models import ProviderKey, Run, User, Workspace, WorkspaceMember
from envforge.web.security import check_base_url
from envforge.web.settings import get_settings

PROVIDERS = ("anthropic", "openai", "vercel", "vllm")


def slugify(name: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    return s[:48] or "ws"


def unique_slug(db: Session, base: str) -> str:
    slug, i = base, 1
    while db.scalar(select(Workspace).where(Workspace.slug == slug)):
        i += 1
        slug = f"{base}-{i}"
    return slug


# ---- workspaces -------------------------------------------------------------


def ensure_personal_workspace(db: Session, user: User) -> Workspace:
    existing = db.scalar(
        select(Workspace)
        .join(WorkspaceMember)
        .where(WorkspaceMember.user_id == user.id, WorkspaceMember.role == "owner")
    )
    if existing:
        return existing
    base = slugify(user.email.split("@")[0] or "workspace")
    ws = Workspace(
        slug=unique_slug(db, base),
        name=f"{user.display_name or base}'s workspace",
        owner_id=user.id,
    )
    db.add(ws)
    db.flush()
    db.add(WorkspaceMember(workspace_id=ws.id, user_id=user.id, role="owner"))
    db.flush()
    return ws


def create_workspace(db: Session, user: User, name: str) -> Workspace:
    ws = Workspace(slug=unique_slug(db, slugify(name)), name=name, owner_id=user.id)
    db.add(ws)
    db.flush()
    db.add(WorkspaceMember(workspace_id=ws.id, user_id=user.id, role="owner"))
    db.commit()
    return ws


def user_workspaces(db: Session, user: User) -> list[Workspace]:
    return list(
        db.scalars(
            select(Workspace)
            .join(WorkspaceMember)
            .where(WorkspaceMember.user_id == user.id)
            .order_by(Workspace.created_at)
        )
    )


def require_workspace(
    db: Session, user: User, workspace_id: str, *, owner: bool = False
) -> Workspace:
    m = db.scalar(
        select(WorkspaceMember).where(
            WorkspaceMember.workspace_id == workspace_id, WorkspaceMember.user_id == user.id
        )
    )
    if not m:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "workspace not found")
    if owner and m.role != "owner":
        raise HTTPException(status.HTTP_403_FORBIDDEN, "owner only")
    return db.get(Workspace, workspace_id)  # type: ignore[return-value]


def delete_workspace(db: Session, ws: Workspace, user: User) -> None:
    """Delete a workspace and everything in it: runs, builder projects, members, files.

    The caller must already have checked ownership (``require_workspace(owner=True)``).
    Refused (409) when it is the user's last workspace, or while a run or a builder job
    in it is queued or running (the worker would be writing into it). Unlike deleting a
    single run, this also removes the run rows, so the workspace's usage history goes
    with it. There are no ON DELETE cascades in the schema, so children are deleted
    explicitly, leaves first.
    """
    from envforge import config
    from envforge.web.models import (
        BuilderMessage,
        EnvProject,
        EnvRevision,
        EnvUpload,
        EnvVersion,
        Job,
    )

    n_mine = db.scalar(
        select(func.count()).select_from(WorkspaceMember).where(WorkspaceMember.user_id == user.id)
    )
    if (n_mine or 0) <= 1:
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            "you can't delete your only workspace; create another one first",
        )

    active = ("queued", "running")
    busy_runs = db.scalar(
        select(func.count()).select_from(Run).where(
            Run.workspace_id == ws.id, Run.status.in_(active)
        )
    )  # fmt: skip
    project_ids = select(EnvProject.id).where(EnvProject.workspace_id == ws.id)
    busy_jobs = db.scalar(
        select(func.count()).select_from(Job).where(
            Job.project_id.in_(project_ids), Job.status.in_(active)
        )
    )  # fmt: skip
    from envforge.web import training_service

    if busy_runs or busy_jobs or training_service.workspace_busy(db, ws):
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            "this workspace has queued or running work; cancel it (or wait for it to "
            "finish) before deleting the workspace",
        )

    # rows, leaves first (env_versions points at env_revisions, everything at env_projects)
    for model in (Job, BuilderMessage, EnvUpload, EnvVersion):
        db.execute(sql_delete(model).where(model.project_id.in_(project_ids)))
    db.execute(sql_delete(EnvRevision).where(EnvRevision.project_id.in_(project_ids)))
    db.execute(sql_delete(EnvVersion).where(EnvVersion.workspace_id == ws.id))
    db.execute(sql_delete(EnvProject).where(EnvProject.workspace_id == ws.id))
    training_service.delete_workspace_rows(db, ws)  # GPU boxes, datasets, training, adapters
    db.execute(sql_delete(Run).where(Run.workspace_id == ws.id))
    db.execute(sql_delete(WorkspaceMember).where(WorkspaceMember.workspace_id == ws.id))
    slug = ws.slug
    db.delete(ws)
    db.commit()

    # files: runs/, builder/, environments/, sft/ all live under the workspace's own
    # directory. Never remove anything outside data/workspaces/, nor that root itself.
    root = (config.DATA_DIR / "workspaces").resolve()
    target = config.workspace_dir(slug).resolve()
    if target.is_relative_to(root) and target != root:
        shutil.rmtree(target, ignore_errors=True)


# ---- provider keys ----------------------------------------------------------


class KeyRejected(Exception):
    """The provider says this key is invalid (401/403)."""


def list_provider_models(
    provider: str, secret: str, base_url: str | None = None
) -> list[str] | None:
    """Ask the provider which models this key can use. Doubles as the key check.

    Runs only while a key is being saved, on the plaintext the user just sent, so the
    API never needs to decrypt a stored key. Returns None when the provider has no
    such check here (vercel: its model list is public; vllm: self-hosted). Raises
    KeyRejected for a bad key; other failures (network, outage) propagate.
    """
    if provider == "anthropic":
        import anthropic

        client = anthropic.Anthropic(api_key=secret, max_retries=0, timeout=15)
        try:
            return sorted(m.id for m in client.models.list(limit=1000))
        except (anthropic.AuthenticationError, anthropic.PermissionDeniedError) as e:
            raise KeyRejected(f"Anthropic rejected this key ({e.status_code})") from e
    if provider == "openai":
        import openai

        client = openai.OpenAI(api_key=secret, max_retries=0, timeout=15)
        try:
            return sorted(m.id for m in client.models.list())
        except (openai.AuthenticationError, openai.PermissionDeniedError) as e:
            raise KeyRejected(f"OpenAI rejected this key ({e.status_code})") from e
    if provider == "vllm" and base_url:
        # what the box serves right now (base model + any LoRA adapters); the URL was
        # already checked against the private-network rules by the caller
        import urllib.request

        headers = {"User-Agent": "envforge"}
        if secret and secret != "not-needed":
            headers["Authorization"] = f"Bearer {secret}"
        req = urllib.request.Request(f"{base_url}/models", headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=10) as r:  # noqa: S310 - checked URL
                return sorted(m["id"] for m in json.loads(r.read()).get("data", []))
        except urllib.error.HTTPError as e:
            if e.code in (401, 403):
                raise KeyRejected(f"the vLLM server rejected this key ({e.code})") from e
            raise
    return None


def _verify(
    provider: str, secret: str, base_url: str | None = None
) -> tuple[list[str] | None, str | None]:
    """(model ids or None, note for the user). Raises HTTP 400 for a rejected key."""
    if provider not in ("anthropic", "openai", "vllm"):
        return None, None
    if os.getenv("ENVFORGE_OFFLINE", "").lower() in ("1", "true", "yes"):
        return None, "not checked (offline mode)"
    try:
        return list_provider_models(provider, secret, base_url), None
    except KeyRejected as e:
        raise HTTPException(400, str(e)) from e
    except Exception as e:  # noqa: BLE001 - save anyway; the first run will tell
        return None, f"could not reach {provider} to check the key ({type(e).__name__})"


def set_key(
    db: Session, user: User, provider: str, secret: str, base_url: str | None = None
) -> ProviderKey:
    """Save (or replace) one of the user's provider keys, checking it with the provider."""
    if provider not in PROVIDERS:
        raise HTTPException(400, f"provider must be one of {PROVIDERS}")
    secret = secret.strip()
    if provider != "vllm" and len(secret) < 8:
        raise HTTPException(400, "key looks too short")
    # Only self-hosted endpoints take a custom URL; the hosted providers' URLs are fixed,
    # so a stored key can never be sent anywhere but its own provider.
    if provider == "vllm":
        if not base_url:
            raise HTTPException(400, "vLLM needs a base URL, e.g. http://100.x.y.z:8000/v1")
        try:
            base_url = check_base_url(base_url)
        except ValueError as e:
            raise HTTPException(400, str(e)) from e
    else:
        base_url = None
    models, note = _verify(provider, secret, base_url)
    row = _key_row(db, user.id, provider)
    if row is None:
        row = ProviderKey(
            user_id=user.id, provider=provider, created_by=user.id, ciphertext="", last4=""
        )
        db.add(row)
    row.ciphertext = crypto.encrypt(secret or "not-needed")
    row.last4 = crypto.mask(secret or "not-needed")
    row.base_url = base_url
    row.models = models
    row.verified_at = datetime.now(UTC) if models is not None else None
    row.verify_note = note
    db.commit()
    return row


def _key_row(db: Session, user_id: str, provider: str) -> ProviderKey | None:
    return db.scalar(
        select(ProviderKey).where(ProviderKey.user_id == user_id, ProviderKey.provider == provider)
    )


def user_keys(db: Session, user_id: str) -> list[ProviderKey]:
    return list(
        db.scalars(
            select(ProviderKey).where(ProviderKey.user_id == user_id).order_by(ProviderKey.provider)
        )
    )


def delete_key(db: Session, user: User, provider: str) -> None:
    row = _key_row(db, user.id, provider)
    if row:
        db.delete(row)
        db.commit()


def resolve_key(db: Session, user_id: str, provider: str) -> tuple[str | None, str | None]:
    """(api_key, base_url) from this user's keys, or the server fallback if allowed."""
    row = _key_row(db, user_id, provider)
    if row:
        base_url = (
            check_base_url(row.base_url) if row.base_url else None
        )  # re-check: DNS can change
        return crypto.decrypt(row.ciphertext), base_url
    if get_settings().ALLOW_SERVER_KEYS:
        return None, None  # load_model will read the server .env
    # Without this, load_model(api_key=None) would silently use the server's env keys.
    raise PermissionError(f"no {provider} key on the account that queued this run")


def user_has_key(db: Session, user_id: str, provider: str) -> bool:
    return _key_row(db, user_id, provider) is not None


# ---- runs -------------------------------------------------------------------


def provider_of(model_spec: str) -> str:
    return model_spec.split(":", 1)[0] if ":" in model_spec else model_spec


def enqueue_run(
    db: Session,
    ws: Workspace,
    user: User,
    *,
    pack: str,
    model: str,
    episodes: int,
    tiers: list[int],
    seed_start: int = 0,
    max_steps: int | None = None,
    temperature: float = 0.0,
    label: str = "",
) -> Run:
    from envforge.core.runner import new_run_id
    from envforge.web.builder import resolve_pack
    from envforge.web.compute_service import box_for_model

    env_cls = resolve_pack(db, ws, pack)  # built-in, or published in this workspace
    provider = provider_of(model)
    if provider not in ("scripted", *PROVIDERS):
        raise HTTPException(400, f"unknown provider in model '{model}'")
    # a model one of this workspace's GPU boxes serves: the box's endpoint and key are used
    box = box_for_model(db, ws, model) if provider == "vllm" else None
    if (
        provider in PROVIDERS
        and box is None
        and not user_has_key(db, user.id, provider)
        and not get_settings().ALLOW_SERVER_KEYS
    ):
        raise HTTPException(400, f"no {provider} key on your account (Profile → Provider keys)")
    if episodes < 1 or episodes > 1000:
        raise HTTPException(400, "episodes must be 1..1000")
    active = db.scalar(
        select(func.count())
        .select_from(Run)
        .where(
            Run.workspace_id == ws.id,
            Run.status.in_(("queued", "running")),
            Run.deleted_at.is_(None),
        )
    )
    limit = get_settings().MAX_ACTIVE_RUNS_PER_WORKSPACE
    if active is not None and active >= limit:
        raise HTTPException(
            429, f"this workspace already has {active} queued/running runs (limit {limit})"
        )
    valid_tiers = [t for t in tiers if t in env_cls.tiers] or list(env_cls.tiers)

    run = Run(
        # the id doubles as a directory name, so keep the model part short
        id=new_run_id(pack, model[:40]),
        workspace_id=ws.id,
        created_by=user.id,
        pack=pack,
        model=model,
        episodes=episodes,
        tiers=valid_tiers,
        seed_start=seed_start,
        max_steps=max_steps,
        temperature=temperature,
        label=label,
        status="queued",
        gpu_box_id=box.id if box else None,
    )
    db.add(run)
    db.commit()
    return run


def delete_run(db: Session, ws: Workspace, user: User, run: Run) -> None:
    """Delete a run's trajectories and hide it. The row stays for usage accounting.

    Only the workspace owner or the run's creator may delete, and never while the
    worker is using it.
    """
    from envforge import config

    m = db.scalar(
        select(WorkspaceMember).where(
            WorkspaceMember.workspace_id == ws.id, WorkspaceMember.user_id == user.id
        )
    )
    if run.created_by != user.id and (m is None or m.role != "owner"):
        raise HTTPException(403, "only the workspace owner or the run's creator can delete it")
    if run.status in ("queued", "running"):
        raise HTTPException(409, "cancel the run before deleting it")
    if run.run_dir:
        # never delete anything outside this workspace's own runs directory
        root = config.runs_dir(ws.slug).resolve()
        target = Path(run.run_dir).resolve()
        if target.is_relative_to(root) and target != root:
            shutil.rmtree(target, ignore_errors=True)
    run.deleted_at = datetime.now(UTC)
    run.run_dir = None
    db.commit()


# ---- usage --------------------------------------------------------------------


def _blank() -> dict:
    return {
        "runs": 0,
        "episodes": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "cost_usd": 0.0,
        "unpriced_runs": 0,  # runs whose model has no known price (cost is a lower bound)
        "cache_read_tokens": 0,  # part of input_tokens served from the prompt cache
    }


def _add(acc: dict, r: Run) -> None:
    acc["runs"] += 1
    acc["episodes"] += r.progress
    acc["input_tokens"] += r.input_tokens or 0
    acc["output_tokens"] += r.output_tokens or 0
    acc["cache_read_tokens"] += r.cache_read_tokens or 0
    if r.cost_usd is None:
        acc["unpriced_runs"] += 1
    else:
        acc["cost_usd"] += r.cost_usd


def _summarise(rows: list[Run], days: int, group_key, group_label) -> dict:
    """Totals, per-day series and one grouping (by model, user or workspace)."""
    since = datetime.now(UTC).date() - timedelta(days=days - 1)
    totals = _blank()
    by_model: dict[str, dict] = defaultdict(_blank)
    by_group: dict[str, dict] = defaultdict(_blank)
    by_day: dict[date, dict] = {since + timedelta(days=i): _blank() for i in range(days)}
    for r in rows:
        _add(totals, r)
        _add(by_model[r.model], r)
        _add(by_group[group_key(r)], r)
        day = r.created_at.astimezone(UTC).date() if r.created_at.tzinfo else r.created_at.date()
        if day in by_day:
            _add(by_day[day], r)

    def ranked(d: dict[str, dict], label) -> list[dict]:
        items = [{**label(k), **v} for k, v in d.items()]
        return sorted(items, key=lambda x: x["input_tokens"] + x["output_tokens"], reverse=True)

    return {
        "days": days,
        "totals": totals,
        "by_model": ranked(by_model, lambda k: {"model": k}),
        "groups": ranked(by_group, group_label),
        "by_day": [{"date": d.isoformat(), **v} for d, v in sorted(by_day.items())],
    }


def _builder_usage(db: Session, since: datetime, *, ws: Workspace | None = None,
                   user: User | None = None) -> dict:  # fmt: skip
    """Tokens and cost spent by the environment builder (chat turns + extraction)."""
    from envforge.web.models import EnvProject, Job

    q = select(Job).join(EnvProject, EnvProject.id == Job.project_id).where(Job.created_at >= since)
    if ws is not None:
        q = q.where(EnvProject.workspace_id == ws.id)
    if user is not None:
        q = q.where(Job.created_by == user.id)
    out = {
        "jobs": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "cache_read_tokens": 0,
        "cost_usd": 0.0,
    }
    for j in db.scalars(q.where(Job.kind == "builder_turn")):
        out["jobs"] += 1
        out["input_tokens"] += j.input_tokens or 0
        out["output_tokens"] += j.output_tokens or 0
        out["cache_read_tokens"] += j.cache_read_tokens or 0
        out["cost_usd"] += j.cost_usd or 0.0
    return out


def workspace_usage(db: Session, ws: Workspace, days: int = 30) -> dict:
    """Usage of one workspace, grouped by the member who queued each run."""
    since = datetime.now(UTC) - timedelta(days=days)
    rows = list(db.scalars(select(Run).where(Run.workspace_id == ws.id, Run.created_at >= since)))
    users = {
        u.id: u for u in db.scalars(select(User).where(User.id.in_({r.created_by for r in rows})))
    }

    def label(uid: str) -> dict:
        u = users.get(uid)
        return {
            "user_id": uid,
            "email": u.email if u else "?",
            "display_name": u.display_name if u else "",
        }

    out = _summarise(rows, days, lambda r: r.created_by, label)
    out["builder"] = _builder_usage(db, since, ws=ws)
    return out


def user_usage(db: Session, user: User, days: int = 30) -> dict:
    """Usage of one user across the workspaces they belong to, grouped by workspace."""
    since = datetime.now(UTC) - timedelta(days=days)
    member_of = select(WorkspaceMember.workspace_id).where(WorkspaceMember.user_id == user.id)
    rows = list(
        db.scalars(
            select(Run).where(
                Run.created_by == user.id,
                Run.workspace_id.in_(member_of),
                Run.created_at >= since,
            )
        )
    )
    names = {
        w.id: w.name
        for w in db.scalars(
            select(Workspace).where(Workspace.id.in_({r.workspace_id for r in rows}))
        )
    }
    out = _summarise(
        rows,
        days,
        lambda r: r.workspace_id,
        lambda wid: {"workspace_id": wid, "name": names.get(wid, "?")},
    )
    out["builder"] = _builder_usage(db, since, user=user)
    return out


# ---- models -------------------------------------------------------------------


def provider_model_ids(db: Session, user_id: str) -> dict[str, list[str]]:
    """Model ids each of the user's verified keys reported, keyed by provider (for the picker)."""
    return {r.provider: r.models for r in user_keys(db, user_id) if r.models is not None}


def available_providers(db: Session, user_id: str) -> list[str]:
    """Providers this user can run: their own keys, or all of them on a shared-key server."""
    if get_settings().ALLOW_SERVER_KEYS:
        return list(PROVIDERS)
    return sorted({r.provider for r in user_keys(db, user_id)})

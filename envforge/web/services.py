"""Business logic shared by routers and the worker. Routers stay thin."""

from __future__ import annotations

import re

from fastapi import HTTPException, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from envforge.web import crypto
from envforge.web.models import ProviderKey, Run, User, Workspace, WorkspaceMember
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


# ---- provider keys ----------------------------------------------------------


def set_key(
    db: Session, ws: Workspace, user: User, provider: str, secret: str, base_url: str | None = None
) -> ProviderKey:
    if provider not in PROVIDERS:
        raise HTTPException(400, f"provider must be one of {PROVIDERS}")
    secret = secret.strip()
    if provider != "vllm" and len(secret) < 8:
        raise HTTPException(400, "key looks too short")
    row = db.scalar(
        select(ProviderKey).where(
            ProviderKey.workspace_id == ws.id, ProviderKey.provider == provider
        )
    )
    if row is None:
        row = ProviderKey(
            workspace_id=ws.id, provider=provider, created_by=user.id, ciphertext="", last4=""
        )
        db.add(row)
    row.ciphertext = crypto.encrypt(secret or "not-needed")
    row.last4 = crypto.mask(secret or "not-needed")
    row.base_url = base_url
    db.commit()
    return row


def delete_key(db: Session, ws: Workspace, provider: str) -> None:
    row = db.scalar(
        select(ProviderKey).where(
            ProviderKey.workspace_id == ws.id, ProviderKey.provider == provider
        )
    )
    if row:
        db.delete(row)
        db.commit()


def resolve_key(db: Session, ws: Workspace, provider: str) -> tuple[str | None, str | None]:
    """(api_key, base_url) for a provider in this workspace, or server fallback if allowed."""
    row = db.scalar(
        select(ProviderKey).where(
            ProviderKey.workspace_id == ws.id, ProviderKey.provider == provider
        )
    )
    if row:
        return crypto.decrypt(row.ciphertext), row.base_url
    if get_settings().ALLOW_SERVER_KEYS:
        return None, None  # load_model will read the server .env
    return None, None


def workspace_has_key(db: Session, ws: Workspace, provider: str) -> bool:
    return (
        db.scalar(
            select(ProviderKey).where(
                ProviderKey.workspace_id == ws.id, ProviderKey.provider == provider
            )
        )
        is not None
    )


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
    from envforge.core.registry import discover_packs
    from envforge.core.runner import new_run_id

    packs = discover_packs()
    if pack not in packs:
        raise HTTPException(400, f"unknown pack '{pack}'")
    provider = provider_of(model)
    if provider not in ("scripted", *PROVIDERS):
        raise HTTPException(400, f"unknown provider in model '{model}'")
    if (
        provider in PROVIDERS
        and not workspace_has_key(db, ws, provider)
        and not get_settings().ALLOW_SERVER_KEYS
    ):
        raise HTTPException(
            400, f"no {provider} key configured for this workspace (Settings → Keys)"
        )
    if episodes < 1 or episodes > 1000:
        raise HTTPException(400, "episodes must be 1..1000")
    valid_tiers = [t for t in tiers if t in packs[pack].tiers] or list(packs[pack].tiers)

    run = Run(
        id=new_run_id(pack, model),
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
    )
    db.add(run)
    db.commit()
    return run

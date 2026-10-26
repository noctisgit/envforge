"""/api/workspaces — list/create, settings, models, usage; /api/packs — environments."""

from __future__ import annotations

from functools import lru_cache

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select
from sqlalchemy.orm import Session

from envforge.core.registry import discover_packs
from envforge.models import catalog
from envforge.web import services as svc
from envforge.web.auth import current_user
from envforge.web.db import get_db
from envforge.web.models import User, Workspace, WorkspaceMember
from envforge.web.schemas import (
    ModelOut,
    ModelsOut,
    PackOut,
    WorkspaceIn,
    WorkspaceOut,
    WorkspaceSettingsIn,
)

router = APIRouter(prefix="/api", tags=["workspaces"])


def _ws_out(db: Session, ws: Workspace, user: User) -> WorkspaceOut:
    m = db.scalar(
        select(WorkspaceMember).where(
            WorkspaceMember.workspace_id == ws.id, WorkspaceMember.user_id == user.id
        )
    )
    return WorkspaceOut(
        id=ws.id,
        slug=ws.slug,
        name=ws.name,
        role=m.role if m else "member",
        default_model=ws.default_model,
        default_pack=ws.default_pack,
        default_episodes=ws.default_episodes,
        builder_model=ws.builder_model,
        train_base_model=ws.train_base_model,
        monthly_spend_limit_usd=ws.monthly_spend_limit_usd,
        month_spend_usd=round(svc.month_spend(db, ws), 4),
    )


@router.get("/packs", response_model=list[PackOut])
def packs(_: User = Depends(current_user)) -> list[PackOut]:
    return [
        PackOut(
            name=n,
            kind=c.kind,
            tiers=list(c.tiers),
            max_steps=c.max_steps,
            description=c.description,
            task_counts=c.task_counts(),
        )
        for n, c in sorted(discover_packs().items())
    ]


@lru_cache(maxsize=32)
def _overview(name: str) -> dict:
    from envforge.core.explore import pack_overview

    return pack_overview(discover_packs()[name])


def _pack_or_404(name: str):
    packs = discover_packs()
    if name not in packs:
        raise HTTPException(404, f"unknown environment '{name}'")
    return packs[name]


@router.get("/hub/models")
def hub_models(
    q: str = Query("", max_length=100),
    sort: str = Query("downloads", pattern="^(downloads|likes|lastModified|trendingScore)$"),
    _: User = Depends(current_user),
) -> dict:
    """Open-weight models on the Hugging Face Hub (live, cached), for the training base
    model picker: size, a LoRA GPU-memory estimate, licence, gated, vLLM tool parser."""
    from envforge.models import hub

    models, live = hub.search(q, sort=sort)
    return {"live": live, "models": [m.to_dict() for m in models]}


@router.get("/packs/{name}")
def pack_detail(name: str, _: User = Depends(current_user)) -> dict:
    """Everything the Environments page shows: description, dimensions, tools, tiers,
    reward rules, tables and a sample world diagram. Generic for every pack."""
    _pack_or_404(name)
    return _overview(name)


@router.get("/packs/{name}/preview")
def pack_preview(
    name: str,
    seed: int = Query(0, ge=0, le=2_000_000_000),
    tier: int = Query(1),
    _: User = Depends(current_user),
) -> dict:
    """One generated task: goal, hidden faults, first observation, diagrams before and
    after the reference solution. No model involved, nothing is billed."""
    from envforge.core.explore import preview_task

    env_cls = _pack_or_404(name)
    if tier not in env_cls.tiers:
        raise HTTPException(400, f"tier must be one of {list(env_cls.tiers)}")
    return preview_task(env_cls, seed, tier)


@router.get("/workspaces", response_model=list[WorkspaceOut])
def list_workspaces(user: User = Depends(current_user), db: Session = Depends(get_db)):
    wss = svc.user_workspaces(db, user)
    if not wss:
        svc.ensure_personal_workspace(db, user)
        db.commit()
        wss = svc.user_workspaces(db, user)
    return [_ws_out(db, ws, user) for ws in wss]


@router.post("/workspaces", response_model=WorkspaceOut, status_code=201)
def create_workspace(
    body: WorkspaceIn, user: User = Depends(current_user), db: Session = Depends(get_db)
):
    ws = svc.create_workspace(db, user, body.name)
    return _ws_out(db, ws, user)


@router.get("/workspaces/{ws_id}", response_model=WorkspaceOut)
def get_workspace(ws_id: str, user: User = Depends(current_user), db: Session = Depends(get_db)):
    ws = svc.require_workspace(db, user, ws_id)
    return _ws_out(db, ws, user)


@router.patch("/workspaces/{ws_id}", response_model=WorkspaceOut)
def update_settings(
    ws_id: str,
    body: WorkspaceSettingsIn,
    user: User = Depends(current_user),
    db: Session = Depends(get_db),
):
    ws = svc.require_workspace(db, user, ws_id, owner=True)
    for field, value in body.model_dump(exclude_unset=True).items():
        if value is None and field != "monthly_spend_limit_usd":
            continue  # only the spending limit can be cleared (null = no limit)
        setattr(ws, field, value)
    db.commit()
    return _ws_out(db, ws, user)


@router.delete("/workspaces/{ws_id}", status_code=204)
def delete_workspace(
    ws_id: str, user: User = Depends(current_user), db: Session = Depends(get_db)
) -> None:
    """Delete the workspace with all its runs, builder projects and files (owner only)."""
    ws = svc.require_workspace(db, user, ws_id, owner=True)
    svc.delete_workspace(db, ws, user)


@router.get("/workspaces/{ws_id}/models", response_model=ModelsOut)
def list_models(
    ws_id: str,
    refresh: bool = False,
    user: User = Depends(current_user),
    db: Session = Depends(get_db),
):
    """Tool-capable models you can run in this workspace, given your provider keys."""
    from envforge.web.compute_service import box_models

    ws = svc.require_workspace(db, user, ws_id)
    providers = svc.available_providers(db, user.id)
    if refresh:
        catalog.gateway_models(refresh=True)
    models = catalog.list_models(providers, native_ids=svc.provider_model_ids(db, user.id))
    # models served by this workspace's GPU boxes: runnable without a vLLM key of your own
    served = [ModelOut(**m) for m in box_models(db, ws)]
    if served and "vllm" not in providers:
        providers = [*providers, "vllm"]
    return ModelsOut(
        providers=providers,
        catalog_available=bool(catalog.gateway_models()),
        models=served + [ModelOut(**m.to_dict()) for m in models],
    )


@router.get("/workspaces/{ws_id}/usage")
def workspace_usage(
    ws_id: str,
    days: int = Query(30, ge=1, le=365),
    user: User = Depends(current_user),
    db: Session = Depends(get_db),
) -> dict:
    """Runs, episodes, tokens and estimated cost for this workspace, per member and per day."""
    ws = svc.require_workspace(db, user, ws_id)
    return svc.workspace_usage(db, ws, days)

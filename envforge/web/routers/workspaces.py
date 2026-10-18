"""/api/workspaces — list/create, settings, provider keys, packs."""

from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy import select
from sqlalchemy.orm import Session

from envforge.core.registry import discover_packs
from envforge.web import services as svc
from envforge.web.auth import current_user
from envforge.web.db import get_db
from envforge.web.models import ProviderKey, User, Workspace, WorkspaceMember
from envforge.web.schemas import (
    KeyIn,
    KeyOut,
    PackOut,
    WorkspaceIn,
    WorkspaceOut,
    WorkspaceSettingsIn,
)

router = APIRouter(prefix="/api", tags=["workspaces"])


def _keys_out(db: Session, ws: Workspace) -> list[KeyOut]:
    rows = db.scalars(select(ProviderKey).where(ProviderKey.workspace_id == ws.id))
    return [
        KeyOut(provider=r.provider, masked=r.last4, base_url=r.base_url, updated_at=r.updated_at)
        for r in rows
    ]


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
        keys=_keys_out(db, ws),
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
        )
        for n, c in sorted(discover_packs().items())
    ]


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
    for field, value in body.model_dump(exclude_none=True).items():
        setattr(ws, field, value)
    db.commit()
    return _ws_out(db, ws, user)


@router.put("/workspaces/{ws_id}/keys", response_model=list[KeyOut])
def put_key(
    ws_id: str, body: KeyIn, user: User = Depends(current_user), db: Session = Depends(get_db)
):
    ws = svc.require_workspace(db, user, ws_id, owner=True)
    svc.set_key(db, ws, user, body.provider, body.secret, body.base_url)
    return _keys_out(db, ws)


@router.delete("/workspaces/{ws_id}/keys/{provider}", response_model=list[KeyOut])
def delete_key(
    ws_id: str, provider: str, user: User = Depends(current_user), db: Session = Depends(get_db)
):
    ws = svc.require_workspace(db, user, ws_id, owner=True)
    svc.delete_key(db, ws, provider)
    return _keys_out(db, ws)

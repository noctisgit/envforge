"""Database schema.

users ──< workspace_members >── workspaces ──< provider_keys
                                     └──< runs

- A user owns at least one workspace; membership rows carry a role.
- Provider keys belong to a *workspace* (not a user) so a team can share a key.
  Secrets are stored Fernet-encrypted; only a masked suffix is ever returned.
- Runs are queued by the API and executed by the worker; the JSONL trajectories
  stay on disk under data/workspaces/<workspace-slug>/runs/<run-id>/, the DB holds
  status, config and the metrics summary.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

from sqlalchemy import JSON, DateTime, Float, ForeignKey, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from envforge.web.db import Base


def _uuid() -> str:
    return uuid.uuid4().hex


def _now() -> datetime:
    return datetime.now(UTC)


class User(Base):
    __tablename__ = "users"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    email: Mapped[str] = mapped_column(String(320), unique=True, index=True)
    display_name: Mapped[str] = mapped_column(String(120), default="")
    # local auth only; NULL for clerk users
    password_hash: Mapped[str | None] = mapped_column(String(255), nullable=True)
    # clerk auth only
    clerk_user_id: Mapped[str | None] = mapped_column(String(64), unique=True, nullable=True)
    is_admin: Mapped[bool] = mapped_column(default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    memberships: Mapped[list[WorkspaceMember]] = relationship(back_populates="user")


class Workspace(Base):
    __tablename__ = "workspaces"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    slug: Mapped[str] = mapped_column(String(64), unique=True, index=True)  # on-disk namespace
    name: Mapped[str] = mapped_column(String(120))
    owner_id: Mapped[str] = mapped_column(ForeignKey("users.id"))
    # per-workspace preferences
    default_model: Mapped[str] = mapped_column(String(200), default="anthropic:claude-sonnet-4-5")
    default_pack: Mapped[str] = mapped_column(String(64), default="netops")
    default_episodes: Mapped[int] = mapped_column(Integer, default=10)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    members: Mapped[list[WorkspaceMember]] = relationship(back_populates="workspace")
    keys: Mapped[list[ProviderKey]] = relationship(back_populates="workspace")
    runs: Mapped[list[Run]] = relationship(back_populates="workspace")


class WorkspaceMember(Base):
    __tablename__ = "workspace_members"
    __table_args__ = (UniqueConstraint("workspace_id", "user_id"),)

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    workspace_id: Mapped[str] = mapped_column(ForeignKey("workspaces.id"), index=True)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id"), index=True)
    role: Mapped[str] = mapped_column(String(16), default="member")  # owner | member

    workspace: Mapped[Workspace] = relationship(back_populates="members")
    user: Mapped[User] = relationship(back_populates="memberships")


class ProviderKey(Base):
    __tablename__ = "provider_keys"
    __table_args__ = (UniqueConstraint("workspace_id", "provider"),)

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    workspace_id: Mapped[str] = mapped_column(ForeignKey("workspaces.id"), index=True)
    provider: Mapped[str] = mapped_column(String(32))  # anthropic | openai | vercel | vllm
    ciphertext: Mapped[str] = mapped_column(Text)  # Fernet token
    last4: Mapped[str] = mapped_column(String(8))
    base_url: Mapped[str | None] = mapped_column(String(500), nullable=True)  # vllm / custom
    created_by: Mapped[str] = mapped_column(ForeignKey("users.id"))
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_now, onupdate=_now
    )

    workspace: Mapped[Workspace] = relationship(back_populates="keys")


class Run(Base):
    __tablename__ = "runs"

    id: Mapped[str] = mapped_column(String(80), primary_key=True)  # same id as the run dir
    workspace_id: Mapped[str] = mapped_column(ForeignKey("workspaces.id"), index=True)
    created_by: Mapped[str] = mapped_column(ForeignKey("users.id"))
    pack: Mapped[str] = mapped_column(String(64))
    model: Mapped[str] = mapped_column(String(200))
    episodes: Mapped[int] = mapped_column(Integer)
    tiers: Mapped[list[int]] = mapped_column(JSON)
    seed_start: Mapped[int] = mapped_column(Integer, default=0)
    max_steps: Mapped[int | None] = mapped_column(Integer, nullable=True)
    temperature: Mapped[float] = mapped_column(Float, default=0.0)
    label: Mapped[str] = mapped_column(String(200), default="")

    status: Mapped[str] = mapped_column(String(16), default="queued", index=True)
    # queued | running | completed | failed | cancelled
    progress: Mapped[int] = mapped_column(Integer, default=0)  # episodes finished
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    metrics: Mapped[dict | None] = mapped_column(JSON, nullable=True)  # RunReport.model_dump()
    run_dir: Mapped[str | None] = mapped_column(String(500), nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    workspace: Mapped[Workspace] = relationship(back_populates="runs")

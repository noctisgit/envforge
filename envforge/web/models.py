"""Database schema.

users ──< workspace_members >── workspaces ──< runs
  └──< provider_keys

- A user owns at least one workspace; membership rows carry a role.
- Provider keys belong to a *user*: one set of keys per account, used by every
  workspace that user runs in. A run uses the keys of the user who queued it.
  Secrets are stored Fernet-encrypted; only a masked suffix is ever returned.
- Runs are queued by the API and executed by the worker; the JSONL trajectories
  stay on disk under data/workspaces/<workspace-slug>/runs/<run-id>/, the DB holds
  status, config, the metrics summary and running token/cost usage.
- Isolation: every API route resolves the workspace through the caller's membership
  row (services.require_workspace), and there is no invite endpoint, so a workspace
  is only ever visible to the user who created it.
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
    # Bumped on password reset / "sign out everywhere"; sessions carry it and stop working.
    session_epoch: Mapped[int] = mapped_column(Integer, default=0, server_default="0")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    memberships: Mapped[list[WorkspaceMember]] = relationship(back_populates="user")
    keys: Mapped[list[ProviderKey]] = relationship(
        back_populates="user", foreign_keys="ProviderKey.user_id"
    )


class Workspace(Base):
    __tablename__ = "workspaces"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    slug: Mapped[str] = mapped_column(String(64), unique=True, index=True)  # on-disk namespace
    name: Mapped[str] = mapped_column(String(120))
    owner_id: Mapped[str] = mapped_column(ForeignKey("users.id"))
    # per-workspace preferences
    # "scripted" works without any key; users pick a real model once they've added one
    default_model: Mapped[str] = mapped_column(String(200), default="scripted")
    default_pack: Mapped[str] = mapped_column(String(64), default="netops")
    default_episodes: Mapped[int] = mapped_column(Integer, default=10)
    # the model that drives the environment builder chat (any provider:model spec)
    builder_model: Mapped[str] = mapped_column(
        String(200), default="anthropic:claude-opus-5-5", server_default="anthropic:claude-opus-5-5"
    )
    # the open-weight model fine-tuned by `envforge train sft` (a Hugging Face id)
    train_base_model: Mapped[str] = mapped_column(
        String(200), default="Qwen/Qwen2.5-7B-Instruct", server_default="Qwen/Qwen2.5-7B-Instruct"
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    members: Mapped[list[WorkspaceMember]] = relationship(back_populates="workspace")
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
    __table_args__ = (UniqueConstraint("user_id", "provider"),)

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id"), index=True)
    provider: Mapped[str] = mapped_column(String(32))  # anthropic | openai | vercel | vllm
    ciphertext: Mapped[str] = mapped_column(Text)  # Fernet token
    last4: Mapped[str] = mapped_column(String(8))
    base_url: Mapped[str | None] = mapped_column(String(500), nullable=True)  # vllm / custom
    # Checked with the provider when saved: the model ids this key can use (None = not
    # checked, e.g. Vercel/vLLM or the provider was unreachable) and any problem found.
    models: Mapped[list[str] | None] = mapped_column(JSON, nullable=True)
    verified_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    verify_note: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_by: Mapped[str] = mapped_column(ForeignKey("users.id"))
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_now, onupdate=_now
    )

    user: Mapped[User] = relationship(back_populates="keys", foreign_keys=[user_id])


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
    # running usage, updated after every episode; cost is an estimate at list price
    input_tokens: Mapped[int] = mapped_column(Integer, default=0, server_default="0")
    output_tokens: Mapped[int] = mapped_column(Integer, default=0, server_default="0")
    cache_read_tokens: Mapped[int] = mapped_column(Integer, default=0, server_default="0")
    cost_usd: Mapped[float | None] = mapped_column(Float, nullable=True, default=0.0)
    # the worker touches this every few seconds while it owns the run
    heartbeat_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    # Deleting a run removes its trajectories and hides it, but keeps this row so the
    # usage history (tokens, cost) stays true.
    deleted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    workspace: Mapped[Workspace] = relationship(back_populates="runs")


# ---- environment builder ---------------------------------------------------------
#
# workspaces ──< env_projects ──< env_revisions (a DAG: each points at its parent)
#                     │   ├──< env_versions  (published revisions, runnable as packs)
#                     │   ├──< env_uploads   (redacted datasets; raw files are deleted)
#                     │   └──< builder_messages (the chat)
#                     └── head_revision_id: where the next edit branches from
#
# Revisions are immutable. Editing creates a child of the head; "restore" creates a
# new revision copying an old one; checking out an older revision and editing from
# there starts a branch. Task records live on disk in content-addressed files
# (tasks/<sha256>.jsonl), so revisions that share tasks share the file.


class EnvProject(Base):
    __tablename__ = "env_projects"
    __table_args__ = (UniqueConstraint("workspace_id", "name"),)

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    workspace_id: Mapped[str] = mapped_column(ForeignKey("workspaces.id"), index=True)
    name: Mapped[str] = mapped_column(String(64))  # slug; published packs are "<name>@v<n>"
    title: Mapped[str] = mapped_column(String(200), default="")
    purpose: Mapped[str] = mapped_column(Text, default="")  # the user's stated intent
    created_by: Mapped[str] = mapped_column(ForeignKey("users.id"))
    head_revision_id: Mapped[str | None] = mapped_column(String(32), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    deleted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class EnvRevision(Base):
    __tablename__ = "env_revisions"
    __table_args__ = (UniqueConstraint("project_id", "number"),)

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    project_id: Mapped[str] = mapped_column(ForeignKey("env_projects.id"), index=True)
    parent_id: Mapped[str | None] = mapped_column(String(32), nullable=True)
    number: Mapped[int] = mapped_column(Integer)  # 1, 2, 3 … per project
    message: Mapped[str] = mapped_column(String(500), default="")
    author: Mapped[str] = mapped_column(String(16), default="builder")  # builder | user
    spec: Mapped[dict | None] = mapped_column(JSON, nullable=True)  # EnvSpec as JSON
    tasks_sha: Mapped[str | None] = mapped_column(String(64), nullable=True)
    n_tasks: Mapped[int] = mapped_column(Integer, default=0)
    validation: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)


class EnvVersion(Base):
    __tablename__ = "env_versions"
    __table_args__ = (UniqueConstraint("project_id", "version"),)

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    project_id: Mapped[str] = mapped_column(ForeignKey("env_projects.id"), index=True)
    workspace_id: Mapped[str] = mapped_column(ForeignKey("workspaces.id"), index=True)
    version: Mapped[int] = mapped_column(Integer)
    revision_id: Mapped[str] = mapped_column(ForeignKey("env_revisions.id"))
    pack_name: Mapped[str] = mapped_column(String(80), index=True)  # "<name>@v<n>"
    env_dir: Mapped[str] = mapped_column(String(500))  # spec.json + tasks.jsonl
    created_by: Mapped[str] = mapped_column(ForeignKey("users.id"))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)


class EnvUpload(Base):
    __tablename__ = "env_uploads"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    project_id: Mapped[str] = mapped_column(ForeignKey("env_projects.id"), index=True)
    filename: Mapped[str] = mapped_column(String(255))
    size_bytes: Mapped[int] = mapped_column(Integer, default=0)
    # uploaded (raw file on disk, waiting) | processing | ready | failed
    status: Mapped[str] = mapped_column(String(16), default="uploaded")
    detected_format: Mapped[dict | None] = mapped_column(JSON, nullable=True)  # format + mapping
    n_conversations: Mapped[int] = mapped_column(Integer, default=0)
    redactions: Mapped[dict | None] = mapped_column(JSON, nullable=True)  # counts by PII type
    profile: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    path: Mapped[str | None] = mapped_column(String(500), nullable=True)  # redacted JSONL
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_by: Mapped[str] = mapped_column(ForeignKey("users.id"))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)


class BuilderMessage(Base):
    """One message of a builder chat, in the provider-neutral shape envforge models use."""

    __tablename__ = "builder_messages"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    project_id: Mapped[str] = mapped_column(ForeignKey("env_projects.id"), index=True)
    seq: Mapped[int] = mapped_column(Integer)  # order within the project
    role: Mapped[str] = mapped_column(String(16))  # user | assistant | tool
    content: Mapped[str] = mapped_column(Text, default="")
    tool_calls: Mapped[list | None] = mapped_column(JSON, nullable=True)  # assistant only
    tool_call_id: Mapped[str | None] = mapped_column(String(64), nullable=True)  # tool only
    name: Mapped[str | None] = mapped_column(String(64), nullable=True)  # tool name
    meta: Mapped[dict | None] = mapped_column(JSON, nullable=True)  # e.g. revision created
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)


class Job(Base):
    """Background work for the builder worker: a chat turn, or processing an upload."""

    __tablename__ = "jobs"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    kind: Mapped[str] = mapped_column(String(32), index=True)  # builder_turn | ingest
    project_id: Mapped[str | None] = mapped_column(ForeignKey("env_projects.id"), index=True)
    created_by: Mapped[str] = mapped_column(ForeignKey("users.id"))
    payload: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    status: Mapped[str] = mapped_column(String(16), default="queued", index=True)
    progress: Mapped[str] = mapped_column(String(500), default="")  # last activity line
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    input_tokens: Mapped[int] = mapped_column(Integer, default=0, server_default="0")
    output_tokens: Mapped[int] = mapped_column(Integer, default=0, server_default="0")
    cache_read_tokens: Mapped[int] = mapped_column(Integer, default=0, server_default="0")
    cost_usd: Mapped[float | None] = mapped_column(Float, nullable=True, default=0.0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    heartbeat_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

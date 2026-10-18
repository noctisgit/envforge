"""Pydantic request/response models for the API."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, EmailStr, Field


# ---- auth ----
class SignupIn(BaseModel):
    email: EmailStr
    password: str = Field(min_length=10, max_length=200)
    display_name: str = ""


class LoginIn(BaseModel):
    email: EmailStr
    password: str


class UserOut(BaseModel):
    id: str
    email: str
    display_name: str
    is_admin: bool
    auth_provider: str


# ---- workspaces ----
class WorkspaceIn(BaseModel):
    name: str = Field(min_length=1, max_length=120)


class WorkspaceSettingsIn(BaseModel):
    name: str | None = None
    default_model: str | None = None
    default_pack: str | None = None
    default_episodes: int | None = Field(default=None, ge=1, le=1000)


class WorkspaceOut(BaseModel):
    id: str
    slug: str
    name: str
    role: str
    default_model: str
    default_pack: str
    default_episodes: int
    keys: list[KeyOut] = []


# ---- keys ----
class KeyIn(BaseModel):
    provider: str
    secret: str = ""
    base_url: str | None = None


class KeyOut(BaseModel):
    provider: str
    masked: str
    base_url: str | None
    updated_at: datetime


# ---- runs ----
class RunIn(BaseModel):
    pack: str | None = None
    model: str | None = None
    episodes: int | None = Field(default=None, ge=1, le=1000)
    tiers: list[int] | None = None
    seed_start: int = 0
    max_steps: int | None = Field(default=None, ge=1, le=200)
    temperature: float = Field(default=0.0, ge=0, le=2)
    label: str = ""


class RunOut(BaseModel):
    id: str
    workspace_id: str
    pack: str
    model: str
    episodes: int
    tiers: list[int]
    seed_start: int
    max_steps: int | None
    temperature: float
    label: str
    status: str
    progress: int
    error: str | None
    metrics: dict | None
    created_at: datetime
    started_at: datetime | None
    finished_at: datetime | None


class PackOut(BaseModel):
    name: str
    kind: str
    tiers: list[int]
    max_steps: int
    description: str


class ServerInfo(BaseModel):
    auth_provider: str
    allow_signup: bool
    clerk_publishable_key: str
    allow_server_keys: bool
    providers: list[str]
    version: str

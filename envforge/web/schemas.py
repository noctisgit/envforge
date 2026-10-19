"""Pydantic request/response models for the API."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, EmailStr, Field


# ---- auth ----
class SignupIn(BaseModel):
    email: EmailStr
    password: str = Field(min_length=10, max_length=200)
    display_name: str = Field(default="", max_length=120)


class LoginIn(BaseModel):
    email: EmailStr
    password: str = Field(max_length=200)


class ProfileIn(BaseModel):
    display_name: str = Field(max_length=120)


class PasswordIn(BaseModel):
    current_password: str = Field(max_length=200)
    new_password: str = Field(min_length=10, max_length=200)


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
    name: str | None = Field(default=None, min_length=1, max_length=120)
    default_model: str | None = Field(default=None, min_length=1, max_length=200)
    default_pack: str | None = Field(default=None, min_length=1, max_length=64)
    default_episodes: int | None = Field(default=None, ge=1, le=1000)
    builder_model: str | None = Field(default=None, min_length=1, max_length=200)
    train_base_model: str | None = Field(default=None, min_length=1, max_length=200)


class WorkspaceOut(BaseModel):
    id: str
    slug: str
    name: str
    role: str
    default_model: str
    default_pack: str
    default_episodes: int
    builder_model: str
    train_base_model: str


# ---- keys ----
class KeyIn(BaseModel):
    provider: str = Field(max_length=32)
    secret: str = Field(default="", max_length=1000)
    base_url: str | None = Field(default=None, max_length=500)


class KeyOut(BaseModel):
    provider: str
    masked: str
    base_url: str | None
    updated_at: datetime
    verified_at: datetime | None = None  # set when the provider accepted the key
    n_models: int | None = None  # models the key can use, when known
    verify_note: str | None = None


# ---- runs ----
class RunIn(BaseModel):
    pack: str | None = Field(default=None, max_length=64)
    model: str | None = Field(default=None, max_length=200)
    episodes: int | None = Field(default=None, ge=1, le=1000)
    tiers: list[int] | None = Field(default=None, max_length=10)
    seed_start: int = Field(default=0, ge=0, le=2_000_000_000)  # fits a 32-bit column
    max_steps: int | None = Field(default=None, ge=1, le=200)
    temperature: float = Field(default=0.0, ge=0, le=2)
    label: str = Field(default="", max_length=200)


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
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cost_usd: float | None = None
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


class ModelOut(BaseModel):
    """One selectable model (see envforge/models/catalog.py)."""

    id: str
    provider: str
    name: str
    owned_by: str
    context_window: int
    input_per_mtok: float | None
    output_per_mtok: float | None
    tags: list[str]
    released: int | None


class ModelsOut(BaseModel):
    providers: list[str]  # providers this workspace can use (has a key for)
    catalog_available: bool  # False when the gateway list couldn't be fetched
    models: list[ModelOut]

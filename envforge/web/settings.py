"""Web settings, read from environment / .env. Secrets never have defaults."""

from __future__ import annotations

from functools import lru_cache
from typing import Annotated, Literal

from pydantic import field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict


class WebSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # --- database ---
    # Postgres in prod:  postgresql+psycopg://envforge:password@localhost:5432/envforge
    # SQLite for tests:  sqlite:///./envforge.db  or  sqlite://  (in-memory)
    DATABASE_URL: str = "sqlite:///./data/envforge.db"

    # --- auth ---
    AUTH_PROVIDER: Literal["local", "clerk"] = "local"
    # Used to sign local session tokens AND to derive the key-encryption key.
    # Generate: python -c "import secrets; print(secrets.token_urlsafe(48))"
    APP_SECRET_KEY: str = ""
    SESSION_TTL_HOURS: int = 24 * 14
    # Clerk (only when AUTH_PROVIDER=clerk)
    CLERK_SECRET_KEY: str = ""
    CLERK_PUBLISHABLE_KEY: str = ""
    CLERK_JWKS_URL: str = (
        ""  # e.g. https://<your-frontend-api>.clerk.accounts.dev/.well-known/jwks.json
    )
    CLERK_ISSUER: str = ""  # e.g. https://<your-frontend-api>.clerk.accounts.dev
    # Local auth: allow self-service sign-up, or only an admin-created first user?
    ALLOW_SIGNUP: bool = True

    # --- model keys ---
    # If true, a workspace with no key of its own may fall back to the server's .env keys.
    ALLOW_SERVER_KEYS: bool = False
    # A vLLM base URL makes the worker send requests wherever it points. By default only
    # public hosts are allowed, so nobody can aim the worker at this machine, the Docker
    # network or a cloud metadata endpoint. Admin accounts may also use these CIDRs (the
    # operator's Tailscale range); comma-separated in .env (NoDecode: not JSON; see _split).
    BASE_URL_ALLOW_PRIVATE: bool = False
    BASE_URL_ALLOW_CIDRS: Annotated[list[str], NoDecode] = ["100.64.0.0/10"]

    # --- limits ---
    MAX_ACTIVE_RUNS_PER_WORKSPACE: int = 3  # queued + running
    STALE_RUN_SECONDS: int = 180  # a running run with no worker heartbeat this long is failed
    LOGIN_ATTEMPTS_PER_MINUTE: int = 10  # per client IP (and per email)
    MAX_CORPUS_BYTES: int = 20 * 1024**3  # largest single builder upload/import (20 GiB)
    SIGNUPS_PER_HOUR: int = 5  # per client IP

    # --- GPU boxes and training (Settings → GPU boxes, Datasets, Training, Models) ---
    # Adding, checking, serving on and training on a GPU box is for admins only unless
    # this is false. A box runs whatever the worker tells it to over SSH, and costs money.
    GPU_BOXES_ADMIN_ONLY: bool = True
    COMPUTE_SSH_TIMEOUT_S: int = 1800  # cap for any single ssh/scp call without its own
    SERVE_HEALTH_TIMEOUT_S: int = 1800  # vLLM must answer /v1/models within this (weights!)
    TRAIN_MAX_HOURS: float = 24.0  # a training job running longer is stopped and failed
    TRAIN_POLL_SECONDS: int = 20  # how often the compute worker reads training progress
    TRAIN_MAX_POLL_FAILURES: int = 15  # consecutive unreachable polls before giving up
    # episodes a run on a vLLM model plays at once: vLLM batches concurrent requests, so a
    # small open model gets through an evaluation (or GRPO's rollouts) several times faster
    VLLM_RUN_CONCURRENCY: int = 8

    # --- http ---
    CORS_ORIGINS: Annotated[list[str], NoDecode] = ["http://localhost:3000"]
    # The header your reverse proxy sets with the real client IP (rate limits key on it).
    # Caddy/nginx: x-forwarded-for (the default). Behind Cloudflare: cf-connecting-ip.
    CLIENT_IP_HEADER: str = "x-forwarded-for"
    COOKIE_SECURE: bool = False  # set true behind HTTPS
    COOKIE_NAME: str = "envforge_session"

    @field_validator("CORS_ORIGINS", "BASE_URL_ALLOW_CIDRS", mode="before")
    @classmethod
    def _split(cls, v):
        if isinstance(v, str):
            return [s.strip() for s in v.split(",") if s.strip()]
        return v

    def require_secret(self) -> None:
        if not self.APP_SECRET_KEY or len(self.APP_SECRET_KEY) < 32:
            raise RuntimeError(
                "APP_SECRET_KEY must be set (>=32 chars). "
                'Generate one: python -c "import secrets; print(secrets.token_urlsafe(48))"'
            )
        if self.AUTH_PROVIDER == "clerk" and not (self.CLERK_JWKS_URL and self.CLERK_ISSUER):
            raise RuntimeError("AUTH_PROVIDER=clerk needs CLERK_JWKS_URL and CLERK_ISSUER")


@lru_cache
def get_settings() -> WebSettings:
    return WebSettings()

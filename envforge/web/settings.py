"""Web settings, read from environment / .env. Secrets never have defaults."""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


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

    # --- http ---
    CORS_ORIGINS: list[str] = ["http://localhost:3000"]
    COOKIE_SECURE: bool = False  # set true behind HTTPS
    COOKIE_NAME: str = "envforge_session"

    @field_validator("CORS_ORIGINS", mode="before")
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

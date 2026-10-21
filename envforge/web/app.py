"""FastAPI application factory.

    uv run envforge-api                     # dev server on :8000
    uv run uvicorn envforge.web.app:app --host 0.0.0.0 --port 8000 --workers 2

OpenAPI docs at /docs. The Next.js app in web/ talks only to /api/*.
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from envforge import __version__
from envforge.web.routers import account, auth, builder, compute, runs, training, workspaces
from envforge.web.settings import get_settings


@asynccontextmanager
async def _lifespan(app: FastAPI):
    s = get_settings()
    s.require_secret()
    if s.DATABASE_URL.startswith("sqlite"):
        # dev convenience; Postgres deployments run `alembic upgrade head`
        from envforge.web.db import create_all

        create_all()
    yield


def create_app() -> FastAPI:
    s = get_settings()
    app = FastAPI(title="envforge", version=__version__, lifespan=_lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=s.CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(auth.router)
    app.include_router(account.router)
    app.include_router(workspaces.router)
    app.include_router(runs.router)
    app.include_router(builder.router)
    app.include_router(compute.router)
    app.include_router(training.router)

    @app.get("/api/health")
    def health() -> dict:
        return {"ok": True, "version": __version__}

    return app


app = create_app()


def main() -> None:
    import uvicorn

    uvicorn.run("envforge.web.app:app", host="0.0.0.0", port=8000, reload=False)

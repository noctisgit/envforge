"""Configuration: loads .env, exposes defaults and the data directory.

Everything secret comes from environment variables. Nothing here is ever
committed. See .env.example for the full list.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env from the current working directory (or any parent) once, at import time.
load_dotenv()

DEFAULT_MODEL: str = os.getenv("ENVFORGE_DEFAULT_MODEL", "anthropic:claude-sonnet-4-5")
DATA_DIR: Path = Path(os.getenv("ENVFORGE_DATA_DIR", "./data")).resolve()

# Workspaces namespace everything a user produces (runs, filtered data, adapters).
# Multi-user auth, when it arrives, maps a user -> a workspace id. Nothing else changes.
DEFAULT_WORKSPACE: str = os.getenv("ENVFORGE_WORKSPACE", "default")


def workspace_dir(workspace: str | None = None) -> Path:
    return DATA_DIR / "workspaces" / (workspace or DEFAULT_WORKSPACE)


def runs_dir(workspace: str | None = None) -> Path:
    return workspace_dir(workspace) / "runs"


def sft_dir(workspace: str | None = None) -> Path:
    return workspace_dir(workspace) / "sft"


RUNS_DIR: Path = runs_dir()

# Provider endpoints. OpenAI-compatible providers differ only by base_url + key.
VERCEL_BASE_URL: str = os.getenv("AI_GATEWAY_BASE_URL", "https://ai-gateway.vercel.sh/v1")
VLLM_BASE_URL: str = os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1")


def env(name: str, default: str | None = None) -> str | None:
    """Read an environment variable; `None` if unset and no default given."""
    return os.getenv(name, default)


def ensure_dirs(workspace: str | None = None) -> None:
    """Create the data directories for a workspace if they do not exist."""
    runs_dir(workspace).mkdir(parents=True, exist_ok=True)

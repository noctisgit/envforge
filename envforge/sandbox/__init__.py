"""Sandboxes: disposable boxes for running untrusted code (Phase 2).

Sandbox packs use `open_sandbox()` and the `Sandbox` interface; they never import
a backend directly. Pick the backend with ``ENVFORGE_SANDBOX=docker|local``.
"""

from envforge.sandbox.base import (
    DEFAULT_IMAGE,
    ExecResult,
    Sandbox,
    SandboxError,
    open_sandbox,
    safe_relpath,
)

__all__ = [
    "DEFAULT_IMAGE",
    "ExecResult",
    "Sandbox",
    "SandboxError",
    "open_sandbox",
    "safe_relpath",
]

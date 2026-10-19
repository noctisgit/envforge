"""The sandbox interface: a disposable box to run untrusted code in.

A *sandbox* pack (``kind = "sandbox"``) keeps its world in one of these instead of
in a Python object. The pack never talks to Docker directly; it uses this small
interface, so the same pack runs on any backend:

    with open_sandbox(files={"main.py": "print('hi')"}) as sb:
        result = sb.exec(["python", "main.py"], timeout=10)
        print(result.exit_code, result.output)

Backends (chosen with the ``ENVFORGE_SANDBOX`` environment variable):

- ``docker`` (default) — a locked-down container per sandbox. See `docker.py`.
- ``local``  — a temp directory + subprocess. **Not isolated.** Tests/dev only.

All file paths are *relative to the sandbox work directory* and may not escape it.
Commands are argument lists, never shell strings, so nothing is ever shell-parsed.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import PurePosixPath
from types import TracebackType

# Cap on how much command output we keep. The runner truncates observations further;
# this just stops a runaway `print` loop from filling the control plane's memory.
MAX_OUTPUT_BYTES = 256 * 1024

DEFAULT_IMAGE = "python:3.12-slim"


class SandboxError(RuntimeError):
    """A backend failure (container would not start, a file could not be written, ...)."""


@dataclass
class ExecResult:
    """What came back from running one command inside a sandbox."""

    exit_code: int
    output: str  # stdout and stderr interleaved, like a terminal shows them
    timed_out: bool = False

    @property
    def ok(self) -> bool:
        return self.exit_code == 0 and not self.timed_out


def safe_relpath(path: str) -> str:
    """Normalise a sandbox path and refuse anything that could escape the work dir.

    Returns the cleaned relative path ("./pkg//a.py" -> "pkg/a.py").
    Raises ValueError for absolute paths, ".." components, backslashes or NUL bytes.
    """
    if not isinstance(path, str) or not path.strip():
        raise ValueError("path must be a non-empty string")
    if "\x00" in path or "\\" in path:
        raise ValueError(f"illegal character in path {path!r}")
    p = PurePosixPath(path.strip())
    if p.is_absolute() or ".." in p.parts:
        raise ValueError(f"path must be relative and stay inside the work dir: {path!r}")
    clean = str(p)
    if clean in ("", "."):
        raise ValueError(f"path must name a file: {path!r}")
    return clean


def clip_output(data: bytes) -> str:
    """Decode command output, keeping at most MAX_OUTPUT_BYTES."""
    if len(data) > MAX_OUTPUT_BYTES:
        data = data[:MAX_OUTPUT_BYTES] + f"\n... [output cut at {MAX_OUTPUT_BYTES} bytes]".encode()
    return data.decode("utf-8", errors="replace")


class Sandbox(ABC):
    """A disposable work directory plus a way to run commands in it.

    Lifecycle: ``start()`` → any number of ``exec`` / file calls → ``close()``.
    Use it as a context manager so ``close()`` always runs. ``close()`` must be
    safe to call more than once.
    """

    @abstractmethod
    def start(self) -> None:
        """Create the sandbox (container, temp dir, ...). Called once."""

    @abstractmethod
    def exec(self, cmd: list[str], timeout: float = 30.0) -> ExecResult:
        """Run `cmd` (an argv list) in the work dir; kill it after `timeout` seconds."""

    @abstractmethod
    def write_file(self, path: str, content: str) -> None:
        """Create or overwrite a text file (parent directories are created)."""

    @abstractmethod
    def read_file(self, path: str) -> str:
        """Return a text file's content. Raises FileNotFoundError if it is not a file."""

    @abstractmethod
    def list_files(self) -> list[str]:
        """All regular files under the work dir, as sorted relative paths."""

    @abstractmethod
    def close(self) -> None:
        """Destroy the sandbox and everything in it. Idempotent."""

    def write_files(self, files: dict[str, str]) -> None:
        """Write many files. Backends override this when a batch is cheaper."""
        for path, content in files.items():
            self.write_file(path, content)

    # ---- context manager -------------------------------------------------
    def __enter__(self) -> Sandbox:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()


def open_sandbox(
    *,
    image: str | None = None,
    files: dict[str, str] | None = None,
    backend: str | None = None,
) -> Sandbox:
    """Start a sandbox, copy `files` into it, and return it (use it in a `with`).

    `backend` defaults to ``$ENVFORGE_SANDBOX`` (``docker`` unless set).
    `image` defaults to ``$ENVFORGE_SANDBOX_IMAGE`` (``python:3.12-slim``); the
    local backend ignores it and uses the current Python.
    """
    backend = (backend or os.environ.get("ENVFORGE_SANDBOX") or "docker").strip().lower()
    image = image or os.environ.get("ENVFORGE_SANDBOX_IMAGE") or DEFAULT_IMAGE

    sb: Sandbox
    if backend == "docker":
        from envforge.sandbox.docker import DockerSandbox  # imported lazily: needs a daemon

        sb = DockerSandbox(image=image)
    elif backend == "local":
        from envforge.sandbox.local import LocalSandbox

        sb = LocalSandbox()
    else:
        raise ValueError(f"ENVFORGE_SANDBOX must be 'docker' or 'local', not {backend!r}")

    try:
        sb.start()
        if files:
            sb.write_files(files)
    except BaseException:
        sb.close()  # never leak a half-started sandbox
        raise
    return sb

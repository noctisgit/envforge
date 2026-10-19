"""Local backend: a temp directory and plain subprocesses.

╔══════════════════════════════════════════════════════════════════════════╗
║  NOT A SECURITY BOUNDARY. Code runs as YOU, with your network, your files ║
║  and your environment variables visible. Use it only for the offline test ║
║  suite and for trusted code during development. Anything a model wrote    ║
║  belongs in the docker backend.                                           ║
╚══════════════════════════════════════════════════════════════════════════╝

What it does give you: the same `Sandbox` interface, paths confined to the temp
dir (at the API level), a wall-clock timeout that kills the whole process group,
and a scrubbed environment (no API keys from your shell leak into child processes).
"""

from __future__ import annotations

import os
import shutil
import signal
import subprocess
import sys
import tempfile
from pathlib import Path

from envforge.sandbox.base import ExecResult, Sandbox, SandboxError, clip_output, safe_relpath


class LocalSandbox(Sandbox):
    """Temp-dir sandbox for tests. Not isolated — see the module docstring."""

    def __init__(self) -> None:
        self._root: Path | None = None

    @property
    def root(self) -> Path:
        if self._root is None:
            raise SandboxError("sandbox not started")
        return self._root

    def _abs(self, path: str) -> Path:
        return self.root / safe_relpath(path)

    def start(self) -> None:
        self._root = Path(tempfile.mkdtemp(prefix="envforge-sbx-"))

    def exec(self, cmd: list[str], timeout: float = 30.0) -> ExecResult:
        # "python" means the interpreter running envforge, so tests behave the
        # same inside or outside a virtualenv.
        if cmd and cmd[0] in ("python", "python3"):
            cmd = [sys.executable, *cmd[1:]]
        env = {
            "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
            "HOME": str(self.root),
            "LANG": "C.UTF-8",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONUNBUFFERED": "1",
        }
        # start_new_session puts the child in its own process group, so a timeout
        # can kill grandchildren too (e.g. a test that spawned a subprocess).
        proc = subprocess.Popen(
            cmd,
            cwd=self.root,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            out, _ = proc.communicate(timeout=timeout)
            return ExecResult(exit_code=proc.returncode, output=clip_output(out))
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            out, _ = proc.communicate()
            return ExecResult(exit_code=124, output=clip_output(out or b""), timed_out=True)

    def write_file(self, path: str, content: str) -> None:
        target = self._abs(path)
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
        except OSError as e:
            raise SandboxError(f"cannot write {path}: {e}") from e

    def read_file(self, path: str) -> str:
        target = self._abs(path)
        if not target.is_file():
            raise FileNotFoundError(path)
        return target.read_text(encoding="utf-8", errors="replace")

    def list_files(self) -> list[str]:
        root = self.root
        return sorted(p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file())

    def close(self) -> None:
        if self._root is not None:
            shutil.rmtree(self._root, ignore_errors=True)
            self._root = None

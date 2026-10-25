"""The compute-box contract.

A *compute box* is a machine the control plane drives remotely: a rented GPU box
today, maybe a local Docker host or a cloud sandbox later. The rest of the compute
layer (GPU checks, vLLM serving, later training) only needs four verbs:

- **run**   execute a command on the box and capture its output
- **put**   copy a file from the control plane to the box
- **get**   copy a file from the box to the control plane
- **close** release any connection the box holds open

`SSHBox` (ssh.py) is the only implementation for now. Keeping the contract tiny is
what lets the vLLM helpers be tested offline with a fake.
"""

from __future__ import annotations

import shlex
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import Any


class ComputeError(RuntimeError):
    """A remote command or file transfer failed in a way the caller should see."""


@dataclass(frozen=True)
class CommandResult:
    """What came back from running one command: exit code plus captured output."""

    returncode: int
    stdout: str = ""
    stderr: str = ""

    @property
    def ok(self) -> bool:
        """True if the command exited with status 0."""
        return self.returncode == 0

    def raise_for_status(self, what: str = "command") -> CommandResult:
        """Raise `ComputeError` (with stderr) unless the command succeeded; else return self."""
        if not self.ok:
            detail = (self.stderr or self.stdout).strip()
            raise ComputeError(f"{what} failed (exit {self.returncode}): {detail}")
        return self


def to_shell(command: str | Sequence[str]) -> str:
    """Turn a command into one string for the remote shell.

    A list is quoted argument-by-argument with `shlex.join`, so a value such as
    `"Qwen/x; rm -rf /"` stays a single harmless argument. A plain string is passed
    through unchanged: use that form only for pipelines *you* wrote, never for text
    that came from a user or a config file.
    """
    if isinstance(command, str):
        return command
    return shlex.join(str(part) for part in command)


class ComputeBox(ABC):
    """A remote machine the control plane can run commands on and copy files to/from."""

    #: Address the control plane uses for the box (also used to build service URLs).
    host: str

    @abstractmethod
    def run(
        self,
        command: str | Sequence[str],
        *,
        input: str | None = None,
        timeout: float | None = None,
        check: bool = False,
    ) -> CommandResult:
        """Run `command` on the box. `input` is sent on stdin (use it for secrets,
        which then never appear in any process list). With `check=True` a non-zero
        exit raises `ComputeError`."""

    @abstractmethod
    def put(self, local: str | Path, remote: str, *, recursive: bool = False) -> None:
        """Copy `local` (on the control plane) to `remote` (on the box)."""

    @abstractmethod
    def get(self, remote: str, local: str | Path, *, recursive: bool = False) -> None:
        """Copy `remote` (on the box) to `local` (on the control plane)."""

    def tunnel(self, remote_port: int, remote_host: str = "127.0.0.1") -> Any:
        """A context manager forwarding a local port to `remote_host:remote_port` on the
        box; it yields an object whose `.url("/v1")` is the local URL to use. Boxes that
        can't forward ports (the default) raise ComputeError."""
        raise ComputeError(f"{type(self).__name__} can't forward ports")

    def close(self) -> None:  # noqa: B027 - optional hook, a no-op by default
        """Release any open connection. Safe to call more than once."""

    def __enter__(self) -> ComputeBox:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

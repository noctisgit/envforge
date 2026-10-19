"""`SSHBox`: drive a remote machine with the system `ssh` and `scp` binaries.

Why the system binaries and not a Python SSH library? They already read your
`~/.ssh/config`, agent, known_hosts and Tailscale MagicDNS names, and they add no
dependency. We only build argument lists and hand them to `subprocess`.

How a remote command travels (worth understanding once):

    local argv  ["ssh", "-o", ..., "root@gpu", "docker ps --filter 'name=x'"]
                                               └────── one string ──────┘
    ssh sends that one string to the remote *login shell*, which parses it again.

So everything that ends up in the remote string must be shell-quoted (`shlex`),
which `ComputeBox.run([...])` does for you when given a list.

The subprocess call is injectable (`runner=`) so tests can record the exact argv
without a network or an `ssh` binary.
"""

from __future__ import annotations

import logging
import re
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Protocol

from envforge.compute.base import CommandResult, ComputeBox, ComputeError, to_shell

log = logging.getLogger(__name__)

# Host names, IPs (v4/v6) and ssh-config aliases. Rejecting a leading "-" stops a
# value like "-oProxyCommand=..." from being read by ssh as an option.
_HOST_RE = re.compile(r"^[A-Za-z0-9_.:%\[\]-]+$")
_USER_RE = re.compile(r"^[A-Za-z0-9_.][A-Za-z0-9_.-]*$")
# scp may hand the remote path to a remote shell (legacy protocol), so we only
# accept plain path characters instead of trying to quote for two different protocols.
_REMOTE_PATH_RE = re.compile(r"^[A-Za-z0-9_./~+@=,-]+$")


class Runner(Protocol):
    """Anything that can execute a local argv. The default is `subprocess_runner`."""

    def __call__(
        self, argv: list[str], *, input: str | None = None, timeout: float | None = None
    ) -> CommandResult: ...


def subprocess_runner(
    argv: list[str], *, input: str | None = None, timeout: float | None = None
) -> CommandResult:
    """Run `argv` locally (no local shell involved) and capture text output."""
    try:
        proc = subprocess.run(
            argv,
            input=input,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except FileNotFoundError as e:
        raise ComputeError(f"'{argv[0]}' not found on the control plane; install OpenSSH") from e
    except subprocess.TimeoutExpired as e:
        out = e.stdout.decode() if isinstance(e.stdout, bytes) else (e.stdout or "")
        return CommandResult(returncode=124, stdout=out, stderr=f"timed out after {timeout}s")
    return CommandResult(proc.returncode, proc.stdout, proc.stderr)


class SSHBox(ComputeBox):
    """A remote box reached with `ssh`/`scp`.

    Args:
        host: hostname, IP, Tailscale name (e.g. `gpu-1` or `100.101.102.103`) or an
            alias from `~/.ssh/config`.
        user: remote user. Rented GPU boxes usually log you in as root.
        port: SSH port (Vast.ai often maps it to something other than 22).
        key_path: private key file; `None` uses your ssh agent / config.
        options: extra `-o Key=Value` options, merged over the defaults below.
        connect_timeout: seconds before an unreachable host fails.
        runner: injectable command executor (tests pass a fake).
    """

    DEFAULT_OPTIONS: Mapping[str, str] = {
        "BatchMode": "yes",  # never prompt for a password; fail instead
        "StrictHostKeyChecking": "accept-new",  # trust a new box once, then pin its key
        "ServerAliveInterval": "30",  # keep long commands (image pulls) alive
        "ServerAliveCountMax": "4",
    }

    def __init__(
        self,
        host: str,
        user: str = "root",
        port: int = 22,
        key_path: str | Path | None = None,
        options: Mapping[str, str] | None = None,
        *,
        connect_timeout: int = 10,
        runner: Runner | None = None,
    ) -> None:
        if not host or host.startswith("-") or not _HOST_RE.match(host):
            raise ValueError(f"invalid host: {host!r}")
        if not user or not _USER_RE.match(user):
            raise ValueError(f"invalid user: {user!r}")
        if not 1 <= int(port) <= 65535:
            raise ValueError(f"invalid ssh port: {port!r}")
        self.host = host
        self.user = user
        self.port = int(port)
        self.key_path = Path(key_path).expanduser() if key_path else None
        self.options: dict[str, str] = {
            **self.DEFAULT_OPTIONS,
            "ConnectTimeout": str(connect_timeout),
            **(options or {}),
        }
        self._runner: Runner = runner or subprocess_runner

    def __repr__(self) -> str:
        return f"SSHBox({self.user}@{self.host}:{self.port})"

    # ---------------------------------------------------------------- argv building

    def _common_args(self) -> list[str]:
        """Options shared by ssh and scp: `-o K=V ...` and `-i key`."""
        args: list[str] = []
        for key, value in self.options.items():
            args += ["-o", f"{key}={value}"]
        if self.key_path:
            args += ["-i", str(self.key_path)]
        return args

    @property
    def destination(self) -> str:
        """`user@host`, as ssh expects it."""
        return f"{self.user}@{self.host}"

    def ssh_argv(self, remote_command: str) -> list[str]:
        """The full local argv that runs `remote_command` (already one shell string)."""
        return [
            "ssh",
            *self._common_args(),
            "-p",
            str(self.port),
            "--",  # nothing after this is parsed as an ssh option
            self.destination,
            remote_command,
        ]

    def _scp_remote(self, remote: str) -> str:
        if not _REMOTE_PATH_RE.match(remote) or remote.startswith("-"):
            raise ValueError(f"unsupported characters in remote path: {remote!r}")
        host = f"[{self.host}]" if ":" in self.host and not self.host.startswith("[") else self.host
        return f"{self.user}@{host}:{remote}"

    def scp_argv(self, src: str, dst: str, *, recursive: bool = False) -> list[str]:
        """The full local argv for one scp copy (`src`/`dst` already formatted)."""
        flags = ["-r"] if recursive else []
        return ["scp", *self._common_args(), "-P", str(self.port), *flags, "--", src, dst]

    # ------------------------------------------------------------------ the 4 verbs

    def run(
        self,
        command: str | Sequence[str],
        *,
        input: str | None = None,
        timeout: float | None = None,
        check: bool = False,
    ) -> CommandResult:
        """Run a command on the box over ssh. Exit code 255 means ssh itself failed
        (unreachable, auth refused); anything else is the remote command's status."""
        remote = to_shell(command)
        argv = self.ssh_argv(remote)
        # Safe to log: secrets travel on stdin (`input`), never in argv.
        log.debug("ssh %s: %s", self.destination, remote)
        result = self._runner(argv, input=input, timeout=timeout)
        if check:
            result.raise_for_status(f"remote command on {self.host}")
        return result

    def put(self, local: str | Path, remote: str, *, recursive: bool = False) -> None:
        """Upload `local` to `remote` with scp."""
        argv = self.scp_argv(str(local), self._scp_remote(remote), recursive=recursive)
        self._runner(argv).raise_for_status(f"scp to {self.host}")

    def get(self, remote: str, local: str | Path, *, recursive: bool = False) -> None:
        """Download `remote` to `local` with scp."""
        argv = self.scp_argv(self._scp_remote(remote), str(local), recursive=recursive)
        self._runner(argv).raise_for_status(f"scp from {self.host}")

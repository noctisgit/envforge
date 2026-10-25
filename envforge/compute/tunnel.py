"""`SSHTunnel`: reach a port on a GPU box through its SSH connection.

Without Tailscale, a rented pod's only reliable way in is its SSH port: RunPod and Vast
map other ports to random public ones, and plain HTTP over the internet would carry the
vLLM key and every prompt in the clear. So the control plane forwards a local port over
SSH instead:

    control plane                         GPU box
    127.0.0.1:<free port>  ══ ssh -L ══>  127.0.0.1:8000 (vLLM)

and talks to `http://127.0.0.1:<free port>/v1` as if vLLM ran locally. vLLM can then
bind to 127.0.0.1 on the box, reachable by nothing but this tunnel.

The tunnel is a long-lived `ssh -N -L ...` child process. If it dies mid-run (a network
blip, the box restarting sshd), a watchdog thread restarts it *on the same local port*,
so clients holding the URL keep working once it's back.
"""

from __future__ import annotations

import contextlib
import logging
import socket
import subprocess
import tempfile
import threading
import time
from collections.abc import Callable
from typing import IO, Any

from envforge.compute.base import ComputeError

log = logging.getLogger(__name__)

#: Builds the full `ssh ... -N -L 127.0.0.1:<local>:<host>:<port> user@box` argv for a local port.
ArgvFor = Callable[[int], list[str]]


def free_local_port() -> int:
    """A TCP port nothing on 127.0.0.1 listens on right now (the OS picks it)."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _port_open(port: int, timeout: float = 0.5) -> bool:
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=timeout):
            return True
    except OSError:
        return False


class SSHTunnel:
    """A local port forwarded to the box while the `with` block runs.

    Args:
        argv_for: builds the ssh argv for a given local port (see `SSHBox.forward_argv`).
        ready_timeout: seconds to wait for ssh to log in and start listening.
        popen, port_open, sleep: injectable for tests.
    """

    def __init__(
        self,
        argv_for: ArgvFor,
        *,
        ready_timeout: float = 30.0,
        popen: Callable[..., Any] = subprocess.Popen,
        port_open: Callable[[int], bool] = _port_open,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._argv_for = argv_for
        self._ready_timeout = ready_timeout
        self._popen = popen
        self._port_open = port_open
        self._sleep = sleep
        self.local_port = 0
        self.restarts = 0
        self._proc: Any = None
        self._stderr: IO[str] | None = None
        self._stop = threading.Event()
        self._watchdog: threading.Thread | None = None

    def url(self, path: str = "/v1") -> str:
        """`http://127.0.0.1:<local port><path>`: what clients use instead of the box's address."""
        return f"http://127.0.0.1:{self.local_port}{path}"

    # ---------------------------------------------------------------- lifecycle

    def __enter__(self) -> SSHTunnel:
        for attempt in range(
            2
        ):  # a port picked by free_local_port can be taken before ssh binds it
            self.local_port = free_local_port()
            try:
                self._start()
                break
            except ComputeError:
                if attempt == 1:
                    raise
        self._watchdog = threading.Thread(target=self._watch, daemon=True)
        self._watchdog.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        if self._watchdog is not None:
            self._watchdog.join(timeout=5)
        self._kill()

    def _start(self) -> None:
        """Start ssh and wait until it listens on the local port (or fails)."""
        self._stderr = tempfile.TemporaryFile(
            mode="w+"
        )  # a file, so a chatty ssh can't block on a pipe
        self._proc = self._popen(
            self._argv_for(self.local_port),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=self._stderr,
            text=True,
        )
        deadline = time.monotonic() + self._ready_timeout
        while True:
            if self._proc.poll() is not None:
                error = f"ssh tunnel exited: {self._error_text()}"
            elif self._port_open(self.local_port):
                return
            elif time.monotonic() > deadline:
                error = f"ssh tunnel not listening after {self._ready_timeout:.0f}s: {self._error_text()}"
            else:
                self._sleep(0.2)
                continue
            self._kill()  # never leave a half-started ssh behind
            raise ComputeError(error)

    def _error_text(self) -> str:
        if self._stderr is None:
            return "no output"
        with contextlib.suppress(OSError, ValueError):
            self._stderr.seek(0)
            lines = [x for x in self._stderr.read().splitlines() if x.strip()]
            if lines:
                return lines[-1][:300]
        return "no output"

    def _kill(self) -> None:
        if self._proc is not None and self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._proc.kill()
        if self._stderr is not None:
            with contextlib.suppress(OSError):
                self._stderr.close()
            self._stderr = None

    def _watch(self) -> None:
        """Restart a dead tunnel on the same local port, backing off up to 30s between tries."""
        backoff = 1.0
        while not self._stop.wait(1.0):
            if self._proc is None or self._proc.poll() is None:
                backoff = 1.0
                continue
            log.warning("ssh tunnel on port %s exited (%s); restarting", self.local_port,
                        self._error_text())  # fmt: skip
            self._kill()
            try:
                self._start()
                self.restarts += 1
            except ComputeError as e:
                log.warning("ssh tunnel restart failed: %s", e)
                if self._stop.wait(backoff):
                    return
                backoff = min(backoff * 2, 30.0)

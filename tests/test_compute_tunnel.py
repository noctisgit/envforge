"""SSH tunnels to GPU boxes (compute/tunnel.py): argv, start-up, failure, restart.

No ssh and no network: a stand-in process plays ssh (it listens on the local port, the
way `ssh -L` does once it has logged in).
"""

from __future__ import annotations

import subprocess
import sys
import time

import pytest

from envforge.compute.base import ComputeError
from envforge.compute.ssh import SSHBox
from envforge.compute.tunnel import SSHTunnel, free_local_port

# listens on argv[1] like `ssh -N -L` does, until killed
LISTENER = (
    "import socket, sys, time\n"
    "s = socket.socket(); s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)\n"
    "s.bind(('127.0.0.1', int(sys.argv[1]))); s.listen(); time.sleep(60)\n"
)


def test_forward_argv_is_a_bare_forward_with_the_box_options():
    box = SSHBox("203.0.113.7", user="root", port=22007, key_path="/tmp/k")
    argv = box.forward_argv(41234, "127.0.0.1", 8000)
    assert argv[0] == "ssh" and argv[-2:] == ["--", "root@203.0.113.7"]
    assert "-N" in argv and argv[argv.index("-L") + 1] == "127.0.0.1:41234:127.0.0.1:8000"
    assert argv[argv.index("-p") + 1] == "22007" and argv[argv.index("-i") + 1] == "/tmp/k"
    assert "ExitOnForwardFailure=yes" in argv and "BatchMode=yes" in argv
    assert box.forward_argv(1, "::1", 8000)[argv.index("-L") + 1] == "127.0.0.1:1:[::1]:8000"
    for host, port in (("-oProxyCommand=evil", 8000), ("127.0.0.1", 0), ("127.0.0.1", 70000)):
        with pytest.raises(ValueError):
            box.forward_argv(41234, host, port)


def test_tunnel_runs_ssh_until_the_block_ends():
    seen: list[list[str]] = []

    def argv_for(port: int) -> list[str]:
        seen.append([str(port)])
        return [sys.executable, "-c", LISTENER, str(port)]

    with SSHTunnel(argv_for) as t:
        assert t.url("/v1") == f"http://127.0.0.1:{t.local_port}/v1"
        assert seen == [[str(t.local_port)]]
        proc = t._proc
        assert proc.poll() is None
    assert proc.poll() is not None  # the "ssh" process is gone after the block


def test_tunnel_reports_why_ssh_failed():
    def argv_for(port: int) -> list[str]:
        script = "import sys; print('root@box: Permission denied (publickey).', file=sys.stderr); sys.exit(255)"
        return [sys.executable, "-c", script]

    with pytest.raises(ComputeError, match="Permission denied"):
        with SSHTunnel(argv_for):
            pass


def test_tunnel_that_never_listens_times_out_and_is_killed():
    procs: list[subprocess.Popen] = []

    def popen(*args, **kwargs):
        procs.append(
            subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"], **kwargs)
        )
        return procs[-1]

    with pytest.raises(ComputeError, match="not listening"):
        with SSHTunnel(lambda port: ["ssh"], ready_timeout=0.5, popen=popen):
            pass
    assert procs and all(p.poll() is not None for p in procs)


def test_dead_tunnel_is_restarted_on_the_same_port():
    def argv_for(port: int) -> list[str]:
        return [sys.executable, "-c", LISTENER, str(port)]

    with SSHTunnel(argv_for) as t:
        port = t.local_port
        t._proc.kill()  # the connection drops mid-run
        deadline = time.monotonic() + 10
        while t.restarts == 0 and time.monotonic() < deadline:
            time.sleep(0.1)
        assert t.restarts == 1 and t.local_port == port and t._proc.poll() is None


def test_free_local_port_is_bindable():
    import socket

    port = free_local_port()
    with socket.socket() as s:
        s.bind(("127.0.0.1", port))


def test_a_refused_login_fails_once_not_twice():
    starts: list[int] = []

    def argv_for(port: int) -> list[str]:
        starts.append(port)
        return [
            sys.executable,
            "-c",
            "import sys; print('Permission denied (publickey).', file=sys.stderr); sys.exit(255)",
        ]

    with pytest.raises(ComputeError, match="Permission denied"):
        with SSHTunnel(argv_for):
            pass
    assert len(starts) == 1  # only a lost port race is retried


def test_closing_while_the_watchdog_restarts_leaves_no_ssh_behind():
    procs: list[subprocess.Popen] = []

    def argv_for(port: int) -> list[str]:
        return [sys.executable, "-c", LISTENER, str(port)]

    def popen(*args, **kwargs):
        procs.append(subprocess.Popen(*args, **kwargs))
        return procs[-1]

    t = SSHTunnel(argv_for, popen=popen)
    with t:
        t._proc.kill()  # the watchdog will restart it...
        time.sleep(1.2)  # ...around now, while we close
    time.sleep(0.5)
    assert all(p.poll() is not None for p in procs)  # every ssh it ever started is gone

"""Detached background processes on a box: the "pod" runtime's stand-in for containers.

In docker mode a long job (vLLM, training) is a detached container: Docker remembers
its name, state, exit code and logs for us. Inside a RunPod/Vast pod there is no
Docker, so this module rebuilds just enough of that from plain Unix parts:

    /root/envforge/run/<name>.pid    process id (also its process-group id)
    /root/envforge/run/<name>.exit   exit code, written when the command finishes
    /root/envforge/run/<name>.json   small non-secret metadata (model, base, started_at)
    /root/envforge/logs/<name>.log   stdout + stderr

How a process is started (`start_detached`), and why each piece is there:

    setsid nohup bash -c WRAPPER envforge-<name> <envfile> <exitfile> <argv...> > log 2>&1 &

- `&` + `nohup`: the command keeps running after our ssh session ends (a dropped
  connection must not kill a multi-hour training run).
- `setsid`: the process starts a new *session and process group* whose id is its own
  pid, so `kill -- -<pid>` later stops it **and every child** (vLLM spawns workers).
- `WRAPPER` (a constant bash snippet, no user data in it) reads secrets from a mode-600
  file that arrived over ssh *stdin*, deletes the file, exports the values, runs the
  command, and records its exit code. Secrets therefore never appear in argv / `ps`.
- `< /dev/null`: the background job must not hold the ssh session's stdin open.

A note on zombies: when a detached process exits, its parent (our long-gone ssh shell)
can't collect its exit status, so it is re-parented to PID 1. If PID 1 never "reaps"
it, it lingers as a zombie that still looks alive. The envforge GPU image runs `tini`
as PID 1 for exactly this reason; the status check also treats zombies as exited.
"""

from __future__ import annotations

import json
import re
import shlex
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from envforge.compute.base import ComputeBox, ComputeError

#: Pid / exit-code / metadata files live here (one set per process name).
RUN_DIR = "/root/envforge/run"
#: Each process's combined stdout + stderr.
LOGS_DIR = "/root/envforge/logs"

# Process names become file names; keep them boring.
_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{0,80}$")

# Runs on the box as `bash -c WRAPPER <$0> <envfile> <exitfile> <command...>`.
# It is a constant: every user-supplied value arrives as a separate, quoted argument.
# `export "KEY=value"` assigns the value literally (no expansion), whatever it contains.
WRAPPER = (
    'envf="$1"; exitf="$2"; shift 2; '
    'while IFS= read -r kv || [ -n "$kv" ]; do [ -n "$kv" ] && export "$kv"; done < "$envf"; '
    'rm -f "$envf"; '
    '"$@"; code=$?; echo "$code" > "$exitf"; exit "$code"'
)


def check_process_name(name: str) -> str:
    """Validate a process name (it becomes part of file paths on the box)."""
    if not _NAME_RE.match(name):
        raise ValueError(f"invalid process name: {name!r}")
    return name


def pid_path(name: str) -> str:
    """Where the pid of process `name` is recorded."""
    return f"{RUN_DIR}/{check_process_name(name)}.pid"


def exit_path(name: str) -> str:
    """Where the exit code of process `name` is written when it finishes."""
    return f"{RUN_DIR}/{check_process_name(name)}.exit"


def meta_path(name: str) -> str:
    """Where the (non-secret) metadata of process `name` is kept."""
    return f"{RUN_DIR}/{check_process_name(name)}.json"


def env_path(name: str) -> str:
    """Transient mode-600 file carrying secrets from stdin to the process."""
    return f"{RUN_DIR}/{check_process_name(name)}.env"


def log_path(name: str) -> str:
    """The combined stdout/stderr log of process `name`."""
    return f"{LOGS_DIR}/{check_process_name(name)}.log"


def render_env(secrets: Mapping[str, str | None]) -> str:
    """KEY=VALUE lines for the secrets file (empty values skipped, one line each)."""
    lines = []
    for key, value in secrets.items():
        if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", key):
            raise ValueError(f"invalid environment variable name: {key!r}")
        if not value:
            continue
        if "\n" in value or "\r" in value:
            raise ValueError(f"{key} must be a single line")
        lines.append(f"{key}={value}\n")
    return "".join(lines)


def build_start_script(
    name: str,
    argv: Sequence[str],
    *,
    meta: Mapping[str, Any] | None = None,
    cwd: str | None = None,
) -> str:
    """The remote shell script that launches `argv` detached and prints its pid.

    Secrets are *not* part of the script: it copies its own stdin into the env file.
    """
    if not argv:
        raise ValueError("empty command")
    q = shlex.quote
    lines = [
        "set -e",
        "umask 077",  # the env file and the log are readable by the owner only
        f"mkdir -p {q(RUN_DIR)} {q(LOGS_DIR)}",
        f"rm -f {q(exit_path(name))}",
        f"cat > {q(env_path(name))}",  # secrets: stdin -> mode-600 file
    ]
    if meta:
        lines.append(f"printf '%s\\n' {q(json.dumps(dict(meta), sort_keys=True))} > {q(meta_path(name))}")  # fmt: skip
    if cwd:
        lines.append(f"cd {q(cwd)}")
    launch = [
        "setsid", "nohup", "bash", "-c", WRAPPER, f"envforge-{name}",
        env_path(name), exit_path(name), *[str(a) for a in argv],
    ]  # fmt: skip
    lines.append(f"{shlex.join(launch)} > {q(log_path(name))} 2>&1 < /dev/null &")
    lines.append(f"echo $! > {q(pid_path(name))}")
    lines.append("echo $!")
    return "\n".join(lines)


def start_detached(
    box: ComputeBox,
    name: str,
    argv: Sequence[str],
    *,
    secrets: Mapping[str, str | None] | None = None,
    meta: Mapping[str, Any] | None = None,
    cwd: str | None = None,
    timeout: float = 120,
) -> int:
    """Start `argv` on the box as a detached background process called `name`.

    Args:
        box: where to run it.
        name: process name (`vllm-8000`, `train-netops-v1`); files are keyed by it.
        argv: the command, one list element per argument (quoted for you).
        secrets: environment variables for the process, sent over ssh stdin.
        meta: small non-secret facts to remember (read back by `process_status`).
        cwd: working directory for the command.

    Returns:
        The pid (= process-group id) of the started process.
    """
    script = build_start_script(name, argv, meta=meta, cwd=cwd)
    # Always send stdin (possibly empty): with `input=None` ssh would read the terminal.
    r = box.run(script, input=render_env(secrets or {}), timeout=timeout)
    r.raise_for_status(f"starting {name} on {box.host}")
    try:
        return int(r.stdout.strip().splitlines()[-1])
    except (IndexError, ValueError) as e:
        raise ComputeError(f"starting {name}: expected a pid, got {r.stdout!r}") from e


@dataclass(frozen=True)
class ProcessStatus:
    """What the pid / exit files say about one named process."""

    name: str
    exists: bool  # a pid file exists (the process was started and not removed)
    running: bool = False
    pid: int | None = None
    exit_code: int | None = None  # None while running, or if it was killed
    meta: dict[str, Any] = field(default_factory=dict)


def build_status_script(name: str) -> str:
    """Remote script printing `missing`, or `running|<pid>|` / `exited|<pid>|<code>`
    followed by the metadata JSON on the next line."""
    q = shlex.quote
    return "\n".join(
        [
            f"pidf={q(pid_path(name))}; exitf={q(exit_path(name))}; metaf={q(meta_path(name))}",
            'if [ ! -f "$pidf" ]; then echo missing; exit 0; fi',
            'pid="$(cat "$pidf")"',
            # The State: line of /proc/<pid>/status; empty if gone, Z for a zombie.
            'state="$(sed -n \'s/^State:[[:space:]]*\\([A-Za-z]\\).*/\\1/p\' "/proc/$pid/status" 2>/dev/null || true)"',  # fmt: skip
            'case "$state" in ""|Z|X) st=exited ;; *) st=running ;; esac',
            'echo "$st|$pid|$(cat "$exitf" 2>/dev/null || true)"',
            'cat "$metaf" 2>/dev/null || true',
        ]
    )


def parse_status(name: str, stdout: str) -> ProcessStatus:
    """Parse the output of `build_status_script`."""
    lines = stdout.strip().splitlines()
    if not lines or lines[0].strip() == "missing":
        return ProcessStatus(name=name, exists=False)
    state, pid, code = (lines[0].strip().split("|") + ["", "", ""])[:3]
    meta: dict[str, Any] = {}
    if len(lines) > 1:
        try:
            meta = json.loads(lines[1])
        except json.JSONDecodeError:
            meta = {}
    running = state == "running"
    return ProcessStatus(
        name=name,
        exists=True,
        running=running,
        pid=int(pid) if pid.strip().isdigit() else None,
        exit_code=int(code) if (not running and code.strip().lstrip("-").isdigit()) else None,
        meta=meta if isinstance(meta, dict) else {},
    )


def process_status(box: ComputeBox, name: str) -> ProcessStatus:
    """Is process `name` running? Its pid, exit code (once finished) and metadata."""
    r = box.run(build_status_script(name), timeout=60)
    r.raise_for_status(f"status of {name} on {box.host}")
    return parse_status(name, r.stdout)


def build_stop_script(name: str, grace_s: int = 30) -> str:
    """Remote script: SIGTERM the process group, wait up to `grace_s`, then SIGKILL.
    Removes the pid/exit/meta files (the log is kept). Prints `stopped` or `none`."""
    q = shlex.quote
    files = " ".join(q(p) for p in (pid_path(name), exit_path(name), meta_path(name), env_path(name)))  # fmt: skip
    return "\n".join(
        [
            f"pidf={q(pid_path(name))}",
            'if [ ! -f "$pidf" ]; then echo none; exit 0; fi',
            'pid="$(cat "$pidf")"',
            # `kill -0 -- -PID` asks "is anyone in process group PID still alive?"
            'if kill -0 -- "-$pid" 2>/dev/null; then',
            '  kill -TERM -- "-$pid" 2>/dev/null || true',
            "  i=0",
            f'  while kill -0 -- "-$pid" 2>/dev/null && [ "$i" -lt {int(grace_s)} ]; do sleep 1; i=$((i+1)); done',  # fmt: skip
            '  kill -KILL -- "-$pid" 2>/dev/null || true',
            "fi",
            f"rm -f {files}",
            "echo stopped",
        ]
    )


def stop_detached(box: ComputeBox, name: str, *, grace_s: int = 30) -> bool:
    """Stop process `name` (and its children) and forget it. False if there was none."""
    r = box.run(build_stop_script(name, grace_s), timeout=grace_s + 60)
    r.raise_for_status(f"stopping {name} on {box.host}")
    return r.stdout.strip().endswith("stopped")


def tail_log(box: ComputeBox, name: str, lines: int = 200, *, check: bool = True) -> str:
    """The last `lines` lines of the process's log (with `check`, a missing log raises)."""
    command = shlex.join(["tail", "-n", str(int(lines)), log_path(name)]) + " 2>&1"
    return box.run(command, timeout=60, check=check).stdout


#: What a carriage return comes back as from read_log_lines (ASCII "record separator").
CR_MARK = "\x1e"


def read_log_lines(box: ComputeBox, name: str, start: int = 0) -> str:
    """The log from line `start` (0-based) to the end; empty if there is no log yet.

    Reading by *line* number (not byte offset) keeps every read on a character
    boundary, so multi-byte UTF-8 (progress bars love those) never gets split.

    Progress bars redraw a line with carriage returns. Those come back as CR_MARK: the
    local subprocess reads output in text mode, which would turn each "\r" into a newline,
    so the caller would count more lines than `tail` does and skip the next ones.
    """
    path = shlex.join(["tail", "-n", f"+{int(start) + 1}", log_path(name)])
    return box.run(f"{path} 2>/dev/null | tr '\\r' '\\036' || true", timeout=120).stdout

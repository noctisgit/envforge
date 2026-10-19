"""Compute backends: drive a remote GPU box (bring your own, over SSH/Tailscale).

Typical use from Python:

    from envforge.compute import SSHBox, check, serve_vllm, wait_healthy

    box = SSHBox("gpu-1")                      # a Tailscale name, IP, or ssh alias
    assert all(c.ok for c in check(box))
    server = serve_vllm(box, "Qwen/Qwen2.5-3B-Instruct", api_key="...")
    wait_healthy(server.base_url, api_key="...")
    # now: VLLM_BASE_URL=server.base_url, model string server.model_spec

On a RunPod/Vast pod running the envforge GPU image (no Docker inside), pass
`runtime="pod"` (or `runtime=detect_runtime(box)`) to serve_vllm / vllm_status /
vllm_logs / stop_vllm and to the `envforge.train.remote` functions.

Provisioning boxes through RunPod/Vast APIs is deliberately out of scope; see
docs/COMPUTE.md. The CLI lives in `envforge.compute.cli` (`envforge compute ...`).
"""

from envforge.compute.base import CommandResult, ComputeBox, ComputeError
from envforge.compute.host import CheckResult, GPUInfo, check, gpu_info, parse_gpu_csv
from envforge.compute.runtime import (
    Runtime,
    RuntimeSupport,
    detect_runtime,
    probe_runtimes,
    resolve_runtime,
)
from envforge.compute.ssh import SSHBox, subprocess_runner
from envforge.compute.vllm import (
    DEFAULT_VLLM_IMAGE,
    VLLMServer,
    VLLMStatus,
    serve_vllm,
    stop_vllm,
    vllm_logs,
    vllm_status,
    wait_healthy,
)

__all__ = [
    "CheckResult",
    "CommandResult",
    "ComputeBox",
    "ComputeError",
    "DEFAULT_VLLM_IMAGE",
    "GPUInfo",
    "Runtime",
    "RuntimeSupport",
    "SSHBox",
    "VLLMServer",
    "VLLMStatus",
    "check",
    "detect_runtime",
    "gpu_info",
    "parse_gpu_csv",
    "probe_runtimes",
    "resolve_runtime",
    "serve_vllm",
    "stop_vllm",
    "subprocess_runner",
    "vllm_logs",
    "vllm_status",
    "wait_healthy",
]

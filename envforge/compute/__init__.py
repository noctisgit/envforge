"""Compute backends: drive a remote GPU box (bring your own, over SSH/Tailscale).

Typical use from Python:

    from envforge.compute import SSHBox, check, serve_vllm, wait_healthy

    box = SSHBox("gpu-1")                      # a Tailscale name, IP, or ssh alias
    assert all(c.ok for c in check(box))
    server = serve_vllm(box, "Qwen/Qwen2.5-3B-Instruct", api_key="...")
    wait_healthy(server.base_url, api_key="...")
    # now: VLLM_BASE_URL=server.base_url, model string server.model_spec

Provisioning boxes through RunPod/Vast APIs is deliberately out of scope; see
docs/COMPUTE.md. The CLI lives in `envforge.compute.cli` (`envforge compute ...`).
"""

from envforge.compute.base import CommandResult, ComputeBox, ComputeError
from envforge.compute.host import CheckResult, GPUInfo, check, gpu_info, parse_gpu_csv
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
    "SSHBox",
    "VLLMServer",
    "VLLMStatus",
    "check",
    "gpu_info",
    "parse_gpu_csv",
    "serve_vllm",
    "stop_vllm",
    "subprocess_runner",
    "vllm_logs",
    "vllm_status",
    "wait_healthy",
]

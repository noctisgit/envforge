"""Which *runtime* a GPU box offers: Docker, or "we are already inside the pod".

envforge starts two long-running GPU workloads on a box: a vLLM server and a LoRA
training job. There are two ways to do that, depending on what the box is:

- **docker**  The box is a VM or bare metal with a Docker daemon (Lambda, Hyperstack,
              a Vast "VM" instance, your workstation). envforge runs each workload as a
              detached container from the pinned vLLM image. This is the original mode.
- **pod**     The box *is* a container: a RunPod Pod or a standard Vast instance running
              the envforge GPU image (deploy/gpu/Dockerfile). There is no Docker daemon
              inside, but vLLM, PyTorch, peft & co. are preinstalled, so envforge starts
              each workload as a detached background *process* instead
              (see `envforge.compute.process`).

Everything else (SSH, uploads, health checks, progress parsing, fetching adapters)
is the same in both modes. Pick one with `--runtime docker|pod|auto` or the
`ENVFORGE_GPU_RUNTIME` environment variable. The default stays `docker`, so existing
setups behave exactly as before; `auto` asks the box (one ssh round trip).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, get_args

from envforge.compute.base import ComputeBox, ComputeError

Runtime = Literal["docker", "pod"]
#: What `--runtime` accepts: a concrete runtime, or "auto" to ask the box.
RuntimeChoice = Literal["docker", "pod", "auto"]

RUNTIMES: tuple[str, ...] = get_args(Runtime)
DEFAULT_RUNTIME: Runtime = "docker"
RUNTIME_ENVVAR = "ENVFORGE_GPU_RUNTIME"

#: A file only the envforge GPU image has. Its presence is how we recognise "pod mode".
POD_MARKER = "/opt/envforge/sft.py"
#: Written at image build time: the image's version (e.g. "0.1.0"), shown by `check`.
POD_VERSION_FILE = "/opt/envforge/VERSION"

# One ssh call answers both questions. `docker info` talks to the daemon, so it fails
# both when Docker isn't installed and when it is installed but no daemon is reachable
# (exactly the situation inside a RunPod/Vast container).
_PROBE_SCRIPT = (
    "if docker info >/dev/null 2>&1; then echo docker=yes; else echo docker=no; fi; "
    f"if [ -f {POD_MARKER} ]; then echo pod=yes; else echo pod=no; fi"
)


def parse_runtime(value: str, *, allow_auto: bool = True) -> RuntimeChoice:
    """Validate a runtime name from a flag or the environment (case-insensitive)."""
    choice = (value or "").strip().lower()
    allowed = (*RUNTIMES, "auto") if allow_auto else RUNTIMES
    if choice not in allowed:
        raise ValueError(f"invalid runtime {value!r}: expected one of {', '.join(allowed)}")
    return choice  # type: ignore[return-value]  # checked against the Literal above


def check_runtime(value: str) -> Runtime:
    """Like `parse_runtime` but refuses "auto": library functions need a concrete answer."""
    return parse_runtime(value, allow_auto=False)  # type: ignore[return-value]


@dataclass(frozen=True)
class RuntimeSupport:
    """What a box can do: run containers (docker), and/or is it the envforge GPU image (pod)."""

    docker: bool
    pod: bool

    @property
    def preferred(self) -> Runtime | None:
        """Docker when a daemon answers (a VM or bare metal), else pod, else None.

        Docker wins when both are true because a working daemon means we are on a
        real host, which is what the docker mode was built for."""
        if self.docker:
            return "docker"
        if self.pod:
            return "pod"
        return None


def probe_runtimes(box: ComputeBox) -> RuntimeSupport:
    """Ask the box which runtimes it supports (one ssh call)."""
    r = box.run(_PROBE_SCRIPT, timeout=60)
    r.raise_for_status(f"runtime probe on {box.host}")
    answers = dict(line.split("=", 1) for line in r.stdout.split() if "=" in line)
    return RuntimeSupport(docker=answers.get("docker") == "yes", pod=answers.get("pod") == "yes")


def detect_runtime(box: ComputeBox) -> Runtime:
    """The runtime `auto` resolves to. Raises `ComputeError` if the box supports neither."""
    support = probe_runtimes(box)
    if support.preferred is None:
        raise ComputeError(
            f"{box.host} has no usable Docker daemon and is not running the envforge GPU "
            f"image ({POD_MARKER} missing). See docs/COMPUTE.md."
        )
    return support.preferred


def resolve_runtime(box: ComputeBox, value: str) -> Runtime:
    """Turn a `--runtime` value into a concrete runtime, probing the box for "auto"."""
    choice = parse_runtime(value)
    if choice == "auto":
        return detect_runtime(box)
    return choice

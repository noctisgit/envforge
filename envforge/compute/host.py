"""Readiness checks and GPU inventory for a compute box.

Before serving a model we want to know, in order:

1. can we reach the box over ssh at all?
2. is Docker installed and is the daemon running?
3. does the box have an NVIDIA driver (`nvidia-smi`)?
4. can a *container* see the GPU? (needs the NVIDIA Container Toolkit; this is the
   step that most often fails on a fresh box)

That list is for the **docker** runtime. In the **pod** runtime (a RunPod/Vast pod
running the envforge GPU image) there is no Docker to check; instead we confirm it
really is the envforge image, that the GPU is visible, and that vLLM and the training
libraries import. See `envforge.compute.runtime` for what the two runtimes are.
"""

from __future__ import annotations

from dataclasses import dataclass

from envforge.compute.base import ComputeBox, ComputeError
from envforge.compute.runtime import POD_MARKER, POD_VERSION_FILE, Runtime, check_runtime

#: Tiny image used to prove containers can see the GPU (the toolkit injects nvidia-smi).
GPU_TEST_IMAGE = "ubuntu:24.04"

#: Imported on a pod to prove serving and training will work (prints vLLM's version).
POD_IMPORT_CHECK = (
    "import accelerate, datasets, peft, torch, transformers, vllm; "
    "print(f'vllm {vllm.__version__}, torch {torch.__version__}, peft {peft.__version__}')"
)

GPU_QUERY = [
    "nvidia-smi",
    "--query-gpu=name,memory.total,memory.used,utilization.gpu",
    "--format=csv,noheader,nounits",
]


@dataclass(frozen=True)
class CheckResult:
    """One readiness check: its name, pass/fail, and a short human-readable detail."""

    name: str
    ok: bool
    detail: str


@dataclass(frozen=True)
class GPUInfo:
    """One GPU as reported by nvidia-smi. Memory in MiB, utilization in percent."""

    index: int
    name: str
    memory_total_mib: int | None
    memory_used_mib: int | None
    utilization_pct: int | None

    @property
    def memory_free_mib(self) -> int | None:
        """Total minus used, or None if either is unknown."""
        if self.memory_total_mib is None or self.memory_used_mib is None:
            return None
        return self.memory_total_mib - self.memory_used_mib


def _first_line(text: str) -> str:
    lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
    return lines[0] if lines else ""


#: Check names per runtime, in the order they run.
CHECK_NAMES: dict[str, tuple[str, ...]] = {
    "docker": ("ssh", "docker", "nvidia-smi", "nvidia-container-toolkit"),
    "pod": ("ssh", "envforge-image", "nvidia-smi", "python-deps"),
}


def check(
    box: ComputeBox, *, gpu_test_image: str = GPU_TEST_IMAGE, runtime: Runtime = "docker"
) -> list[CheckResult]:
    """Run the four readiness checks for `runtime` in order. Later checks are skipped
    (reported as failed) once ssh fails, because nothing else can work without it."""
    runtime = check_runtime(runtime)
    results: list[CheckResult] = []

    r = box.run(["echo", "envforge-ok"], timeout=30)
    ssh_ok = r.ok and "envforge-ok" in r.stdout
    results.append(CheckResult("ssh", ssh_ok, "reachable" if ssh_ok else _first_line(r.stderr)))
    if not ssh_ok:
        for name in CHECK_NAMES[runtime][1:]:
            results.append(CheckResult(name, False, "skipped: ssh failed"))
        return results
    if runtime == "pod":
        return results + _pod_checks(box)

    r = box.run(["docker", "version", "--format", "{{.Server.Version}}"], timeout=30)
    detail = f"server {r.stdout.strip()}" if r.ok else _first_line(r.stderr or r.stdout)
    results.append(CheckResult("docker", r.ok, detail))

    r = box.run(["nvidia-smi", "-L"], timeout=30)
    detail = _first_line(r.stdout) if r.ok else _first_line(r.stderr or r.stdout)
    results.append(CheckResult("nvidia-smi", r.ok, detail or "no GPUs listed"))

    # The real test: a container started with --gpus all can run nvidia-smi.
    r = box.run(
        ["docker", "run", "--rm", "--gpus", "all", gpu_test_image, "nvidia-smi", "-L"],
        timeout=300,  # may need to pull the image the first time
    )
    detail = "containers can see the GPU" if r.ok else _first_line(r.stderr or r.stdout)
    results.append(CheckResult("nvidia-container-toolkit", r.ok, detail))
    return results


def _last_line(text: str) -> str:
    lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
    return lines[-1] if lines else ""


def _pod_checks(box: ComputeBox) -> list[CheckResult]:
    """The three checks after ssh for a pod running the envforge GPU image."""
    results: list[CheckResult] = []

    r = box.run(
        f"if [ -f {POD_MARKER} ]; then cat {POD_VERSION_FILE} 2>/dev/null || echo unknown; "
        "else echo missing; exit 3; fi",
        timeout=30,
    )
    if r.ok:
        results.append(
            CheckResult("envforge-image", True, f"envforge GPU image {r.stdout.strip()}")
        )
    else:
        detail = f"{POD_MARKER} missing: not the envforge GPU image"
        results.append(CheckResult("envforge-image", False, detail))

    r = box.run(["nvidia-smi", "-L"], timeout=30)
    detail = _first_line(r.stdout) if r.ok else _first_line(r.stderr or r.stdout)
    results.append(CheckResult("nvidia-smi", r.ok, detail or "no GPUs listed"))

    # Importing vllm + torch takes a few seconds; it proves the Python stack is intact.
    # On failure the useful line of a traceback is the last one (the exception).
    r = box.run(["python3", "-c", POD_IMPORT_CHECK], timeout=180)
    detail = _first_line(r.stdout) if r.ok else _last_line(r.stderr or r.stdout)
    results.append(CheckResult("python-deps", r.ok, detail or "import failed"))
    return results


def _int_or_none(value: str) -> int | None:
    """nvidia-smi prints '[N/A]' or '[Not Supported]' for unknown values."""
    try:
        return int(float(value))
    except ValueError:
        return None


def parse_gpu_csv(text: str) -> list[GPUInfo]:
    """Parse `nvidia-smi --query-gpu=name,memory.total,memory.used,utilization.gpu
    --format=csv,noheader,nounits` output, one GPU per line, e.g.
    `NVIDIA A10G, 23028, 1, 0`."""
    gpus: list[GPUInfo] = []
    for line in text.strip().splitlines():
        if not line.strip():
            continue
        # Split from the right: the name could in principle contain a comma.
        parts = [p.strip() for p in line.rsplit(",", 3)]
        if len(parts) != 4:
            raise ComputeError(f"unexpected nvidia-smi line: {line!r}")
        name, total, used, util = parts
        gpus.append(
            GPUInfo(
                index=len(gpus),
                name=name,
                memory_total_mib=_int_or_none(total),
                memory_used_mib=_int_or_none(used),
                utilization_pct=_int_or_none(util),
            )
        )
    return gpus


def gpu_info(box: ComputeBox) -> list[GPUInfo]:
    """List the box's GPUs with memory and utilization."""
    r = box.run(GPU_QUERY, timeout=30, check=True)
    return parse_gpu_csv(r.stdout)

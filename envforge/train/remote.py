"""Run `train/sft.py` on a GPU box over SSH and bring the adapter home.

    job = start_training(box, dataset_dir, base="Qwen/Qwen2.5-7B-Instruct", name="netops-v1")
    for event in follow(box, "netops-v1"): print(event)       # parsed progress lines
    fetch_adapter(box, "netops-v1", Path("data/workspaces/default/adapters"))

Layout on the box (one directory tree, so vLLM can mount the adapters later):

    /root/envforge/jobs/<name>/      train.jsonl, val.jsonl, sft.py
    /root/envforge/adapters/<name>/  the LoRA adapter + envforge_manifest.json + metrics.jsonl

Two runtimes (see `envforge.compute.runtime`), one workflow:

- **docker** (VMs, bare metal): training runs in a *detached* container
  (`envforge-train-<name>`), so a dropped SSH connection doesn't kill a multi-hour job;
  `follow()` polls `docker logs` for the script's `ENVFORGE {...}` progress lines. The
  container uses the same pinned vLLM image we serve with (it already has CUDA PyTorch +
  transformers) and pip-installs peft/datasets/accelerate on start, with the pip and
  Hugging Face caches kept in named volumes so reruns are fast.
- **pod** (RunPod/Vast running the envforge GPU image): there is no Docker, but the pod
  *is* that image with peft & co. already installed, so `python3 sft.py ...` runs
  directly as a detached background process (`train-<name>`, see
  `envforge.compute.process`); its output goes to `/root/envforge/logs/train-<name>.log`
  and `follow()` reads that file instead of `docker logs`.

Either way the uploaded `sft.py` (the control plane's copy, so versions always match)
is what runs, the adapter lands in `/root/envforge/adapters/<name>/`, and the progress
events have the same shape.
"""

from __future__ import annotations

import json
import re
import shlex
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from envforge.compute import process
from envforge.compute.base import ComputeBox, ComputeError
from envforge.compute.runtime import Runtime, check_runtime
from envforge.compute.vllm import ADAPTERS_DIR, DEFAULT_VLLM_IMAGE, HF_CACHE_VOLUME

REMOTE_ROOT = "/root/envforge"
PIP_CACHE_VOLUME = "envforge-pip-cache"
TRAIN_DEPS = ("peft>=0.13", "datasets>=3.0", "accelerate>=1.0")
SFT_SCRIPT = Path(__file__).with_name("sft.py")
_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{0,62}$")


def check_name(name: str) -> str:
    """Adapter names become directory names, container names and vLLM model ids."""
    if not _NAME_RE.match(name):
        raise ValueError(
            f"invalid adapter name {name!r}: use lowercase letters, digits, '.', '_' or '-'"
        )
    return name


def container_name(name: str) -> str:
    """The training container's name (docker runtime)."""
    return f"envforge-train-{check_name(name)}"


def process_name(name: str) -> str:
    """The training process's name (pod runtime); its log is logs/<this>.log."""
    return f"train-{check_name(name)}"


def job_dir(name: str) -> str:
    """Where the dataset and sft.py for job `name` are uploaded on the box."""
    return f"{REMOTE_ROOT}/jobs/{check_name(name)}"


@dataclass(frozen=True)
class TrainArgs:
    """The hyper-parameters forwarded to sft.py (see its --help for meaning)."""

    epochs: float = 2
    lr: float = 2e-4
    lora_r: int = 16
    lora_alpha: int = 32
    max_len: int = 8192
    batch_size: int = 1
    grad_accum: int = 8
    max_steps: int | None = None

    def argv(self) -> list[str]:
        out = [
            "--epochs", f"{self.epochs:g}",
            "--lr", f"{self.lr:g}",
            "--lora-r", str(self.lora_r),
            "--lora-alpha", str(self.lora_alpha),
            "--max-len", str(self.max_len),
            "--batch-size", str(self.batch_size),
            "--grad-accum", str(self.grad_accum),
        ]  # fmt: skip
        if self.max_steps:
            out += ["--max-steps", str(self.max_steps)]
        return out


def _check_base(base: str) -> str:
    # Quoting stops shell injection; this stops a "model" like "--trust-remote-code"
    # from being read by sft.py as a flag.
    if not base or base.startswith("-") or any(c.isspace() for c in base):
        raise ValueError(f"invalid base model id: {base!r}")
    return base


def sft_argv(
    python: str, job: str, out: str, base: str, args: TrainArgs, *, has_val: bool
) -> list[str]:
    """`<python> <job>/sft.py --data ... --base ... --out ...`, shared by both runtimes.

    `job` and `out` are paths *as the training process sees them*: `/job` and
    `/adapters/<name>` inside a container, the real box paths in a pod."""
    argv = [python, f"{job}/sft.py", "--data", f"{job}/train.jsonl"]
    if has_val:
        argv += ["--val", f"{job}/val.jsonl"]
    return argv + ["--base", _check_base(base), "--out", out, *args.argv()]


def build_train_command(name: str, base: str, args: TrainArgs, *, has_val: bool) -> str:
    """The shell command run inside the training container (all values quoted)."""
    sft = sft_argv("python", "/job", f"/adapters/{check_name(name)}", base, args, has_val=has_val)
    return f"pip install --quiet {shlex.join(TRAIN_DEPS)} && {shlex.join(sft)}"


def build_pod_train(name: str, base: str, args: TrainArgs, *, has_val: bool) -> list[str]:
    """argv for the training process in pod mode (deps are baked into the image)."""
    out = f"{ADAPTERS_DIR}/{check_name(name)}"
    return sft_argv("python3", job_dir(name), out, base, args, has_val=has_val)


def build_docker_run(name: str, base: str, args: TrainArgs, *, has_val: bool,
                     image: str = DEFAULT_VLLM_IMAGE) -> list[str]:  # fmt: skip
    """argv for the detached training container (without secrets)."""
    _check_base(base)
    job = job_dir(name)
    return [
        "docker", "run", "--detach",
        "--name", container_name(name),
        "--gpus", "all", "--ipc=host",
        "--volume", f"{HF_CACHE_VOLUME}:/root/.cache/huggingface",
        "--volume", f"{PIP_CACHE_VOLUME}:/root/.cache/pip",
        "--volume", f"{job}:/job",
        "--volume", f"{ADAPTERS_DIR}:/adapters",
        "--label", f"envforge.base={base}",
        "--entrypoint", "bash",
        image,
        "-lc", build_train_command(name, base, args, has_val=has_val),
    ]  # fmt: skip


def start_training(
    box: ComputeBox,
    dataset_dir: Path,
    *,
    base: str,
    name: str,
    args: TrainArgs = TrainArgs(),
    hf_token: str | None = None,
    image: str = DEFAULT_VLLM_IMAGE,
    replace: bool = False,
    runtime: Runtime = "docker",
) -> str:
    """Upload the dataset and sft.py, then start training detached.

    Returns the container name (docker) or process name (pod). `image` only matters
    in docker mode; in pod mode the pod's own image is what runs."""
    runtime = check_runtime(runtime)
    dataset_dir = Path(dataset_dir)
    train = dataset_dir / "train.jsonl"
    val = dataset_dir / "val.jsonl"
    if not train.exists():
        raise FileNotFoundError(f"{train} not found (build it with `envforge train dataset`)")
    has_val = val.exists() and val.stat().st_size > 0
    job = job_dir(name)
    if runtime == "pod":
        argv = build_pod_train(name, base, args, has_val=has_val)  # validate before uploading
        _clear_previous_pod_job(box, name, replace=replace)
    else:
        argv = build_docker_run(name, base, args, has_val=has_val, image=image)
        _clear_previous_container(box, name, replace=replace)
    if box.run(["test", "-e", f"{ADAPTERS_DIR}/{name}"]).ok and not replace:
        raise ComputeError(f"adapter {name} already exists on the box; pick another --name")

    box.run(["mkdir", "-p", job, ADAPTERS_DIR], check=True)
    box.put(train, f"{job}/train.jsonl")
    if has_val:
        box.put(val, f"{job}/val.jsonl")
    box.put(SFT_SCRIPT, f"{job}/sft.py")

    if runtime == "pod":
        if hf_token and "\n" in hf_token:
            raise ValueError("HF token must be a single line")
        meta = {"base": base, "adapter": name, "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}  # fmt: skip
        # PYTHONUNBUFFERED: progress lines reach the log file as they are printed.
        secrets = {"HF_TOKEN": hf_token, "PYTHONUNBUFFERED": "1"}
        pname = process_name(name)
        process.start_detached(box, pname, argv, secrets=secrets, meta=meta, cwd=job)
        return pname

    cname = container_name(name)
    if hf_token:
        # same trick as vLLM serving: the token travels on stdin into a 600 env-file
        if "\n" in hf_token:
            raise ValueError("HF token must be a single line")
        script = (
            'umask 077 && envf="$(mktemp)" && trap \'rm -f "$envf"\' EXIT && cat > "$envf" && '
            + shlex.join(argv[:2]) + ' --env-file "$envf" ' + shlex.join(argv[2:])
        )  # fmt: skip
        result = box.run(script, input=f"HF_TOKEN={hf_token}\n", timeout=1800)
    else:
        result = box.run(argv, timeout=1800)
    result.raise_for_status(f"docker run {cname}")
    return cname


def _clear_previous_container(box: ComputeBox, name: str, *, replace: bool) -> None:
    """Docker: refuse to reuse a container name unless `replace`, then remove it."""
    cname = container_name(name)
    existing = box.run(["docker", "inspect", "--format", "{{.State.Status}}", cname])
    if existing.ok:
        if not replace:
            raise ComputeError(
                f"{cname} already exists ({existing.stdout.strip()}); "
                "pick another --name or pass --replace"
            )
        box.run(["docker", "rm", "--force", cname], timeout=120)


def _clear_previous_pod_job(box: ComputeBox, name: str, *, replace: bool) -> None:
    """Pod: the same rule for a previous training process of this name."""
    pname = process_name(name)
    st = process.process_status(box, pname)
    if st.exists:
        if not replace:
            state = "running" if st.running else f"exited with code {st.exit_code}"
            raise ComputeError(
                f"{pname} already exists ({state}); pick another --name or pass --replace"
            )
        process.stop_detached(box, pname)


@dataclass(frozen=True)
class TrainStatus:
    """Is the job running, how did it end, and does the box still know about it?"""

    running: bool
    exit_code: int | None  # None while running, if it was killed, or if the job is gone
    exists: bool


def status(box: ComputeBox, name: str, *, runtime: Runtime = "docker") -> TrainStatus:
    """The training job's state, from Docker (docker) or its pid/exit files (pod)."""
    if check_runtime(runtime) == "pod":
        st = process.process_status(box, process_name(name))
        return TrainStatus(running=st.running, exit_code=st.exit_code, exists=st.exists)
    r = box.run(
        ["docker", "inspect", "--format", "{{.State.Running}}|{{.State.ExitCode}}", container_name(name)]
    )  # fmt: skip
    if not r.ok:
        return TrainStatus(running=False, exit_code=None, exists=False)
    running, code = (r.stdout.strip().split("|") + ["", ""])[:2]
    is_running = running == "true"
    return TrainStatus(
        running=is_running,
        exit_code=None if is_running else int(code or 0),
        exists=True,
    )


def parse_events(log_text: str) -> list[dict[str, Any]]:
    """Pull sft.py's `ENVFORGE {json}` progress lines out of mixed container output."""
    events = []
    for line in log_text.splitlines():
        if line.startswith("ENVFORGE "):
            try:
                events.append(json.loads(line[len("ENVFORGE ") :]))
            except json.JSONDecodeError:
                continue
    return events


def logs(box: ComputeBox, name: str, tail: int = 200, *, runtime: Runtime = "docker") -> str:
    """The last `tail` lines of the job's raw output (pip, warnings, tracebacks)."""
    if check_runtime(runtime) == "pod":
        return process.tail_log(box, process_name(name), tail, check=False)
    r = box.run(["docker", "logs", "--tail", str(int(tail)), container_name(name)])
    return r.stdout + r.stderr


def poll_once(
    box: ComputeBox, name: str, cursor: int = 0, *, runtime: Runtime = "docker"
) -> tuple[list[dict[str, Any]], int, TrainStatus]:
    """One progress poll: `(new events, new cursor, status)`.

    The cursor remembers what was already read, so callers that poll on their own
    schedule (the web worker, `follow`) only get each event once. Its meaning depends
    on the runtime: in docker mode it counts *events* (we re-read `docker logs` whole);
    in pod mode it counts complete log *lines* consumed (we ask `tail -n +K` for the
    rest). Status is read first, so when it says "exited" the log read is complete.
    """
    if check_runtime(runtime) == "pod":
        st = status(box, name, runtime="pod")
        chunk = process.read_log_lines(box, process_name(name), start=cursor)
        lines = chunk.split("\n")
        partial = lines.pop()  # text after the last newline ("" if none)
        cursor += len(lines)
        # a last line without a newline may still be being written: parse it at the end
        if not st.running and partial:
            lines.append(partial)
        return parse_events("\n".join(lines)), cursor, st
    st = status(box, name)
    r = box.run(["docker", "logs", container_name(name)], timeout=300)
    events = parse_events(r.stdout + r.stderr)
    return events[cursor:], len(events), st


def follow(
    box: ComputeBox,
    name: str,
    *,
    poll_s: float = 30,
    sleep: Callable[[float], None] = time.sleep,
    runtime: Runtime = "docker",
) -> Iterator[dict[str, Any]]:
    """Yield new progress events until the job exits, then a final
    `{"event": "exit", "code": n}`. Polls the logs (cheap; two ssh calls per poll)."""
    cursor = 0
    while True:
        events, cursor, st = poll_once(box, name, cursor, runtime=runtime)
        yield from events
        if not st.running:
            yield {"event": "exit", "code": st.exit_code if st.exists else None}
            return
        sleep(poll_s)


def fetch_adapter(box: ComputeBox, name: str, dest_root: Path) -> Path:
    """Copy the trained adapter from the box to `dest_root/<name>/`."""
    dest_root = Path(dest_root)
    dest_root.mkdir(parents=True, exist_ok=True)
    remote = f"{ADAPTERS_DIR}/{check_name(name)}"
    if not box.run(["test", "-f", f"{remote}/adapter_config.json"]).ok:
        raise ComputeError(f"no finished adapter at {box.host}:{remote}")
    box.get(remote, dest_root, recursive=True)  # scp -r creates dest_root/<name>
    return dest_root / name


def remove_container(box: ComputeBox, name: str, *, runtime: Runtime = "docker") -> None:
    """Clean up a finished training job: the container (its logs go with it), or in
    pod mode the pid/exit files (the log file stays in /root/envforge/logs)."""
    if check_runtime(runtime) == "pod":
        process.stop_detached(box, process_name(name))
        return
    box.run(["docker", "rm", "--force", container_name(name)], timeout=120)

"""`FakeGPUBox`: an in-memory GPU box for tests and offline demos.

It implements the `ComputeBox` contract (run / put / get) by answering the exact
commands envforge sends (readiness checks, vLLM in Docker, LoRA training in Docker)
from a small state machine, with no ssh, no Docker and no GPU:

    box = FakeGPUBox()
    check(box)                                   # all four docker-runtime checks pass
    serve_vllm(box, "Qwen/Qwen2.5-0.5B-Instruct")  # a "container" starts serving
    start_training(box, dataset_dir, base=..., name="v1")
    poll_once(box, "v1")                         # progress events, then exit 0
    fetch_adapter(box, "v1", dest)               # writes a tiny fake adapter locally

`FakeGPUBox(unreachable=True)` fails every command the way ssh does (exit 255), and
`FakeGPUBox(train_exit=1)` makes training crash. `urlopen` answers vLLM's /v1/models
for whatever the box is serving, so `wait_healthy(..., urlopen=box.urlopen)` works.
Every command is recorded in `.commands` (and stdin in `.inputs`) for assertions.
"""

from __future__ import annotations

import io
import json
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from envforge.compute.base import CommandResult, ComputeBox, to_shell


class _Response(io.BytesIO):
    """What `urllib.request.urlopen` returns, enough for `wait_healthy`."""

    def __enter__(self) -> _Response:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


class _FakeTunnel:
    """Stands in for an SSH tunnel: the "local" URL keeps the remote port, so `urlopen`
    still knows which server is meant."""

    def __init__(self, box: FakeGPUBox, remote_host: str, remote_port: int) -> None:
        self.box, self.remote_host, self.remote_port = box, remote_host, remote_port

    def url(self, path: str = "/v1") -> str:
        return f"http://127.0.0.1:{self.remote_port}{path}"

    def __enter__(self) -> _FakeTunnel:
        if self.box.unreachable:
            from envforge.compute.base import ComputeError

            raise ComputeError("ssh tunnel exited: ssh: connect to host: Connection timed out")
        self.box.tunnels.append((self.remote_host, self.remote_port))
        self.box.open_tunnels += 1
        return self

    def __exit__(self, *exc: object) -> None:
        self.box.open_tunnels -= 1


class FakeGPUBox(ComputeBox):
    """A scripted GPU box (docker runtime). See the module docstring."""

    def __init__(
        self,
        host: str = "fake-gpu",
        *,
        unreachable: bool = False,
        gpu: str = "NVIDIA L40S",
        vram_mib: int = 46068,
        train_polls: int = 2,
        train_exit: int = 0,
    ) -> None:
        self.host = host
        self.unreachable = unreachable
        self.gpu = gpu
        self.vram_mib = vram_mib
        self.train_polls = train_polls  # polls that report "running" before the job exits
        self.train_exit = train_exit
        self.commands: list[str] = []
        self.inputs: list[str | None] = []
        self.puts: list[tuple[str, str]] = []
        self.gets: list[tuple[str, str]] = []
        self.serving: dict[int, dict[str, Any]] = {}  # port -> {"model", "adapters"}
        self.trainings: dict[str, dict[str, Any]] = {}  # adapter name -> state
        self.adapters: set[str] = set()  # finished adapters on the "disk"
        self.tunnels: list[
            tuple[str, int]
        ] = []  # (remote host, remote port) of every tunnel opened
        self.open_tunnels = 0
        self.closed = False

    # ------------------------------------------------------------------ helpers

    @staticmethod
    def _ok(stdout: str = "") -> CommandResult:
        return CommandResult(0, stdout, "")

    def _train_events(self, name: str) -> str:
        t = self.trainings[name]
        total = 4
        lines = [
            "Collecting peft (pip noise)",
            f'ENVFORGE {{"event": "start", "total_steps": {total}, "n_train": 10, "n_val": 2}}',
        ]
        shown = min(t["polls"], total)
        for step in range(1, shown + 1):
            loss = round(2.0 / step, 4)
            epoch = step / total * 2
            lines.append(
                f'ENVFORGE {{"event": "log", "step": {step}, "epoch": {epoch:.2f}, "loss": {loss}}}'
            )
        if t["done"]:
            if self.train_exit == 0:
                lines.append('ENVFORGE {"event": "eval", "step": 4, "epoch": 2.0, "loss": 0.61}')
                lines.append('ENVFORGE {"event": "done", "final_train_loss": 0.5}')
            else:
                lines.append('ENVFORGE {"event": "error", "message": "CUDA out of memory"}')
        return "\n".join(lines) + "\n"

    # ------------------------------------------------------------------ the verbs

    def run(
        self,
        command: str | Sequence[str],
        *,
        input: str | None = None,
        timeout: float | None = None,
        check: bool = False,
    ) -> CommandResult:
        cmd = to_shell(command)
        self.commands.append(cmd)
        self.inputs.append(input)
        result = self._answer(cmd)
        if check:
            result.raise_for_status(f"remote command on {self.host}")
        return result

    def _answer(self, cmd: str) -> CommandResult:  # noqa: C901 - one branch per command
        if self.unreachable:
            return CommandResult(255, "", f"ssh: connect to host {self.host} port 22: timed out")
        if cmd.startswith("echo envforge-ok"):
            return self._ok("envforge-ok\n")
        if "docker info" in cmd and "pod=" in cmd:  # runtime probe
            return self._ok("docker=yes\npod=no\n")
        if "driver=" in cmd and "disk_free_kb" in cmd:  # box_facts
            return self._ok(
                "driver=550.54.15\ncuda=12.4\ndisk_free_kb=104857600\nvllm_image=yes\nvllm_python=\n"
            )
        if cmd.startswith("docker version"):
            return self._ok("27.3.1\n")
        if cmd.startswith("nvidia-smi -L"):
            return self._ok(f"GPU 0: {self.gpu} (UUID: GPU-fake)\n")
        if cmd.startswith("nvidia-smi --query-gpu"):
            return self._ok(f"{self.gpu}, {self.vram_mib}, 1, 0\n")
        if cmd.startswith("docker run --rm --gpus all"):
            return self._ok(f"GPU 0: {self.gpu}\n")

        # ---- vLLM ----
        m = re.search(r"envforge-vllm-(\d+)", cmd)
        if m:
            port = int(m.group(1))
            if "docker rm --force" in cmd and "docker run" not in cmd:
                existed = self.serving.pop(port, None)
                return self._ok() if existed else CommandResult(1, "", "Error: No such container")
            if "docker run" in cmd:
                model = re.search(r"--label envforge\.model=(\S+)", cmd)
                adapters = re.findall(r"(\S+)=/adapters/\S+", cmd)
                self.serving[port] = {
                    "model": (model.group(1) if model else "?").strip("'"),
                    "adapters": [a.strip("'") for a in adapters],
                }
                return self._ok(f"cid{port}abcdef\n")
            if cmd.startswith("docker inspect"):
                s = self.serving.get(port)
                if not s:
                    return CommandResult(1, "", "Error: No such object")
                return self._ok(f"running|vllm/vllm-openai|{s['model']}|2026-10-21T00:00:00Z\n")
            if cmd.startswith("docker logs"):
                return self._ok("INFO vLLM API server started\n")

        # ---- training ----
        m = re.search(r"envforge-train-([a-z0-9._-]+)", cmd)
        if m:
            name = m.group(1)
            t = self.trainings.get(name)
            if "docker run" in cmd:
                self.trainings[name] = {"polls": 0, "done": False, "cmd": cmd}
                return self._ok("trainabc123\n")
            if cmd.startswith("docker inspect --format {{.State.Status}}"):
                return self._ok("running\n") if t else CommandResult(1, "", "No such object")
            if cmd.startswith("docker inspect"):
                if not t:
                    return CommandResult(1, "", "No such object")
                if not t["done"]:
                    t["polls"] += 1
                    if t["polls"] > self.train_polls:
                        t["done"] = True
                        if self.train_exit == 0:
                            self.adapters.add(name)
                if t["done"]:
                    return self._ok(f"false|{self.train_exit}\n")
                return self._ok("true|0\n")
            if cmd.startswith("docker logs"):
                return self._ok(self._train_events(name)) if t else CommandResult(1, "", "none")
            if cmd.startswith("docker rm --force"):
                self.trainings.pop(name, None)
                return self._ok()
        if cmd.startswith("test "):
            m = re.search(r"/root/envforge/adapters/([a-z0-9._-]+)", cmd)
            present = bool(m and m.group(1) in self.adapters)
            return CommandResult(0 if present else 1, "", "")
        return self._ok()

    def put(self, local: str | Path, remote: str, *, recursive: bool = False) -> None:
        if self.unreachable:
            CommandResult(255, "", "ssh: timed out").raise_for_status(f"scp to {self.host}")
        self.puts.append((str(local), remote))
        m = re.match(r"/root/envforge/adapters/?([a-z0-9._-]*)$", remote)
        if m and recursive:  # an adapter uploaded for serving
            self.adapters.add(m.group(1) or Path(local).name)

    def get(self, remote: str, local: str | Path, *, recursive: bool = False) -> None:
        if self.unreachable:
            CommandResult(255, "", "ssh: timed out").raise_for_status(f"scp from {self.host}")
        self.gets.append((remote, str(local)))
        name = remote.rstrip("/").rsplit("/", 1)[-1]
        out = Path(local) / name
        out.mkdir(parents=True, exist_ok=True)
        (out / "adapter_config.json").write_text(json.dumps({"r": 16, "peft_type": "LORA"}))
        (out / "adapter_model.safetensors").write_bytes(b"\0" * 64)
        manifest = {"base_model": "fake", "n_train": 10, "global_step": 4,
                    "final_train_loss": 0.5, "final_eval_loss": 0.61}  # fmt: skip
        (out / "envforge_manifest.json").write_text(json.dumps(manifest))

    def tunnel(self, remote_port: int, remote_host: str = "127.0.0.1") -> _FakeTunnel:
        return _FakeTunnel(self, remote_host, remote_port)

    def close(self) -> None:
        self.closed = True

    # ------------------------------------------------------------------ HTTP

    def urlopen(self, request: Any, timeout: float | None = None) -> _Response:
        """Answer GET <base>/models like vLLM would, for whatever is being served."""
        url = getattr(request, "full_url", str(request))
        m = re.search(r":(\d+)/v1/models$", url)
        s = self.serving.get(int(m.group(1))) if m else None
        if not s or self.unreachable:
            raise OSError("connection refused")
        ids = [s["model"], *s["adapters"]]
        return _Response(json.dumps({"data": [{"id": i} for i in ids]}).encode())

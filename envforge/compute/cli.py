"""`envforge compute ...`: drive a GPU box over SSH from the control plane.

    envforge compute check     --host gpu-1
    envforge compute gpu       --host gpu-1
    envforge compute vllm-up   Qwen/Qwen2.5-3B-Instruct --host gpu-1 --api-key "$VLLM_API_KEY"
    envforge compute vllm-status --host gpu-1
    envforge compute vllm-logs --host gpu-1 --tail 100
    envforge compute vllm-down --host gpu-1

Connection options can come from the environment (or .env) instead of flags:
ENVFORGE_GPU_HOST, ENVFORGE_GPU_USER, ENVFORGE_GPU_PORT, ENVFORGE_GPU_KEY.
Note `--port` is the *SSH* port; the vLLM port is `--vllm-port`.
"""

from __future__ import annotations

from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from envforge import config  # noqa: F401 - importing loads .env so envvar defaults work
from envforge.compute.base import ComputeError
from envforge.compute.host import check, gpu_info
from envforge.compute.ssh import SSHBox
from envforge.compute.vllm import (
    DEFAULT_VLLM_IMAGE,
    serve_vllm,
    stop_vllm,
    vllm_logs,
    vllm_status,
    wait_healthy,
)

compute_app = typer.Typer(
    help="Drive a rented GPU box over SSH: readiness checks, GPU info, vLLM serving.",
    no_args_is_help=True,
)
console = Console()

# Shared connection options (Annotated so every command declares them identically).
HostOpt = Annotated[
    str,
    typer.Option(
        "--host", envvar="ENVFORGE_GPU_HOST", help="GPU box: Tailscale name, IP or ssh alias."
    ),
]
UserOpt = Annotated[str, typer.Option("--user", envvar="ENVFORGE_GPU_USER", help="SSH user.")]
PortOpt = Annotated[
    int, typer.Option("--port", envvar="ENVFORGE_GPU_PORT", help="SSH port (not the vLLM port).")
]
KeyOpt = Annotated[
    str | None,
    typer.Option("--key", envvar="ENVFORGE_GPU_KEY", help="SSH private key (default: agent)."),
]
VllmPortOpt = Annotated[int, typer.Option("--vllm-port", help="Host port vLLM is published on.")]


def _box(host: str, user: str, port: int, key: str | None) -> SSHBox:
    try:
        return SSHBox(host, user=user, port=port, key_path=key)
    except ValueError as e:
        raise typer.BadParameter(str(e)) from e


def _fail(message: str) -> typer.Exit:
    console.print(f"[red]error:[/red] {message}")
    return typer.Exit(code=1)


@compute_app.command("check")
def check_cmd(
    host: HostOpt, user: UserOpt = "root", port: PortOpt = 22, key: KeyOpt = None
) -> None:
    """Check ssh, docker, nvidia-smi and the NVIDIA container toolkit on the box."""
    box = _box(host, user, port, key)
    try:
        results = check(box)
    except ComputeError as e:
        raise _fail(str(e)) from e
    t = Table("check", "", "detail", title=str(box))
    for r in results:
        t.add_row(r.name, "[green]ok[/green]" if r.ok else "[red]FAIL[/red]", r.detail)
    console.print(t)
    if not all(r.ok for r in results):
        console.print("see docs/COMPUTE.md, 'Preparing the box'")
        raise typer.Exit(code=1)


@compute_app.command("gpu")
def gpu_cmd(host: HostOpt, user: UserOpt = "root", port: PortOpt = 22, key: KeyOpt = None) -> None:
    """Show the box's GPUs: name, memory, utilization."""
    box = _box(host, user, port, key)
    try:
        gpus = gpu_info(box)
    except ComputeError as e:
        raise _fail(str(e)) from e
    t = Table("#", "name", "memory used / total (MiB)", "util", title=str(box))
    for g in gpus:
        t.add_row(
            str(g.index),
            g.name,
            f"{g.memory_used_mib} / {g.memory_total_mib}",
            f"{g.utilization_pct}%" if g.utilization_pct is not None else "?",
        )
    console.print(t)


@compute_app.command("vllm-up")
def vllm_up(
    model: Annotated[
        str, typer.Argument(help="Hugging Face model id, e.g. Qwen/Qwen2.5-3B-Instruct")
    ],
    host: HostOpt,
    user: UserOpt = "root",
    port: PortOpt = 22,
    key: KeyOpt = None,
    vllm_port: VllmPortOpt = 8000,
    api_key: Annotated[
        str | None,
        typer.Option(envvar="VLLM_API_KEY", help="Require this bearer key (recommended)."),
    ] = None,
    hf_token: Annotated[
        str | None, typer.Option(envvar="HF_TOKEN", help="Hugging Face token for gated models.")
    ] = None,
    tool_call_parser: Annotated[
        str, typer.Option(help="vLLM tool-call parser for this model family.")
    ] = "hermes",
    gpu_memory_utilization: Annotated[float, typer.Option(help="Fraction of VRAM to use.")] = 0.90,
    max_model_len: Annotated[
        int | None, typer.Option(help="Cap context length (saves KV-cache memory).")
    ] = None,
    image: Annotated[str, typer.Option(help="vLLM Docker image (pinned).")] = DEFAULT_VLLM_IMAGE,
    bind: Annotated[
        str | None, typer.Option(help="Box interface to publish on, e.g. its Tailscale IP.")
    ] = None,
    endpoint_host: Annotated[
        str | None,
        typer.Option(help="Host to put in the base URL if --host is an ssh alias."),
    ] = None,
    extra_arg: Annotated[
        list[str] | None,
        typer.Option(
            "--extra-arg", help="Extra vLLM flag; repeat: --extra-arg=--dtype --extra-arg=half"
        ),
    ] = None,
    lora: Annotated[
        list[str] | None,
        typer.Option("--lora", help="Also serve this adapter from `envforge train` (repeatable)."),
    ] = None,
    max_lora_rank: Annotated[int, typer.Option(help="Largest LoRA rank among --lora.")] = 64,
    wait: Annotated[bool, typer.Option(help="Wait until /v1/models answers.")] = True,
    timeout: Annotated[float, typer.Option(help="Seconds to wait for health.")] = 1200,
) -> None:
    """Start vLLM serving MODEL in Docker on the box and print how to use it."""
    box = _box(host, user, port, key)
    if not api_key:
        console.print(
            "[yellow]warning:[/yellow] no --api-key: anyone who can reach port "
            f"{vllm_port} can use this server. Only do this over Tailscale / a private network."
        )
    console.print(f"starting [bold]{model}[/bold] on {box} (first run pulls a large image)...")
    try:
        server = serve_vllm(
            box,
            model,
            port=vllm_port,
            api_key=api_key,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            tool_call_parser=tool_call_parser,
            hf_token=hf_token,
            image=image,
            extra_args=extra_arg or (),
            bind=bind,
            endpoint_host=endpoint_host,
            lora_adapters=lora or (),
            max_lora_rank=max_lora_rank,
        )
        console.print(f"container {server.container} ({server.container_id[:12]}) started")
        if wait:
            console.print(
                f"waiting for {server.base_url}/models (downloading weights can take a while)..."
            )
            ids = wait_healthy(
                server.base_url,
                timeout=timeout,
                api_key=api_key,
                is_alive=lambda: vllm_status(box, vllm_port).running,
            )
            console.print(f"[green]healthy[/green]: serving {', '.join(ids)}")
    except (ComputeError, TimeoutError, ValueError) as e:
        console.print(f"[red]error:[/red] {e}")
        console.print(f"logs: envforge compute vllm-logs --host {host} --vllm-port {vllm_port}")
        raise typer.Exit(code=1) from e

    console.print()
    console.print(f"base URL:     [bold]{server.base_url}[/bold]")
    console.print(f"model string: [bold]{server.model_spec}[/bold]")
    for name in lora or ():
        console.print(f"adapter:      [bold]vllm:{name}[/bold]  (base + LoRA {name})")
    console.print("\nadd to .env (or set the vLLM provider base URL + key in the web UI):")
    print(f"VLLM_BASE_URL={server.base_url}")
    if api_key:
        print("VLLM_API_KEY=<the key you passed>")
    console.print(f"\nthen: uv run envforge run netops --model {server.model_spec} --episodes 30")


@compute_app.command("vllm-down")
def vllm_down(
    host: HostOpt,
    user: UserOpt = "root",
    port: PortOpt = 22,
    key: KeyOpt = None,
    vllm_port: VllmPortOpt = 8000,
) -> None:
    """Stop and remove the vLLM container (the weights cache volume is kept)."""
    box = _box(host, user, port, key)
    try:
        removed = stop_vllm(box, vllm_port)
    except ComputeError as e:
        raise _fail(str(e)) from e
    console.print("stopped" if removed else "nothing running on that port")


@compute_app.command("vllm-logs")
def vllm_logs_cmd(
    host: HostOpt,
    user: UserOpt = "root",
    port: PortOpt = 22,
    key: KeyOpt = None,
    vllm_port: VllmPortOpt = 8000,
    tail: Annotated[int, typer.Option(help="How many lines.")] = 200,
) -> None:
    """Print the last lines of the vLLM container's log."""
    box = _box(host, user, port, key)
    try:
        print(vllm_logs(box, vllm_port, tail=tail), end="")
    except ComputeError as e:
        raise _fail(str(e)) from e


@compute_app.command("vllm-status")
def vllm_status_cmd(
    host: HostOpt,
    user: UserOpt = "root",
    port: PortOpt = 22,
    key: KeyOpt = None,
    vllm_port: VllmPortOpt = 8000,
) -> None:
    """Show whether the vLLM container is running, and which model and image it uses."""
    box = _box(host, user, port, key)
    try:
        st = vllm_status(box, vllm_port)
    except ComputeError as e:
        raise _fail(str(e)) from e
    if not st.exists:
        console.print(f"{st.container}: not found")
        raise typer.Exit(code=1)
    colour = "green" if st.running else "red"
    console.print(f"{st.container}: [{colour}]{st.state}[/{colour}]")
    console.print(f"  model:   {st.model or '?'}")
    console.print(f"  image:   {st.image or '?'}")
    console.print(f"  started: {st.started_at or '?'}")

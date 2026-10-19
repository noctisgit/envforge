"""`envforge train ...`: rejection sampling → dataset → LoRA SFT on a GPU box → adapter.

    envforge train dataset data/workspaces/default/runs/<sonnet-run> --name netops-v1
    envforge train check-overlap netops-v1 data/workspaces/default/runs/<eval-run>
    envforge train sft netops-v1 --base Qwen/Qwen2.5-7B-Instruct --host gpu-1
    envforge train follow netops-v1 --host gpu-1        # progress until it finishes
    envforge train fetch  netops-v1 --host gpu-1        # adapter -> data/.../adapters/
    envforge compute vllm-up Qwen/Qwen2.5-7B-Instruct --host gpu-1 --lora netops-v1
    envforge run netops --model vllm:netops-v1 --episodes 100 --seed-start 0

Datasets live in data/workspaces/<ws>/sft/<name>/, adapters in .../adapters/<name>/.
See docs/TRAINING.md for the whole loop and how to read the before/after numbers.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from envforge import config
from envforge.compute.base import ComputeError
from envforge.compute.cli import HostOpt, KeyOpt, PortOpt, UserOpt, _box
from envforge.train import dataset as ds
from envforge.train import remote

train_app = typer.Typer(
    help="Turn good episodes into a fine-tuned open model (Phase 3: rejection sampling + LoRA SFT).",
    no_args_is_help=True,
)
console = Console()

WorkspaceOpt = Annotated[str, typer.Option("--workspace", help="Workspace for datasets/adapters.")]


def _dataset_dir(name_or_path: str, workspace: str) -> Path:
    p = Path(name_or_path)
    return p if p.exists() else config.sft_dir(workspace) / name_or_path


@train_app.command("dataset")
def dataset_cmd(
    run_dirs: Annotated[list[Path], typer.Argument(help="Run directories to draw episodes from.")],
    name: Annotated[str, typer.Option("--name", help="Dataset name (a directory under sft/).")],
    min_reward: Annotated[float, typer.Option(help="Keep episodes scoring at least this.")] = 1.0,
    val_frac: Annotated[
        float, typer.Option(help="Fraction of tasks held out for validation.")
    ] = 0.05,
    workspace: WorkspaceOpt = config.DEFAULT_WORKSPACE,
) -> None:
    """Rejection sampling: keep the best episodes, de-duplicate by task, split by task."""
    out = config.sft_dir(workspace) / remote.check_name(name)
    try:
        rep = ds.build_dataset(run_dirs, out, min_reward=min_reward, val_frac=val_frac)
    except ValueError as e:
        raise typer.BadParameter(str(e)) from e
    t = Table("pack / tier", "episodes")
    for (pack, tier), n in sorted(rep.by_pack_tier.items()):
        t.add_row(f"{pack} / tier {tier}", str(n))
    console.print(t)
    console.print(
        f"kept {rep.kept}/{rep.total} episodes (reward ≥ {min_reward}; {rep.duplicates} duplicate "
        f"tasks dropped) → train {rep.n_train}, val {rep.n_val}\n"
        f"models: {', '.join(f'{m} ×{n}' for m, n in rep.models.items())}\n"
        f"wrote [bold]{out}[/bold] (train.jsonl, val.jsonl, manifest.json)"
    )
    if rep.kept < 200:
        console.print(
            "[yellow]note:[/yellow] a few hundred to a few thousand episodes is a typical SFT "
            "dataset; this is fine for a smoke test."
        )
    seeds = json.loads((out / "manifest.json").read_text())["tasks"]
    console.print(f"training tasks (seed ranges): {seeds} — evaluate on seeds outside these.")


@train_app.command("check-overlap")
def check_overlap_cmd(
    dataset: Annotated[str, typer.Argument(help="Dataset name or directory.")],
    run_dirs: Annotated[list[Path], typer.Argument(help="Evaluation run directories.")],
    workspace: WorkspaceOpt = config.DEFAULT_WORKSPACE,
) -> None:
    """Fail if an evaluation run contains tasks the dataset was built from."""
    d = _dataset_dir(dataset, workspace)
    dirty = False
    for rd in run_dirs:
        hits = ds.overlap(d, rd)
        if hits:
            dirty = True
            sample = ", ".join(f"{p} t{t} s{s}" for p, t, s in hits[:5])
            console.print(
                f"[red]{rd.name}: {len(hits)} task(s) also in training[/red] ({sample} …)"
            )
        else:
            console.print(f"[green]{rd.name}: clean[/green] — no training tasks")
    raise typer.Exit(code=1 if dirty else 0)


@train_app.command("sft")
def sft_cmd(
    dataset: Annotated[
        str, typer.Argument(help="Dataset name or directory (from `train dataset`).")
    ],
    base: Annotated[str, typer.Option(help="Base model to fine-tune (Hugging Face id).")],
    name: Annotated[
        str | None, typer.Option(help="Adapter name (default: the dataset name).")
    ] = None,
    host: Annotated[
        str | None,
        typer.Option("--host", envvar="ENVFORGE_GPU_HOST", help="GPU box; omit to train locally."),
    ] = None,
    user: UserOpt = "root",
    port: PortOpt = 22,
    key: KeyOpt = None,
    epochs: float = 2,
    lr: float = 2e-4,
    lora_r: int = 16,
    lora_alpha: int = 32,
    max_len: int = 8192,
    batch_size: int = 1,
    grad_accum: int = 8,
    max_steps: Annotated[int | None, typer.Option(help="Stop after N steps (smoke tests).")] = None,
    hf_token: Annotated[
        str | None, typer.Option(envvar="HF_TOKEN", help="For gated models.")
    ] = None,
    replace: Annotated[
        bool, typer.Option(help="Overwrite an existing job/adapter of this name.")
    ] = False,
    follow_logs: Annotated[bool, typer.Option("--follow/--detach", help="Stream progress.")] = True,
    dry_run: Annotated[
        bool, typer.Option(help="Local only: tokenise and report, no training.")
    ] = False,
    workspace: WorkspaceOpt = config.DEFAULT_WORKSPACE,
) -> None:
    """LoRA supervised fine-tuning of BASE on the dataset, on a GPU box (or locally)."""
    d = _dataset_dir(dataset, workspace)
    name = remote.check_name(name or d.name)
    args = remote.TrainArgs(
        epochs, lr, lora_r, lora_alpha, max_len, batch_size, grad_accum, max_steps
    )

    if host is None or dry_run:
        # Local: needs `uv sync --extra train` (torch, transformers, peft, ...) and, for
        # anything but --dry-run or a tiny model, a CUDA GPU.
        argv = [sys.executable, str(remote.SFT_SCRIPT), "--data", str(d / "train.jsonl")]
        if (d / "val.jsonl").exists() and (d / "val.jsonl").stat().st_size:
            argv += ["--val", str(d / "val.jsonl")]
        argv += ["--base", base, "--out", str(config.workspace_dir(workspace) / "adapters" / name)]
        argv += args.argv() + (["--dry-run"] if dry_run else [])
        console.print("[dim]" + " ".join(argv) + "[/dim]")
        raise typer.Exit(code=subprocess.call(argv))

    box = _box(host, user, port, key)
    try:
        cname = remote.start_training(
            box, d, base=base, name=name, args=args, hf_token=hf_token, replace=replace
        )
    except (ComputeError, ValueError, FileNotFoundError) as e:
        console.print(f"[red]error:[/red] {e}")
        raise typer.Exit(code=1) from e
    console.print(
        f"started [bold]{cname}[/bold] on {box} (first run installs deps and downloads {base})"
    )
    if follow_logs:
        _follow(box, name)
    else:
        console.print(f"follow with: envforge train follow {name} --host {host}")


def _follow(box, name: str) -> None:
    for ev in remote.follow(box, name):
        kind = ev.get("event")
        if kind == "exit":
            code = ev.get("code")
            if code == 0:
                console.print(
                    f"[green]training finished[/green] — fetch with: envforge train fetch {name} --host {box.host}"
                )
            else:
                console.print(
                    f"[red]training exited with code {code}[/red]; see: envforge train logs {name} --host {box.host}"
                )
                raise typer.Exit(code=1)
            return
        console.print(json.dumps(ev))


@train_app.command("follow")
def follow_cmd(
    name: str, host: HostOpt, user: UserOpt = "root", port: PortOpt = 22, key: KeyOpt = None
) -> None:
    """Stream a training job's progress until it finishes."""
    _follow(_box(host, user, port, key), remote.check_name(name))


@train_app.command("logs")
def logs_cmd(name: str, host: HostOpt, user: UserOpt = "root", port: PortOpt = 22, key: KeyOpt = None,
             tail: int = 200) -> None:  # fmt: skip
    """Raw container output (pip install, warnings, tracebacks) for a training job."""
    print(remote.logs(_box(host, user, port, key), remote.check_name(name), tail))


@train_app.command("fetch")
def fetch_cmd(
    name: str,
    host: HostOpt,
    user: UserOpt = "root",
    port: PortOpt = 22,
    key: KeyOpt = None,
    cleanup: Annotated[bool, typer.Option(help="Remove the finished training container.")] = True,
    workspace: WorkspaceOpt = config.DEFAULT_WORKSPACE,
) -> None:
    """Copy a finished adapter to data/workspaces/<ws>/adapters/<name>/ (it also stays on the box for vLLM)."""
    box = _box(host, user, port, key)
    name = remote.check_name(name)
    st = remote.status(box, name)
    if st.running:
        raise typer.BadParameter(f"{name} is still training")
    try:
        path = remote.fetch_adapter(box, name, config.workspace_dir(workspace) / "adapters")
    except ComputeError as e:
        console.print(f"[red]error:[/red] {e}")
        raise typer.Exit(code=1) from e
    if cleanup and st.exists:
        remote.remove_container(box, name)
    manifest = path / "envforge_manifest.json"
    if manifest.exists():
        m = json.loads(manifest.read_text())
        console.print(
            f"base {m.get('base_model')} · {m.get('n_train')} train examples · "
            f"final train loss {m.get('final_train_loss')} · eval loss {m.get('final_eval_loss')}"
        )
    console.print(f"adapter at [bold]{path}[/bold]")
    console.print(f"serve it: envforge compute vllm-up <base> --host {host} --lora {name}")

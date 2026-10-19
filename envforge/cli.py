"""envforge command line.

envforge packs
envforge tasks netops --tier 2 --n 3
envforge run netops --model scripted --episodes 10
envforge run netops --model anthropic:claude-sonnet-4-5 --episodes 30 --tiers 1,2,3
envforge eval data/runs/<run-id>
envforge compare data/runs/<a> data/runs/<b> ...
envforge show data/runs/<run-id> --episode 0
envforge filter data/runs/<run-id> --min-reward 1.0 --format chat -o data/sft/good.jsonl
envforge compute vllm-up Qwen/Qwen2.5-3B-Instruct --host gpu-1      (docs/COMPUTE.md)
envforge train dataset <runs> --name v1 && envforge train sft v1 --base <hf-id> --host gpu-1 (docs/TRAINING.md)
envforge admin reset-password me@example.com                        (web layer)
"""

from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from envforge import config
from envforge.compute.cli import compute_app
from envforge.core.evaluate import Metrics, evaluate_run, filter_trajectories
from envforge.core.registry import discover_packs, get_pack
from envforge.core.runner import run_many
from envforge.core.trajectory import load_run
from envforge.declarative.cli import env_app
from envforge.models import PROVIDER_HELP, load_model
from envforge.train.cli import train_app

app = typer.Typer(
    help="Pluggable RL environments + eval-to-training pipeline for LLM agents.",
    no_args_is_help=True,
)
app.add_typer(compute_app, name="compute")
app.add_typer(train_app, name="train")
app.add_typer(env_app, name="env")
try:  # the admin commands need the web extra (uv sync --extra web)
    from envforge.web.admin_cli import admin_app

    app.add_typer(admin_app, name="admin")
except ImportError:
    pass
console = Console()


def _parse_tiers(s: str | None, default: tuple[int, ...]) -> list[int]:
    if not s:
        return list(default)
    return [int(x) for x in s.split(",") if x.strip()]


@app.command()
def packs() -> None:
    """List available environment packs."""
    t = Table("pack", "kind", "tiers", "max_steps", "description")
    for name, cls in sorted(discover_packs().items()):
        t.add_row(
            name, cls.kind, ",".join(map(str, cls.tiers)), str(cls.max_steps), cls.description
        )
    console.print(t)


@app.command()
def tasks(
    pack: str,
    tier: int = typer.Option(1, help="difficulty tier"),
    n: int = typer.Option(3, help="how many tasks to preview"),
    seed_start: int = typer.Option(0),
    hidden: bool = typer.Option(False, help="also print the hidden fault descriptions"),
) -> None:
    """Preview generated tasks for a pack (no model calls)."""
    cls = get_pack(pack)
    for i in range(n):
        task = cls.generate_task(seed=seed_start + i, tier=tier)
        console.rule(f"[bold]{task.id}")
        console.print(task.goal)
        if hidden:
            console.print("[dim]hidden:[/dim]", json.dumps(task.hidden, indent=2))


@app.command()
def run(
    pack: str,
    model: str = typer.Option(config.DEFAULT_MODEL, help=PROVIDER_HELP),
    episodes: int = typer.Option(10),
    tiers: str = typer.Option(None, help="comma list, e.g. 1,2,3 (default: all)"),
    seed_start: int = typer.Option(0),
    max_steps: int = typer.Option(None),
    temperature: float = typer.Option(0.0),
    run_id: str = typer.Option(None),
    workspace: str = typer.Option(config.DEFAULT_WORKSPACE, help="namespace for this user's data"),
    runs_dir: Path = typer.Option(None, help="override; default data/workspaces/<workspace>/runs"),
) -> None:
    """Run episodes of a pack with a model and write trajectories to a run directory."""
    cls = get_pack(pack)
    mdl = load_model(model, temperature=temperature)
    tier_list = _parse_tiers(tiers, cls.tiers)
    runs_dir = runs_dir or config.runs_dir(workspace)
    console.print(
        f"[bold]run[/bold] pack={pack} model={mdl.model_id} episodes={episodes} "
        f"tiers={tier_list} workspace={workspace}"
    )

    def progress(traj):
        mark = "[green]✓[/green]" if traj.success else "[red]✗[/red]"
        extra = ""
        if traj.verdict and traj.verdict.details.get("collateral"):
            extra = f" [yellow]collateral x{len(traj.verdict.details['collateral'])}[/yellow]"
        if traj.error:
            extra += f" [red]{traj.error}[/red]"
        console.print(
            f"  {mark} ep{traj.episode:>3} tier{traj.task.tier} steps={len(traj.steps):>2} "
            f"reward={traj.reward:.2f} {traj.finished_reason}{extra}"
        )

    out = run_many(
        cls,
        mdl,
        runs_dir=runs_dir,
        episodes=episodes,
        tiers=tier_list,
        seed_start=seed_start,
        max_steps=max_steps,
        run_id=run_id,
        progress=progress,
    )
    console.print(f"\nwrote [bold]{out}[/bold]")
    _print_report(evaluate_run(out))


def _metrics_row(label: str, m: Metrics) -> list[str]:
    return [
        label,
        str(m.n),
        f"{m.solve_rate:.0%}",
        f"{m.mean_reward:.2f}",
        f"{m.mean_steps:.1f}",
        f"{m.mean_tool_calls:.1f}",
        f"{m.tool_error_rate:.0%}",
        f"{m.collateral_rate:.0%}",
        f"{m.max_steps_rate:.0%}",
        f"{m.input_tokens + m.output_tokens:,}",
        _cost(m.cost_usd),
    ]


def _cost(usd: float | None) -> str:
    if usd is None:
        return "?"
    return "$0" if usd == 0 else f"${usd:,.4f}" if usd < 1 else f"${usd:,.2f}"


def _print_report(report) -> None:
    t = Table(
        "",
        "n",
        "solve",
        "reward",
        "steps",
        "calls",
        "tool err",
        "collateral",
        "budget out",
        "tokens",
        "cost",
        title=f"{report.meta.get('pack', '?')} · {report.meta.get('model', '?')}",
    )
    t.add_row(*_metrics_row("overall", report.overall), style="bold")
    for tier, m in report.by_tier.items():
        t.add_row(*_metrics_row(f"tier {tier}", m))
    console.print(t)


@app.command("eval")
def eval_cmd(run_dir: Path, as_json: bool = typer.Option(False, "--json")) -> None:
    """Compute metrics for a run directory."""
    report = evaluate_run(run_dir)
    if as_json:
        print(report.model_dump_json(indent=2))
    else:
        _print_report(report)


@app.command()
def compare(run_dirs: list[Path]) -> None:
    """Side-by-side metrics for several runs (e.g. different models or before/after training)."""
    t = Table("run", "model", "n", "solve", "reward", "steps", "collateral", "tokens", "cost")
    for rd in run_dirs:
        r = evaluate_run(rd)
        m = r.overall
        t.add_row(
            rd.name[:28],
            str(r.meta.get("model", "?")),
            str(m.n),
            f"{m.solve_rate:.0%}",
            f"{m.mean_reward:.2f}",
            f"{m.mean_steps:.1f}",
            f"{m.collateral_rate:.0%}",
            f"{m.input_tokens + m.output_tokens:,}",
            _cost(m.cost_usd),
        )
    console.print(t)


@app.command()
def show(run_dir: Path, episode: int = typer.Option(0), hidden: bool = typer.Option(True)) -> None:
    """Print one trajectory as a readable transcript."""
    trajs = load_run(run_dir)
    if episode >= len(trajs):
        raise typer.BadParameter(f"run has {len(trajs)} episodes")
    tj = trajs[episode]
    console.rule(
        f"[bold]{tj.task.id} · {tj.model} · {'SUCCESS' if tj.success else 'FAIL'} · reward {tj.reward}"
    )
    console.print("[bold]GOAL[/bold]", tj.task.goal)
    if hidden:
        console.print("[dim]hidden faults:[/dim]", tj.task.hidden.get("faults"))
    for s in tj.steps:
        if s.assistant_text:
            console.print(f"\n[cyan]assistant:[/cyan] {s.assistant_text}")
        for call, obs, err in zip(s.tool_calls, s.observations, s.errors, strict=False):
            args = ", ".join(f"{k}={v}" for k, v in call.args.items())
            console.print(f"[magenta]→ {call.name}({args})[/magenta]")
            console.print(("[red]" if err else "[dim]") + obs + ("[/red]" if err else "[/dim]"))
    if tj.verdict:
        console.print("\n[bold]VERDICT[/bold]", json.dumps(tj.verdict.details, indent=2))
    console.print(
        f"finished: {tj.finished_reason}  tokens: {tj.usage.input_tokens}+{tj.usage.output_tokens}  {tj.wall_time_s:.1f}s"
    )


@app.command("filter")
def filter_cmd(
    run_dirs: list[Path],
    min_reward: float = typer.Option(1.0),
    workspace: str = typer.Option(config.DEFAULT_WORKSPACE),
    output: Path = typer.Option(
        None, "-o", "--output", help="default data/workspaces/<ws>/sft/filtered.jsonl"
    ),
    fmt: str = typer.Option(
        "trajectory", "--format", help="trajectory (lossless) | chat (OpenAI messages, for SFT)"
    ),
) -> None:
    """Rejection sampling: keep high-reward trajectories from one or more runs into one JSONL."""
    from envforge.train.filter import dataset_lines, tool_schemas

    if fmt not in ("trajectory", "chat"):
        raise typer.BadParameter("--format must be trajectory or chat")
    output = output or (config.sft_dir(workspace) / "filtered.jsonl")
    kept = []
    total = 0
    packs = set()
    for rd in run_dirs:
        trajs = load_run(rd)
        total += len(trajs)
        kept.extend(filter_trajectories(trajs, min_reward=min_reward))
        packs.add(evaluate_run(rd).meta.get("pack"))
    tools = tool_schemas(get_pack(packs.pop())) if fmt == "chat" and len(packs) == 1 else None
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        f.writelines(dataset_lines(kept, fmt=fmt, tools=tools))  # type: ignore[arg-type]
    console.print(f"kept {len(kept)}/{total} trajectories -> {output} ({fmt})")


if __name__ == "__main__":
    app()

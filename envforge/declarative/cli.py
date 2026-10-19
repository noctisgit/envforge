"""`envforge env ...`: inspect declarative environment directories from the terminal.

    envforge env validate envforge/declarative/examples/isp_support
    envforge env preview  envforge/declarative/examples/isp_support --seed 3 --tier 2

(`env_app` is mounted on the main CLI in envforge/cli.py.)
"""

from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from envforge.declarative.store import is_env_dir, load_spec_and_tasks
from envforge.declarative.validate import validate_dataset

env_app = typer.Typer(
    help="Declarative environments (spec.json + tasks.jsonl).", no_args_is_help=True
)
console = Console()


def _require_dir(path: Path) -> None:
    if not is_env_dir(path):
        console.print(f"[red]{path} is not an environment directory (no spec.json)[/red]")
        raise typer.Exit(2)


@env_app.command()
def validate(path: Path = typer.Argument(..., help="environment directory")) -> None:
    """Check the spec and every task (each reference solution must reach reward 1.0)."""
    _require_dir(path)
    spec, tasks = load_spec_and_tasks(path)
    summary = validate_dataset(spec, tasks)
    console.print(f"[bold]{spec.title}[/bold] ({spec.name}) — {spec.description}")
    console.print(
        f"{len(spec.collections)} collections, {len(spec.tools)} tools, "
        f"{len(spec.policies)} policies, {len(spec.resolution_categories)} resolution categories"
    )
    if summary["spec_problems"]:
        console.print("[red]spec problems:[/red]")
        for p in summary["spec_problems"]:
            console.print(f"  - {p}")
    else:
        console.print("[green]spec: ok[/green]")
    for task_id, problems in summary["problems"].items():
        console.print(f"[red]task {task_id}:[/red]")
        for p in problems:
            console.print(f"  - {p}")
    if summary["duplicate_ids"]:
        console.print(f"[red]duplicate task ids: {summary['duplicate_ids']}[/red]")
    tiers = ", ".join(f"tier {t}: {n}" for t, n in summary["tiers"].items())
    colour = "green" if summary["ok"] == summary["n_tasks"] else "yellow"
    console.print(f"[{colour}]{summary['ok']}/{summary['n_tasks']} tasks ok[/{colour}] ({tiers})")
    if summary["spec_problems"] or summary["problems"] or summary["duplicate_ids"]:
        raise typer.Exit(1)


@env_app.command()
def preview(
    path: Path = typer.Argument(..., help="environment directory"),
    seed: int = typer.Option(0, help="task seed"),
    tier: int = typer.Option(1, help="difficulty tier"),
) -> None:
    """Show the task for (seed, tier): goal, hidden data, and the reference solution's result."""
    from envforge.core.explore import preview_task
    from envforge.declarative.store import load_env_dir

    _require_dir(path)
    env_cls = load_env_dir(path)
    p = preview_task(env_cls, seed=seed, tier=tier)
    task = p["task"]
    console.print(f"[bold]{task['id']}[/bold] (tier {task['tier']})")
    console.print(f"[bold]goal:[/bold] {task['goal']}")
    console.print("[bold]hidden:[/bold]")
    console.print(json.dumps(task["hidden"], indent=2, ensure_ascii=False))
    console.print("[bold]first observation:[/bold]")
    console.print(p["first_observation"], markup=False)
    t = Table("#", "call", "observation (first line)")
    for i, step in enumerate(p["solution"], start=1):
        args = ", ".join(f"{k}={v!r}" for k, v in step["args"].items())
        first = (step["observation"] or "").splitlines()[0] if step["observation"] else ""
        t.add_row(str(i), f"{step['name']}({args})", first)
    console.print(t)
    v = p["verdict"]
    colour = "green" if v["reward"] == 1.0 else "red"
    console.print(f"[{colour}]reference: success={v['success']} reward={v['reward']}[/{colour}]")
    for c in v["details"]["success_checks"]:
        mark = "✓" if c["passed"] else "✗"
        console.print(f"  {mark} {c['check']}" + (f" ({c['detail']})" if c["detail"] else ""))
    for c in v["details"]["collateral"]:
        console.print(f"  [red]{c}[/red]")

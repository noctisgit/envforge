"""Environment directories on disk: `<dir>/spec.json` + `<dir>/tasks.jsonl`.

    save_env_dir("envs/isp", spec, tasks)
    env_cls = load_env_dir("envs/isp")     # -> an Environment subclass

`get_pack()` in envforge/core/registry.py accepts such a directory path anywhere a
pack name is expected, e.g. `envforge run envs/isp --model scripted`.
"""

from __future__ import annotations

import json
from pathlib import Path

from envforge.core.env import Environment
from envforge.declarative.runtime import make_environment
from envforge.declarative.spec import EnvSpec, TaskRecord

SPEC_FILE = "spec.json"
TASKS_FILE = "tasks.jsonl"


def is_env_dir(path: str | Path) -> bool:
    """True if `path` is a directory holding a spec.json."""
    p = Path(path)
    return p.is_dir() and (p / SPEC_FILE).is_file()


def load_spec_and_tasks(path: str | Path) -> tuple[EnvSpec, list[TaskRecord]]:
    """Read and validate (as pydantic models) the spec and every task record."""
    p = Path(path)
    spec = EnvSpec.model_validate_json((p / SPEC_FILE).read_text(encoding="utf-8"))
    tasks: list[TaskRecord] = []
    tasks_path = p / TASKS_FILE
    if tasks_path.exists():
        for n, line in enumerate(tasks_path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            try:
                tasks.append(TaskRecord.model_validate_json(line))
            except ValueError as e:
                raise ValueError(f"{tasks_path}:{n}: invalid task record: {e}") from e
    return spec, tasks


def load_env_dir(path: str | Path, *, name: str | None = None) -> type[Environment]:
    """Build the Environment class for an environment directory."""
    spec, tasks = load_spec_and_tasks(path)
    return make_environment(spec, tasks, name=name)


def save_env_dir(path: str | Path, spec: EnvSpec, tasks: list[TaskRecord]) -> Path:
    """Write spec.json (pretty) and tasks.jsonl (one record per line); returns the dir."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    (p / SPEC_FILE).write_text(spec.model_dump_json(indent=2) + "\n", encoding="utf-8")
    lines = [json.dumps(t.model_dump(mode="json"), ensure_ascii=False) for t in tasks]
    (p / TASKS_FILE).write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return p

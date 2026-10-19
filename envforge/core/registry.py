"""Pack discovery.

A pack is a subpackage of `envforge.packs` whose `__init__.py` exposes a module
attribute `PACK` pointing at an `Environment` subclass. Third-party packs can
also register via the `envforge.packs` entry-point group in their pyproject.

`get_pack()` also accepts the path of a declarative environment directory
(`<dir>/spec.json` + `<dir>/tasks.jsonl`, see envforge/declarative/store.py).
"""

from __future__ import annotations

import importlib
import pkgutil
from importlib.metadata import entry_points
from pathlib import Path

from envforge.core.env import Environment


def discover_packs() -> dict[str, type[Environment]]:
    packs: dict[str, type[Environment]] = {}

    # built-in packs
    import envforge.packs as pkgs

    for mod in pkgutil.iter_modules(pkgs.__path__):
        if not mod.ispkg:
            continue
        module = importlib.import_module(f"envforge.packs.{mod.name}")
        cls = getattr(module, "PACK", None)
        if cls is not None and issubclass(cls, Environment):
            packs[cls.name] = cls

    # third-party packs via entry points
    for ep in entry_points(group="envforge.packs"):
        cls = ep.load()
        if issubclass(cls, Environment):
            packs[cls.name] = cls

    return packs


def get_pack(name: str) -> type[Environment]:
    """A pack by name, or a declarative environment by directory path."""
    if Path(name).is_dir() and (Path(name) / "spec.json").is_file():
        # imported lazily: declarative is generic infrastructure, not a pack
        from envforge.declarative.store import load_env_dir

        return load_env_dir(name)
    packs = discover_packs()
    if name not in packs:
        raise KeyError(f"Unknown pack '{name}'. Available: {sorted(packs)}")
    return packs[name]

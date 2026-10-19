"""repofix self-description: describe() and the file-tree diagram, with no sandbox.

Previews of sandbox packs never start a container, so everything here runs with
ENVFORGE_SANDBOX unset and `open_sandbox` booby-trapped: if anything tried to open a
sandbox (Docker or local), the test would fail loudly.
"""

import pytest

from envforge.core.env import Diagram, EnvDescription
from envforge.core.explore import pack_overview, preview_task
from envforge.core.registry import get_pack
from envforge.packs.repofix import PACK
from envforge.packs.repofix import env as repofix_env
from envforge.packs.repofix.corpus import get_project
from envforge.packs.repofix.mutations import CATALOG, by_id, enclosing_function

SEEDS = range(6)


@pytest.fixture(autouse=True)
def no_sandbox(monkeypatch):
    """Unset the backend and make opening any sandbox an error."""
    monkeypatch.delenv("ENVFORGE_SANDBOX", raising=False)

    def refuse(*args, **kwargs):
        raise AssertionError("a sandbox was opened during a no-container preview")

    monkeypatch.setattr(repofix_env, "open_sandbox", refuse)


def test_describe_validates_and_every_tool_has_a_category():
    desc = PACK.describe()
    EnvDescription.model_validate(desc.model_dump())
    assert desc.kind == "sandbox"
    assert {t.name: t.category for t in desc.tools} == {
        "list_files": "observe",
        "read_file": "observe",
        "search": "observe",
        "run_tests": "observe",
        "write_file": "act",
        "replace_in_file": "act",
    }
    assert sorted(r.reward for r in desc.reward_rules) == [0.0, 0.3, 1.0]
    assert set(desc.tier_notes) == set(PACK.tiers)
    titles = [t.title for t in desc.tables]
    assert titles == ["Corpus", "Mutation catalogue", "Anti-cheating rules"]
    for table in desc.tables:
        assert all(len(row) == len(table.columns) for row in table.rows)
    catalogue = desc.tables[1]
    assert sum(int(row[2]) for row in catalogue.rows) == len(CATALOG)


def test_every_mutation_site_is_inside_a_function():
    for m in CATALOG:
        assert enclosing_function(m), m.id


@pytest.mark.parametrize("tier", PACK.tiers)
def test_diagram_is_a_tree_without_reset(tier):
    for seed in SEEDS:
        task = PACK.generate_task(seed, tier)
        d = PACK(task).diagram()  # no reset(): no sandbox exists
        assert isinstance(d, Diagram) and d.layout == "tree"
        assert d.edges == []
        ids = [n.id for n in d.nodes]
        assert len(ids) == len(set(ids))
        roots = [n for n in d.nodes if n.parent is None]
        assert len(roots) == 1 and roots[0].kind == "dir"
        by = {n.id: n for n in d.nodes}
        for n in d.nodes:
            if n.parent is not None:
                assert by[n.parent].kind == "dir", n.id
        assert {n.kind for n in d.nodes} <= {"dir", "module", "test", "hidden-test", "file"}

        # every injected bug's file is broken and badged; nothing else is broken
        mutated = {by_id(i).path for i in task.hidden["mutations"]}
        assert {n.id for n in d.nodes if n.state == "broken"} == mutated
        for path in mutated:
            assert "mutated" in by[path].badges
            for i in task.hidden["mutations"]:
                if by_id(i).path == path:
                    assert f"{enclosing_function(by_id(i))}()" in by[path].detail

        # visible tests carry "test", hidden tests "hidden", decoy files "decoy"
        project = get_project(task.hidden["project"])
        for path in project.test_files():
            assert by[path].kind == "test" and "test" in by[path].badges
        hidden = [n for n in d.nodes if n.kind == "hidden-test"]
        assert len(hidden) == 2 and all("hidden" in n.badges for n in hidden)
        for entry in task.hidden["protected"]:
            path, fn = entry.split(":", 1)
            assert "decoy" in by[path].badges and f"{fn}()" in by[path].detail
        # every visible repo file is drawn
        assert set(project.repo_files()) <= set(ids)


def test_diagram_is_deterministic():
    a = PACK(PACK.generate_task(3, 3)).diagram()
    b = PACK(PACK.generate_task(3, 3)).diagram()
    assert a == b


@pytest.mark.parametrize("tier", PACK.tiers)
def test_preview_task_works_without_docker(tier):
    pack = get_pack("repofix")
    for seed in SEEDS:
        p = preview_task(pack, seed, tier)
        assert p["diagram_before"]["layout"] == "tree"
        assert p["first_observation"] is None  # sandbox packs are not reset in previews
        names = [s["name"] for s in p["solution"]]
        assert names == ["replace_in_file"] * tier + ["run_tests"]
        assert p["verdict"] is None


def test_pack_overview_works_without_docker():
    o = pack_overview(get_pack("repofix"))
    assert o["world_diagram"]["layout"] == "tree"
    assert o["replayable"] is False
    assert any("4 observe, 2 act" in row[1] for row in o["dimensions"])

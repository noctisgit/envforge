"""repofix pack tests, run offline on the local sandbox backend.

Same checklist as netops (determinism, broken-initially, reference passes, partial
fails, collateral penalised) plus the sandbox-specific ones: the mutation catalog is
sound, hidden tests catch overfit fixes, and tool paths can't escape the repo.
"""

from itertools import combinations

import pytest

from envforge.core.env import ToolCall, ToolError
from envforge.core.registry import get_pack
from envforge.core.runner import run_episode
from envforge.models import load_model
from envforge.packs.repofix import PACK
from envforge.packs.repofix import env as repofix_env
from envforge.packs.repofix.corpus import HIDDEN_API_PATH, HIDDEN_BEHAVIOUR_PATH, PROJECTS
from envforge.packs.repofix.mutations import CATALOG, apply_mutations, generate, mutations_for
from envforge.sandbox import open_sandbox

SEEDS = range(8)


@pytest.fixture(autouse=True)
def local_sandbox(monkeypatch):
    """Use the temp-dir backend: no Docker needed, and fast."""
    monkeypatch.setenv("ENVFORGE_SANDBOX", "local")


def _call(name: str, **args) -> ToolCall:
    return ToolCall(id=f"t_{name}", name=name, args=args)


def _solved_env(seed: int, tier: int):
    env = PACK(PACK.generate_task(seed, tier))
    env.reset()
    for call in env.reference_solve():
        env.step(call)
    return env


def _task_with(mutation_id: str, tier: int = 1):
    """First seed whose task contains `mutation_id`."""
    for seed in range(500):
        task = PACK.generate_task(seed, tier)
        if mutation_id in task.hidden["mutations"]:
            return task
    raise AssertionError(f"no seed found for {mutation_id}")


# ---- generation and the catalog ---------------------------------------------


def test_registry_finds_repofix_as_sandbox_pack():
    assert get_pack("repofix") is PACK
    assert PACK.kind == "sandbox"


def test_task_generation_is_deterministic():
    assert PACK.generate_task(seed=7, tier=3) == PACK.generate_task(seed=7, tier=3)
    project_a, muts_a = generate(7, 3)
    project_b, muts_b = generate(7, 3)
    assert project_a == project_b and muts_a == muts_b
    files = project_a.repo_files()
    assert apply_mutations(files, muts_a) == apply_mutations(files, muts_b)


def test_tiers_have_the_promised_shape():
    for seed in range(60):
        _, t1 = generate(seed, 1)
        _, t2 = generate(seed, 2)
        _, t3 = generate(seed, 3)
        assert len(t1) == 1
        assert len(t2) == 2 and t2[0].path != t2[1].path
        assert len(t3) == 3 and len({m.id for m in t3}) == 3
        assert any(m.decoy_twin for m in t3) and len({m.path for m in t3}) >= 2


def test_catalog_snippets_combine_and_revert_cleanly():
    """Any subset of a file's mutations can be applied together and swapped back exactly."""
    for project in PROJECTS:
        pristine = project.repo_files()
        pool = mutations_for(project.name)
        assert pool, project.name
        for path in {m.path for m in pool}:
            in_file = [m for m in pool if m.path == path]
            for r in range(1, len(in_file) + 1):
                for subset in combinations(in_file, r):
                    broken = apply_mutations(pristine, list(subset))[path]
                    for m in subset:
                        assert broken.count(m.after) == 1, (m.id, [s.id for s in subset])
                    for m in subset:
                        broken = broken.replace(m.after, m.before)
                    assert broken == pristine[path]


def test_pristine_projects_pass_all_their_tests():
    for project in PROJECTS:
        with open_sandbox(files={**project.repo_files(), **project.hidden_files()}) as sb:
            res = sb.exec(["python", "-m", "unittest"])
            assert res.ok, f"{project.name}\n{res.output}"
            for path in (HIDDEN_BEHAVIOUR_PATH, HIDDEN_API_PATH):
                assert path in sb.list_files()


def test_every_mutation_breaks_visible_and_hidden_tests():
    """Tier 1 promise: a visible test points at the bug. And hidden tests catch it too."""
    for m in CATALOG:
        project = next(p for p in PROJECTS if p.name == m.project)
        files = apply_mutations(project.repo_files(), [m])
        with open_sandbox(files={**files, **project.hidden_files()}) as sb:
            visible = [p for p in project.test_files() if "/test_" in p]
            mods = [p.removesuffix(".py").replace("/", ".") for p in visible]
            assert not sb.exec(["python", "-m", "unittest", *mods]).ok, m.id
            assert not sb.exec(["python", "-m", "unittest", "tests.test_hidden_behaviour"]).ok, m.id
            # decoy tests must not depend on any mutation site, or an unfinished fix
            # would be misreported as collateral
            assert sb.exec(["python", "-m", "unittest", "tests.test_hidden_api"]).ok, m.id


# ---- the episode contract ---------------------------------------------------


def test_task_is_broken_initially_and_not_collateral():
    for seed in SEEDS:
        for tier in PACK.tiers:
            env = PACK(PACK.generate_task(seed, tier))
            env.reset()
            v = env.verify()
            env.close()
            assert not v.success and v.reward == 0.0, f"seed={seed} tier={tier} not broken"
            assert v.details["collateral"] == [], v.details


def test_initial_observation_shows_failing_tests():
    env = PACK(PACK.generate_task(0, 1))
    obs = env.reset()
    env.close()
    assert "Repository files:" in obs.text and "FAILED" in obs.text


@pytest.mark.parametrize("tier", PACK.tiers)
def test_reference_solution_passes_cleanly(tier):
    for seed in SEEDS:
        env = _solved_env(seed, tier)
        v = env.verify()
        env.close()
        assert v.success and v.reward == 1.0, f"seed={seed}: {v.details}"
        assert v.details["tests_ran"] == v.details["tests_expected"] > 0


def test_partial_fix_fails():
    env = PACK(PACK.generate_task(seed=3, tier=3))
    env.reset()
    fixes = [c for c in env.reference_solve() if c.name == "replace_in_file"]
    for call in fixes[:-1]:
        env.step(call)
    v = env.verify()
    env.close()
    assert not v.success and v.reward == 0.0
    assert v.details["failing_tests"]


def test_verify_is_repeatable_and_leaves_the_repo_alone():
    env = _solved_env(1, 2)
    files_before = env.sandbox.list_files()
    assert env.verify().success and env.verify().success
    assert env.sandbox.list_files() == files_before  # hidden tests never land here
    env.close()


# ---- collateral and reward hacking ------------------------------------------


def test_editing_tests_is_penalised():
    env = _solved_env(2, 1)
    env.step(_call("write_file", path="tests/__init__.py", content="# touched\n"))
    v = env.verify()
    env.close()
    assert not v.success and v.reward == 0.3
    assert any("modified test file" in c for c in v.details["collateral"])


def test_gutting_tests_instead_of_fixing_scores_zero():
    env = PACK(PACK.generate_task(4, 1))
    env.reset()
    always_green = (
        "import unittest\n\n\nclass T(unittest.TestCase):\n    def test_ok(self):\n        pass\n"
    )
    for path in env.sandbox.list_files():
        if path.startswith("tests/test_"):
            env.step(_call("write_file", path=path, content=always_green))
    assert "OK" in env.step(_call("run_tests")).text  # looks green to the agent...
    v = env.verify()  # ...but the verifier restores the real tests
    env.close()
    assert not v.success and v.reward == 0.0 and v.details["collateral"]


def test_special_casing_the_visible_test_fails_hidden_tests():
    task = _task_with("iv-quarter")
    env = PACK(task)
    env.reset()
    env.step(
        _call(
            "replace_in_file",
            path="intervals/dates.py",
            old="return d.month // 3 + 1",
            new="return 1 if d.month == 3 else d.month // 3 + 1",
        )
    )
    assert "OK" in env.step(_call("run_tests")).text  # visible suite is green
    v = env.verify()
    env.close()
    assert not v.success and v.reward == 0.0
    assert any("hidden" in t for t in v.details["failing_tests"])


def test_deleting_a_decoy_is_collateral():
    task = _task_with("ts-median-div", tier=3)
    env = _solved_env(task.seed, 3)
    source = env.sandbox.read_file("textstats/stats.py")
    gutted = source[: source.index("def median_low")]
    env.step(_call("write_file", path="textstats/stats.py", content=gutted))
    v = env.verify()
    env.close()
    assert not v.success and v.reward == 0.3
    assert any("removed public function median_low" in c for c in v.details["collateral"])


def test_fixing_a_decoy_that_was_not_broken_is_collateral():
    task = _task_with("iv-overlaps", tier=3)
    env = _solved_env(task.seed, 3)
    env.step(
        _call(
            "replace_in_file",
            path="intervals/ranges.py",
            old="return a.start <= b.end and b.start <= a.end",
            new="return a.start < b.end and b.start < a.end",
        )
    )
    v = env.verify()
    env.close()
    assert not v.success and v.reward == 0.3
    assert any("broke public API" in c for c in v.details["collateral"])


def test_shadowing_the_stdlib_is_tampering():
    env = _solved_env(0, 1)
    env.step(_call("write_file", path="unittest.py", content="print('Ran 99 tests in 0s')\n"))
    v = env.verify()
    env.close()
    assert not v.success and v.reward == 0.0
    assert any("shadows the standard-library" in c for c in v.details["collateral"])


# ---- tools ------------------------------------------------------------------


def test_path_traversal_is_rejected():
    env = PACK(PACK.generate_task(0, 1))
    env.reset()
    bad_paths = ["../outside.py", "/etc/passwd", "pkg/../../x.py", "a\\..\\b.py", ""]
    for path in bad_paths:
        with pytest.raises(ToolError):
            env.step(_call("read_file", path=path))
        with pytest.raises(ToolError):
            env.step(_call("write_file", path=path, content="x"))
        with pytest.raises(ToolError):
            env.step(_call("replace_in_file", path=path, old="a", new="b"))
    with pytest.raises(ToolError):
        env.step(_call("run_tests", test="tests; rm -rf /"))
    env.close()


def test_tool_errors_are_recoverable():
    env = PACK(PACK.generate_task(0, 1))
    env.reset()
    readme = "README.md"
    with pytest.raises(ToolError):
        env.step(_call("read_file", path="nope.py"))
    with pytest.raises(ToolError):
        env.step(_call("read_file", path=readme, start_line=5, end_line=2))
    with pytest.raises(ToolError):
        env.step(_call("replace_in_file", path=readme, old="no such text", new="x"))
    with pytest.raises(ToolError):
        env.step(_call("replace_in_file", path=readme, old="t", new="x"))  # many matches
    with pytest.raises(ToolError):
        env.step(_call("search", pattern="("))
    with pytest.raises(ToolError):
        env.step(_call("read_file"))  # missing argument
    # still usable afterwards
    assert "README.md" in env.step(_call("list_files")).text
    assert "unittest" in env.step(_call("search", pattern="python -m unittest")).text
    env.close()


def test_tool_specs_are_valid_json_schema_shapes():
    env = PACK(PACK.generate_task(seed=0, tier=1))
    names = set()
    for t in env.tools():
        assert t.name not in names
        names.add(t.name)
        assert t.parameters["type"] == "object"
        assert t.description
    assert names == {
        "list_files",
        "read_file",
        "search",
        "write_file",
        "replace_in_file",
        "run_tests",
    }


# ---- lifecycle --------------------------------------------------------------


def test_runner_closes_every_sandbox(monkeypatch):
    opened = []

    def tracking_open_sandbox(**kwargs):
        sb = open_sandbox(**kwargs)
        opened.append(sb)
        return sb

    monkeypatch.setattr(repofix_env, "open_sandbox", tracking_open_sandbox)
    traj = run_episode(PACK, PACK.generate_task(6, 3), load_model("scripted"))
    assert traj.success and traj.reward == 1.0
    assert len(opened) == 2  # the agent's sandbox + the verifier's fresh one
    assert all(sb._root is None for sb in opened)  # LocalSandbox clears it on close()


def test_reset_twice_does_not_leak():
    env = PACK(PACK.generate_task(0, 1))
    env.reset()
    first = env.sandbox
    env.reset()
    assert first._root is None and env.sandbox is not first
    env.close()
    env.close()  # idempotent

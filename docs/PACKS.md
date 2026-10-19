# Writing a pack

A pack is a subpackage of `envforge/packs/` that exports `PACK`, an `Environment` subclass.
The core discovers it automatically (`envforge packs`). Third-party packs can instead
register an entry point in the `envforge.packs` group.

## Checklist

- [ ] `envforge/packs/<name>/__init__.py` exports `PACK`
- [ ] class attributes: `name`, `kind` (`"sim"` | `"sandbox"`), `max_steps`, `tiers`, `description`
- [ ] `generate_task(seed, tier)` is deterministic and returns a `Task` with a `goal` the agent sees and `hidden` metadata it doesn't
- [ ] `reset()` rebuilds state from `self.task` (regenerate from the seed — don't try to serialise your world into the Task)
- [ ] `tools()` returns `ToolSpec`s with JSON-Schema `parameters`
- [ ] `step(call)` applies one tool; raises `ToolError` on misuse (never a bare exception)
- [ ] `verify()` is pure code; returns `Verdict(success, reward, details)`; put collateral findings in `details["collateral"]` (a list) so the evaluator counts them
- [ ] `system_prompt()` describes the role and rules; keep tool semantics in the tool descriptions
- [ ] `reference_solve()` returns tool calls that make `verify()` pass — the tests and the scripted model depend on it
- [ ] optional but recommended: `category="observe"|"act"` on each `ToolSpec`, `describe()` and `diagram()` (below) so the Environments page can explain the pack
- [ ] `tests/test_<name>_pack.py`: determinism, broken-initially, reference-passes, partial-fails, collateral-penalised
- [ ] `envforge run <name> --model scripted --episodes 100` is 100% / 1.00 reward with 0% collateral

## Template

```python
# envforge/packs/mydomain/__init__.py
from envforge.packs.mydomain.env import MyEnvironment

PACK = MyEnvironment
```

```python
# envforge/packs/mydomain/env.py
from __future__ import annotations
import random
from typing import ClassVar
from envforge.core.env import Environment, Observation, Task, ToolCall, ToolError, ToolSpec, Verdict


class MyEnvironment(Environment):
    name: ClassVar[str] = "mydomain"
    kind: ClassVar[str] = "sim"
    max_steps: ClassVar[int] = 20
    tiers: ClassVar[tuple[int, ...]] = (1, 2)
    description: ClassVar[str] = "One line for `envforge packs`."

    @classmethod
    def generate_task(cls, seed: int, tier: int) -> Task:
        rng = random.Random(seed)
        # build the problem deterministically from rng and tier
        return Task(
            id=f"{cls.name}-t{tier}-s{seed}",
            pack=cls.name,
            tier=tier,
            seed=seed,
            goal="What the agent is asked to do.",
            hidden={"answer": "..."},
        )

    def reset(self) -> Observation:
        rng = random.Random(self.task.seed)
        self.state = {...}  # rebuild from seed
        return Observation(text="initial view of the world")

    def tools(self) -> list[ToolSpec]:
        return [
            ToolSpec(
                name="look", description="...", parameters={"type": "object", "properties": {}}
            )
        ]

    def step(self, call: ToolCall) -> Observation:
        if call.name == "look":
            return Observation(text=str(self.state))
        raise ToolError(f"unknown tool {call.name}")

    def verify(self) -> Verdict:
        ok = ...
        collateral = [...]
        return Verdict(
            success=ok and not collateral,
            reward=1.0 if ok and not collateral else (0.3 if ok else 0.0),
            details={"collateral": collateral},
        )

    def system_prompt(self) -> str:
        return "You are ..."

    def reference_solve(self) -> list[ToolCall]:
        return [ToolCall(id="ref1", name="...", args={...})]
```

## Showing up in the Environments page

Every pack appears in the web UI's Environments explorer with no extra code: the default
`describe()` lists its tools, tiers and budget, and `tests/test_explore.py` checks every
pack. To make it *explain* your domain, implement two optional hooks (see
`envforge/core/env.py` and the netops/sipops/repofix packs for examples):

- `describe()` — start from `super().describe()` and `model_copy(update={...})`: an
  `overview`, `world` facts, `observation` / `action` sentences, `tier_notes`,
  `reward_rules`, and any `tables` (`InfoTable`: fault catalogues, policies, corpora…).
- `diagram()` — the world as a `Diagram` of `DiagramNode`s and `DiagramEdge`s:
  `layout="graph"` (give `x`/`y` in [0, 1] for a hand-made layout, or leave them out for
  automatic layout; `group` draws zones) or `layout="tree"` (set `parent`, e.g. a file
  system). Use the standard states so the UI colours them: nodes `ok|broken|target|inactive`,
  edges `ok|broken|path|inactive`; add free-form `badges` and a `detail` tooltip, and
  `notes` for one-line captions. Sim packs: reflect *live* state — it is redrawn after every
  replayed tool call. Sandbox packs: build it from the task alone (no `reset()`).

Task previews and replays are generic (`envforge/core/explore.py`): they regenerate the
task from its seed and re-apply tool calls, which is why determinism matters.

## Design advice

**Make the reward verifiable.** If success can't be decided by code from the end state,
pick a narrower task. "Call completes end-to-end" is verifiable; "the agent was helpful" is not.

**Punish collateral.** The most useful behaviour to train is *restraint*. Give the verifier a
list of things that must still work and things that must stay blocked, and dock reward when
they change.

**Put decoys in from tier 2.** Something that looks like the problem but is protected
policy. Agents that "delete until it works" should score badly.

**Observations are text an operator would recognise.** Format tool output the way the real
tool would. Models have seen a lot of `show ip route`; they have not seen your JSON.

**Keep `hidden` honest.** It is for debugging and `envforge show`; never leak it into
`goal` or an observation.

**Tiers should be qualitatively different**, not just "more faults": one fault with an obvious
symptom → two faults where one masks the other → faults on the return path plus decoys.

**Sandbox packs** follow the same contract; `reset()` starts a container, `step()` execs
commands, `close()` tears it down. Keep the Docker image and any fixtures inside the pack
directory. Phase 2 adds `envforge/sandbox/` helpers for this.

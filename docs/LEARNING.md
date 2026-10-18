# LEARNING.md — a guided tour of the code and the ML behind it

This file is for the maintainer. It walks the repository in the order the pieces
depend on each other, and for each module says (a) what it does, (b) the ML
concept it implements, and (c) one or two things worth doing by hand to make
the Python stick.

Read the code alongside this. Every module is short on purpose.

---

## 0. The map

```
Task ──► Environment ──► Observation
            ▲   │
   tool call│   │ tool result            one *step*
            │   ▼
          Model (policy)                 a list of steps = one *episode* = one Trajectory
                                         Trajectory + Verdict → reward
```

Vocabulary you will see everywhere in RL papers, translated:

| RL word | Plain word | In this repo |
|---|---|---|
| environment | the world + its rules | `Environment` subclass (a pack) |
| state | everything true about the world right now | the pack's internal objects |
| observation | the slice of state the agent is shown | `Observation.text` |
| action | what the agent does | a `ToolCall` |
| policy | the thing that chooses actions | the model (`ChatModel`) |
| episode / rollout | one attempt at a task, start to finish | `run_episode()` → `Trajectory` |
| reward | a number that says how well it went | `Verdict.reward` from `verify()` |
| return | total reward over the episode | here: just the final reward |
| horizon / budget | max steps per episode | `Environment.max_steps` |

---

## 1. `envforge/core/env.py` — the contract

**Does:** declares the five things a domain must provide (state, tools, task
generator, verifier, instructions) as an abstract base class plus the small
pydantic models that flow between core and pack.

**ML concept:** this *is* the definition of an environment. Gym/Gymnasium does the
same with `reset()` / `step()`; the difference here is that actions are structured
tool calls (what LLMs emit) rather than integers or vectors.

**Python to notice:**
- `ABC` + `@abstractmethod`: a class you can't instantiate until every abstract
  method is implemented. Try: make a subclass missing `verify()` and instantiate it.
- `ClassVar[...]` on `name`, `kind`, `max_steps`: attributes that belong to the
  class, not each instance.
- `pydantic.BaseModel`: data classes that validate on construction and serialise to
  JSON for free (`model_dump_json()` / `model_validate_json()`). That's why
  trajectories round-trip through JSONL without any hand-written parsing.

**Do by hand:** in a Python REPL, build a `ToolSpec`, dump it to JSON, load it back.
Then make an invalid one (`parameters="oops"`) and read the validation error.

---

## 2. `envforge/packs/netops/sim.py` — the world

**Does:** a tiny network: routers with static routes and ACLs, links with up/down
state, hosts, a DNS table. `Network.trace()` walks a packet hop by hop and returns
*why* it stopped. `build_base_network()` constructs a working topology.

**ML concept:** this is a **simulator**. Fast, deterministic, no I/O. Training needs
tens of thousands of episodes; anything that touches real systems can't deliver that.
Note how `trace()` returns a reason string — that string becomes the agent's observation,
so the quality of your simulator's error messages directly limits what the agent can learn.

**Python to notice:**
- `@dataclass` for plain state holders (`Router`, `Link`, `AclRule`).
- the `ipaddress` stdlib module: `ip in network` membership tests and `prefixlen` give you
  longest-prefix-match in a few lines (`Router.lookup`).
- `_shortest_paths()` is a breadth-first search producing next-hop tables — the same
  algorithm as a routing protocol's SPF, in 25 lines.

**Do by hand:** write a function `all_pairs(n: Network) -> dict[tuple[str,str], bool]`
that returns reachability for every host pair. Then break one route and diff the result.

---

## 3. `envforge/packs/netops/faults.py` — the task generator

**Does:** from a `(seed, tier)` builds hosts, picks a target pair, injects one to three
faults (missing route, wrong route, link down, ACL deny, stale DNS, missing return route),
adds decoy "security policy" rules, and records which other paths must keep working.
Every fault also knows its own fix, which is what `reference_solve()` returns.

**ML concept:** **synthetic data / environment pipeline** and **curriculum**.
- *Synthetic* because tasks are generated, not collected. Infinite tasks for free.
- *Seeded* because the same seed must give the same task — otherwise you can't compare two
  models on the same problems, and you can't reproduce a bug.
- *Tiered* because a curriculum (easy → hard) is how you both measure progress and,
  later, schedule training.

**Python to notice:**
- `random.Random(seed)`: a private RNG, so the generator is deterministic no matter what
  else used `random` before it.
- small dataclass subclasses of `Fault` each with `apply()`; the generator composes them.
  Adding a new fault type is adding one class and one `elif`.

**Do by hand:** add a fault type `acl_wrong_order` (a permit rule that sits *below* a
deny so it never matches; the fix is to re-add it at the top). Make the tests pass.

---

## 4. `envforge/packs/netops/env.py` — wiring the pack to the contract

**Does:** declares the tools (`TOOLS`), implements `step()` as a dispatch on the tool
name, and `verify()` as: target reachable both ways **and** no collateral.

**ML concept: reward design.** Read `verify()` slowly. Three things make this reward
trainable:
1. it is computed by code from end state (*verifiable*, can't be argued with);
2. it penalises collateral (0.3 instead of 1.0) so *restraint* is learnable;
3. it checks outcomes not actions, so there is more than one correct solution.

A reward that only said "did you call `set_route R2 …`" would teach memorisation. This one
teaches diagnosis.

**Python to notice:**
- `raise ToolError(...) from e`: exception chaining — the original error stays attached.
- the `try/except KeyError` around the dispatch converts a missing argument into a
  `ToolError` the model can recover from. The runner relies on that: a `ToolError` is an
  observation, any other exception ends the episode.

**Do by hand:** change the collateral reward from 0.3 to 0.0 and think about what
behaviour that would encourage or discourage during training. (Answer: with 0.0, "fix it
but break something" and "didn't fix it" look identical to the learner; partial credit
keeps the gradient pointing somewhere.)

---

## 5. `envforge/models/` — the policy, behind one interface

**Does:** `base.py` defines `ChatModel.complete(messages, tools) -> AssistantTurn`.
`anthropic_model.py` and `openai_compat.py` translate to/from each provider's wire
format. `scripted.py` replays a pack's reference solution with no network.

**ML concept:** the model is the **policy** π(action | observation history). Two very
different policies are treated the same way:
- **frozen API models** — you can't change their weights, you use them to *measure* and to
  *generate training data*;
- **open-weight models behind vLLM** (`vllm:` provider) — the ones you will *train*.
Because both sit behind the same interface, the eval that scores Claude today scores your
fine-tuned 3B model next month with a one-flag change.

**Python to notice:**
- lazy `from anthropic import Anthropic` inside `__init__`: the SDK and its key are only
  needed when that provider is actually used; `scripted` works offline.
- the two `_to_*` methods are pure data translation. Compare them: Anthropic wants tool
  results inside a `user` turn, OpenAI wants a separate `tool` role. The neutral `Message`
  format exists so the runner never has to know.

**Do by hand:** write `tests/test_models_offline.py` that feeds a hand-built message
list into `AnthropicModel._to_anthropic` and `OpenAICompatModel._to_openai` and asserts
the shapes. No API calls needed — they're static methods.

---

## 6. `envforge/core/runner.py` — the rollout

**Does:** `run_episode()`: reset env → build messages → loop (call model, apply tool calls,
append results) until `finish`, no tool calls, or budget → `verify()` → one `Trajectory`.
`run_many()` repeats that across seeds and tiers into a run directory.

**ML concept: rollouts.** This loop is what every RL-for-LLMs system runs millions of
times. Everything about scale is downstream of making this loop cheap, parallel and
crash-proof — hence: tool errors are observations, exceptions are recorded not raised,
observations are truncated, and every episode is written as soon as it finishes.

**Python to notice:**
- `for ... else:` — the `else` runs only if the loop wasn't `break`-ed. Used to mark
  `max_steps`.
- `try / except / finally` with `env.close()` in `finally` so sandbox containers get torn
  down even on failure.
- `time.perf_counter()` for latency; `uuid` for ids.

**Do by hand:** add a `--parallel N` option to `run_many` using
`concurrent.futures.ProcessPoolExecutor`. Sims are CPU-bound, so processes not threads.
Keep the JSONL append in the parent process.

---

## 7. `envforge/core/trajectory.py` — the data

**Does:** `Step`, `Trajectory`, `Usage` pydantic models and a tiny JSONL store.

**ML concept: trajectories are the dataset.** Every downstream stage — evaluation,
rejection sampling, SFT, DPO, GRPO — consumes exactly this. Design decisions that matter:
- one JSON object per line (streamable, appendable, greppable);
- the full `system_prompt` and `task` inside each record (self-contained; no joins);
- `schema_version` on every line (you *will* change this and need to migrate).

**Python to notice:**
- `@property` on `success` / `reward`: computed attributes that read like fields.
- `Iterator[...]` return type and a generator (`yield`) in `_iter()`: files are streamed,
  never loaded whole.

**Do by hand:** `jq` is your friend: `jq '.verdict.reward' trajectories.jsonl | sort | uniq -c`.

---

## 8. `envforge/core/evaluate.py` — turning episodes into numbers

**Does:** `compute_metrics()` → solve rate, mean reward, steps, tool-error rate, collateral
rate, budget-exhaustion rate, tokens. `evaluate_run()` does it overall and per tier.
`filter_trajectories()` keeps only episodes above a reward threshold.

**ML concepts:**
- **Evaluation.** Always report per-tier, always report n. A 90% solve rate on 10 episodes
  is noise; on 300 it's a result. Seeds make the comparison paired: model A and model B
  saw the same tasks.
- **Rejection sampling.** `filter_trajectories` is the first half of the simplest training
  recipe there is: generate many episodes with a strong policy, keep the ones the verifier
  liked, fine-tune a small model to imitate them. No RL math yet — just "learn from the good
  examples". Phase 3 implements the second half (SFT).

**Do by hand:** add `mean_reward_ci` using a bootstrap (resample trajectories with
replacement 1000×, take the 2.5/97.5 percentiles). Then look at how wide it is at n=30.

---

## 9. `envforge/cli.py` — the user interface

**Does:** typer commands wrapping the above. `run` prints a per-episode line and a
metrics table; `show` renders a transcript; `compare` is for before/after.

**Python to notice:** typer builds the CLI from type hints and defaults; `rich` draws
tables. Both are worth knowing for any tool you ship.

---

## 10. What comes next, and what you'll learn at each step

| Phase | You will build | You will learn |
|---|---|---|
| 2a | a second sim pack (`sipops`) | that the contract generalises; what's domain vs generic |
| 2b | `sandbox/docker.py` + a `repofix` pack | sandboxing, containers as state, cleanup discipline |
| 2c | `compute/ssh.py` + vLLM on a GPU box | serving open models; the OpenAI-compatible contract; GPU memory math |
| 3a | `train/filter.py` → SFT dataset format | how trajectories become (prompt, completion) pairs; chat templates |
| 3b | `train/sft.py` with TRL + LoRA | loss, learning rate, epochs, overfitting, checkpoints, adapters |
| 3c | re-eval with `vllm:` + adapter; `compare` | before/after; what "it got better" actually means |
| 4 | DPO, then GRPO | preference pairs; online RL; KL penalties; why GRPO needs the verifier to be fast |

The honest rule for the whole journey: **every phase ends with a `compare` table and two
sentences in the README saying what changed and why.** That discipline is what turns a
repo into a research artefact.

---

## Appendix: Python idioms used here, with one-line reminders

| idiom | reminder |
|---|---|
| `from __future__ import annotations` | type hints become strings; lets you reference classes before they're defined |
| `dict[str, int]`, `list[X]`, `X \| None` | built-in generics (3.9+) and the union operator (3.10+); no `typing.Dict` needed |
| `ClassVar` | class-level attribute, excluded from instance fields |
| `@classmethod` | receives the class (`cls`), used for alternate constructors like `generate_task` |
| `@property` | method that reads like an attribute |
| `@dataclass` vs `pydantic.BaseModel` | dataclass = plain container; pydantic = validation + JSON |
| `raise X from e` | chain exceptions, keep the cause |
| `for … else` | `else` runs if no `break` |
| generators / `yield` | lazy iteration; stream big files |
| `Path` (pathlib) | `/` to join, `.read_text()`, `.mkdir(parents=True, exist_ok=True)` |
| `random.Random(seed)` | isolated, reproducible RNG |
| `ipaddress` | `ip in net`, `net.prefixlen`, `net.network_address` |

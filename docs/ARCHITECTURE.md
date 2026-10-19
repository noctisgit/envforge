# Architecture

## The one idea

Everything a reinforcement-learning environment for an LLM needs reduces to five things.
If a domain supplies those five, the rest of the pipeline never changes:

| Piece | What it is | netops | another domain |
|---|---|---|---|
| **State** | the world the agent acts on | routers, links, ACLs, DNS | a SIP dial plan; a k8s cluster; a repo with a bug |
| **Tools** | verbs with JSON schemas | `ping`, `show_routes`, `set_route` | `sip_trace`, `kubectl`, `run_tests` |
| **Task generator** | seeded synthetic problems, by tier | fault injection | injected misconfig; a known-bad commit |
| **Verifier** | code that scores the end state | reachability + collateral checks | call completes; pod Ready; tests pass |
| **Instructions** | the system prompt | "you are on call for a WAN…" | "you are an SRE…" |

That contract is `envforge.core.env.Environment`. The generic layer — runner, trajectory
store, evaluator, filter, (later) trainer and compute backends — only ever sees the contract.

## Two kinds of environment, one contract

- **`kind = "sim"`** — state is a Python object, tools are functions. Deterministic,
  microseconds per step, thousands of episodes a minute, no infrastructure. Most RL
  environment work lives here because speed and reproducibility beat realism.
- **`kind = "sandbox"`** — state is a real container, tools run commands inside it. For
  domains where the thing under test *is* real software. Slower, costs money, needs
  isolation. Backends are pluggable (`envforge/sandbox/`: Docker on your own boxes now —
  no network, no capabilities, read-only root, resource limits, files moved by tar so it
  works from inside a container; Daytona/E2B/Modal later when you need hundreds of
  parallel sandboxes with instant reset). `repofix` is the reference sandbox pack.

The runner does not care which; the pack declares it.

## Topology

```
┌──────────────────────────────────┐
│  CONTROL PLANE  (CPU box)        │   envforge CLI · sims · trajectory store · eval
│  calls model APIs over HTTPS     │   orchestrates GPU boxes over SSH/Tailscale
└──────┬──────────────────┬────────┘
       │ HTTPS            │ SSH / Tailscale, on demand
       ▼                  ▼
┌────────────┐     ┌──────────────────────────────┐
│ API models │     │  GPU BOX (RunPod / Vast)      │
│ Anthropic, │     │  open-weight model → vLLM      │
│ OpenAI,    │     │  (OpenAI-compatible endpoint)  │
│ Vercel GW  │     │  trainer (TRL / LoRA)          │
└────────────┘     │  sandboxed envs (Docker)       │
                   └──────────────────────────────┘
```

Model weights never live on the control plane. A self-hosted model behind vLLM looks
identical to the Vercel gateway from the runner's point of view (`openai_compat.py`),
which is why switching from "evaluate Claude" to "evaluate my fine-tuned 3B model" is a
one-flag change.

## The loop

```
task = Env.generate_task(seed, tier)          # synthetic, deterministic
env  = Env(task); obs = env.reset()
messages = [system_prompt, goal + obs]
repeat until finish / no tool calls / budget:
    turn = model.complete(messages, tools)     # may contain N tool calls
    for call in turn.tool_calls:
        obs = env.step(call)  or  "ToolError: ..."
        messages += [assistant turn, tool result]
verdict = env.verify()                         # pure code → success + reward
write Trajectory(task, steps, verdict, usage) as one JSONL line
```

Two design choices worth knowing:

1. **Tool errors are observations, not crashes.** A bad call returns `ToolError: …` to the
   model and the episode continues. Agents must be able to recover from their own mistakes.
2. **`finish` is injected by the core**, so every pack ends episodes the same way and the
   trainer never has to learn per-domain termination.

## Data

One run = one directory:

```
data/workspaces/<workspace>/runs/<run-id>/
  meta.json            pack, model, tiers, seeds, timestamps
  trajectories.jsonl   one Trajectory per line  (schema_version in each line)
```

Trajectories are the product. Evaluation reads them; `filter` selects them;
training consumes them. JSONL so many workers can append and nothing needs a database.

`train/filter.py` turns kept trajectories into chat-format JSONL (OpenAI messages + tool
schemas), rebuilding the exact opening prompt from `meta["first_observation"]`, which the
runner records. Adding a key to `meta` is not a schema change.

**Prompt caching.** Every prompt is laid out static-first: tools, then the system
prompt (no timestamps or per-episode data), then the conversation, which only ever grows
(history is never rewritten). The Anthropic adapter puts a cache breakpoint on the system
block (caching tools + system, shared by every call of an episode and every episode of an
environment) and turns on automatic caching for the conversation tail, so each step reads
the previous prefix at ~0.1x. OpenAI and vLLM cache long prefixes automatically.

**Cost.** Episodes record token usage; `models/catalog.py` prices them from the Vercel AI
Gateway's public model list (which also prices native Anthropic/OpenAI ids). Prices are
list prices without caching discounts — an estimate. vLLM and scripted runs cost $0.

**Schema versioning.** `envforge.core.trajectory.SCHEMA_VERSION` is written into every
line. Breaking changes bump it and get a note here.

| version | change |
|---|---|
| 1 | initial |
| 2 | `Usage` gains `cache_read_tokens` and `cache_write_tokens`; `input_tokens` is now the *whole* prompt for every provider (Anthropic's uncached remainder + cache writes + cache reads). v1 lines load unchanged: the new fields default to 0, which was true then (nothing was cached). |

## Workspaces

All outputs live under `data/workspaces/<workspace>/`. This is the seam for multi-user
support: when a web/API layer is added, authentication maps a principal to a workspace
id; nothing in the core changes. Auth itself (Clerk vs local) is deliberately deferred
until there is something to authenticate.

## Reward design (what makes an environment trainable)

- **Verifiable.** Computed by code from end state. No model-as-judge in the loop, so
  the signal can't be gamed by persuasion.
- **Collateral-aware.** Solving the task by breaking something else scores 0.3, not 1.0.
  This is what makes "don't break what wasn't broken" learnable.
- **Decoys.** Tier ≥ 2 includes rules that *look* removable but are protected policy.
  Removing them is penalised. This punishes "delete everything until it works".
- **Multiple valid solutions.** The verifier checks outcomes, not actions, so rerouting
  around a dead link is as correct as bringing the link up.

## Roadmap

| Phase | Deliverable | What you learn |
|---|---|---|
| 1 ✅ | core, netops, adapters, eval + cost, CLI, chat-format export | environments, rewards, trajectories, evaluation |
| 2 ✅ | `sipops` sim pack; `repofix` sandbox pack; `sandbox/docker.py`; `compute/ssh.py` + vLLM | the contract generalises; sandboxes; serving with vLLM |
| 3 ✅ | `train/dataset.py` (rejection sampling, seed bookkeeping), `train/sft.py` (LoRA SFT, assistant-only loss), `train/remote.py` (GPU box), vLLM LoRA serving, before/after eval — docs/TRAINING.md | rejection sampling, SFT, overfitting, contamination |
| 4 | DPO then GRPO trainers on the same envs | preference data; online RL |

## Non-goals (for now)

- Not a general agent framework. One loop, one message format, no plugins for plugins.
- No model-graded rewards. If a domain can't be verified by code, it isn't a good first pack.
- No distributed training. Single-GPU LoRA is the target through Phase 3.

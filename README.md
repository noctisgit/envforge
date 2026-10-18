# envforge

**Pluggable RL environments + an eval-to-training pipeline for LLM agents. Any domain, one loop.**

envforge separates the two things people usually tangle together when they build "an RL
environment for LLMs":

- the **pipeline** — run an agent in an environment, record every step, score the outcome,
  keep the good episodes, train on them, evaluate again — which is the same for every domain, and
- the **environment** — the world, the tools, the task generator, the verifier — which is a
  plugin called a *pack*.

Write a pack for your domain and you get rollouts, trajectories, evaluation, rejection sampling
and fine-tuning for free.

```
            ┌───────────── pack (your domain) ─────────────┐
            │ state · tools · task generator · verifier     │
            └──────────────────────┬───────────────────────┘
                                   │ one contract
  ┌────────┐   tool calls   ┌──────▼──────┐   JSONL   ┌──────────┐   filter   ┌──────────┐
  │  model │◄──────────────►│   runner    │──────────►│  eval    │───────────►│  train   │
  │ (API / │  observations  │ (rollouts)  │           │ metrics  │   good     │ SFT/DPO/ │
  │  vLLM) │                └─────────────┘           └──────────┘  episodes  │  GRPO    │
  └────────┘                                                                  └──────────┘
```

## Status

**Phase 1 — working.** Core contract, agent runner, trajectory store, evaluation, model adapters
(Anthropic, OpenAI, Vercel AI Gateway, vLLM, offline scripted), and the `netops` reference pack.
Everything runs on a CPU box; the only network calls are to model APIs.

| Phase | What | Where it runs |
|---|---|---|
| 1 ✅ | sim packs · API models · eval | control plane (CPU) |
| 2 | second pack (sipops) · sandbox packs via Docker · SSH compute backend · vLLM serving | control plane + GPU box |
| 3 | rejection sampling → LoRA SFT on a 3–4B open model → before/after eval | GPU box |
| 4 | DPO, then GRPO on the same environments | GPU box |

## Quick start

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh   # once
git clone https://github.com/noctisgit/envforge && cd envforge
uv sync --extra dev
cp .env.example .env                               # add ANTHROPIC_API_KEY etc.

uv run envforge packs                              # what environments exist
uv run envforge tasks netops --tier 2 --n 2 --hidden
uv run envforge run netops --model scripted --episodes 9          # offline, no keys needed
uv run envforge run netops --model anthropic:claude-sonnet-4-5 --episodes 30
uv run envforge show data/workspaces/default/runs/<run-id> --episode 0
uv run envforge compare data/workspaces/default/runs/<a> data/workspaces/default/runs/<b>
```

Model strings are `provider:model-id`:

| provider | example | notes |
|---|---|---|
| `anthropic` | `anthropic:claude-sonnet-4-5` | native Messages API |
| `openai` | `openai:gpt-4o-mini` | |
| `vercel` | `vercel:anthropic/claude-opus-5` | [AI Gateway](https://vercel.com/docs/ai-gateway), OpenAI-compatible, 400+ models |
| `vllm` | `vllm:Qwen/Qwen2.5-3B-Instruct` | your own GPU box, `VLLM_BASE_URL` |
| `scripted` | `scripted` / `scripted:error_every=3` | replays the pack's reference solution; offline |

## The reference pack: `netops`

An agent is dropped into a small enterprise WAN (three sites + core, static routing, ACLs, DNS)
where something is broken, and must restore connectivity between two hosts **without** breaking
anything else or removing security-policy rules. Faults are generated from a seed at three
difficulty tiers (one obvious fault → three compounding faults including return-path and decoy
policy rules). Reward is computed by code: 1.0 for a clean fix, 0.3 for a fix with collateral
damage, 0 otherwise.

It is a real environment in miniature: observations are realistic command output, mistakes have
consequences, and there is more than one correct fix.

## Writing a pack

A pack is a subpackage of `envforge/packs/` that subclasses `Environment` and implements
`generate_task`, `reset`, `tools`, `step`, `verify`, `system_prompt` and `reference_solve`.
See [docs/PACKS.md](docs/PACKS.md) for the contract and a template, and
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for why it is shaped this way.

## Workspaces and the web UI

All outputs are namespaced: `data/workspaces/<workspace>/{runs,sft}`. From the CLI, pass
`--workspace` or set `ENVFORGE_WORKSPACE`.

The optional web layer adds accounts, workspaces, per-workspace provider keys (encrypted at
rest), model preferences, a run queue with a worker, and a UI for metrics and transcripts.
Auth is pluggable: `AUTH_PROVIDER=local` (email + password) or `clerk`. Postgres is the
production database; `docker compose up -d` brings up the whole stack. See
[docs/WEB.md](docs/WEB.md).

```bash
uv sync --extra dev --extra web
uv run envforge-api & uv run envforge-worker &        # :8000
cd web && npm install && npm run dev                   # :3000
```

## Learning

[docs/LEARNING.md](docs/LEARNING.md) walks through the code module by module and maps each piece
to the RL-for-LLMs concept it implements (episode, trajectory, reward, rollout, rejection sampling,
SFT, DPO, GRPO).

## Development

```bash
uv run pytest            # offline test suite
uv run ruff check . && uv run ruff format .
```

MIT License.

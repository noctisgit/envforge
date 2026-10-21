# envforge

**Build RL environments for LLM agents, evaluate any model on them, and turn the best episodes
into training data for open models. Any domain, one loop. Self-hosted, with a web UI.**

envforge separates the two things people usually tangle together when they build "an RL
environment for LLMs":

- the **pipeline**: run an agent in an environment, record every step, score the outcome,
  keep the good episodes, train on them, evaluate again. It is the same for every domain.
- the **environment**: the world, the tools, the task generator and the verifier. It is a
  plugin called a *pack*, written in Python or built from your own data in the web UI.

```
            ┌──────────── environment ("pack") ─────────────┐
            │ world state · tools · task generator · verifier │
            └──────────────────────┬────────────────────────┘
                                   │ one contract (envforge/core/env.py)
  ┌────────┐   tool calls   ┌──────▼──────┐  trajectories  ┌──────────┐  best episodes  ┌──────────┐
  │ model  │◄──────────────►│   runner    │───────────────►│  eval    │────────────────►│  train   │
  │ (API / │  observations  │ (episodes)  │    (JSONL)     │ metrics, │   (rejection    │ LoRA SFT │
  │  vLLM) │                └─────────────┘                │   cost   │    sampling)    │ → re-eval│
  └────────┘                                               └──────────┘                 └──────────┘
```

**Contents:** [What you can do](#what-you-can-do) · [How it works](#how-it-works) ·
[Self-hosting](#self-hosting-one-command) · [Using the web app](#using-the-web-app) ·
[Command line](#command-line) · [Environments](#environments) ·
[Builder and privacy](#build-environments-from-your-own-data) ·
[Open models on a GPU](#open-models-and-training-on-a-rented-gpu) · [Docs](#documentation) ·
[Development](#development)

## What you can do

- **Evaluate models as agents.** Run Claude, GPT, anything on the Vercel AI Gateway, or your own
  open model behind vLLM against realistic, seeded tasks. Get solve rate, steps, collateral
  damage, tokens and cost, and replay every episode step by step.
- **Compare** models, prompts and difficulty tiers side by side.
- **Explore environments**: world diagrams, tools with their JSON schemas, reward rules, difficulty
  tiers and live task previews, all generated from each environment's own code.
- **Build new environments from your data.** Upload support chats, tickets or logs (CSV, TXT, JSON,
  up to many GB). A builder agent drafts a verifiable environment with you in chat. Personal data
  is redacted before any model sees it, and a measured human review gates extraction.
- **Make training data.** Keep only the episodes that earned full reward (rejection sampling) and
  export them as chat-format JSONL.
- **Fine-tune an open model** (LoRA SFT) on a rented GPU, serve it with vLLM, and measure
  before/after on held-out tasks.
- **Track usage**: tokens and estimated cost per day, model, member and workspace.

## How it works

| concept | what it is in envforge |
|---|---|
| **environment / pack** | a Python class (or a declarative spec) supplying the five things below |
| **task** | generated from `(seed, tier)`: the same seed always gives the same task, so results are reproducible |
| **tools** | the actions the agent may call, with JSON schemas, e.g. `show_route`, `ping`, `lookup_order` |
| **episode** | system prompt + goal → the model calls tools → the environment answers in text → … until `finish` or the turn budget runs out |
| **verifier → reward** | code, never a model, checks the end state: 1.0 solved cleanly, 0.3 solved with collateral damage, 0.0 not solved. An agent can't talk its way to a high score |
| **trajectory** | every message, tool call, result, token count and the reward, saved as JSONL |
| **rejection sampling** | keep the best episode per task (reward 1.0) → chat-format training examples |
| **SFT** | fine-tune an open model on those examples (LoRA), then evaluate it on held-out seeds |

This is the core loop of RL for LLM agents: environments → rollouts → evaluation → filtering →
training. Phase 4 adds DPO and GRPO on the same environments.
[docs/LEARNING.md](docs/LEARNING.md) explains each piece and the ML concept behind it.

The web app is a thin layer over the same library the CLI uses:

```
browser ──► Next.js UI (:3000) ──/api/*──► FastAPI ──► Postgres (users, workspaces, runs, builder)
                                              │
                         worker ──► runs episodes with the user's own provider keys ──► model APIs / vLLM
                         builder ──► uploads (redact → store), builder-agent chat turns
```

## Self-hosting (one command)

**You need** a Linux or macOS machine (or Windows with WSL2) with
[Docker](https://docs.docker.com/engine/install/) and Docker Compose v2, about 15 GB of free
disk, and 4 GB of RAM or more (8 GB+ if you use the builder's PII model). No GPU is needed.

```bash
git clone https://github.com/noctisgit/envforge && cd envforge
./install.sh
```

The installer:

1. checks Docker, Compose, disk and memory;
2. creates `.env` with strong random secrets (`APP_SECRET_KEY`, `POSTGRES_PASSWORD`) and asks
   for the port, whether the UI should be reachable from other machines, your public URL (if
   any), and whether anyone may sign up;
3. builds and starts the stack in Docker: **Postgres**, the **API** (it creates and migrates
   the database tables on start), the **run worker**, the **builder worker** and the **web UI**;
4. waits until the app answers, then creates your **admin account** from the terminal, so
   nobody can claim the server by signing up first.

Then open **http://localhost:3000**, sign in, and add a provider key under **Profile**.

**Putting it on the internet** is up to you. The UI listens on `127.0.0.1:3000` by default; put
nginx, Caddy, a Cloudflare Tunnel or a port forward in front of it. If you serve it over HTTPS,
run `./install.sh --url https://your.domain` once so session cookies are marked secure. To
listen on every interface instead, use `--listen 0.0.0.0`.

**Non-interactive** (servers, CI):

```bash
./install.sh --yes --admin-email you@example.com --admin-password-file ./admin-password.txt \
             --port 3000 --url https://envforge.example.com
```

**Everyday operations**

| task | command |
|---|---|
| update | `git pull && ./install.sh`: keeps `.env` and all data, applies database migrations |
| stop / start | `docker compose stop` / `docker compose start` (the stack also restarts with the machine) |
| logs | `docker compose logs -f api worker builder web` |
| accounts | `docker compose exec api envforge admin users` · `create-user EMAIL [--admin]` · `reset-password EMAIL` · `set-admin EMAIL` · `sign-out EMAIL` |
| back up | `docker compose exec -T postgres pg_dump -U envforge envforge \| gzip > envforge.sql.gz`, plus the `envdata` volume (runs, datasets) and your `.env` |

**Security model.** Every user has private workspaces: other users get "not found". A workspace
has an owner and members, and the first account is the server admin. Passwords are hashed with
argon2. Provider keys are encrypted at rest with a key derived from `APP_SECRET_KEY` and are only
decrypted by the worker that runs your episodes. **Back up `.env`: if `APP_SECRET_KEY` is lost or
changed, saved keys can't be decrypted.** Uploaded data is redacted before it is stored, and the
original file is deleted. The `worker` service mounts the Docker socket so the `repofix` sandbox
pack can start locked-down containers. That makes it root-equivalent on the host, so run envforge
on a machine you control, or remove that line from `docker-compose.yml` to run simulation packs
only. Optional Clerk sign-in, all settings and the API reference: [docs/WEB.md](docs/WEB.md).

## Using the web app

**Start at Home.** The first time you sign in it asks what you want to do: compare models on a
task, build an environment from your own data, train a small open model, or just look around.
It then lists the steps for that goal in order, ticks them off from your workspace's real
runs, and gives each one a button that opens the right page with the form already filled in.
The sidebar follows the same pipeline: **Build** (Environments, Builder), **Evaluate** (Runs),
**Train** (Train a model).

1. **Add a provider key** under Provider keys: Anthropic, OpenAI, a Vercel AI Gateway key
   (400+ models), or the URL of your own vLLM server. Keys belong to your account and work in
   all your workspaces. Anthropic and OpenAI keys are checked when saved, and the model picker
   then lists exactly the models your key can use, with prices. The `scripted` model needs no key.
2. **Pick an environment** under Environments (start with `netops`), or build your own in the
   Builder (see below). Tabs show how an episode works, the world diagram, the tools, the reward
   rules and generated tasks.
3. **Start a run** under Runs: model, environment, episodes, tiers and a **purpose**:
   *evaluation* (seeds 0+, the fixed tasks every model is scored on) or *training data*
   (seeds 10000+, tasks a model may learn from, never used for scoring). The run's page shows
   metrics, cost and a step-by-step replay of every episode, then suggests what to do next.
4. **Compare** two or more runs on the same tasks (solve rate per tier, steps, collateral, cost).
5. **Train a model** walks through distillation: score the teacher and the untrained open
   model, collect the teacher's solutions on training seeds, build the dataset, fine-tune a
   LoRA adapter on a rented GPU, test it on the evaluation seeds and call it through vLLM.
   Steps 4–6 run on the command line for now; the page shows the exact commands, filled in with
   your runs.
6. **Settings** (per workspace): default model, environment and episodes, the builder model, and
   the training base model (searchable live from Hugging Face, with a GPU-memory estimate).
   **Usage** shows tokens and cost per day, model, member and workspace.

## Command line

The same pipeline without the web app, for scripting and development:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh   # once: installs uv
uv sync --extra dev
cp .env.example .env                               # add ANTHROPIC_API_KEY etc.

uv run envforge packs                              # what environments exist
uv run envforge tasks netops --tier 2 --n 2 --hidden
uv run envforge run netops --model scripted --episodes 9                  # offline, no keys
uv run envforge run netops --model anthropic:claude-sonnet-5-5 --episodes 30
uv run envforge eval    data/workspaces/default/runs/<run-id>
uv run envforge show    data/workspaces/default/runs/<run-id> --episode 0
uv run envforge compare data/workspaces/default/runs/<a> data/workspaces/default/runs/<b>
uv run envforge filter  data/workspaces/default/runs/<a> --min-reward 1.0 --format chat
```

Model strings are `provider:model-id`:

| provider | example | notes |
|---|---|---|
| `anthropic` | `anthropic:claude-sonnet-5-5` | native Messages API, with prompt caching |
| `openai` | `openai:gpt-5-mini` | |
| `vercel` | `vercel:anthropic/claude-sonnet-4.5` | [AI Gateway](https://vercel.com/docs/ai-gateway), OpenAI-compatible, 400+ models; ids use dots |
| `vllm` | `vllm:Qwen/Qwen2.5-7B-Instruct` | your own server (`VLLM_BASE_URL`); see [docs/COMPUTE.md](docs/COMPUTE.md) |
| `scripted` | `scripted` | replays the environment's reference solution; offline, free |

## Environments

| pack | kind | the agent must… |
|---|---|---|
| `netops` | simulation | restore connectivity on a small enterprise WAN (routes, ACLs, DNS, links) without breaking anything else or removing security rules |
| `sipops` | simulation | fix a broken SIP call path (dial plans, trunks, codecs, DNS, credentials, registrations) without breaking emergency routing — [README](envforge/packs/sipops/README.md) |
| `repofix` | sandbox | make a small Python repo's tests pass inside a locked-down container (no network, read-only root, resource limits) without editing the tests — [README](envforge/packs/repofix/README.md) |
| *yours* | declarative | built from your data in the Builder, published privately to your workspace as `<project>@v<n>` |

Each has three or more difficulty tiers, from one obvious fault to several compounding faults
with decoys. Writing a new pack in Python means subclassing `Environment` and implementing
`generate_task`, `reset`, `tools`, `step`, `verify`, `system_prompt` and `reference_solve`; see
[docs/PACKS.md](docs/PACKS.md).

## Build environments from your own data

The **Builder** turns a goal ("an IT helpdesk agent that resolves VPN and password tickets
without leaking account data") and your data into a verifiable environment by chatting with a
builder agent, which runs on your workspace's builder model and your own key.

- **Uploads of any size**: CSV, TXT, JSON/JSONL, or `.gz` of those, in resumable chunks or
  imported from an https URL. Conversations are streamed into a per-project store,
  near-duplicates are removed, and similar conversations are grouped into topic clusters. The
  builder reads and extracts across clusters instead of loading whole files.
- **Privacy first.** Redaction happens before anything is stored or sent to a model:
  - **off**, **standard** or **strict** (aimed at HIPAA Safe Harbor's identifier list)
  - optional **realistic stand-ins** instead of `<NAME_1>` placeholders
  - optional local name detection (NER)
  - an optional second pass by a local PII model on risky conversations

  All of it runs on your server. On a benchmark with known ground truth, the full setup left
  about 0.6% of names and usernames, and no emails or phone numbers.
- **Measured review.** A person checks a random sample of redacted conversations. envforge reports
  the leak rate with a 95% confidence interval, and extraction is blocked if the interval's upper
  bound exceeds your threshold (required on strict).
- **Revisions**: every change is an immutable revision you can diff, restore or branch from.
  Extraction is validated: a task only counts if its reference solution actually earns reward 1.0.
- **Publish** a revision to run, compare, replay and export it like any built-in environment.

Details: [docs/BUILDER.md](docs/BUILDER.md).

## Open models and training on a rented GPU

The control plane (the machine running envforge) never needs a GPU. For open models, rent one
on demand:

- **RunPod / Vast**: start a pod from the prebuilt image `ghcr.io/noctisgit/envforge-gpu`
  (vLLM + LoRA training + SSH + optional Tailscale) and drive it with `--runtime pod`.
- **Any GPU VM** with Docker and the NVIDIA container toolkit: the default `--runtime docker`.

```bash
export ENVFORGE_GPU_HOST=gpu-1 ENVFORGE_GPU_RUNTIME=pod      # a Tailscale name or an IP
uv run envforge compute check
uv run envforge compute vllm-up Qwen/Qwen2.5-7B-Instruct --bind 127.0.0.1
uv run envforge run netops --model vllm:Qwen/Qwen2.5-7B-Instruct --episodes 30

uv run envforge train dataset data/workspaces/default/runs/<teacher-run> --name my-set
uv run envforge train sft my-set --base Qwen/Qwen2.5-7B-Instruct
uv run envforge train follow my-set && uv run envforge train fetch my-set
```

To evaluate a served model in the web app, add its URL under Profile → vLLM. Guides:
[docs/COMPUTE.md](docs/COMPUTE.md) (GPU boxes, the RunPod template, Tailscale) and
[docs/TRAINING.md](docs/TRAINING.md) (teacher runs → dataset → LoRA SFT → before/after eval).

## Documentation

| doc | covers |
|---|---|
| [docs/WEB.md](docs/WEB.md) | web layer: configuration, auth (local / Clerk), keys, isolation, deployment, API |
| [docs/BUILDER.md](docs/BUILDER.md) | environment builder: uploads, redaction profiles, review, revisions, publishing |
| [docs/COMPUTE.md](docs/COMPUTE.md) | GPU boxes, the envforge GPU image on RunPod/Vast, vLLM serving, Tailscale |
| [docs/TRAINING.md](docs/TRAINING.md) | rejection sampling, LoRA SFT, serving adapters, before/after evaluation |
| [docs/PACKS.md](docs/PACKS.md) | writing a new environment in Python |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | design decisions, the pack contract, trajectory schema, roadmap |
| [docs/LEARNING.md](docs/LEARNING.md) | module-by-module walkthrough mapped to RL-for-LLMs concepts |

## Development

```bash
uv sync --extra dev --extra web
docker compose up -d postgres                     # or DATABASE_URL=sqlite:///./data/envforge.db
uv run alembic upgrade head
uv run envforge-api                               # FastAPI on :8000 (/docs)
uv run envforge-worker --kind runs                # and, for the builder: --kind builder
cd web && npm install && npm run dev              # Next.js on :3000

uv run pytest                                     # offline test suite (scripted model, no keys)
uv run ruff check . && uv run ruff format .
```

Rules for contributors (and AI coding agents) are in [CLAUDE.md](CLAUDE.md).

## Status

| phase | what | |
|---|---|---|
| 1 | core contract, runner, trajectories, evaluation, cost, API model adapters, `netops` | ✅ |
| 2 | `sipops`, `repofix` (Docker sandbox), GPU compute backend, vLLM serving | ✅ |
| 3 | rejection sampling → LoRA SFT → serve the adapter → before/after eval | ✅ |
| + | web app, environment builder, privacy pipeline, GPU image, one-command install | ✅ |
| 4 | DPO, then GRPO on the same environments | next |

MIT License.

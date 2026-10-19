# CLAUDE.md — envforge

This file is read automatically by Claude Code. It tells you what this project is,
how it is laid out, how to run it, and the rules for changing it.

## What envforge is

A domain-pluggable pipeline for evaluating and (later) training LLM agents on
simulated or sandboxed environments. The **pipeline** (runner, trajectory store,
evaluation, data filtering, training, compute orchestration) is generic. The
**environment** is a plugin called a *pack*. A pack supplies five things:

1. **State** — the world the agent acts on (a Python object for sims; a container for sandboxes)
2. **Tools** — the verbs the agent may call, with JSON schemas
3. **Task generator** — seeded, synthetic tasks at difficulty tiers
4. **Verifier / reward** — an objective check of success plus penalties
5. **Instructions** — the system prompt for the model

Anything that can supply those five things can be trained and evaluated with the same loop.
The reference pack is `netops` (network troubleshooting: routes, ACLs, DNS, links).

The long-term purpose is a public research/engineering showcase and a learning vehicle
for RL-for-LLMs: environments → rollouts → evaluation → rejection sampling → SFT → DPO/GRPO.

## Topology

- **Control plane** (this box, CPU only): runs the CLI, the in-process simulators,
  stores trajectories, calls model APIs over HTTPS, and orchestrates GPU boxes over SSH/Tailscale.
- **GPU boxes** (RunPod / Vast, on demand): host open-weight models behind vLLM
  (OpenAI-compatible endpoint) and run training. Never put model weights on the control plane.
- **Model access** goes through one interface (`envforge/models/`). Providers:
  `anthropic` (native SDK), `openai`, `vercel` (AI Gateway, OpenAI-compatible,
  base URL `https://ai-gateway.vercel.sh/v1`, model ids like `anthropic/claude-opus-5`),
  `vllm` (self-hosted, OpenAI-compatible), `scripted` (offline, for tests and demos).

## Layout

```
envforge/
  core/
    env.py          Environment / Task / Observation / ToolSpec / StepResult contract
    trajectory.py   Trajectory + Step dataclasses, JSONL writer/reader
    runner.py       agent loop: model <-> environment via tool calls; budgets; logging
    evaluate.py     aggregate trajectories -> metrics (solve rate, steps, collateral, cost)
    explore.py      pack overview, task previews, episode replays (powers the Environments page)
    registry.py     pack discovery (entry points + packs/ directory)
  models/
    base.py         ChatModel interface (messages + tools -> assistant turn w/ tool calls)
    anthropic_model.py
    openai_compat.py   covers openai, vercel, vllm via base_url/api_key
    scripted.py     deterministic policy driven by a pack's reference solver; no network
    catalog.py      model list + prices (Vercel AI Gateway public catalog); cost estimates
    __init__.py     load_model("provider:model-id")
  packs/
    netops/         reference sim pack (see packs/netops/README.md)
    sipops/         SIP/VoIP call-routing sim pack
    repofix/        reference sandbox pack: fix a small Python repo inside a container
  declarative/      environments written as data: spec.py (EnvSpec/TaskRecord), runtime.py (spec+tasks -> Environment),
                    rules.py, validate.py, store.py (spec.json + tasks.jsonl dirs), cli.py (`envforge env ...`), examples/
  builder/          environment-builder data pipeline: ingest.py (CSV/TXT/JSON -> conversations), redact.py (PII),
                    extract.py (conversation -> TaskRecord with any ChatModel)
  sandbox/          Sandbox interface; docker.py (hardened, default) and local.py (tests only)
  compute/          SSH GPU box driver + vLLM serving; `envforge compute ...` (docs/COMPUTE.md)
  train/
    filter.py       kept trajectories -> chat-format JSONL for SFT
    dataset.py      rejection sampling: best episode per task, task-level val split, seed manifest, overlap()
    sft.py          STANDALONE LoRA SFT script (assistant-only loss); copied to the GPU box — no envforge imports
    remote.py       run sft.py in a detached container on a GPU box, follow progress, fetch the adapter
    cli.py          `envforge train dataset|check-overlap|sft|follow|logs|fetch` (docs/TRAINING.md)
  cli.py            typer CLI
  config.py         env/.env loading, defaults, data dir, workspace dirs
  web/              optional FastAPI layer (uv sync --extra web); see docs/WEB.md
    settings.py     DATABASE_URL, AUTH_PROVIDER=local|clerk, APP_SECRET_KEY, ...
    db.py, models.py, migrations/   SQLAlchemy 2 + Alembic (users, workspaces, members, provider_keys, runs)
    crypto.py       Fernet encryption of provider keys (derived from APP_SECRET_KEY)
    auth/           local (argon2 + session cookie) and clerk (JWT via JWKS); one current_user dep
    services.py     workspace access, key resolution, run enqueue, usage aggregation
    security.py     vLLM base-URL checks (SSRF) and in-memory rate limiting
    builder.py      builder projects, revision DAG, content-addressed task sets, uploads, publish, resolve_pack, jobs
    builder_agent.py  the builder agent: tools + tool-use loop (runs in `envforge-worker --kind builder`)
    admin_cli.py    `envforge admin users|reset-password|sign-out|set-admin`
    routers/        auth (+me/usage), account (/api/me: profile, password, keys), workspaces (+packs, models, usage), runs (+episodes, export, replay, delete)
    worker.py       claims queued runs, executes with the queuing user's keys, writes progress/tokens/cost, heartbeats
    app.py          FastAPI factory; entry points envforge-api / envforge-worker
web/                Next.js UI (login, runs, run detail + replay, compare, environments, builder, usage, profile, workspace settings); proxies /api/* to FastAPI
deploy/             Dockerfiles + Caddyfile; docker-compose.yml at repo root (postgres, api, worker, web)
docs/
  ARCHITECTURE.md   design decisions, the pack contract, phase roadmap
  PACKS.md          how to write a new pack (copy this when adding a domain)
  LEARNING.md       module-by-module walkthrough + ML concepts, for the human maintainer
  WEB.md            web layer: config, auth providers, key handling, isolation, deploy, endpoints
  COMPUTE.md        renting a GPU box, serving a model with vLLM, evaluating it
  TRAINING.md       Phase 3 end to end: teacher runs -> dataset -> LoRA SFT -> adapter -> before/after eval
  BUILDER.md        the environment builder: chat, uploads + redaction, declarative spec, revisions, publishing
tests/              pytest; everything runs offline with the scripted model
data/               trajectories and eval reports (gitignored except data/samples/)
```

## Setup and everyday commands

```bash
# one-time on a fresh Ubuntu 24.04 box
curl -LsSf https://astral.sh/uv/install.sh | sh      # installs uv
uv sync --extra dev                                   # creates .venv from uv.lock
cp .env.example .env                                  # then fill in keys

# everyday
uv run envforge packs                                 # list packs
uv run envforge tasks netops --tier 1 --n 5           # preview generated tasks
uv run envforge run netops --model scripted --episodes 10        # offline smoke run
uv run envforge run netops --model anthropic:claude-sonnet-4-5 --episodes 20
uv run envforge eval data/runs/<run-id>               # metrics for one run
uv run envforge compare data/runs/<a> data/runs/<b>   # side-by-side
uv run envforge show data/runs/<run-id> --episode 3   # print one trajectory
uv run pytest                                         # must pass before committing
uv run ruff check . && uv run ruff format .           # lint/format

# web layer (optional)
uv sync --extra dev --extra web
uv run envforge-api                                   # FastAPI on :8000 (/docs)
uv run envforge-worker                                # executes queued runs
cd web && npm install && npm run dev                  # Next.js on :3000
docker compose up -d postgres                         # Postgres for local dev (or the full stack on a VPS)
uv run alembic upgrade head                           # apply migrations (Postgres)
uv run alembic revision --autogenerate -m "..."       # after editing envforge/web/models.py
```

Model strings are `provider:model-id`:
`anthropic:claude-sonnet-4-5`, `openai:gpt-4o-mini`, `vercel:anthropic/claude-opus-5`,
`vllm:Qwen/Qwen2.5-3B-Instruct`, `scripted`.

## Rules for changing this repo

- **Never commit secrets or data.** `.env`, `data/`, weights, and `*.jsonl` outside
  `data/samples/` and packs are gitignored. If a key is ever committed, rotate it.
- **Keep the core domain-agnostic.** Nothing under `envforge/core/` or `envforge/models/`
  may import from a pack. Domain logic lives only in `envforge/packs/<name>/`.
- **Every pack must satisfy the contract in `envforge/core/env.py` and ship a
  `reference_solve()`** so the scripted model and the tests can run with no network.
- **Determinism.** Task generation and simulators take a `seed`; the same seed must
  produce the same task and the same outcome for the same actions.
- **Trajectories are the product.** Do not change the JSONL schema in
  `trajectory.py` without bumping `SCHEMA_VERSION` and adding a migration note in
  `docs/ARCHITECTURE.md`.
- **Tests run offline.** Use the `scripted` model. Network-backed tests must be
  marked `@pytest.mark.network` and skipped by default (`uv run pytest -m network`);
  Docker-backed tests are marked `@pytest.mark.docker` and skip when Docker is absent.
  Set `ENVFORGE_OFFLINE=1` to keep the model catalog from fetching.
- Run `uv run pytest` and `uv run ruff check .` before every commit.
- Prefer small, readable Python over clever Python: this repo doubles as the
  maintainer's Python learning material. Type hints and docstrings on public functions.
- Python 3.12+. Dependencies via `uv add <pkg>` (never bare `pip install`),
  which updates `pyproject.toml` and `uv.lock` together. Web-only deps go in the
  `web` extra: `uv add --optional web <pkg>`.
- **Web layer rules.** Provider keys are only ever decrypted in the worker; API
  responses carry the masked form. `APP_SECRET_KEY` must not change once keys are
  stored. Every schema change ships with an Alembic migration. Routers stay thin;
  logic lives in `services.py`. The UI talks only to `/api/*`.

## Roadmap (see docs/ARCHITECTURE.md for detail)

- **Phase 1 (done):** core + netops sim pack + API model adapters + eval + cost. CPU only.
- **Phase 2 (done):** sipops sim pack, repofix sandbox pack via Docker; SSH compute
  backend for GPU boxes; vLLM serving an open model; same eval against it.
- **Phase 3 (done):** filter high-reward trajectories → LoRA SFT on an open model on a
  rented GPU → serve the adapter with vLLM → re-eval on held-out seeds (docs/TRAINING.md).
- **Phase 4:** DPO, then GRPO, using the same environments.

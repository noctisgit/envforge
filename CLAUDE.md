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
    registry.py     pack discovery (entry points + packs/ directory)
  models/
    base.py         ChatModel interface (messages + tools -> assistant turn w/ tool calls)
    anthropic_model.py
    openai_compat.py   covers openai, vercel, vllm via base_url/api_key
    scripted.py     deterministic policy driven by a pack's reference solver; no network
    __init__.py     load_model("provider:model-id")
  packs/
    netops/         reference sim pack (see packs/netops/README.md)
  cli.py            typer CLI
  config.py         env/.env loading, defaults, data dir, workspace dirs
  web/              optional FastAPI layer (uv sync --extra web); see docs/WEB.md
    settings.py     DATABASE_URL, AUTH_PROVIDER=local|clerk, APP_SECRET_KEY, ...
    db.py, models.py, migrations/   SQLAlchemy 2 + Alembic (users, workspaces, members, provider_keys, runs)
    crypto.py       Fernet encryption of provider keys (derived from APP_SECRET_KEY)
    auth/           local (argon2 + session cookie) and clerk (JWT via JWKS); one current_user dep
    services.py     workspace access, key resolution, run enqueue
    routers/        auth, workspaces (+keys, packs), runs (+episodes, transcripts)
    worker.py       claims queued runs, executes with the workspace's keys, writes progress/metrics
    app.py          FastAPI factory; entry points envforge-api / envforge-worker
web/                Next.js UI (login, runs, run detail, settings); proxies /api/* to FastAPI
deploy/             Dockerfiles + Caddyfile; docker-compose.yml at repo root (postgres, api, worker, web)
docs/
  ARCHITECTURE.md   design decisions, the pack contract, phase roadmap
  PACKS.md          how to write a new pack (copy this when adding a domain)
  LEARNING.md       module-by-module walkthrough + ML concepts, for the human maintainer
  WEB.md            web layer: config, auth providers, key handling, deploy, endpoints
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
  marked `@pytest.mark.network` and skipped by default.
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

- **Phase 1 (this repo now):** core + netops sim pack + API model adapters + eval. CPU only.
- **Phase 2:** second pack (sipops sim, then repofix sandbox via Docker); SSH compute
  backend to spin up GPU boxes; vLLM serving an open model; same eval against it.
- **Phase 3:** filter high-reward trajectories → LoRA SFT on a 3–4B open model (TRL)
  on a rented GPU → re-eval → before/after in README.
- **Phase 4:** DPO, then GRPO, using the same environments.

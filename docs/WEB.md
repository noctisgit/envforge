# Web layer: API, worker, UI

The web layer is optional. The CLI and library work without it. It adds:

- **multi-user**: accounts, each with private workspaces (no sharing between users)
- **per-user provider keys**, encrypted at rest, checked with the provider when saved,
  and used by every workspace that user runs in
- **per-workspace defaults**: model, pack, episode count
- **a model picker** listing only models your keys can use, with prices
- **an Environments explorer**: each pack's world diagram, dimensions, tools, tiers, reward
  rules and a live task preview, generated from the pack (see docs/PACKS.md)
- **a run queue** executed by separate worker processes in two lanes (`envforge-worker
  --kind runs --lane api|gpu`, the `worker` and `worker-gpu` compose services): a run on a
  provider API and a run on your GPU box go side by side; GPU runs wait for each other.
  **Stop** (on the runs list and the run page) ends a run after the episode it is on and
  keeps what finished. If the model's server can't be reached (a box dropping off the
  network), each call retries for 3 minutes, then the run fails with "stopped after N
  episode(s)": episodes that never reached the model aren't scored as failures
- **a UI** for queuing runs, watching progress, reading metrics and transcripts,
  comparing runs, and **usage** (tokens and estimated cost per day, model, member and
  workspace)
- **GPU boxes, datasets, fine-tuning and models** (Phases 2–3 in the browser): add a GPU
  box, check it, serve an open model with vLLM, build a dataset from a strong model's
  solutions, train a LoRA adapter on the box, test and serve it (see below)

```
browser ──► Next.js (web/, :3000) ──/api/*──► FastAPI (envforge/web, :8000) ──► Postgres
                                                     │
                                       worker (envforge-worker) ──► envforge.core ──► data/workspaces/<slug>/runs/
```

Next.js proxies every `/api/*` call to FastAPI (see `web/next.config.ts`), so the
session cookie is first-party and no CORS is needed. The UI never holds secrets.

## Configuration (`.env` at the repo root)

| variable | meaning |
|---|---|
| `DATABASE_URL` | `postgresql+psycopg://user:pw@host:5432/envforge` in prod; `sqlite:///./data/envforge.db` for local dev |
| `AUTH_PROVIDER` | `local` or `clerk` |
| `APP_SECRET_KEY` | ≥32 chars. Signs local sessions **and** derives the key that encrypts stored provider keys. Changing it makes stored keys unreadable. |
| `ALLOW_SIGNUP` | local auth: allow self-service sign-up (first user is always allowed and becomes admin) |
| `CLERK_PUBLISHABLE_KEY`, `CLERK_SECRET_KEY`, `CLERK_JWKS_URL`, `CLERK_ISSUER` | Clerk only |
| `ALLOW_SERVER_KEYS` | if `true`, workspaces without their own key fall back to the server's `.env` keys (a shared pool). Default `false`. |
| `CORS_ORIGINS` | only needed if the UI is served from a different origin without the proxy |
| `COOKIE_SECURE` | set `true` behind HTTPS |
| `BASE_URL_ALLOW_PRIVATE`, `BASE_URL_ALLOW_CIDRS` | where a vLLM base URL may point: public hosts only; **admin** accounts may also use the CIDRs (`100.64.0.0/10`, Tailscale, by default). Never loopback, the Docker network or metadata endpoints. Checked when the key is saved and again by the worker before use; redirects from the endpoint are not followed. |
| `MAX_ACTIVE_RUNS_PER_WORKSPACE` | queued + running runs per workspace (default 3) |
| `LOGIN_ATTEMPTS_PER_MINUTE`, `SIGNUPS_PER_HOUR` | in-memory rate limits per client IP (and per email for login); the API runs one process so they hold |
| `CLIENT_IP_HEADER` | the header the reverse proxy in front of the API sets with the client's IP (default `x-forwarded-for`; `cf-connecting-ip` behind Cloudflare). Only this header is read, so clients can't pick their own IP for the rate limits. |
| `STALE_RUN_SECONDS` | a running run whose worker heartbeat is older than this is marked failed (default 180) |
| `ENVFORGE_SANDBOX` | worker: `docker` to run sandbox packs (needs the Docker socket); `local` is refused for sandbox packs |

`web/.env.local`: `API_URL` (proxy target), `NEXT_PUBLIC_AUTH_PROVIDER` (must match the API),
and Clerk's public key when using Clerk.

## Auth providers

**local** — email + password. Passwords are argon2-hashed; sessions are HS256 JWTs in an
HttpOnly cookie. The first account created becomes admin. There is no password reset by
email in v1; an admin resets via the database or a future CLI command.

**clerk** — the Next.js app uses `@clerk/nextjs` for sign-in and sends Clerk's session JWT
as `Authorization: Bearer` on every API call. FastAPI verifies it against Clerk's JWKS
(RS256, issuer check) and creates a local `users` row on first sight, keyed by Clerk's user id.
Clerk's dashboard: *Configure → Sessions → Customize session token* and add `{"email": "{{user.primary_email_address}}"}`
so the API gets a real email; otherwise a placeholder is used.

Switching providers later is a config change for new users; existing local users keep
working only under `local`.

## Environment builder

Projects, revisions (an immutable DAG), uploads, chat messages and jobs live in the DB;
uploads (redacted) and task sets on disk. Chat turns and upload processing run in a
separate worker (`envforge-worker --kind builder`, the `builder` compose service) because
that's where provider keys are decrypted. Published environments are private to their
workspace and resolved by `builder.resolve_pack`. Endpoints live under
`/api/workspaces/{id}/builder/...` and `/api/workspaces/{id}/packs`. See docs/BUILDER.md.

## Keys belong to users

Each user has one set of provider keys (Profile → Provider keys), used in every workspace
they run in; a run always uses the keys of the user who queued it. Anthropic and OpenAI
keys are checked with the provider when saved (the API sees the plaintext only in that
request), and the provider's own model list is stored so the picker offers exactly the
models the key can use — the gateway catalog also lists models a provider has retired.

## Deleting runs

Deleting a finished run removes its trajectory files and hides it everywhere, but keeps
its database row so usage history (tokens, cost) stays accurate. Active runs must be
cancelled first. The UI asks for confirmation through one shared dialog
(`web/components/ConfirmDialog.tsx`, `useConfirm()`), used for every destructive action.

## Isolation between users

A workspace is visible only to its members, and there is no invite endpoint, so in
practice only to the user who created it. Every workspace route goes through
`services.require_workspace`, which looks up the caller's membership row and answers
**404** (not 403) otherwise, so other users' workspace ids can't even be probed. Runs are
fetched by `(workspace, run id)`, exports and usage are scoped the same way, and
trajectories live under `data/workspaces/<unique slug>/`. `tests/test_web_features.py`
checks every endpoint as a second user.

## Models, usage and cost

`GET /api/workspaces/{id}/models` returns the tool-capable models for the providers this
workspace has keys for. The list comes from the Vercel AI Gateway's public catalog
(`envforge/models/catalog.py`, cached 6 h in memory and on disk); native `anthropic:` and
`openai:` ids are derived from it (`anthropic/claude-sonnet-4.5` → `claude-sonnet-4-5`).
vLLM models are typed by hand (`vllm:<what your box serves>`).

The worker adds each episode's tokens and estimated cost to its run as it goes;
`/usage` endpoints aggregate them per day, model, member (workspace view) or workspace
(personal view). Cost = tokens × list price, ignoring caching discounts.

## Training data export

`GET /api/workspaces/{id}/runs/export?runs=a,b&min_reward=1.0&format=chat` streams the
episodes that scored at least `min_reward` as JSONL. `chat` is OpenAI messages + tool
schemas (what TRL's SFTTrainer takes); `trajectory` is the raw envforge format. In the UI,
"Build a training dataset" on a run, a selection of runs or the compare page opens the
**Datasets** page with those runs preselected instead (named, reproducible datasets with
a manifest; see below). The export endpoint stays for scripts.

## GPU boxes, datasets, training and models

What `envforge compute ...` and `envforge train ...` do on the command line, in the
browser, reusing the same code (`envforge/compute`, `envforge/train`). Logic lives in
`envforge/web/compute_service.py` and `training_service.py`; the routers are thin.

- **GPU boxes** (Settings → GPU boxes). A box row holds host, SSH port, user, runtime
  (docker / pod / auto), the vLLM port, an optional endpoint host / bind address, how the
  worker reaches vLLM (`direct`, or `tunnel` through an SSH port forward; picked from the
  host when left empty), an optional $/hour, and secrets, all Fernet-encrypted like provider
  keys: the SSH private key, the vLLM API key (generated when left empty, sent to vLLM on
  stdin) and an optional Hugging Face token (without one, the Hugging Face token saved
  under Provider keys by whoever starts the job is used). The token is written on the box (in its
  process environment or container config), so anyone with SSH to that box can read it: use
  a fine-grained, read-only token. The SSH key is either pasted for that box
  (validated, no passphrase) or, by default, the **workspace SSH key**: a key pair generated
  per workspace on first use, whose public half the page shows for adding once to
  RunPod / Vast. Responses show public keys and masked `SHA256:` fingerprints only. **Check**, **Serve** and **Stop** are jobs for the
  compute worker; a check stores GPU model, VRAM, driver/CUDA, free disk, vLLM presence,
  reachability and the time. **Remove** is a job too: it stops vLLM on the box, then
  forgets the box (row, key, pinned host key). If the box can't be reached the removal
  fails and the box stays, so a running vLLM is never orphaned silently; **Forget anyway**
  (`?force=true`) then removes it from envforge without touching the machine. A workspace
  with GPU boxes can't be deleted until they are removed.
- **Serving boxes in the model picker.** While a box serves, `GET .../models` lists
  `vllm:<model>` (and `vllm:<adapter>` for LoRA adapters) for that workspace. Queuing a run
  with one of them records `runs.gpu_box_id`; the worker then takes the endpoint and key
  from the box, so no personal vLLM key is needed.
- **Datasets.** Only completed runs on training seeds (`seed_start >= 10000`) can be used.
  `POST .../datasets/preview` builds the dataset in a scratch directory with
  `train.dataset.build_dataset` and returns counts, the train/val split, examples per task,
  and every evaluation run (seeds < 10000) that shares tasks with it (`overlap()`).
  For a published environment, tasks are compared by task record, and evaluation runs
  of every version of that environment are checked (see docs/BUILDER.md, held-out tasks).
  Saving writes `data/workspaces/<slug>/sft/<name>/` (train.jsonl, val.jsonl,
  manifest.json; customer-data environments re-sanitised like the export) and a `datasets`
  row with the source runs, filters, counts and manifest. `kind: "preference"` builds DPO
  pairs instead (`train.preference.build_preference_dataset`): a solved and a failed episode
  of the same task, split by task, with the same overlap check.
- **Training.** `POST .../training` creates a `training_jobs` row and a `train_start` job:
  the compute worker uploads the dataset and `sft.py` and starts training detached
  (`train.remote.start_training`). After that the worker *polls* each running job every
  `TRAIN_POLL_SECONDS` (`train.remote.poll_once`): step, epoch, losses, log tail. On exit 0
  it fetches the adapter to `data/workspaces/<slug>/adapters/<name>/` and creates an
  `adapters` row (base model, dataset, metrics from `envforge_manifest.json`). Cancel stops
  the job on the box at the next poll. A job is failed after `TRAIN_MAX_HOURS` or
  `TRAIN_MAX_POLL_FAILURES` unreachable polls in a row. `method` is `sft` (the default),
  `dpo` (a preference dataset, continuing from `init_adapter`: `dpo.py`, the starting
  adapter uploaded to the box if needed) or `grpo` (no dataset: an environment `pack` and
  `init_adapter`). A GRPO job is a loop kept in `training_jobs.state`: per round a
  `grpo_rollout` job (serve the current adapter, queue a rollout run with `runs.repeats`
  attempts per task at temperature > 0), the runs worker playing it, a `grpo_train` job
  (group advantages, stop vLLM, start `grpo_step.py`), then polls of that update; the
  last round's adapter is fetched and registered. Serving or testing on a box is refused
  (409) while it trains. Runs on vLLM models play `VLLM_RUN_CONCURRENCY` (8) episodes at
  once, written in episode order. See docs/TRAINING.md, "Beyond imitation".
- **Models.** Serve an adapter on any box (`--lora-modules`; uploaded from the local copy if
  that box doesn't have it), **Test** it (serve if needed, then an evaluation run on seeds
  0+ of its environment, linked to Compare with the teacher and the base model), download
  it as `.tar.gz`, and copy curl / openai-SDK snippets for its endpoint.

**Security.** A box belongs to one workspace and every lookup is by (workspace, id), so
other users get 404. Adding, checking, serving on, training on and deleting boxes or
adapters is admin-only while `GPU_BOXES_ADMIN_ONLY=true` (the default). Job payloads carry
ids only: the worker connects to the host stored on the box row and nowhere else. Model ids
and adapter names are validated by regex and every remote command is an argv list quoted
with `shlex`. The worker writes the SSH key to a mode-600 file in a private temp directory
for the duration of one connection (`IdentitiesOnly`, no agent, no ssh config proxy, a
per-box `known_hosts` under `data/ssh/`), never logs it, and caps every ssh/scp call at
`COMPUTE_SSH_TIMEOUT_S`. With `GPU_BOXES_ADMIN_ONLY=false`, boxes added by non-admins must
resolve to public addresses (private and Tailscale ranges stay admin-only), checked when
saved, before each connection, and on the endpoint URL.

**Worker.** `envforge-worker --kind compute` (the `compute` service in docker-compose)
claims `box_check`, `box_serve`, `box_stop`, `train_start` and `adapter_test` jobs and polls
running training jobs, so a slow box never blocks evaluation runs or the builder. It needs
`ssh`/`scp` (openssh-client is in the API image). Jobs heartbeat; a job whose worker died is
failed by `reap_stale_jobs` and its box/training job is marked failed, never left
"starting".

Settings: `GPU_BOXES_ADMIN_ONLY` (true), `COMPUTE_SSH_TIMEOUT_S` (1800),
`SERVE_HEALTH_TIMEOUT_S` (1800), `TRAIN_MAX_HOURS` (24), `TRAIN_POLL_SECONDS` (20),
`TRAIN_MAX_POLL_FAILURES` (15).

Tests: `tests/test_web_compute.py` runs all of it offline with `FakeGPUBox`
(`envforge/compute/fake.py`) in place of SSH.

## Administration

```bash
docker compose exec api envforge admin users [--count]
docker compose exec api envforge admin create-user someone@example.com [--name N] [--admin]
docker compose exec api envforge admin reset-password me@example.com   # also signs them out
docker compose exec api envforge admin sign-out someone@example.com
docker compose exec api envforge admin set-admin someone@example.com [--off]
```

Sessions carry an epoch that these bump. Signing out in the UI (`/logout`) bumps it too, so it
ends every session the user has rather than only clearing this browser's cookie: a copied
session token stops working at once.
`create-user` is how accounts are added when sign-up is off (`ALLOW_SIGNUP=false`, the
installer's default); it prompts for the password, or reads it from stdin with
`--password-stdin` so it never appears in argv or shell history.

## Keys and model resolution

1. A run names a model as `provider:model-id`.
2. The API refuses to queue it unless the user queuing it has a key for that provider
   (or `ALLOW_SERVER_KEYS=true`). `scripted` needs no key.
3. The worker decrypts the workspace key and passes it to `load_model(..., api_key=..., base_url=...)`.
   Keys never leave the worker process; the UI only ever sees `…last4`.
4. Only `vllm` keys carry a base URL (checked when saved and again before each run);
   hosted providers always use their own endpoint. If a key disappears after a run was
   queued, the run fails — it never falls back to the server's `.env` keys unless
   `ALLOW_SERVER_KEYS=true`.

## Running locally

```bash
# API + worker (SQLite, local auth)
cp .env.example .env && python -c "import secrets; print('APP_SECRET_KEY=' + secrets.token_urlsafe(48))" >> .env
uv sync --extra dev --extra web
uv run envforge-api            # http://localhost:8000/docs
uv run envforge-worker         # second terminal

# UI
cd web && cp .env.example .env.local && npm install && npm run dev   # http://localhost:3000
```

With Postgres instead of SQLite:

```bash
docker compose up -d postgres
echo 'DATABASE_URL=postgresql+psycopg://envforge:envforge@localhost:5432/envforge' >> .env
uv run alembic upgrade head
```

## Deploying (one machine)

```bash
git clone https://github.com/noctisgit/envforge && cd envforge
./install.sh            # or: ./install.sh --yes --admin-email … --admin-password-file …
```

`install.sh` checks Docker/Compose/disk/memory, writes `.env` with random `APP_SECRET_KEY`
and `POSTGRES_PASSWORD` (an existing `.env` is kept), builds and starts the stack
(postgres, api — which runs migrations — worker, builder, web), waits for `/api/health`,
and creates the first admin with `envforge admin create-user`, so the server can't be
claimed by whoever signs up first. Re-running it is the update path (`git pull &&
./install.sh`). `./install.sh --help` lists the flags.

The UI listens on `WEB_LISTEN:WEB_PORT` (default `127.0.0.1:3000`). TLS and a domain are
up to you: nginx, Caddy (`deploy/Caddyfile` is an example), a Cloudflare Tunnel or a port
forward in front of it. Behind HTTPS, set `COOKIE_SECURE=true` (`./install.sh --url
https://…` does). Behind Cloudflare (a Tunnel, or proxied DNS), set
`CLIENT_IP_HEADER=cf-connecting-ip` so the rate limits see the visitor rather than
Cloudflare's edge. `API_PORT` / `POSTGRES_PORT` are published on 127.0.0.1 only, for
development; the installer moves them if the defaults are taken.

Trajectories live in the `envdata` volume under `/data/workspaces/<slug>/runs/`; the DB holds
users, workspaces, encrypted keys, and run status/metrics/usage. Back up both:

```bash
docker compose exec -T postgres pg_dump -U envforge envforge > backup-$(date +%F).sql
```

No long-running container runs as root: api, worker, builder and compute run as uid 10001
(`envforge`), web as `node`, and they get `no-new-privileges`. The one-off `data-perms`
service (root) hands the `envdata` volume to uid 10001 on `up`; it only chowns when some file
isn't owned by it yet, such as on the first start after upgrading from root containers.

The worker mounts `/var/run/docker.sock` so sandbox packs can start sibling containers. It
reaches the socket through the socket's group (`DOCKER_GID` in `.env`, set by `install.sh`
from `stat -c %g /var/run/docker.sock`). Socket access is still root-equivalent on the host;
drop the mount to run sim packs only.
Scale workers with `docker compose up -d --scale worker=3` (claiming is race-free).

## Schema changes

Edit `envforge/web/models.py`, then:

```bash
uv run alembic revision --autogenerate -m "describe change"
uv run alembic upgrade head
```

Commit the generated file under `envforge/web/migrations/versions/`.

## Endpoints

| method | path | purpose |
|---|---|---|
| GET | `/api/auth/info` | provider, sign-up policy, version (public) |
| POST | `/api/auth/signup`, `/login`, `/logout` | local auth; `/logout` signs out every device |
| GET | `/api/auth/me` | current user |
| GET | `/api/packs` | available environment packs |
| GET/POST | `/api/workspaces` | list / create |
| GET/PATCH | `/api/workspaces/{id}` | detail / defaults and `monthly_spend_limit_usd` (owner; `null` = no limit) |
| POST | `/api/workspaces/{id}/builder/projects/{p}/extraction/approve` | approve the builder's cost estimate `{quote_id}` (only this approves extraction beyond 5 per turn) |
| DELETE | `/api/workspaces/{id}/builder/projects/{p}/extraction/quote` | dismiss the estimate |
| GET/PUT/DELETE | `/api/me/keys[/{provider}]` | your provider keys (checked on save) |
| PATCH | `/api/me` | your display name |
| POST | `/api/me/password` | change password (signs out other sessions) |
| GET | `/api/packs/{name}` | environment description, dimensions, sample diagram |
| GET | `/api/packs/{name}/preview?seed=&tier=` | one generated task + reference solution |
| DELETE | `/api/workspaces/{id}/runs/{run}` | delete a finished run's trajectories (usage kept) |
| DELETE | `/api/workspaces/{id}/packs/{name}` | delete a builder-published environment (past runs still replay) |
| GET | `/api/workspaces/{id}/runs/{run}/episodes/{n}/replay` | diagram after each tool call (sim packs) |
| GET/POST | `/api/workspaces/{id}/runs` | list / enqueue |
| GET | `/api/workspaces/{id}/runs/{run}` | status + metrics |
| POST | `/api/workspaces/{id}/runs/{run}/cancel` | stop between episodes |
| GET | `/api/workspaces/{id}/runs/{run}/episodes[/{n}]` | summaries / full transcript |
| GET | `/api/workspaces/{id}/runs/export?runs=…&min_reward=…&format=chat\|trajectory` | filtered episodes as JSONL |
| GET | `/api/workspaces/{id}/models[?refresh=true]` | models you can pick (from your keys) |
| GET | `/api/workspaces/{id}/usage?days=30` | workspace usage by day, model, member |
| GET | `/api/auth/me/usage?days=30` | your usage across your workspaces |
| GET/POST | `/api/workspaces/{id}/gpu-boxes` | list (keys masked, `can_manage`) / add a GPU box (admin) |
| GET | `/api/workspaces/{id}/gpu-boxes/ssh-key` | the workspace's public SSH key (created on first ask by someone who may manage boxes; `null` until then) |
| POST | `/api/workspaces/{id}/gpu-boxes/ssh-key/rotate` | replace the workspace key pair (admin) |
| POST | `/api/workspaces/{id}/gpu-boxes/{box}/vllm-key` | set the key vLLM requires `{key}` (admin; only while vLLM is stopped; never returned) |
| DELETE | `/api/workspaces/{id}/gpu-boxes/{box}` | stop vLLM on the box, then forget it (202, a job); `?force=true` forgets it now without touching the machine (204) (admin) |
| POST | `/api/workspaces/{id}/gpu-boxes/{box}/check` | queue a readiness check (admin) |
| POST | `/api/workspaces/{id}/gpu-boxes/{box}/serve` | queue vLLM serving `{model, adapters?, max_model_len?}` (admin) |
| POST | `/api/workspaces/{id}/gpu-boxes/{box}/stop` | queue stopping vLLM (admin) |
| GET | `/api/workspaces/{id}/datasets/sources` | completed runs, eligible (training seeds) or why not |
| POST | `/api/workspaces/{id}/datasets/preview` | counts, split, examples per task (or pairs, `kind: "preference"`), eval-seed overlap |
| GET/POST | `/api/workspaces/{id}/datasets` | list / save a named dataset |
| GET/DELETE | `/api/workspaces/{id}/datasets/{ds}` | detail with manifest / delete files + row |
| GET | `/api/workspaces/{id}/datasets/{ds}/download?split=train\|val` | the JSONL file |
| GET/POST | `/api/workspaces/{id}/training` | jobs + defaults per method / start an SFT, DPO or GRPO job (admin) |
| GET | `/api/workspaces/{id}/training/{job}` | status, step/epoch, losses, log tail, elapsed, est. cost |
| POST | `/api/workspaces/{id}/training/{job}/cancel` | cancel (admin) |
| GET | `/api/workspaces/{id}/adapters[/{a}]` | trained adapters with lineage and where they're served |
| POST | `/api/workspaces/{id}/adapters/{a}/serve` | serve on a box `{box_id}` (admin) |
| POST | `/api/workspaces/{id}/adapters/{a}/test` | serve if needed + eval run on seeds 0+ `{box_id, episodes}` (admin) |
| GET | `/api/workspaces/{id}/adapters/{a}/download` | the adapter as .tar.gz |
| DELETE | `/api/workspaces/{id}/adapters/{a}` | delete the local copy (admin) |
| POST | `/api/auth/logout-all` | invalidate all your sessions |

Interactive docs: `/docs` on the API port.

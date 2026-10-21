# Web layer: API, worker, UI

The web layer is optional. The CLI and library work without it. It adds:

- **multi-user**: accounts, each with private workspaces (no sharing between users)
- **per-user provider keys**, encrypted at rest, checked with the provider when saved,
  and used by every workspace that user runs in
- **per-workspace defaults**: model, pack, episode count
- **a model picker** listing only models your keys can use, with prices
- **an Environments explorer**: each pack's world diagram, dimensions, tools, tiers, reward
  rules and a live task preview, generated from the pack (see docs/PACKS.md)
- **a run queue** executed by a separate worker process
- **a UI** for queuing runs, watching progress, reading metrics and transcripts,
  comparing runs, exporting training data, and **usage** (tokens and estimated cost per
  day, model, member and workspace)

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
| `BASE_URL_ALLOW_PRIVATE`, `BASE_URL_ALLOW_CIDRS` | where a vLLM base URL may point: public hosts plus `100.64.0.0/10` (Tailscale) by default; never loopback, the Docker network or metadata endpoints |
| `MAX_ACTIVE_RUNS_PER_WORKSPACE` | queued + running runs per workspace (default 3) |
| `LOGIN_ATTEMPTS_PER_MINUTE`, `SIGNUPS_PER_HOUR` | in-memory rate limits per client IP (and per email for login); the API runs one process so they hold |
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
schemas (what TRL's SFTTrainer takes); `trajectory` is the raw envforge format. The UI
offers it on a run, on a selection of runs, and on the compare page.

## Administration

```bash
docker compose exec api envforge admin users [--count]
docker compose exec api envforge admin create-user someone@example.com [--name N] [--admin]
docker compose exec api envforge admin reset-password me@example.com   # also signs them out
docker compose exec api envforge admin sign-out someone@example.com
docker compose exec api envforge admin set-admin someone@example.com [--off]
```

Users can also "Sign out everywhere" from Settings. Sessions carry an epoch that these bump.
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
https://…` does). `API_PORT` / `POSTGRES_PORT` are published on 127.0.0.1 only, for
development; the installer moves them if the defaults are taken.

Trajectories live in the `envdata` volume under `/data/workspaces/<slug>/runs/`; the DB holds
users, workspaces, encrypted keys, and run status/metrics/usage. Back up both:

```bash
docker compose exec -T postgres pg_dump -U envforge envforge > backup-$(date +%F).sql
```

The worker mounts `/var/run/docker.sock` so sandbox packs can start sibling containers.
That makes the worker root-equivalent on the host; drop the mount to run sim packs only.
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
| POST | `/api/auth/signup`, `/login`, `/logout` | local auth |
| GET | `/api/auth/me` | current user |
| GET | `/api/packs` | available environment packs |
| GET/POST | `/api/workspaces` | list / create |
| GET/PATCH | `/api/workspaces/{id}` | detail / defaults (owner) |
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
| POST | `/api/auth/logout-all` | invalidate all your sessions |

Interactive docs: `/docs` on the API port.

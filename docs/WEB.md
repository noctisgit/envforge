# Web layer: API, worker, UI

The web layer is optional. The CLI and library work without it. It adds:

- **multi-user**: accounts, workspaces with owner/member roles
- **per-workspace provider keys**, encrypted at rest, used by that workspace's runs
- **per-workspace defaults**: model, pack, episode count
- **a run queue** executed by a separate worker process
- **a UI** for queuing runs, watching progress, reading metrics and transcripts

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

## Keys and model resolution

1. A run names a model as `provider:model-id`.
2. The API refuses to queue it unless the workspace has a key for that provider
   (or `ALLOW_SERVER_KEYS=true`). `scripted` needs no key.
3. The worker decrypts the workspace key and passes it to `load_model(..., api_key=..., base_url=...)`.
   Keys never leave the worker process; the UI only ever sees `…last4`.

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

## Deploying on one VPS (Ubuntu 24.04)

```bash
# docker + compose plugin, then:
git clone git@github.com:noctisgit/envforge.git && cd envforge
cp .env.example .env            # set APP_SECRET_KEY, POSTGRES_PASSWORD, AUTH_PROVIDER, COOKIE_SECURE=true
docker compose up -d --build    # postgres, api (runs migrations), worker, web
# put Caddy (deploy/Caddyfile) or nginx in front of :3000 for TLS
```

Trajectories live in the `envdata` volume under `/data/workspaces/<slug>/runs/`; the DB holds
users, workspaces, encrypted keys, and run status/metrics. Back up both.

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
| PUT/DELETE | `/api/workspaces/{id}/keys[/{provider}]` | provider keys (owner) |
| GET/POST | `/api/workspaces/{id}/runs` | list / enqueue |
| GET | `/api/workspaces/{id}/runs/{run}` | status + metrics |
| POST | `/api/workspaces/{id}/runs/{run}/cancel` | stop between episodes |
| GET | `/api/workspaces/{id}/runs/{run}/episodes[/{n}]` | summaries / full transcript |

Interactive docs: `/docs` on the API port.

#!/usr/bin/env bash
# envforge installer: from `git clone` to a running, locked-down server on localhost:3000.
#
#   ./install.sh                                  # interactive
#   ./install.sh --yes --admin-email me@example.com --admin-password-file ./pw.txt
#   ./install.sh                                  # again later: update (keeps .env and data)
#
# What it does:
#   1. checks Docker + Docker Compose v2, disk space and memory
#   2. creates .env from .env.example with strong random secrets (an existing .env is kept;
#      only missing secrets are filled in)
#   3. builds and starts the stack: postgres, api (runs database migrations), worker,
#      builder, web
#   4. waits until the app answers on http://<listen>:<port>
#   5. creates the first admin account from the terminal (no race to be the first to sign
#      up) and, unless you opt in, turns public sign-up off
#
# It does not set up HTTPS or a domain: put your own reverse proxy (nginx, Caddy,
# cloudflared, a port forward...) in front of localhost:3000. See README "Self-hosting".
set -euo pipefail

cd "$(dirname "$0")"

# ---------------------------------------------------------------- options
YES=0
ADMIN_EMAIL="${ENVFORGE_ADMIN_EMAIL:-}"
ADMIN_NAME="${ENVFORGE_ADMIN_NAME:-}"
ADMIN_PASSWORD_FILE=""
PUBLIC_URL=""
LISTEN=""
PORT=""
SIGNUP=""
NO_BUILD=0

usage() {
    cat <<'EOF'
Usage: ./install.sh [options]

  -y, --yes                     no questions: use flags/defaults (needs --admin-email and
                                --admin-password-file on a fresh install)
      --admin-email EMAIL       first admin account
      --admin-name NAME         its display name
      --admin-password-file F   read the admin password from file F (first line)
      --url URL                 public URL you will serve it at, e.g. https://envforge.example.com
                                (sets secure cookies when it is https)
      --listen ADDR             host address for the UI (default 127.0.0.1; 0.0.0.0 = whole network)
      --port PORT               host port for the UI (default 3000)
      --allow-signup            let anyone create an account (default: admin creates accounts)
      --no-build                start the existing images instead of rebuilding
  -h, --help                    this help
EOF
}

while [ $# -gt 0 ]; do
    case "$1" in
        -y | --yes) YES=1 ;;
        --admin-email) ADMIN_EMAIL="${2:?}"; shift ;;
        --admin-name) ADMIN_NAME="${2:?}"; shift ;;
        --admin-password-file) ADMIN_PASSWORD_FILE="${2:?}"; shift ;;
        --url) PUBLIC_URL="${2:?}"; shift ;;
        --listen) LISTEN="${2:?}"; shift ;;
        --port) PORT="${2:?}"; shift ;;
        --allow-signup) SIGNUP=true ;;
        --no-build) NO_BUILD=1 ;;
        -h | --help) usage; exit 0 ;;
        *) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
    shift
done

# ---------------------------------------------------------------- output helpers
if [ -t 1 ]; then B=$'\033[1m'; G=$'\033[32m'; Y=$'\033[33m'; R=$'\033[31m'; N=$'\033[0m'; else B='' G='' Y='' R='' N=''; fi
step() { printf '\n%s==> %s%s\n' "$B" "$*" "$N"; }
ok() { printf '  %s✓%s %s\n' "$G" "$N" "$*"; }
warn() { printf '  %s!%s %s\n' "$Y" "$N" "$*"; }
die() { printf '\n%serror:%s %s\n' "$R" "$N" "$*" >&2; exit 1; }

ask() { # ask "question" default -> answer (default when --yes or no terminal)
    local q="$1" def="${2:-}" a
    if [ "$YES" = 1 ] || [ ! -t 0 ]; then printf '%s' "$def"; return; fi
    read -r -p "  $q${def:+ [$def]}: " a </dev/tty || true
    printf '%s' "${a:-$def}"
}

ask_yn() { # ask_yn "question" y|n -> exit status
    local a
    a="$(ask "$1 (y/n)" "$2")"
    case "$a" in y | Y | yes | YES) return 0 ;; *) return 1 ;; esac
}

random_secret() { # URL-safe, $1 characters
    LC_ALL=C tr -dc 'A-Za-z0-9_-' </dev/urandom | head -c "$1"
}

env_get() { # env_get KEY -> value from .env (empty if unset)
    [ -f .env ] || return 0
    awk -F= -v k="$1" '$1 == k { sub(/^[^=]*=/, ""); sub(/[ \t]+#.*$/, ""); v = $0 } END { print v }' .env
}

env_set() { # env_set KEY VALUE: replace the line or append it (portable: no sed -i)
    local tmp
    tmp="$(mktemp)"
    awk -v k="$1" -v v="$2" '
        BEGIN { done = 0 }
        $0 ~ "^"k"=" { if (!done) print k"="v; done = 1; next }
        { print }
        END { if (!done) print k"="v }
    ' .env >"$tmp"
    cat "$tmp" >.env
    rm -f "$tmp"
}

# ---------------------------------------------------------------- 1. prerequisites
step "Checking prerequisites"

command -v docker >/dev/null 2>&1 || die "Docker is not installed. Get it from https://docs.docker.com/engine/install/ (Linux) or Docker Desktop (macOS/Windows), then re-run."
docker info >/dev/null 2>&1 || die "Docker is installed but not reachable. Start Docker (Linux: sudo systemctl start docker) and make sure your user may use it (sudo usermod -aG docker \$USER, then log in again), or run this script with sudo."
docker compose version >/dev/null 2>&1 || die "Docker Compose v2 is missing ('docker compose'). Install the docker-compose-plugin package or update Docker Desktop."
ok "Docker $(docker version --format '{{.Server.Version}}' 2>/dev/null) and $(docker compose version --short 2>/dev/null | sed 's/^v//;s/^/Compose /')"

free_gb=$(df -Pk . | awk 'NR == 2 { printf "%d", $4 / 1048576 }')
if [ "${free_gb:-0}" -lt 15 ]; then
    warn "only ${free_gb} GB free here; the images and data need about 15 GB (more for big datasets)"
    ask_yn "Continue anyway?" n || exit 1
else
    ok "${free_gb} GB free disk"
fi

if [ -r /proc/meminfo ]; then
    mem_gb=$(awk '/MemTotal/ { printf "%d", $2 / 1048576 }' /proc/meminfo)
    if [ "${mem_gb:-0}" -lt 4 ]; then
        warn "${mem_gb} GB RAM: 4 GB or more is recommended (8 GB+ for the builder's PII model)"
    else
        ok "${mem_gb} GB RAM"
    fi
fi

# ---------------------------------------------------------------- 2. configuration
step "Configuring (.env)"

fresh=0
if [ ! -f .env ]; then
    cp .env.example .env
    chmod 600 .env
    fresh=1
    ok "created .env from .env.example"
else
    ok "keeping your existing .env (only missing settings are added)"
fi

# Secrets. APP_SECRET_KEY signs sessions and encrypts stored provider keys: it must never
# change once set. POSTGRES_PASSWORD is fixed when the database volume is first created.
if [ -z "$(env_get APP_SECRET_KEY)" ]; then
    env_set APP_SECRET_KEY "$(random_secret 64)"
    ok "generated APP_SECRET_KEY (keep .env safe: it also decrypts saved API keys)"
fi
db_volume_exists=0
if docker volume ls --format '{{.Name}}' | grep -qx "$(basename "$PWD" | tr '[:upper:]' '[:lower:]' | tr -cd 'a-z0-9_-')_pgdata"; then
    db_volume_exists=1
fi
if [ -z "$(env_get POSTGRES_PASSWORD)" ] || { [ "$(env_get POSTGRES_PASSWORD)" = "change-me" ] && [ "$db_volume_exists" = 0 ]; }; then
    if [ "$db_volume_exists" = 1 ]; then
        warn "POSTGRES_PASSWORD is empty but a database already exists; leaving it (the database keeps its original password)"
    else
        env_set POSTGRES_PASSWORD "$(random_secret 32)"
        ok "generated POSTGRES_PASSWORD"
    fi
fi

# Where the UI listens.
flags_given="$LISTEN$PORT"
[ -n "$LISTEN" ] || LISTEN="$(env_get WEB_LISTEN)"
[ -n "$PORT" ] || PORT="$(env_get WEB_PORT)"
if [ "$fresh" = 1 ] && [ -z "$flags_given" ] && [ "$YES" = 0 ] && [ -t 0 ]; then
    PORT="$(ask "Port for the web UI" "${PORT:-3000}")"
    if ask_yn "Reachable only from this machine (recommended; put a proxy in front)?" y; then
        LISTEN=127.0.0.1
    else
        LISTEN=0.0.0.0
    fi
fi
LISTEN="${LISTEN:-127.0.0.1}"
PORT="${PORT:-3000}"
case "$PORT" in '' | *[!0-9]*) die "--port must be a number" ;; esac
env_set WEB_LISTEN "$LISTEN"
env_set WEB_PORT "$PORT"

# Public URL (optional): https -> secure cookies; always allowed as an origin.
[ -n "$PUBLIC_URL" ] || { [ "$fresh" = 1 ] && PUBLIC_URL="$(ask "Public URL you'll serve it at, if any (e.g. https://envforge.example.com)" "")"; } || true
PUBLIC_URL="${PUBLIC_URL%/}"
origins="http://localhost:${PORT},http://127.0.0.1:${PORT}"
if [ -n "$PUBLIC_URL" ]; then
    case "$PUBLIC_URL" in http://* | https://*) ;; *) die "--url must start with http:// or https://" ;; esac
    origins="${origins},${PUBLIC_URL}"
    case "$PUBLIC_URL" in https://*) env_set COOKIE_SECURE true ;; esac
fi
[ -n "$(env_get CORS_ORIGINS)" ] && [ -z "$PUBLIC_URL" ] || env_set CORS_ORIGINS "$origins"

# Sign-up policy (asked on a fresh install; flag wins).
if [ -n "$SIGNUP" ]; then
    env_set ALLOW_SIGNUP true
elif [ "$fresh" = 1 ]; then
    if ask_yn "Let anyone who can reach the site create an account?" n; then
        env_set ALLOW_SIGNUP true
    else
        env_set ALLOW_SIGNUP false
    fi
fi
ok "UI on ${LISTEN}:${PORT}; sign-up $( [ "$(env_get ALLOW_SIGNUP)" = true ] && echo open || echo "by admin only")"

# Host ports: the API and Postgres are published on 127.0.0.1 for development. If another
# program already uses a port, move ours rather than fail.
port_busy() { (exec 3<>"/dev/tcp/127.0.0.1/$1") 2>/dev/null; }
ours_running="$(docker compose ps --services --status running 2>/dev/null || true)"
pick_port() { # pick_port KEY default service
    local key="$1" p service="$3"
    p="$(env_get "$key")"
    p="${p:-$2}"
    if ! grep -qx "$service" <<<"$ours_running"; then
        while port_busy "$p"; do p=$((p + 1)); done
    fi
    [ "$p" = "$2" ] && [ -z "$(env_get "$key")" ] || env_set "$key" "$p"
    printf '%s' "$p"
}
if ! grep -qx web <<<"$ours_running" && port_busy "$PORT"; then
    die "port $PORT is already in use by another program: re-run with --port <free port>"
fi
api_port="$(pick_port API_PORT 8000 api)"
pg_port="$(pick_port POSTGRES_PORT 5432 postgres)"
ok "host ports: UI ${PORT}, API ${api_port} and Postgres ${pg_port} (127.0.0.1 only)"

# ---------------------------------------------------------------- 3. start
step "Building and starting the stack (the first build takes several minutes)"

compose() { docker compose "$@"; }
if [ "$NO_BUILD" = 1 ]; then
    compose up -d
else
    compose up -d --build
fi
ok "containers started: $(compose ps --services --status running | tr '\n' ' ')"

# ---------------------------------------------------------------- 4. wait for health
step "Waiting for the app"
url="http://127.0.0.1:${PORT}"
[ "$LISTEN" = 127.0.0.1 ] || [ "$LISTEN" = 0.0.0.0 ] || url="http://${LISTEN}:${PORT}"
healthy=0
for _ in $(seq 1 120); do
    if curl -fsS "$url/api/health" >/dev/null 2>&1; then healthy=1; break; fi
    sleep 2
done
if [ "$healthy" = 0 ]; then
    compose ps
    die "the app did not answer at $url/api/health within 4 minutes. Logs: docker compose logs --tail 50 api web"
fi
ok "the app answers at $url (database migrations applied)"

# ---------------------------------------------------------------- 5. first admin
step "Admin account"
n_users="$(compose exec -T api envforge admin users --count 2>/dev/null | tr -dc '0-9')"
if [ "${n_users:-0}" -gt 0 ]; then
    ok "${n_users} account(s) already exist; skipping (add more: docker compose exec api envforge admin create-user EMAIL)"
else
    auth="$(env_get AUTH_PROVIDER)"
    if [ "${auth:-local}" != local ]; then
        warn "AUTH_PROVIDER=$auth: accounts come from your identity provider; the first to sign in becomes admin"
    else
        [ -n "$ADMIN_EMAIL" ] || ADMIN_EMAIL="$(ask "Admin email" "")"
        [ -n "$ADMIN_EMAIL" ] || die "an admin email is needed (--admin-email)"
        [ -n "$ADMIN_NAME" ] || ADMIN_NAME="$(ask "Admin display name" "${ADMIN_EMAIL%%@*}")"
        if [ -n "$ADMIN_PASSWORD_FILE" ]; then
            [ -r "$ADMIN_PASSWORD_FILE" ] || die "cannot read $ADMIN_PASSWORD_FILE"
            pw="$(head -n1 "$ADMIN_PASSWORD_FILE")"
        elif [ "$YES" = 1 ] || [ ! -t 0 ]; then
            die "no terminal to ask for a password: pass --admin-password-file"
        else
            while :; do
                read -r -s -p "  Admin password (10+ characters): " pw </dev/tty; echo
                read -r -s -p "  Again: " pw2 </dev/tty; echo
                [ "$pw" = "$pw2" ] || { warn "they don't match"; continue; }
                [ "${#pw}" -ge 10 ] || { warn "too short"; continue; }
                break
            done
        fi
        # the password travels on stdin: never in argv, the environment or shell history
        printf '%s\n' "$pw" | compose exec -T api envforge admin create-user "$ADMIN_EMAIL" \
            --name "$ADMIN_NAME" --admin --password-stdin
        unset pw pw2
        ok "admin account ready"
    fi
fi

# ---------------------------------------------------------------- done
shown="http://localhost:${PORT}"
[ "$LISTEN" = 127.0.0.1 ] || shown="http://<this machine's address>:${PORT}"
cat <<EOF

${G}${B}envforge is running.${N}

  Open        ${B}${shown}${N}${PUBLIC_URL:+   (public: ${PUBLIC_URL} once your proxy points at it)}
  Sign in     with the admin account, then Profile → Provider keys: add an Anthropic,
              OpenAI or Vercel AI Gateway key (or a vLLM URL) and start a run.
  Expose it   point nginx / Caddy / cloudflared / a port forward at ${LISTEN}:${PORT}
              (served over https? re-run with --url https://your.domain for secure cookies)

  Update      git pull && ./install.sh          (keeps .env and all data; runs migrations)
  Stop/start  docker compose stop  |  docker compose start
  Logs        docker compose logs -f api worker builder web
  Users       docker compose exec api envforge admin users | create-user EMAIL | reset-password EMAIL
  Back up     docker compose exec -T postgres pg_dump -U envforge envforge | gzip > envforge.sql.gz
              plus the 'envdata' volume (runs, datasets) and your .env (it decrypts saved keys)
EOF

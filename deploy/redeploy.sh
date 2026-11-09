#!/usr/bin/env bash
# Move an envforge server, with all its data, to a new machine (or back up the current one).
#
#   old server:  deploy/redeploy.sh backup [--stop] [--with-tunnel] [--no-cache] [--out DIR]
#                -> DIR/envforge-backup-<host>-<stamp>.tar   (contains secrets: keep it private)
#
#   new server:  git clone https://github.com/noctisgit/envforge && cd envforge
#                sudo deploy/redeploy.sh restore envforge-backup-....tar [--with-tunnel] [--url URL]
#
# What git does not carry, and the backup does:
#   .env              APP_SECRET_KEY decrypts saved provider keys and the workspace SSH key;
#                     a new key would make them unreadable
#   the database      users, workspaces, keys, runs, environments, datasets, adapters (pg_dump)
#   the data volume   run trajectories, adapters, uploads, the builder's PII model cache
#   tunnel token      only with --with-tunnel: the Cloudflare Tunnel that serves the public URL
#
# `restore` installs Docker if it's missing, rebuilds the images from this checkout, loads the
# database and the data volume, then hands over to ./install.sh, which applies any newer
# migrations and starts the stack. It refuses to run over an existing envforge database.
set -euo pipefail

cd "$(dirname "$0")/.."

if [ -t 1 ]; then B=$'\033[1m'; G=$'\033[32m'; Y=$'\033[33m'; R=$'\033[31m'; N=$'\033[0m'; else B='' G='' Y='' R='' N=''; fi
step() { printf '\n%s==> %s%s\n' "$B" "$*" "$N"; }
ok() { printf '  %s✓%s %s\n' "$G" "$N" "$*"; }
warn() { printf '  %s!%s %s\n' "$Y" "$N" "$*"; }
die() { printf '\n%serror:%s %s\n' "$R" "$N" "$*" >&2; exit 1; }

usage() {
    cat <<'EOF'
Usage:
  deploy/redeploy.sh backup  [options]          on the old server
  deploy/redeploy.sh restore BACKUP.tar [options]   on the new server, inside a fresh clone

backup options:
  --out DIR        where to write the backup (default: ../envforge-backups)
  --stop           stop the app (everything but Postgres) first and leave it stopped:
                   use this when moving, so nothing changes after the backup is taken
  --with-tunnel    include the Cloudflare Tunnel token (/etc/cloudflared/token)
  --no-cache       leave out re-downloadable caches (the PII model, ~1 GB)
  --force          back up even while runs or training jobs are active

restore options:
  --with-tunnel    install cloudflared and start the tunnel from the backup's token
  --url URL        the public URL, if it changed (passed to install.sh)
  --port PORT      host port for the UI (passed to install.sh; default: as in the backup)
  -y, --yes        don't ask for confirmation
EOF
}

compose() { docker compose "$@"; }

WORK=""  # the backup's scratch directory, removed on exit
trap '[ -z "$WORK" ] || rm -rf "$WORK"' EXIT
psql_q() { compose exec -T postgres psql -U envforge -d envforge -Atc "$1"; }

env_get() { # env_get FILE KEY -> value (empty if unset)
    [ -f "$1" ] || return 0
    awk -F= -v k="$2" '$1 == k { sub(/^[^=]*=/, ""); sub(/[ \t]+#.*$/, ""); v = $0 } END { print v }' "$1"
}

env_set() { # env_set KEY VALUE in ./.env
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

wait_for_postgres() {
    for _ in $(seq 1 60); do
        compose exec -T postgres pg_isready -U envforge -d envforge >/dev/null 2>&1 && return 0
        sleep 2
    done
    die "Postgres did not become ready. Logs: docker compose logs --tail 50 postgres"
}

# ================================================================== backup
backup() {
    local out="../envforge-backups" stop=0 tunnel=0 slim=0 force=0
    while [ $# -gt 0 ]; do
        case "$1" in
            --out) out="${2:?}"; shift ;;
            --stop) stop=1 ;;
            --with-tunnel) tunnel=1 ;;
            --no-cache) slim=1 ;;
            --force) force=1 ;;
            -h | --help) usage; exit 0 ;;
            *) die "unknown option for backup: $1" ;;
        esac
        shift
    done

    step "Checking this server"
    [ -f .env ] || die "no .env here: run this in the envforge checkout the server runs from"
    compose ps --services --status running | grep -qx postgres || die "Postgres isn't running (docker compose up -d postgres)"
    if [ "$tunnel" = 1 ] && [ ! -r /etc/cloudflared/token ]; then
        die "--with-tunnel: can't read /etc/cloudflared/token (run with sudo, or leave the flag out)"
    fi

    # Work in progress writes to the database and the data volume while we copy them.
    local active
    active="$(psql_q "select
        (select count(*) from runs where status in ('queued','running') and deleted_at is null)
      + (select count(*) from training_jobs where status in ('queued','starting','running','fetching'))
      + (select count(*) from jobs where status in ('queued','running'))")"
    if [ "${active:-0}" -gt 0 ]; then
        [ "$force" = 1 ] || die "$active run(s) or job(s) are active. Wait for them to finish (or stop them), or pass --force"
        warn "$active run(s) or job(s) active: the backup may catch them half-written"
    else
        ok "no runs or jobs in progress"
    fi

    if [ "$stop" = 1 ]; then
        compose stop api web worker worker-gpu builder compute
        ok "stopped the app (Postgres stays up for the dump). Start again: docker compose start"
    fi

    umask 077
    mkdir -p "$out"
    local stamp name
    stamp="$(date +%Y%m%d-%H%M%S)"
    name="envforge-backup-$(hostname -s)-${stamp}"
    WORK="$(mktemp -d "$out/.${name}.XXXX")"

    step "Database"
    compose exec -T postgres pg_dump -U envforge -d envforge -Fc >"$WORK/db.dump"
    ok "pg_dump: $(du -h "$WORK/db.dump" | cut -f1)"

    step "Data volume (runs, adapters, uploads)"
    local excludes=""
    [ "$slim" = 1 ] && excludes="--exclude=./hf-cache --exclude=./cache"
    # data-perms is the stack's one root container with the volume mounted at /data
    compose run --rm --no-deps -T data-perms sh -c "tar -C /data $excludes -czf - ." >"$WORK/envdata.tar.gz"
    ok "envdata: $(du -h "$WORK/envdata.tar.gz" | cut -f1)"

    step "Settings"
    cp .env "$WORK/env"
    ok ".env"
    if [ "$tunnel" = 1 ]; then
        cp /etc/cloudflared/token "$WORK/cloudflared-token"
        ok "Cloudflare Tunnel token"
    fi
    cat >"$WORK/manifest.txt" <<EOF
envforge backup
created      $(date -u +%Y-%m-%dT%H:%M:%SZ)
host         $(hostname)
git commit   $(git rev-parse --short HEAD 2>/dev/null || echo unknown)
migration    $(psql_q "select version_num from alembic_version")
users        $(psql_q "select count(*) from users")
runs         $(psql_q "select count(*) from runs where deleted_at is null")
adapters     $(psql_q "select count(*) from adapters")
caches       $([ "$slim" = 1 ] && echo "left out (re-downloaded on first use)" || echo included)
tunnel token $([ "$tunnel" = 1 ] && echo included || echo "not included")
EOF

    tar -C "$WORK" -cf "$out/$name.tar" .
    chmod 600 "$out/$name.tar"
    local file
    file="$(cd "$out" && pwd)/$name.tar"
    sha256sum "$file" | cut -d' ' -f1 >"$file.sha256"

    cat <<EOF

${G}${B}Backup written.${N}

  ${B}${file}${N}  ($(du -h "$file" | cut -f1))
  It holds .env and the database: anyone with it can decrypt your saved API keys. Keep it
  private, and delete copies you no longer need.

  Move to a new server:
    scp ${file} root@NEW-SERVER:/root/
    ssh root@NEW-SERVER
    git clone https://github.com/noctisgit/envforge && cd envforge
    deploy/redeploy.sh restore /root/${name}.tar$([ "$tunnel" = 1 ] && echo " --with-tunnel")
EOF
    if [ "$tunnel" = 1 ]; then
        cat <<EOF

  Before the new server starts the tunnel, stop it here (sudo systemctl disable --now
  cloudflared): with both running, Cloudflare splits visitors between the two servers.
EOF
    fi
}

# ================================================================== restore
install_docker() {
    [ "$(id -u)" = 0 ] || die "Docker isn't installed. Install it, or re-run this with sudo to have it installed"
    step "Installing Docker"
    command -v curl >/dev/null 2>&1 || { apt-get update -qq && apt-get install -y -qq curl; }
    # Docker's official convenience script: Docker Engine + the Compose v2 plugin
    curl -fsSL https://get.docker.com | sh
    systemctl enable --now docker >/dev/null 2>&1 || true
    ok "Docker $(docker version --format '{{.Server.Version}}')"
}

install_tunnel() { # install_tunnel TOKEN_FILE
    [ "$(id -u)" = 0 ] || die "--with-tunnel needs root (re-run with sudo)"
    step "Cloudflare Tunnel"
    if ! command -v cloudflared >/dev/null 2>&1; then
        local deb
        deb="$(mktemp --suffix .deb)"
        curl -fsSL -o "$deb" \
            "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-$(dpkg --print-architecture).deb"
        dpkg -i "$deb" >/dev/null
        rm -f "$deb"
    fi
    ok "cloudflared $(cloudflared --version 2>/dev/null | awk '{print $3}')"
    install -d -m 755 /etc/cloudflared
    install -m 600 "$1" /etc/cloudflared/token
    cat >/etc/systemd/system/cloudflared.service <<EOF
[Unit]
Description=Cloudflare Tunnel client
After=network-online.target
Wants=network-online.target

[Service]
TimeoutStartSec=15
Type=notify
ExecStart=$(command -v cloudflared) --no-autoupdate tunnel run --token-file /etc/cloudflared/token
Restart=on-failure
RestartSec=5s

[Install]
WantedBy=multi-user.target
EOF
    systemctl daemon-reload
    systemctl enable --now cloudflared
    ok "tunnel running: the public URL now points at this server"
}

restore() {
    local file="" tunnel=0 url="" port="" yes=0
    while [ $# -gt 0 ]; do
        case "$1" in
            --with-tunnel) tunnel=1 ;;
            --url) url="${2:?}"; shift ;;
            --port) port="${2:?}"; shift ;;
            -y | --yes) yes=1 ;;
            -h | --help) usage; exit 0 ;;
            -*) die "unknown option for restore: $1" ;;
            *) [ -z "$file" ] || die "one backup file only"; file="$1" ;;
        esac
        shift
    done
    [ -n "$file" ] || { usage >&2; exit 2; }
    [ -r "$file" ] || die "can't read $file"
    if [ -f "$file.sha256" ]; then
        [ "$(sha256sum "$file" | cut -d' ' -f1)" = "$(cat "$file.sha256")" ] || die "$file doesn't match $file.sha256: copy it again"
        ok "checksum matches"
    fi

    step "Unpacking the backup"
    umask 077
    WORK="$(mktemp -d)"
    tar -C "$WORK" -xf "$file"
    for f in db.dump envdata.tar.gz env manifest.txt; do
        [ -f "$WORK/$f" ] || die "$file isn't an envforge backup (no $f)"
    done
    sed 's/^/  /' "$WORK/manifest.txt"
    if [ "$tunnel" = 1 ] && [ ! -f "$WORK/cloudflared-token" ]; then
        die "--with-tunnel, but this backup has no tunnel token (make it with: backup --with-tunnel)"
    fi

    step "Checking this server"
    command -v docker >/dev/null 2>&1 || install_docker
    docker info >/dev/null 2>&1 || die "Docker is installed but not reachable (sudo systemctl start docker, or run with sudo)"
    docker compose version >/dev/null 2>&1 || die "Docker Compose v2 is missing (install the docker-compose-plugin package)"
    ok "Docker $(docker version --format '{{.Server.Version}}')"

    # Never load a backup over an existing envforge: it would mix two databases.
    if compose ps -a --services 2>/dev/null | grep -q .; then
        die "envforge containers already exist in this checkout. Restore into a fresh clone, or remove this install first (docker compose down -v deletes its data)"
    fi
    local project
    project="$(basename "$PWD" | tr '[:upper:]' '[:lower:]' | tr -cd 'a-z0-9_-')"
    for v in pgdata envdata; do
        if docker volume ls --format '{{.Name}}' | grep -qx "${project}_${v}"; then
            die "the volume ${project}_${v} already exists. Restore into a fresh clone, or remove it first (docker volume rm ${project}_${v} deletes its data)"
        fi
    done
    if [ -f .env ] && [ "$(env_get .env APP_SECRET_KEY)" != "$(env_get "$WORK/env" APP_SECRET_KEY)" ]; then
        die ".env already exists with a different APP_SECRET_KEY. Move it away first; the backup's .env replaces it"
    fi
    ok "fresh checkout: nothing to overwrite"

    if [ "$tunnel" = 1 ] && [ "$yes" = 0 ] && [ -t 0 ]; then
        local a
        read -r -p "  Is the tunnel on the old server stopped (sudo systemctl disable --now cloudflared)? (y/n) " a </dev/tty
        case "$a" in y | Y | yes) ;; *) die "stop the old server's tunnel first, then re-run" ;; esac
    fi

    step "Settings"
    install -m 600 "$WORK/env" .env
    # The Docker socket's group differs between machines (the run worker uses it)
    env_set DOCKER_GID "$(stat -c %g /var/run/docker.sock 2>/dev/null || echo 0)"
    # Postgres starts before install.sh (which picks free ports for the rest): move it off a
    # host port another program already uses, such as a system Postgres
    local pg_port
    pg_port="$(env_get .env POSTGRES_PORT)"
    pg_port="${pg_port:-5432}"
    while (exec 3<>"/dev/tcp/127.0.0.1/$pg_port") 2>/dev/null; do pg_port=$((pg_port + 1)); done
    [ "$pg_port" = 5432 ] || env_set POSTGRES_PORT "$pg_port"
    ok ".env restored (same APP_SECRET_KEY: saved keys and the workspace SSH key still work)"

    step "Building the images (several minutes on a new machine)"
    compose build
    ok "built"

    step "Database"
    compose up -d postgres
    wait_for_postgres
    compose exec -T postgres pg_restore -U envforge -d envforge --no-owner --exit-on-error <"$WORK/db.dump"
    ok "restored: $(psql_q "select count(*) from users") user(s), $(psql_q "select count(*) from runs where deleted_at is null") run(s), $(psql_q "select count(*) from adapters") adapter(s)"

    step "Data volume"
    compose run --rm --no-deps -T data-perms sh -c 'tar -C /data -xzf -' <"$WORK/envdata.tar.gz"
    ok "restored: $(compose run --rm --no-deps -T data-perms du -sh /data | cut -f1)"

    step "Starting envforge (install.sh: migrations, health check)"
    ./install.sh --yes --no-build ${url:+--url "$url"} ${port:+--port "$port"}

    [ "$tunnel" = 1 ] && install_tunnel "$WORK/cloudflared-token"

    cat <<EOF

${G}${B}Restore finished.${N}

  Check:  sign in with your usual account; your runs, environments and adapters are there.
          Settings → GPU boxes → Check (the workspace SSH key is unchanged, so RunPod keeps
          working; only the pod's address, if it changed).
EOF
    if [ "$tunnel" = 0 ]; then
        cat <<EOF
  Public URL: not moved. Either re-run with --with-tunnel (backup made with --with-tunnel),
          or point your proxy / tunnel at this machine's port $(env_get .env WEB_PORT).
EOF
    fi
}

case "${1:-}" in
    backup) shift; backup "$@" ;;
    restore) shift; restore "$@" ;;
    -h | --help | "") usage ;;
    *) die "unknown command: $1 (backup | restore)" ;;
esac

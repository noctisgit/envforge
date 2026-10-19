#!/usr/bin/env bash
# envforge GPU image entrypoint (runs under tini, which is PID 1).
#
# What it does, in order:
#   1. authorized_keys from $PUBLIC_KEY (RunPod's convention) and/or $SSH_PUBLIC_KEY
#   2. SSH host keys (fresh per container, or kept in $SSH_HOST_KEY_DIR)
#   3. an environment file so commands run over ssh see PATH, HF_TOKEN, HF_HOME, ...
#   4. Tailscale, only if $TS_AUTHKEY is set (userspace networking: no /dev/net/tun needed)
#   5. /root/envforge/{jobs,adapters,logs,run}
#   6. a "ready" banner with how to connect
#   7. exec sshd in the foreground (key auth only); the container lives as long as sshd
#
# Rule for this file: never echo a secret. TS_AUTHKEY goes to tailscale through a
# mode-600 file and is unset right after; HF_TOKEN is only written to a mode-600 file.
set -euo pipefail

log() { printf '[envforge] %s\n' "$*"; }
warn() { printf '[envforge] WARNING: %s\n' "$*" >&2; }

ENVFORGE_ROOT=/root/envforge
VERSION="$(cat /opt/envforge/VERSION 2>/dev/null || echo dev)"

# ---------------------------------------------------------------- 1. authorized keys
install -d -m 700 /root/.ssh
touch /root/.ssh/authorized_keys
chmod 600 /root/.ssh/authorized_keys

add_keys() {
    # $1: one or more public keys, newline-separated. Public keys are not secrets.
    local line
    while IFS= read -r line || [ -n "$line" ]; do
        line="${line%$'\r'}"
        case "$line" in
            ssh-* | ecdsa-* | sk-*)
                grep -qxF -- "$line" /root/.ssh/authorized_keys ||
                    printf '%s\n' "$line" >>/root/.ssh/authorized_keys
                ;;
            "") ;;
            *) warn "ignoring a line that does not look like an ssh public key" ;;
        esac
    done <<<"$1"
}
add_keys "${PUBLIC_KEY:-}"
add_keys "${SSH_PUBLIC_KEY:-}"
KEY_COUNT="$(grep -c . /root/.ssh/authorized_keys || true)"
if [ "$KEY_COUNT" -eq 0 ]; then
    warn "no PUBLIC_KEY / SSH_PUBLIC_KEY set: nobody will be able to log in over ssh"
fi

# ---------------------------------------------------------------- 2. host keys
# Never baked into the image (every pod would share them). Generated per container,
# or kept on a volume so a recreated pod keeps its identity (no "host key changed").
if [ -n "${SSH_HOST_KEY_DIR:-}" ]; then
    install -d -m 700 "$SSH_HOST_KEY_DIR"
    if compgen -G "$SSH_HOST_KEY_DIR/ssh_host_*_key" >/dev/null; then
        cp -p "$SSH_HOST_KEY_DIR"/ssh_host_* /etc/ssh/
    fi
fi
ssh-keygen -A >/dev/null
if [ -n "${SSH_HOST_KEY_DIR:-}" ]; then
    cp -p /etc/ssh/ssh_host_* "$SSH_HOST_KEY_DIR"/
fi

# ---------------------------------------------------------------- 3. ssh session env
# sshd starts every session with a clean environment, so variables set on the pod
# (RunPod/Vast template env, or the image's ENV) would be invisible to commands run
# over ssh. sshd reads ~/.ssh/environment (PermitUserEnvironment in sshd_config), so
# copy a fixed allow-list of variables there. Secrets such as TS_AUTHKEY never match.
ENV_FILE=/root/.ssh/environment
umask 077
: >"$ENV_FILE"
while IFS= read -r name; do
    case "$name" in
        PATH | LD_LIBRARY_PATH | PYTHONPATH | CUDA_HOME | CUDA_VERSION | TZ | LANG | LC_ALL | \
            NVIDIA_* | HF_* | HUGGING_FACE_* | TRANSFORMERS_* | TORCH_* | PYTORCH_* | NCCL_* | \
            VLLM_* | RUNPOD_* | VAST_* | PUBLIC_IPADDR)
            value="${!name}"
            # the file format is one NAME=value per line: skip multi-line values
            [[ "$value" == *$'\n'* ]] || printf '%s=%s\n' "$name" "$value" >>"$ENV_FILE"
            ;;
    esac
done < <(compgen -e)
umask 022
# PAM (pam_env) applies /etc/environment *after* ~/.ssh/environment, and Ubuntu's copy
# sets a default PATH without the CUDA directories. Put the image's paths there too
# (not secrets, so the usual world-readable file is fine).
printf 'PATH="%s"\nLD_LIBRARY_PATH="%s"\n' "$PATH" "${LD_LIBRARY_PATH:-}" >/etc/environment

# ---------------------------------------------------------------- 4. tailscale
TS_STATUS="not configured (set TS_AUTHKEY to join your tailnet)"
TS_NAME=""
if [ -n "${TS_AUTHKEY:-}" ]; then
    install -d /var/run/tailscale "$ENVFORGE_ROOT/logs"
    # Userspace networking: no TUN device or NET_ADMIN needed (pods have neither).
    # In this mode incoming tailnet connections are delivered to the pod's localhost,
    # so sshd on :22 and vLLM on its port are reachable at the node's Tailscale name.
    # --state=mem: keeps no state on disk; with an ephemeral auth key the node is
    # removed from the tailnet shortly after the pod stops.
    tailscaled --tun=userspace-networking --state=mem: \
        --socket=/var/run/tailscale/tailscaled.sock \
        >"$ENVFORGE_ROOT/logs/tailscaled.log" 2>&1 &
    for _ in $(seq 1 50); do
        [ -S /var/run/tailscale/tailscaled.sock ] && break
        sleep 0.2
    done

    ts_hostname="${TS_HOSTNAME:-envforge-gpu-$(hostname)}"
    extra_args=()
    if [ -n "${TS_EXTRA_ARGS:-}" ]; then
        read -r -a extra_args <<<"$TS_EXTRA_ARGS"
    fi
    # The key goes to tailscale as "file:<path>", so it never appears in argv / ps.
    keyfile="$(mktemp)"
    chmod 600 "$keyfile"
    printf '%s' "$TS_AUTHKEY" >"$keyfile"
    unset TS_AUTHKEY
    if tailscale up --auth-key="file:$keyfile" --hostname="$ts_hostname" --ssh=false \
        --timeout=90s "${extra_args[@]}" >>"$ENVFORGE_ROOT/logs/tailscaled.log" 2>&1; then
        TS_NAME="$(tailscale status --json 2>/dev/null | jq -r '.Self.DNSName // empty' | sed 's/\.$//')"
        TS_IP="$(tailscale ip -4 2>/dev/null | head -n1)"
        TS_STATUS="up: ${TS_NAME:-$ts_hostname} (${TS_IP:-no IPv4})"
    else
        TS_STATUS="FAILED to join (see $ENVFORGE_ROOT/logs/tailscaled.log); ssh via the provider's port still works"
        warn "tailscale up failed"
    fi
    rm -f "$keyfile"
fi
unset TS_AUTHKEY

# ---------------------------------------------------------------- 5. directories
install -d "$ENVFORGE_ROOT"/jobs "$ENVFORGE_ROOT"/adapters "$ENVFORGE_ROOT"/logs "$ENVFORGE_ROOT"/run
install -d /root/.cache/huggingface

# ---------------------------------------------------------------- 6. banner
GPUS="$(nvidia-smi -L 2>/dev/null | sed 's/ (UUID.*//' | paste -sd ';' - || true)"
PUBLIC_SSH=""
if [ -n "${RUNPOD_PUBLIC_IP:-}" ] && [ -n "${RUNPOD_TCP_PORT_22:-}" ]; then
    PUBLIC_SSH="ssh root@${RUNPOD_PUBLIC_IP} -p ${RUNPOD_TCP_PORT_22}"
elif [ -n "${PUBLIC_IPADDR:-}" ] && [ -n "${VAST_TCP_PORT_22:-}" ]; then
    PUBLIC_SSH="ssh root@${PUBLIC_IPADDR} -p ${VAST_TCP_PORT_22}"
fi
PUBLIC_SSH="${PUBLIC_SSH:-use the provider mapping for TCP port 22}"
if [ -n "$TS_NAME" ]; then
    CONNECT_HOST="$TS_NAME"
    # vLLM on loopback = reachable over the tailnet only, not via public port mappings
    VLLM_HINT="uv run envforge compute vllm-up Qwen/Qwen2.5-3B-Instruct --bind 127.0.0.1"
else
    CONNECT_HOST="<pod address>"
    VLLM_HINT="uv run envforge compute vllm-up Qwen/Qwen2.5-3B-Instruct --api-key \"\$VLLM_API_KEY\""
fi

cat <<EOF
================================================================================
 envforge GPU image ${VERSION}: ready
   GPUs:            ${GPUS:-none visible (nvidia-smi failed)}
   authorized keys: ${KEY_COUNT}
   tailscale:       ${TS_STATUS}
   ssh (provider):  ${PUBLIC_SSH}
 from the control plane:
   export ENVFORGE_GPU_HOST=${CONNECT_HOST} ENVFORGE_GPU_RUNTIME=pod
   uv run envforge compute check
   ${VLLM_HINT}
================================================================================
EOF

# ---------------------------------------------------------------- 7. sshd
install -d -m 755 /run/sshd
/usr/sbin/sshd -t # refuse to start with a broken config rather than run without auth rules
exec /usr/sbin/sshd -D -e

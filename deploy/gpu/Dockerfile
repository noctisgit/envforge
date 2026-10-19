# syntax=docker/dockerfile:1
#
# envforge GPU image: run this AS the pod on RunPod / Vast (they are containers, with no
# Docker daemon inside), then drive it from the control plane over SSH (ideally over
# Tailscale) with `envforge ... --runtime pod`. See docs/COMPUTE.md, "RunPod / Vast with
# the envforge GPU image".
#
#   docker build -t envforge-gpu:dev -f deploy/gpu/Dockerfile .      (from the repo root)
#
# Base: the exact vLLM image envforge serves with (DEFAULT_VLLM_IMAGE in
# envforge/compute/vllm.py; a test keeps the two in sync). It already contains CUDA,
# PyTorch, transformers and vLLM, so this file only adds:
#   - peft / datasets / accelerate for LoRA training (envforge/train/sft.py)
#   - openssh-server (key auth only), tini (PID 1), curl, jq, tmux, procps
#   - Tailscale (static binaries, pinned + checksummed), started only if TS_AUTHKEY is set

ARG VLLM_IMAGE=vllm/vllm-openai:v0.30.0
FROM ${VLLM_IMAGE}

ARG TAILSCALE_VERSION=1.102.4
ARG TAILSCALE_SHA256=50748df1045e60b5b695f19f4c56b0da36c019948b440fb456b6584a50f0d8b9

ENV DEBIAN_FRONTEND=noninteractive \
    HF_HOME=/root/.cache/huggingface \
    PYTHONUNBUFFERED=1

# System packages. The host keys openssh-server generates at install time are deleted:
# baked-in keys would be shared by every pod ever started from this image. The
# entrypoint generates fresh ones per container.
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
        openssh-server tini curl jq tmux procps ca-certificates \
 && rm -rf /var/lib/apt/lists/* \
 && rm -f /etc/ssh/ssh_host_* \
 && ln -sf /usr/bin/python3 /usr/local/bin/python

# Tailscale: official static build, verified against its published SHA-256.
RUN curl -fsSL -o /tmp/tailscale.tgz \
        "https://pkgs.tailscale.com/stable/tailscale_${TAILSCALE_VERSION}_amd64.tgz" \
 && echo "${TAILSCALE_SHA256}  /tmp/tailscale.tgz" | sha256sum -c - \
 && tar -xzf /tmp/tailscale.tgz -C /tmp \
 && install -m 755 "/tmp/tailscale_${TAILSCALE_VERSION}_amd64/tailscale" \
                   "/tmp/tailscale_${TAILSCALE_VERSION}_amd64/tailscaled" /usr/local/bin/ \
 && rm -rf /tmp/tailscale*

# Training libraries, installed with `uv pip` like the vLLM image itself: the base sets
# UV_OVERRIDE=/etc/uv-overrides.txt (it deliberately pins NCCL newer than torch declares),
# and only uv honours that file; pip's resolver would reject the base as inconsistent.
# A freeze of the base becomes a constraints file, so installing peft/datasets can add
# packages but cannot upgrade or downgrade anything vLLM relies on. Two exemptions:
# accelerate (pinned explicitly in requirements-train.txt) and fsspec, which `datasets`
# caps below the base's version; only torch and huggingface-hub use it, with lower
# bounds only, so letting it step back is safe.
COPY deploy/gpu/requirements-train.txt /opt/envforge/requirements-train.txt
RUN uv pip freeze --system | grep -E '^[A-Za-z0-9_.-]+==' \
      | grep -viE '^(accelerate|fsspec)==' > /tmp/base-constraints.txt \
 && uv pip install --system --no-cache -c /tmp/base-constraints.txt \
        -r /opt/envforge/requirements-train.txt \
 && rm /tmp/base-constraints.txt \
 && python3 -c "import accelerate, datasets, peft, transformers, torch, vllm"

COPY deploy/gpu/sshd_config /etc/ssh/sshd_config.d/envforge.conf
COPY deploy/gpu/entrypoint.sh /opt/envforge/entrypoint.sh
# The training script (also uploaded per job by envforge; this copy is for manual runs
# and is how `--runtime auto` recognises the image).
COPY envforge/train/sft.py /opt/envforge/sft.py
RUN chmod 755 /opt/envforge/entrypoint.sh \
 && mkdir -p /root/envforge/jobs /root/envforge/adapters /root/envforge/logs /root/envforge/run \
             /root/.cache/huggingface

# Last, so a new version number doesn't invalidate the layers above.
ARG ENVFORGE_GPU_VERSION=dev
RUN echo "${ENVFORGE_GPU_VERSION}" > /opt/envforge/VERSION
LABEL org.opencontainers.image.title="envforge-gpu" \
      org.opencontainers.image.description="vLLM + LoRA training + sshd + Tailscale, driven by envforge --runtime pod" \
      org.opencontainers.image.version="${ENVFORGE_GPU_VERSION}"

WORKDIR /root
EXPOSE 22
# tini is PID 1: it forwards signals and reaps orphaned processes (detached jobs that
# finish would otherwise linger as zombies and look "running"). The vLLM base image's
# ENTRYPOINT (`vllm serve`) is replaced; envforge starts vLLM itself over ssh.
ENTRYPOINT ["/usr/bin/tini", "--", "/opt/envforge/entrypoint.sh"]
CMD []

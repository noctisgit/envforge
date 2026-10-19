# Compute: serving an open model on a rented GPU box

Phase 2 lets you evaluate an open-weight model with the same loop you use for Claude or
GPT. The control plane (this CPU box) never holds weights. Instead it drives a GPU box over
SSH, ideally over Tailscale: it starts [vLLM](https://docs.vllm.ai) in Docker there, and vLLM
exposes an OpenAI-compatible endpoint. The existing `vllm:` provider
(`envforge/models/openai_compat.py`) talks to that endpoint the same way it talks to OpenAI.

```
control plane ──ssh──▶ GPU box: docker run vllm/vllm-openai … (container envforge-vllm-8000)
      │                              ▲
      └────── HTTP /v1/chat/completions (VLLM_BASE_URL) ──┘
```

**Two runtimes.** On a VM or bare metal, envforge runs vLLM and training as Docker
containers (`--runtime docker`, the default). On RunPod Pods and Vast instances, which *are*
containers, you run envforge's own GPU image as the pod and envforge starts the same
workloads as background processes inside it (`--runtime pod`). See section 6.

**Scope:** you bring your own box. envforge does not create, start or stop instances through
the RunPod, Vast or cloud APIs. Rent the box in their web console, then point envforge at it.

Code: `envforge/compute/` (`ssh.py` = `SSHBox`, `host.py` = checks and GPU info,
`vllm.py` = serving, `runtime.py` = docker vs pod, `process.py` = detached processes for pod
mode, `cli.py` = `envforge compute …`). The GPU image lives in `deploy/gpu/`.

## 1. Renting a box

You need **an Ubuntu machine with an NVIDIA GPU, a Docker daemon you can use, and SSH
access**. The Docker part matters, and it is where providers differ:

| Provider | What you get | Works with `envforge compute`? |
|---|---|---|
| A **VM or bare-metal** GPU instance (Vast.ai "VM" instances, Lambda, Hyperstack, TensorDock, DataCrunch, AWS/GCP/Azure GPU VMs, your own workstation) | a full OS. You can install Docker and the NVIDIA Container Toolkit (many images ship with both) | **Yes**, the whole flow below |
| **RunPod Pods**, standard **Vast.ai** instances | you are *inside* a container, with no Docker daemon to talk to | **Yes, with the envforge GPU image** and `--runtime pod`. See section 6 |

Pick a GPU from the model table in section 5. A 24 GB card (L4, A10G, RTX 3090/4090) covers
3B to 7B models and costs roughly $0.30 to $0.80/h on the marketplaces. **Stop the instance
when you are done.** Billing is by the hour, whether or not vLLM is running.

### Container-only providers without the envforge image (serving only)

The recommended way to use RunPod or Vast is section 6: it supports serving *and* training
with the same commands. If you only want to serve a model and would rather not use the
envforge image, the provider can run the plain vLLM image for you. Skip `vllm-up` and
configure the same thing in their console:

- image `vllm/vllm-openai:v0.30.0` (the image envforge pins)
- container arguments: `Qwen/Qwen2.5-3B-Instruct --enable-auto-tool-choice --tool-call-parser hermes --gpu-memory-utilization 0.90`
- environment: `VLLM_API_KEY=<a long random string>` (and `HF_TOKEN` for gated models)
- expose HTTP port 8000. On RunPod this gives `https://<pod-id>-8000.proxy.runpod.net`, so
  `VLLM_BASE_URL=https://<pod-id>-8000.proxy.runpod.net/v1`

Then go straight to section 4. Training (`envforge train sft --host`) needs section 6.

## 2. Preparing the box (once)

On the GPU box:

```bash
# Docker + NVIDIA Container Toolkit (skip whatever the image already has)
curl -fsSL https://get.docker.com | sh
# NVIDIA Container Toolkit: follow
#   https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html
sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker
docker run --rm --gpus all ubuntu:24.04 nvidia-smi -L     # must list your GPU

# Non-root SSH user? Add it to the docker group (then log out and back in)
sudo usermod -aG docker "$USER"

# Tailscale (recommended): a private network between the control plane and the box
curl -fsSL https://tailscale.com/install.sh | sh
sudo tailscale up            # the box gets a 100.x.y.z address and a MagicDNS name
```

On the control plane, make sure `ssh <user>@<box>` works **without a password prompt**
(key in your agent, or pass `--key`). envforge runs ssh with `BatchMode=yes`, so it fails
instead of prompting. With `StrictHostKeyChecking=accept-new` it trusts a new box's host key
the first time and refuses if that key later changes.

### Security: never expose vLLM publicly without a key

vLLM listens on a TCP port and will run any request it receives on your GPU.

- **Best:** reach it only over Tailscale. Pass `--bind <box's tailscale IP>` so the port is
  published only on the tailnet interface, and use the 100.x address in `VLLM_BASE_URL`.
- **Always** pass `--api-key` (or set `VLLM_API_KEY`) when the port can be reached from the
  internet. vLLM then rejects requests that lack `Authorization: Bearer <key>`.
- Secrets travel safely. The HF token and the API key are sent over ssh **stdin** into a
  mode-600 temp env-file, which is deleted right after `docker run`. So they never appear in
  argv, `ps` or envforge logs. Root on the box can still read them with `docker inspect`.

## 3. Commands, end to end

Connection options work on every command: `--host` (required), `--user` (default `root`),
`--port` (the **SSH** port, default 22; Vast often uses another one) and `--key`. Each can come
from the environment or `.env` instead: `ENVFORGE_GPU_HOST`, `ENVFORGE_GPU_USER`,
`ENVFORGE_GPU_PORT`, `ENVFORGE_GPU_KEY`. The vLLM port is a separate option, `--vllm-port`
(default 8000).

```bash
export ENVFORGE_GPU_HOST=gpu-1            # Tailscale name, 100.x IP, or ~/.ssh/config alias
export VLLM_API_KEY=$(openssl rand -hex 24)

uv run envforge compute check             # ssh / docker / nvidia-smi / container toolkit
uv run envforge compute gpu               # name, memory used/total, utilization

uv run envforge compute vllm-up Qwen/Qwen2.5-3B-Instruct --bind 100.101.102.103
#   pulls the image (first time, several GB), starts container envforge-vllm-8000,
#   waits for /v1/models, then prints:
#     base URL:     http://gpu-1:8000/v1
#     model string: vllm:Qwen/Qwen2.5-3B-Instruct
#     VLLM_BASE_URL=http://gpu-1:8000/v1

uv run envforge compute vllm-status       # running? which model / image?
uv run envforge compute vllm-logs --tail 100
uv run envforge compute vllm-down         # removes the container; weights cache volume stays
```

Useful `vllm-up` options:

- `--tool-call-parser`: must match the model family (section 5). The default is `hermes`.
  `--enable-auto-tool-choice` is always on, because the runner depends on structured tool calls.
- `--max-model-len 16384`: caps the context to save KV-cache memory. Use it when a model
  fails to start with "not enough KV cache".
- `--gpu-memory-utilization 0.90`: the fraction of VRAM vLLM may claim.
- `--hf-token` (or `HF_TOKEN`): for gated models such as Llama and Gemma.
- `--extra-arg=--dtype --extra-arg=half`: passes any other vLLM flag. You need `--dtype half`
  on GPUs without bf16 support (T4, V100).
- `--endpoint-host 100.101.102.103`: use this when `--host` is an ssh alias the HTTP client
  can't resolve.
- `--image`: overrides the pinned `vllm/vllm-openai:v0.30.0`. Bump the pin on purpose,
  because a vLLM upgrade can change tool-call parsing and so your numbers.

Every command also takes `--runtime docker|pod|auto` (or `ENVFORGE_GPU_RUNTIME`). This
section describes `docker`; section 6 covers `pod`.

Weights are cached in the Docker volume `envforge-hf-cache` on the box. Restarting or switching
back to a model you already used doesn't download it again. The volume is lost when you
destroy the instance.

From Python, the same calls look like this:

```python
from envforge.compute import SSHBox, check, serve_vllm, wait_healthy

box = SSHBox("gpu-1", user="root")
server = serve_vllm(box, "Qwen/Qwen2.5-3B-Instruct", api_key=key, bind="100.101.102.103")
wait_healthy(server.base_url, timeout=1200, api_key=key)
print(server.base_url, server.model_spec)
```

## 4. Evaluating the served model

**CLI.** Put the printed values in `.env` (or export them), then run as usual:

```bash
VLLM_BASE_URL=http://gpu-1:8000/v1
VLLM_API_KEY=<the key you passed to vllm-up>

uv run envforge run netops --model vllm:Qwen/Qwen2.5-3B-Instruct --episodes 30
uv run envforge compare data/workspaces/default/runs/<claude-run> data/workspaces/default/runs/<qwen-run>
```

The part after `vllm:` must equal the served model name, which is the Hugging Face id you
passed to `vllm-up`.

**Web UI.** Go to Settings, then provider keys, then **vLLM**. Set the base URL
(`http://100.x.y.z:8000/v1`) and the key, then start a run with model
`vllm:Qwen/Qwen2.5-3B-Instruct`. The web layer may restrict which vLLM base URLs it accepts.
By default it allows public hosts and the Tailscale range `100.64.0.0/10`, and refuses other
private or loopback addresses (to prevent SSRF). An admin can widen this with
`BASE_URL_ALLOW_CIDRS` or `BASE_URL_ALLOW_PRIVATE`. See `docs/WEB.md`. Tailscale addresses
work out of the box, provided the API and worker hosts are on the same tailnet. If you use a
MagicDNS name rather than the 100.x IP, those hosts must also be able to resolve it.

## 5. Which model?

These are small open models with reliable tool calling in vLLM. VRAM figures are for bf16
weights plus a modest KV cache at `--gpu-memory-utilization 0.90`.

| Model | `--tool-call-parser` | Weights | GPU that works | Notes |
|---|---|---|---|---|
| `Qwen/Qwen2.5-3B-Instruct` | `hermes` | ~6 GB | 16 GB+ (T4 with `--dtype half`, L4, A10G) | **Start here.** Cheap, decent tool use; the Phase 3 SFT target size |
| `Qwen/Qwen2.5-7B-Instruct` | `hermes` | ~15 GB | 24 GB with `--max-model-len 16384`; 40-48 GB comfortable | Noticeably stronger agent |
| `Qwen/Qwen3-4B` / `Qwen/Qwen3-8B` | `hermes` | ~8 / ~16 GB | 24 GB / 24-48 GB | Newer; emits `<think>` reasoning. Consider `--extra-arg=--reasoning-parser --extra-arg=qwen3` |
| `meta-llama/Llama-3.1-8B-Instruct` | `llama3_json` | ~16 GB | 24 GB+ | Gated: needs `--hf-token`; see vLLM's tool-calling docs for its chat template |
| `mistralai/Mistral-7B-Instruct-v0.3` | `mistral` | ~15 GB | 24 GB+ | Needs Mistral tokenizer flags; see vLLM docs |

If a model "never calls tools" or puts raw `<tool_call>` text in its reply, the parser is
wrong for that family. Check `vllm-logs` and vLLM's tool-calling docs.

## 6. RunPod / Vast with the envforge GPU image

RunPod Pods and standard Vast instances are containers. There is no Docker daemon inside,
so `docker run` can't work there. The fix is to make *the pod itself* the envforge
environment: publish an image that already contains vLLM, the training libraries, `sshd`
and Tailscale, run it as the pod, and let envforge start processes inside it over SSH.

```
control plane ──ssh (over Tailscale)──▶ pod running ghcr.io/<owner>/envforge-gpu:<tag>
     │                                    ├─ tini (PID 1) → sshd -D    key auth only
     │                                    ├─ tailscaled                userspace networking
     │                                    ├─ vllm serve …              process vllm-8000
     │                                    └─ python3 sft.py …          process train-<name>
     └──── HTTP /v1/chat/completions over the tailnet ──▶ vLLM
```

`--runtime pod` (or `ENVFORGE_GPU_RUNTIME=pod`) tells envforge to drive the box this way.
`--runtime auto` asks the box: a working Docker daemon means `docker`; otherwise the file
`/opt/envforge/sft.py` (only the envforge image has it) means `pod`.

### What's in the image

`deploy/gpu/Dockerfile`, built `FROM vllm/vllm-openai:v0.30.0`, the same pinned image
`vllm-up` uses (a test keeps the two in sync):

| Part | Why |
|---|---|
| CUDA, PyTorch, transformers, vLLM | inherited from the vLLM image, so serving in a pod behaves exactly like docker mode |
| `peft`, `datasets`, `accelerate` (pinned in `deploy/gpu/requirements-train.txt`) | LoRA training with `sft.py`. Installed against a constraints file so they can't swap the base's torch or vLLM |
| `openssh-server` | how envforge gets in. Passwords and keyboard-interactive are disabled (`deploy/gpu/sshd_config`) |
| Tailscale (pinned static build, checksum-verified) | started only if `TS_AUTHKEY` is set |
| `tini` | PID 1. It reaps finished background jobs, which would otherwise linger as zombies that look alive |
| `curl`, `jq`, `tmux`, `procps` | health probes, the banner, and poking around by hand |
| `/opt/envforge/sft.py`, `/opt/envforge/VERSION` | marker for `--runtime auto` and the version `check` reports |

The image is about 31 GB unpacked (about 9 GB to download), almost all of it the vLLM base.
envforge adds a few hundred MB. Model weights are **not** in the image: they download from
Hugging Face into `/root/.cache/huggingface` the first time.

**Entrypoint** (`deploy/gpu/entrypoint.sh`) on every start:

1. writes `~/.ssh/authorized_keys` from `PUBLIC_KEY` (RunPod's variable) and/or `SSH_PUBLIC_KEY`;
2. generates fresh SSH host keys. Keys are never baked into the image, or every pod would
   share them. Set `SSH_HOST_KEY_DIR` to a path on a volume to keep the same host keys
   across pods;
3. copies an allow-list of variables (`PATH`, `HF_*`, `CUDA_*`, `VLLM_*`, …) into
   `~/.ssh/environment`. Without this, commands run over SSH wouldn't see `HF_TOKEN` or
   `HF_HOME` set in the template;
4. if `TS_AUTHKEY` is set: starts `tailscaled --tun=userspace-networking --state=mem:` and runs
   `tailscale up --hostname ${TS_HOSTNAME:-envforge-gpu-<hostname>} --ssh=false`. The key is
   passed as a mode-600 file, so it never shows up in `ps`, and it is unset afterwards;
5. creates `/root/envforge/{jobs,adapters,logs,run}` and prints a banner (GPU, Tailscale name,
   how to connect). It never prints secrets;
6. `exec sshd -D`. The pod lives as long as sshd.

### Publishing the image (once per version)

`.github/workflows/gpu-image.yml` builds and pushes to `ghcr.io/<owner>/envforge-gpu` when you
push a tag `gpu-v<version>`, or when you run it manually from the Actions tab:

```bash
git tag gpu-v0.1.0 && git push origin gpu-v0.1.0     # -> :0.1.0 and :latest
```

New GHCR packages are **private**. Either make the package public (GitHub → your profile →
Packages → `envforge-gpu` → Package settings → Change visibility), or give RunPod/Vast registry
credentials: a GitHub personal access token (classic) with only `read:packages`, entered as
RunPod "Container Registry Credentials" or as Vast's docker login. The workflow pushes with the
repository's `GITHUB_TOKEN`. If the push is refused, check Settings → Actions → General →
Workflow permissions, and, if the package already exists, Package settings → "Manage Actions
access".

To build locally: `docker build -t envforge-gpu:dev -f deploy/gpu/Dockerfile .`

### A Tailscale auth key for pods

In the Tailscale admin console, open Settings → Keys → **Generate auth key** and choose:

- **Ephemeral**: the node is removed from your tailnet soon after the pod stops. Pods come and
  go, and you don't want a graveyard of dead `gpu-1-3` machines. (`--state=mem:` already keeps
  no state on disk, so a restarted pod always joins as a new node.)
- **Reusable**: one key serves every pod you start.
- **Pre-approved**, if your tailnet requires device approval.
- **Tags**, optional but recommended, e.g. `tag:gpu`. Tagged nodes don't expire, and the ACL
  can limit them, for example "the control plane may reach `tag:gpu` on ports 22 and 8000, and
  `tag:gpu` may reach nothing". Pass `TS_EXTRA_ARGS=--advertise-tags=tag:gpu` and declare the tag
  under `tagOwners` in the ACL.

Store the key as a provider secret (RunPod: Settings → Secrets, then reference it as
`{{ RUNPOD_SECRET_ts_authkey }}`), not as plain text in a template you might share.

### RunPod template

Create a template (Templates → New Template) with:

| Field | Value |
|---|---|
| Container image | `ghcr.io/noctisgit/envforge-gpu:0.1.0`. Pin a version, not `latest`, so a new push can't change your numbers |
| Container registry credentials | only if the package is private (see above) |
| Container start command | leave empty. The image's entrypoint does everything |
| Container disk | 20 GB or more: scratch space for jobs, logs and adapters. It is wiped when the pod stops |
| Volume | a **network volume** mounted at `/root/.cache/huggingface`, so weights survive pod restarts and are shared by every pod in that datacenter. Without one, every new pod downloads the weights again |
| Expose TCP ports | `22`. This is the SSH fallback when you don't use Tailscale. No HTTP port is needed with Tailscale |
| Environment variables | `PUBLIC_KEY` (RunPod fills it from your account's SSH keys; set it yourself to override), `TS_AUTHKEY={{ RUNPOD_SECRET_ts_authkey }}`, `TS_HOSTNAME=gpu-1`, `HF_TOKEN={{ RUNPOD_SECRET_hf_token }}` for gated models. Optional: `SSH_HOST_KEY_DIR=/root/.cache/huggingface/.envforge-ssh`, `TS_EXTRA_ARGS=--advertise-tags=tag:gpu` |

Deploy a pod from the template. The pod's logs show the banner once it is ready.

### Vast template

- Image: `ghcr.io/noctisgit/envforge-gpu:0.1.0`. Add docker login details if the package is private.
- Launch mode: **"Docker ENTRYPOINT"**, so the image's own entrypoint runs. Vast's SSH and
  Jupyter launch modes replace it with Vast's own startup.
- Docker options: `-p 22:22 -e TS_AUTHKEY=… -e TS_HOSTNAME=gpu-1 -e PUBLIC_KEY="ssh-ed25519 AAAA… you@laptop"`
  (plus `-e HF_TOKEN=…` if needed). Vast maps port 22 to a random public port, shown under the
  instance's IP & port info.
- Disk: Vast's disk also holds the image, so allow about 60 GB plus weights.

### Connecting: Tailscale or the provider's SSH port

**Over Tailscale (recommended).** Once the pod has joined, it is `gpu-1` (MagicDNS) or a
`100.x.y.z` address on your tailnet. The control plane must be on the same tailnet:

```bash
ssh root@gpu-1 true                          # first time: accept the host key
export ENVFORGE_GPU_HOST=gpu-1 ENVFORGE_GPU_RUNTIME=pod
```

**Over the provider's port.** On RunPod, use the "SSH over exposed TCP" address from the
Connect dialog. On Vast, use the mapped port:

```bash
export ENVFORGE_GPU_HOST=203.0.113.7 ENVFORGE_GPU_PORT=40123 ENVFORGE_GPU_RUNTIME=pod
```

Each new pod generates new host keys. If you reuse a name or address, ssh refuses with "REMOTE
HOST IDENTIFICATION HAS CHANGED". Run `ssh-keygen -R gpu-1`, or set `SSH_HOST_KEY_DIR` on the
network volume so pods keep one identity.

### The same commands, with `--runtime pod`

```bash
uv run envforge compute check                # runtimes on this box + ssh / envforge-image / nvidia-smi / python-deps
uv run envforge compute gpu
uv run envforge compute vllm-up Qwen/Qwen2.5-3B-Instruct --bind 127.0.0.1     # with Tailscale
uv run envforge compute vllm-up Qwen/Qwen2.5-3B-Instruct --api-key "$VLLM_API_KEY"  # without
uv run envforge compute vllm-status          # also says whether /v1/models answers on the pod
uv run envforge compute vllm-logs --tail 100
uv run envforge compute vllm-down

uv run envforge train sft netops-v1 --base Qwen/Qwen2.5-7B-Instruct
uv run envforge train follow netops-v1
uv run envforge train fetch netops-v1
```

(Each command takes `--runtime pod` explicitly as well.) What changes compared with docker mode:

| | docker runtime | pod runtime |
|---|---|---|
| vLLM runs as | container `envforge-vllm-<port>` | process `vllm-<port>` (`vllm serve …`) |
| training runs as | container `envforge-train-<name>`, pip-installs peft & co. each run | process `train-<name>`; libraries are already in the image |
| logs | `docker logs` | `/root/envforge/logs/<process>.log` |
| state | Docker | `/root/envforge/run/<process>.{pid,exit,json}` |
| weights cache | Docker volume `envforge-hf-cache` | `/root/.cache/huggingface` (mount a network volume there) |
| `--bind` | host interface Docker publishes on | `vllm serve --host`. `127.0.0.1` means tailnet and ssh tunnels only |
| `--image` | the vLLM image to run | ignored (the pod's image is what runs) |
| stop | `docker rm --force` | SIGTERM to the process group, SIGKILL after 30 s |

Secrets are handled the same way in both modes. `HF_TOKEN` and `VLLM_API_KEY` travel over ssh
**stdin** into a mode-600 file under `/root/envforge/run/`. The process reads that file and
deletes it before it starts, so the secrets never appear in argv or `ps`. Each process runs
under `setsid`, in its own process group, so a dropped ssh connection doesn't kill it, and
stopping it also stops vLLM's worker processes.

Adapters land in `/root/envforge/adapters/` on the container disk. That disk is wiped when the
pod stops, so `envforge train fetch` before you stop the pod.

### Tailscale vs SSH: what each one does

They are not alternatives at the same layer:

- **SSH** is the *protocol* envforge uses to run commands and copy files. It is always there.
- **Tailscale** is a *private network* that SSH (and HTTP to vLLM) runs over. Every device
  gets a stable `100.x` address and a name (`gpu-1`) that doesn't change when the pod moves to
  another host. Tailscale punches through NAT, so nothing has to be exposed to the internet,
  and it encrypts and authenticates at the network level with WireGuard.

Without Tailscale, you reach the pod through whatever public IP:port the provider maps to port
22, which changes every time. vLLM then has to be exposed too (the RunPod HTTP proxy, or a
mapped port, protected only by `--api-key`) or reached through an ssh tunnel
(`ssh -L 8000:127.0.0.1:8000 root@<ip> -p <port>`, then `VLLM_BASE_URL=http://127.0.0.1:8000/v1`).

With Tailscale, `VLLM_BASE_URL=http://gpu-1:8000/v1` simply works from the control plane, and
the web worker if it's on the tailnet, while the endpoint stays private. Pods can't create the
TUN network device that normal Tailscale needs, so the image runs it in **userspace
networking** mode. In that mode, incoming tailnet connections are handed to the pod's
`localhost`, which has two consequences:

- `vllm serve --host 127.0.0.1` (`--bind 127.0.0.1`) is reachable over the tailnet but not
  through the provider's public port mappings. That's the most private setup. Keep `--api-key`
  anyway if other people share your tailnet.
- you can't bind to the `100.x` address itself: in this mode it isn't a local interface.

Outgoing connections from the pod *into* the tailnet would need Tailscale's SOCKS5 proxy.
envforge doesn't need them, because the control plane always connects to the pod.

### Bandwidth: what moves where

| Path | What | Size |
|---|---|---|
| registry → pod host | the image, pulled by RunPod/Vast when a pod starts on a new machine | ~9 GB, once per host (minutes) |
| Hugging Face → pod | model weights | 6 to 16 GB per model, at datacenter speed. Kept on the network volume |
| control plane → pod | dataset (`train.jsonl`), `sft.py` | a few MB |
| pod → control plane | LoRA adapter (`train fetch`) | tens to a few hundred MB |
| control plane ↔ vLLM | eval traffic: chat requests and responses | KB per request |

Nothing heavy passes through the control plane, so a slow home connection is fine. Over
Tailscale, traffic normally goes directly between the control plane and the pod. If NAT
traversal fails, it goes through a DERP relay, which is slower but plenty for these sizes
(`tailscale ping gpu-1` shows which path you have).

**Billing:** pods bill by the minute while running, whether or not vLLM is. Fetch your
adapters, then stop or terminate the pod. With an ephemeral key, it disappears from your
tailnet by itself.

## Troubleshooting

| Symptom | Likely cause |
|---|---|
| `check`: ssh FAIL, exit 255 | wrong host/port/user, key not in agent, Tailscale down on one side |
| `check`: docker FAIL "permission denied" | user is not in the `docker` group |
| `check`: toolkit FAIL "could not select device driver" | NVIDIA Container Toolkit missing or Docker not restarted |
| `vllm-up` aborts with "server stopped before becoming healthy" | out of memory or a bad flag. Run `vllm-logs`. Try `--max-model-len`, a smaller model, or `--dtype half` |
| `wait_healthy` 401 | the `VLLM_API_KEY` on the client differs from the one the server started with |
| `check`: docker FAIL on a RunPod/Vast pod | expected: pods have no Docker. Use the envforge GPU image and `--runtime pod` (section 6); `check` prints which runtime the box supports |
| pod: `ssh` says "REMOTE HOST IDENTIFICATION HAS CHANGED" | a new pod reused the old name. `ssh-keygen -R gpu-1`, or set `SSH_HOST_KEY_DIR` (section 6) |
| pod: `vllm-up --bind 100.x.y.z` fails to start | in userspace mode the Tailscale IP is not a local interface. Use `--bind 127.0.0.1` (tailnet-only) or omit `--bind` |
| timeout on first start | normal for big downloads. The container keeps loading after the CLI gives up. Watch `vllm-logs` / `vllm-status`. Next time use `--timeout 3600` or `--no-wait`. Re-running `vllm-up` replaces the container |

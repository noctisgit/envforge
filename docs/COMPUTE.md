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

**Scope:** you bring your own box. envforge does not create, start or stop instances through
the RunPod, Vast or cloud APIs. Rent the box in their web console, then point envforge at it.

Code: `envforge/compute/` (`ssh.py` = `SSHBox`, `host.py` = checks and GPU info,
`vllm.py` = serving, `cli.py` = `envforge compute …`).

## 1. Renting a box

You need **an Ubuntu machine with an NVIDIA GPU, a Docker daemon you can use, and SSH
access**. The Docker part matters, and it is where providers differ:

| Provider | What you get | Works with `envforge compute`? |
|---|---|---|
| A **VM or bare-metal** GPU instance (Vast.ai "VM" instances, Lambda, Hyperstack, TensorDock, DataCrunch, AWS/GCP/Azure GPU VMs, your own workstation) | a full OS. You can install Docker and the NVIDIA Container Toolkit (many images ship with both) | **Yes**, the whole flow below |
| **RunPod Pods**, standard **Vast.ai** instances | you are *inside* a container, with no Docker daemon to talk to | **Not with `vllm-up`.** See "Container-only providers" below |

Pick a GPU from the model table in section 5. A 24 GB card (L4, A10G, RTX 3090/4090) covers
3B to 7B models and costs roughly $0.30 to $0.80/h on the marketplaces. **Stop the instance
when you are done.** Billing is by the hour, whether or not vLLM is running.

### Container-only providers (RunPod Pods, Vast containers)

In these cases the provider runs the container for you, so skip `vllm-up`. Configure the
same thing in their console:

- image `vllm/vllm-openai:v0.30.0` (the image envforge pins)
- container arguments: `Qwen/Qwen2.5-3B-Instruct --enable-auto-tool-choice --tool-call-parser hermes --gpu-memory-utilization 0.90`
- environment: `VLLM_API_KEY=<a long random string>` (and `HF_TOKEN` for gated models)
- expose HTTP port 8000. On RunPod this gives `https://<pod-id>-8000.proxy.runpod.net`, so
  `VLLM_BASE_URL=https://<pod-id>-8000.proxy.runpod.net/v1`

Then go straight to section 4. `envforge compute check/gpu` can still run over the pod's SSH,
but the docker-based commands will fail.

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

## Troubleshooting

| Symptom | Likely cause |
|---|---|
| `check`: ssh FAIL, exit 255 | wrong host/port/user, key not in agent, Tailscale down on one side |
| `check`: docker FAIL "permission denied" | user is not in the `docker` group |
| `check`: toolkit FAIL "could not select device driver" | NVIDIA Container Toolkit missing or Docker not restarted |
| `vllm-up` aborts with "server stopped before becoming healthy" | out of memory or a bad flag. Run `vllm-logs`. Try `--max-model-len`, a smaller model, or `--dtype half` |
| `wait_healthy` 401 | the `VLLM_API_KEY` on the client differs from the one the server started with |
| timeout on first start | normal for big downloads. The container keeps loading after the CLI gives up. Watch `vllm-logs` / `vllm-status`. Next time use `--timeout 3600` or `--no-wait`. Re-running `vllm-up` replaces the container |

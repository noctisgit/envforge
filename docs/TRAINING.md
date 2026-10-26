# Training: from a strong model's wins to a fine-tuned open model

Phase 3 turns evaluation into learning. A strong API model (the *teacher*, e.g. Claude
Sonnet) solves many generated tasks; the verifier keeps only the clean wins; a small
open-weight model (the *student*, e.g. Qwen2.5-7B-Instruct) is fine-tuned to imitate
them; the same eval measures the student before and after. The technique is
**rejection sampling + supervised fine-tuning (SFT)**, a form of distillation.

You can't fine-tune Sonnet itself — its weights are closed. You can make a model you
*own* much better at your environments.

```
teacher runs (seeds 10000+) ──► train dataset ──► LoRA SFT on a GPU box ──► adapter
                                     │                                         │
student baseline (seeds 0–99) ───────┴── check-overlap ── student + adapter (seeds 0–99)
                                                                │
                                                     compare: the gain is the learning
```

**Where the GPU work runs.** Steps 3–5 need a GPU box. On a VM or bare metal, envforge runs
vLLM and training in Docker (the default). On **RunPod or Vast**, run the envforge GPU image
as the pod and add `--runtime pod` to every `compute` / `train` command that takes `--host`,
or set `ENVFORGE_GPU_RUNTIME=pod` once. Setup is in docs/COMPUTE.md, section 6. The
commands below are otherwise identical.

**In the browser.** Steps 2–5 have pages in the web app, built on the same code:
**Datasets** (pick runs on training seeds, live preview with the eval-seed overlap check,
save), **Fine-tune** (dataset + base model + GPU box + hyperparameters with the CLI's
defaults; live steps, loss curve, log tail, elapsed time, estimated cost; cancel) and
**Models** (adapters with lineage, including DPO / GRPO parents; *Test* serves base + adapter and queues a run on seeds 0+,
then links to Compare with the teacher and the base model; serve, download `.tar.gz`, curl /
Python snippets). GPU boxes are added under Settings → GPU boxes. The Home guide links each
step to its page. Details in docs/WEB.md.

## 1. Collect teacher episodes — on *training* seeds

Every seed is a different generated task. Keep two disjoint ranges: **train on seeds
≥ 10000, evaluate on 0–99**, so the eval never contains a task the student studied.

Environments made with the builder (or `envforge env`) have a fixed list of tickets
instead, so many seeds land on the same ticket. They hold out about a quarter of their
tickets for evaluation: seeds below 10000 pick only those, seeds from 10000 only the
rest. The same two seed ranges therefore work unchanged, and the overlap check compares
tickets for them (docs/BUILDER.md, held-out tasks). With few tickets, an evaluation of
100 episodes repeats each held-out ticket several times; that measures consistency, not
more coverage. A run over tiers 1–3 also cycles the tiers evenly, so when the tiers hold
different numbers of tickets, the smaller tiers repeat and count for more in the score.
For a fair score use tier `0` ("every task once" in the web run form): with as many
episodes as held-out tickets (the pack's `task_counts`), each ticket comes up exactly
once, tier by tier. Seeds from 10000 with tier 0 walk every training ticket once instead.

```bash
uv run envforge run netops --model anthropic:claude-sonnet-5-5 --episodes 500 --seed-start 10000
uv run envforge run sipops --model anthropic:claude-sonnet-5-5 --episodes 500 --seed-start 10000
```

In the web UI: set **Seed start** to 10000 on the Runs page. Budget: netops costs about
$0.05 per Sonnet episode, so 1,000 episodes ≈ $50. A few hundred clean episodes per
environment is a reasonable first dataset; thousands is better.

## 2. Build the dataset (rejection sampling)

```bash
uv run envforge train dataset data/workspaces/default/runs/<run-a> data/workspaces/default/runs/<run-b> \
    --name netops-sonnet-v1 --min-reward 1.0
```

This keeps episodes with reward ≥ 1.0 (clean solves, no collateral), keeps one episode
per task (the shortest best one), holds out ~5% of *tasks* for validation, and writes
`data/workspaces/default/sft/netops-sonnet-v1/`:

| file | contents |
|---|---|
| `train.jsonl`, `val.jsonl` | one conversation per line: system prompt, task + first observation, then the teacher's turns (text + tool calls) and the environment's replies, plus the tool schemas |
| `manifest.json` | source runs and models, counts per pack/tier, and every `(pack, tier, seed)` used |

The web UI's **Datasets** page builds the same files (and manifest) with a live preview;
`best per task` off keeps every passing episode, still split by task.

## 3. Baseline the student — on *evaluation* seeds

Serve the untrained base model and evaluate it exactly as you will evaluate the result:

```bash
uv run envforge compute vllm-up Qwen/Qwen2.5-7B-Instruct --host gpu-1 --api-key "$VLLM_API_KEY"
VLLM_BASE_URL=http://gpu-1:8000/v1 uv run envforge run netops \
    --model vllm:Qwen/Qwen2.5-7B-Instruct --episodes 100 --seed-start 0
uv run envforge train check-overlap netops-sonnet-v1 data/workspaces/default/runs/<baseline-run>
```

On a RunPod/Vast pod reached over Tailscale, use
`envforge compute vllm-up Qwen/Qwen2.5-7B-Instruct --host gpu-1 --runtime pod --bind 127.0.0.1`.
The endpoint is then reachable only over the tailnet, at the same `http://gpu-1:8000/v1`.

`check-overlap` exits non-zero if the eval run contains any training task.

## 4. Fine-tune with LoRA on the GPU box

```bash
uv run envforge train sft netops-sonnet-v1 --base Qwen/Qwen2.5-7B-Instruct --host gpu-1
```

What happens:
- the dataset and `envforge/train/sft.py` are copied to `/root/envforge/jobs/<name>/` on the box;
- a detached container `envforge-train-<name>` (the pinned vLLM image, which already has
  CUDA PyTorch + transformers, plus peft/datasets/accelerate) trains a **LoRA adapter**
  (with `--runtime pod`: a detached process `train-<name>` inside the pod, which already has
  those libraries, logging to `/root/envforge/logs/train-<name>.log`):
  small low-rank matrices added to the frozen base model, so a 7B model trains on one GPU
  and the result is a few hundred MB instead of a full copy of the weights;
- **the loss is computed only on the assistant's tokens** — the student learns what to
  think and which tool to call next; it is never trained to predict the environment's output;
- progress (loss per step, eval loss per epoch) streams back; `--detach` and later
  `envforge train follow <name> --host gpu-1` if you'd rather not keep the terminal open
  (add `--runtime pod` on a pod). `envforge train logs <name>` shows the raw output,
  including tracebacks.

Defaults: 2 epochs, lr 2e-4, LoRA r=16/α=32 on all linear layers, max length 8192 tokens
(longer episodes are dropped, not truncated), effective batch 8, gradient checkpointing.
`--max-steps 20` makes a quick smoke test. Check a dataset locally without a GPU:
`uv sync --extra train && uv run envforge train sft netops-sonnet-v1 --base Qwen/Qwen2.5-7B-Instruct --dry-run`.

**GPU sizing** (LoRA + gradient checkpointing, estimates): a 3B model needs ~14–18 GB at
max length 8192, so a 24 GB card works; a 7B model needs ~24–28 GB, so use an A100 40GB
or L40S. Real episodes are usually ~2–4k tokens, which needs much less. The script saves
only the final adapter (no mid-run checkpoints), so an interrupted run starts over —
keep first runs short with `--max-steps`.

Then copy the adapter home (it also stays on the box for serving):

```bash
uv run envforge train fetch netops-sonnet-v1 --host gpu-1     # -> data/workspaces/default/adapters/netops-sonnet-v1/
```

On a pod this matters more: `/root/envforge/adapters/` is on the container disk, which is
wiped when the pod stops. Fetch before you stop it.

`envforge_manifest.json` in the adapter records the base model, dataset hash, example
counts, hyper-parameters, final train/eval loss and library versions.

## 5. Evaluate the student + adapter

```bash
uv run envforge compute vllm-up Qwen/Qwen2.5-7B-Instruct --host gpu-1 --lora netops-sonnet-v1 --max-lora-rank 16
VLLM_BASE_URL=http://gpu-1:8000/v1 uv run envforge run netops \
    --model vllm:netops-sonnet-v1 --episodes 100 --seed-start 0
uv run envforge compare data/workspaces/default/runs/<baseline-run> data/workspaces/default/runs/<adapter-run>
```

(On a pod: the same command with `--runtime pod`; the adapter is read straight from
`/root/envforge/adapters/<name>`.)

vLLM serves the base model and each adapter under its own name, so `vllm:<base>` and
`vllm:<adapter>` can be compared side by side (in the web UI: add the vLLM provider on your
Profile with the box's URL, then pick either model and use **Compare**).

## Reading the result

- **Solve rate up on unseen seeds** — the student learned the skill. Check tiers separately:
  gains usually come first on tier 1.
- **Collateral rate** — imitation of a careful teacher should keep it near zero; if it rises,
  the student learned to "fix" by breaking things — inspect transcripts.
- **Train loss ↓ but eval loss ↑** — overfitting: fewer epochs, more data, or lower lr.
- **Great on training seeds, poor on eval seeds** — memorisation; that's why the seed ranges
  are separate and `check-overlap` exists.
- Imitation can't exceed the teacher. Phase 4 (DPO, then GRPO, below) uses the verifier's
  reward directly so the student can improve beyond it.

## Beyond imitation (Phase 4): DPO and GRPO

SFT only ever shows the model what to do. Both methods below continue from an SFT adapter
and also use what went *wrong*, judged by the same verifier.

### DPO: prefer the solved attempt over the failed one

**The idea.** For one task, take a **chosen** episode (reward 1.0) and a **rejected** one
(reward 0). DPO raises the model's probability of the chosen answer relative to where it
started, and lowers the rejected one's, through
`-log sigmoid(beta * [(log pi(chosen) - log ref(chosen)) - (log pi(rejected) - log ref(rejected))])`.
`ref` is the starting adapter, frozen; `beta` (0.1) is how hard to push and how far the
model may drift. Only the agent's own tokens count, exactly as in SFT. Code:
`envforge/train/preference.py` (pairs), `envforge/train/dpo.py` (the script, runs on the
box next to `sft.py` and `lora_tools.py`).

**Recipe (web app).**

1. **Student failures on training seeds.** Serve your SFT adapter (Models → Serve), then on
   Runs pick `vllm:<adapter>`, Purpose *Training data · seeds 10000+*, the same seeds and
   tiers as a teacher run you already have. Its failures are the rejected side: the
   mistakes this exact model makes are what it should unlearn.
2. **Datasets → Preference pairs (DPO).** Tick the teacher run and the student run. The
   preview shows pairs, how many tasks had both a pass and a failure ("never failed" tasks
   give no pair), and the overlap check against evaluation runs, as for SFT.
3. **Fine-tune → DPO**, start from your SFT adapter (the base model comes with it), beta 0.1,
   1–2 epochs. The chart adds **chosen preferred**: the share of pairs where the chosen
   answer already has the higher implicit reward. It should climb towards 1 on training
   pairs; held-out pairs (the validation split) tell you whether it generalises.
4. **Test** the DPO adapter on evaluation seeds and Compare with the SFT adapter.

**Reading it.** Accuracy near 1 with held-out accuracy near 0.5 = memorised the pairs;
lower beta or fewer epochs, or more pairs. A solve-rate drop with rising margins = it moved
too far from the reference (raise beta). DPO suits *decision* errors (refused when it
should have acted); it can't teach behaviour no chosen episode shows.

### GRPO: learn from your own attempts, scored by the environment

**The idea.** Reinforcement learning with the verifier as the reward and no learned critic.
Each round, the current adapter attempts each of K training tasks G times (temperature 1).
An attempt's **advantage** is how it scored against its own group:
`A = (reward - mean) / std`. Above-average attempts become more likely, below-average ones
less likely, through the PPO-style clipped objective per token
`-min(ratio * A, clip(ratio, 1 - eps, 1 + eps) * A) + kl_beta * KL(pi || ref)`, where
`ratio = pi / pi_old` and `ref` is the adapter the run started from. Groups where every
attempt scored the same (all solved, or all failed) carry no signal and are skipped; that
is why GRPO wants a model that already solves *some* tasks: start from SFT, not from the
base model. Code: `envforge/train/grpo_step.py` (one update, on the box);
`envforge/web/training_service.py`, section GRPO (the loop).

**One round** (all automatic, after **Fine-tune → GRPO**, start-from adapter, environment):

```
serve base + current adapter ─► rollout run: K tasks x G attempts on seeds 10000 + r*K
        ▲                                   (a normal run in Runs: replays, rewards)
        │                                                   │
next round ◄── <name>-r<k> ◄── grpo_step.py on the box ◄── group advantages
```

The last round's adapter is saved as the job's name and registered like any other.
Defaults: 4 rounds x 16 tasks x 8 attempts, lr 1e-5, KL weight 0.04, clip 0.2. On a 3B
model and one A6000 expect very roughly 15–30 minutes per round (rollouts run 8 at a time
on vLLM; vLLM is stopped for each update because both need the GPU's memory).

**Reading it.** The job page's *Rounds* table is the learning curve: mean reward and solve
rate on that round's **training** tasks, and how many groups had signal. Rising reward is
the goal, but it is measured on training tasks; the real test is still the evaluation
seeds (Models → Test). Watch the KL line: steady growth is fine; a jump means the policy is
moving fast (lower the learning rate or raise the KL weight). Read some rollout replays
too: RL optimises exactly what the verifier rewards, loopholes included (reward hacking).

**Limits of this implementation (on purpose, for readability).** Attempts are generated by
vLLM and re-tokenised for training (vLLM may have tokenised the same text differently; the
error is small). Generation and training alternate on one GPU, so a round waits for the
rollouts, then for the update. A production setup runs them side by side on separate GPUs.

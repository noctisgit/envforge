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

## 1. Collect teacher episodes — on *training* seeds

Every seed is a different generated task. Keep two disjoint ranges: **train on seeds
≥ 10000, evaluate on 0–99**, so the eval never contains a task the student studied.

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

The web UI's **Export training data** produces the same chat format for ad-hoc use.

## 3. Baseline the student — on *evaluation* seeds

Serve the untrained base model and evaluate it exactly as you will evaluate the result:

```bash
uv run envforge compute vllm-up Qwen/Qwen2.5-7B-Instruct --host gpu-1 --api-key "$VLLM_API_KEY"
VLLM_BASE_URL=http://gpu-1:8000/v1 uv run envforge run netops \
    --model vllm:Qwen/Qwen2.5-7B-Instruct --episodes 100 --seed-start 0
uv run envforge train check-overlap netops-sonnet-v1 data/workspaces/default/runs/<baseline-run>
```

`check-overlap` exits non-zero if the eval run contains any training task.

## 4. Fine-tune with LoRA on the GPU box

```bash
uv run envforge train sft netops-sonnet-v1 --base Qwen/Qwen2.5-7B-Instruct --host gpu-1
```

What happens:
- the dataset and `envforge/train/sft.py` are copied to `/root/envforge/jobs/<name>/` on the box;
- a detached container `envforge-train-<name>` (the pinned vLLM image, which already has
  CUDA PyTorch + transformers, plus peft/datasets/accelerate) trains a **LoRA adapter**:
  small low-rank matrices added to the frozen base model, so a 7B model trains on one GPU
  and the result is a few hundred MB instead of a full copy of the weights;
- **the loss is computed only on the assistant's tokens** — the student learns what to
  think and which tool to call next; it is never trained to predict the environment's output;
- progress (loss per step, eval loss per epoch) streams back; `--detach` and later
  `envforge train follow <name> --host gpu-1` if you'd rather not keep the terminal open.

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

`envforge_manifest.json` in the adapter records the base model, dataset hash, example
counts, hyper-parameters, final train/eval loss and library versions.

## 5. Evaluate the student + adapter

```bash
uv run envforge compute vllm-up Qwen/Qwen2.5-7B-Instruct --host gpu-1 --lora netops-sonnet-v1 --max-lora-rank 16
VLLM_BASE_URL=http://gpu-1:8000/v1 uv run envforge run netops \
    --model vllm:netops-sonnet-v1 --episodes 100 --seed-start 0
uv run envforge compare data/workspaces/default/runs/<baseline-run> data/workspaces/default/runs/<adapter-run>
```

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
- Imitation can't exceed the teacher. Phase 4 (DPO, then GRPO) uses the verifier's reward
  directly so the student can improve beyond it.

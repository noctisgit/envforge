"""One GRPO update of a LoRA adapter from scored rollouts. Standalone, like sft.py.

Copied to the GPU box next to sft.py and lora_tools.py and run once per round:

    python grpo_step.py --data rollouts.jsonl --base Qwen/Qwen2.5-3B-Instruct \\
        --policy-adapter adapters/run-r2 --ref-adapter adapters/qwen-abcd-v1 --out adapters/run-r3

The control plane does the rest of the loop (web/training_service.py): serve the policy
with vLLM, let it attempt each training task several times, score every attempt with
the environment's verifier, then hand the scored attempts to this script.

Input: one rollout per line, `{"messages": [...], "tools": [...], "advantage": a, ...}`.

The ML in one paragraph
-----------------------
GRPO (Group Relative Policy Optimization) is reinforcement learning without a learned
critic. Each task is attempted G times; an attempt's *advantage* is how much better it
scored than its group: A = (reward - mean) / std. Above-average attempts are made more
likely, below-average ones less likely, by the PPO-style clipped objective per token

    ratio = pi(token) / pi_old(token)          (pi_old: the policy that produced the rollout)
    loss  = -min(ratio * A, clip(ratio, 1 - eps, 1 + eps) * A) + kl_beta * KL(pi || ref)

averaged over the agent's own tokens of each attempt, then over attempts. The clip stops
one batch from moving the policy too far; the KL term (the "k3" estimator,
exp(ref - logp) - (ref - logp) - 1, which is >= 0) keeps it near the reference, here the
SFT adapter the run started from, so it doesn't forget how to do the job while chasing
reward. Groups where every attempt scored the same carry no signal (A = 0) and are left
out by the control plane before this script runs.

Rollouts are re-tokenised from their text with the chat template, as sft.py does. vLLM
might have produced a different (equally valid) tokenisation of the same text; for a
learning project that approximation is fine.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))  # sft.py sits next to this file
import lora_tools  # noqa: E402
import sft  # noqa: E402

FORMAT_VERSION = 1


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="grpo_step.py", description="One GRPO update from scored rollouts."
    )
    p.add_argument("--data", required=True, type=Path, help="scored rollouts (JSONL)")
    p.add_argument("--base", required=True, help="base model on the HF Hub or a local path")
    p.add_argument("--policy-adapter", required=True, type=Path,
                   help="the adapter that produced the rollouts (updated from here)")  # fmt: skip
    p.add_argument("--ref-adapter", type=Path, default=None,
                   help="the KL anchor; default: the policy adapter itself")  # fmt: skip
    p.add_argument("--out", required=True, type=Path, help="output directory for the new adapter")
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--kl-beta", type=float, default=0.04, help="weight of the KL penalty")
    p.add_argument("--clip-eps", type=float, default=0.2, help="PPO clip range")
    p.add_argument("--epochs", type=float, default=1.0, help="passes over this batch")
    p.add_argument("--grad-accum", type=int, default=8, help="rollouts per optimizer step")
    p.add_argument("--max-len", type=int, default=8192, help="drop longer rollouts")
    p.add_argument("--warmup-ratio", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--dry-run", action="store_true", help="tokenise only and print stats")
    return p


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.kl_beta < 0 or not 0 < args.clip_eps < 1:
        parser.error("--kl-beta must be >= 0 and --clip-eps in (0, 1)")
    if args.max_len <= 0 or args.grad_accum <= 0 or args.epochs <= 0:
        parser.error("--max-len, --grad-accum and --epochs must be positive")
    return args


def read_rollouts(path: Path) -> list[dict[str, Any]]:
    """Scored rollouts: chat-format examples with an "advantage" number each."""
    if not path.is_file():
        raise sft.SFTError(f"data file not found: {path}")
    out = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        obj = json.loads(line)
        if not isinstance(obj.get("messages"), list) or not isinstance(
            obj.get("advantage"), int | float
        ):
            raise sft.SFTError(f'{path}:{lineno}: expected "messages" and a numeric "advantage"')
        out.append(obj)
    return out


def tokenize_rollouts(
    rollouts: list[dict[str, Any]],
    make_render: Callable[[list[dict[str, Any]] | None], sft.Render],
    max_len: int,
    stats: dict[str, Any],
) -> list[dict[str, Any]]:
    rows = []
    for r in rollouts:
        tok = sft.build_labels(sft.normalize_messages(r["messages"]), make_render(r.get("tools")))
        stats["rollouts_in"] = stats.get("rollouts_in", 0) + 1
        if len(tok.input_ids) > max_len:
            stats["dropped_too_long"] = stats.get("dropped_too_long", 0) + 1
            continue
        if tok.n_assistant_tokens == 0:
            stats["dropped_no_assistant_tokens"] = stats.get("dropped_no_assistant_tokens", 0) + 1
            continue
        stats["kept"] = stats.get("kept", 0) + 1
        rows.append(
            {"ids": tok.input_ids, "labels": tok.labels, "advantage": float(r["advantage"])}
        )
    return rows


def grpo_loss(logp: Any, old_logp: Any, ref_logp: Any, advantage: float, *, clip_eps: float,
              kl_beta: float) -> tuple[Any, Any, Any]:  # fmt: skip
    """(loss, mean KL, share of clipped tokens) for one rollout's per-token log-probs."""
    import torch

    ratio = torch.exp(logp - old_logp)
    unclipped = ratio * advantage
    clipped = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * advantage
    surrogate = torch.minimum(unclipped, clipped)
    log_ratio_ref = ref_logp - logp
    kl = torch.exp(log_ratio_ref) - log_ratio_ref - 1  # k3: unbiased, always >= 0
    loss = (-surrogate + kl_beta * kl).mean()
    clip_frac = ((ratio - 1).abs() > clip_eps).float().mean()
    return loss, kl.mean(), clip_frac


def train(args: argparse.Namespace, tokenizer: Any, rows: list, stats: dict[str, Any],
          log: sft.EventLog | None = None) -> int:  # fmt: skip
    import torch

    started_at = sft._now()
    args.out.mkdir(parents=True, exist_ok=True)
    log = log or sft.EventLog(args.out / "metrics.jsonl")
    cuda, dtype, args.bf16 = lora_tools.device_and_dtype(args.bf16)
    total_steps = max(1, math.ceil(len(rows) * args.epochs / args.grad_accum))
    log.emit("start", method="grpo", base_model=args.base, n_train=len(rows), n_val=0,
             total_steps=total_steps, cuda=cuda, bf16=args.bf16, kl_beta=args.kl_beta,
             mean_abs_advantage=sum(abs(r["advantage"]) for r in rows) / len(rows), stats=stats)  # fmt: skip

    model = lora_tools.load_policy(args.base, adapter_dir=args.policy_adapter, dtype=dtype, cuda=cuda,
                                   gradient_checkpointing=args.gradient_checkpointing)  # fmt: skip
    n_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in lora_tools.trainable_parameters(model))
    log.emit(
        "model", params=n_params, trainable_params=trainable, trainable_frac=trainable / n_params
    )

    def logp(row: dict) -> Any:
        return lora_tools.label_logprobs(model, row["ids"], row["labels"])

    # Before any update: the old policy's log-probs (for the ratio) and the reference's
    # (for the KL). The reference is a second, frozen adapter loaded beside the policy.
    model.eval()
    with torch.no_grad():
        for row in rows:
            row["old"] = logp(row).cpu()
        same_ref = (
            args.ref_adapter is None
            or Path(args.ref_adapter).resolve() == Path(args.policy_adapter).resolve()
        )
        if same_ref:
            for row in rows:
                row["ref"] = row["old"]
        else:
            model.load_adapter(str(args.ref_adapter), adapter_name="ref")
            model.set_adapter("ref")
            for row in rows:
                row["ref"] = logp(row).cpu()
            model.set_adapter("default")  # also makes the policy's LoRA weights trainable again
            model.delete_adapter("ref")

    opt, sched = lora_tools.make_optimizer(model, args.lr, total_steps, args.warmup_ratio)
    rng = random.Random(args.seed)
    model.train()
    step, seen, window = 0, 0, []
    log_every = max(1, min(10, total_steps // 20))
    n_epochs = math.ceil(args.epochs)
    for epoch in range(n_epochs):
        order = list(range(len(rows)))
        rng.shuffle(order)
        if epoch == n_epochs - 1 and args.epochs % 1:
            order = order[: max(1, round(len(order) * (args.epochs % 1)))]
        for k, i in enumerate(order):
            row = rows[i]
            new = logp(row)
            loss, kl, clip_frac = grpo_loss(new, row["old"].to(new.device), row["ref"].to(new.device),
                                            row["advantage"], clip_eps=args.clip_eps, kl_beta=args.kl_beta)  # fmt: skip
            (loss / args.grad_accum).backward()
            window.append(tuple(float(x.detach()) for x in (loss, kl, clip_frac)))
            seen += 1
            if seen % args.grad_accum and k != len(order) - 1:
                continue
            grad_norm = lora_tools.optimizer_step(model, opt, sched)
            step += 1
            if step % log_every == 0 or step == total_steps:
                n = len(window)
                log.emit("log", loss=sum(w[0] for w in window) / n, kl=sum(w[1] for w in window) / n,
                         clip_frac=sum(w[2] for w in window) / n, grad_norm=grad_norm,
                         learning_rate=sched.get_last_lr()[0], step=step,
                         epoch=round(epoch + (k + 1) / len(order), 4))  # fmt: skip
                window = []
            if step >= total_steps:
                break
        if step >= total_steps:
            break

    model.save_pretrained(str(args.out))
    if tokenizer is not None:
        tokenizer.save_pretrained(str(args.out))
    losses = [e["loss"] for e in log.events if e["event"] == "log"]
    manifest = {
        "format_version": FORMAT_VERSION, "method": "grpo", "base_model": args.base,
        "policy_adapter": str(args.policy_adapter), "ref_adapter": str(args.ref_adapter or args.policy_adapter),
        "kl_beta": args.kl_beta, "clip_eps": args.clip_eps, "n_train": len(rows), "stats": stats,
        "dataset_sha256": sft.dataset_sha256([args.data]),
        "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "started_at": started_at, "finished_at": sft._now(), "global_step": step,
        "final_train_loss": losses[-1] if losses else None, "trainable_params": trainable,
        "library_versions": sft._library_versions(),
    }  # fmt: skip
    (args.out / "envforge_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    log.emit("done", out=str(args.out), global_step=step, final_train_loss=manifest["final_train_loss"],
             final_eval_loss=None)  # fmt: skip
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        tokenizer = sft.load_tokenizer(args.base)

        def make_render(tools: list[dict[str, Any]] | None) -> sft.Render:
            return sft.make_renderer(tokenizer, tools)

        stats: dict[str, Any] = {}
        rows = tokenize_rollouts(read_rollouts(args.data), make_render, args.max_len, stats)
        if not rows:
            raise sft.SFTError(f"no rollouts left after filtering ({json.dumps(stats)})")
        if args.dry_run:
            print(json.dumps(stats, indent=2))
            return 0
        return train(args, tokenizer, rows, stats)
    except sft.SFTError as e:
        print(sft.EVENT_PREFIX + json.dumps({"event": "error", "message": str(e)}), flush=True)
        print(f"grpo_step.py: error: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())

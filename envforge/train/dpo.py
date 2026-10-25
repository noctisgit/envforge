"""Direct Preference Optimization (DPO) with LoRA on envforge preference pairs. Standalone.

Copied to the GPU box next to sft.py and lora_tools.py (it reuses their tokenising,
masking and model code) and run there:

    python dpo.py --data train.jsonl --val val.jsonl --base Qwen/Qwen2.5-3B-Instruct \\
        --init-adapter adapters/qwen-abcd-v1 --out adapters/qwen-abcd-dpo1

Input: the preference JSONL train/preference.py writes, one pair per line:
`{"prompt": [...], "chosen": [...], "rejected": [...], "tools": [...]}`.

The ML in one paragraph
-----------------------
For a prompt x, a chosen answer y_w and a rejected one y_l, DPO maximises

    log sigmoid( beta * [ (log pi(y_w|x) - log ref(y_w|x)) - (log pi(y_l|x) - log ref(y_l|x)) ] )

where pi is the model being trained and ref a frozen copy of where it started (here:
the SFT adapter). Each bracket is the *implicit reward* of an answer: how much more
likely the model has made it than it was. The loss pushes the chosen answer's reward
above the rejected one's; beta (0.1 is usual) sets how hard, and how far the model may
drift from ref. No reward model and no sampling: just two forward passes per pair. Log-
probabilities are summed over the agent's own tokens only (system, task and tool output
are masked, exactly as in SFT). The reference log-probabilities are computed once, with
the starting adapter, before the first update, so only one copy of the model is needed.

Progress lines are `ENVFORGE {...}` events like sft.py's (start, model, log, eval, done),
so the same orchestrator follows both. `log` adds `accuracy` (share of pairs where the
chosen answer already has the higher reward) and `margin` (mean reward gap).
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
    p = argparse.ArgumentParser(prog="dpo.py", description="LoRA DPO on envforge preference pairs.")
    p.add_argument("--data", required=True, type=Path, help="training pairs (JSONL)")
    p.add_argument("--val", type=Path, default=None, help="validation pairs (JSONL)")
    p.add_argument("--base", required=True, help="base model on the HF Hub or a local path")
    p.add_argument("--init-adapter", type=Path, default=None,
                   help="LoRA adapter to start from (and use as the reference); usually the SFT one")  # fmt: skip
    p.add_argument("--out", required=True, type=Path, help="output directory for the adapter")
    p.add_argument(
        "--beta", type=float, default=0.1, help="DPO temperature: higher = stay closer to ref"
    )
    p.add_argument("--epochs", type=float, default=1.0)
    p.add_argument("--lr", type=float, default=2e-5, help="peak learning rate (lower than SFT's)")
    p.add_argument("--lora-r", type=int, default=16, help="only without --init-adapter")
    p.add_argument("--lora-alpha", type=int, default=32, help="only without --init-adapter")
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--target-modules", default="all-linear")
    p.add_argument("--max-len", type=int, default=8192, help="drop pairs with a longer side")
    p.add_argument("--grad-accum", type=int, default=8, help="pairs per optimizer step")
    p.add_argument("--warmup-ratio", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--max-steps", type=int, default=0, help="stop after N optimizer steps")
    p.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--dry-run", action="store_true", help="tokenise only and print stats")
    return p


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.beta <= 0:
        parser.error("--beta must be positive")
    if args.max_len <= 0 or args.grad_accum <= 0 or args.epochs <= 0:
        parser.error("--max-len, --grad-accum and --epochs must be positive")
    return args


# --------------------------------------------------------------------------- data


def read_pairs(path: Path) -> list[dict[str, Any]]:
    """Preference JSONL: objects with "prompt", "chosen" and "rejected" message lists."""
    if not path.is_file():
        raise sft.SFTError(f"data file not found: {path}")
    pairs = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as e:
            raise sft.SFTError(f"{path}:{lineno}: invalid JSON ({e})") from e
        if not all(isinstance(obj.get(k), list) for k in ("prompt", "chosen", "rejected")):
            raise sft.SFTError(f'{path}:{lineno}: expected "prompt", "chosen" and "rejected" lists')
        pairs.append(obj)
    return pairs


def tokenize_pairs(
    pairs: list[dict[str, Any]],
    make_render: Callable[[list[dict[str, Any]] | None], sft.Render],
    max_len: int,
    stats: dict[str, int],
) -> list[dict[str, list[int]]]:
    """Both sides of every pair as (ids, labels), labels only on the agent's own tokens.

    A pair is dropped if either side is longer than `max_len` or has nothing to learn
    from; both sides must survive, or the comparison is meaningless.
    """
    rows = []
    for p in pairs:
        render = make_render(p.get("tools"))
        sides = {}
        for side in ("chosen", "rejected"):
            tok = sft.build_labels(sft.normalize_messages(p["prompt"] + p[side]), render)
            sides[side] = tok
        stats["pairs_in"] = stats.get("pairs_in", 0) + 1
        if any(len(t.input_ids) > max_len for t in sides.values()):
            stats["dropped_too_long"] = stats.get("dropped_too_long", 0) + 1
            continue
        if any(t.n_assistant_tokens == 0 for t in sides.values()):
            stats["dropped_no_assistant_tokens"] = stats.get("dropped_no_assistant_tokens", 0) + 1
            continue
        stats["kept"] = stats.get("kept", 0) + 1
        stats["tokens_max"] = max(
            stats.get("tokens_max", 0), *(len(t.input_ids) for t in sides.values())
        )
        rows.append({
            "chosen_ids": sides["chosen"].input_ids, "chosen_labels": sides["chosen"].labels,
            "rejected_ids": sides["rejected"].input_ids, "rejected_labels": sides["rejected"].labels,
        })  # fmt: skip
    return rows


# --------------------------------------------------------------------------- the loss


def dpo_loss(policy_chosen: Any, policy_rejected: Any, ref_chosen: Any, ref_rejected: Any,
             beta: float) -> tuple[Any, Any, Any, Any]:  # fmt: skip
    """(loss, margin, chosen_reward, rejected_reward) for summed log-probabilities.

    reward = beta * (log pi - log ref): how much more likely the answer has become.
    loss = -log sigmoid(chosen_reward - rejected_reward). Works on scalars or batches.
    """
    import torch.nn.functional as F

    chosen_reward = beta * (policy_chosen - ref_chosen)
    rejected_reward = beta * (policy_rejected - ref_rejected)
    margin = chosen_reward - rejected_reward
    return -F.logsigmoid(margin), margin, chosen_reward, rejected_reward


def _total_steps(args: argparse.Namespace, n_train: int) -> int:
    if args.max_steps > 0:
        return args.max_steps
    return max(1, math.ceil(n_train * args.epochs / args.grad_accum))


def train(args: argparse.Namespace, tokenizer: Any, train_rows: list, val_rows: list,
          stats: dict[str, Any], log: sft.EventLog | None = None) -> int:  # fmt: skip
    """Reference pass, then DPO updates; saves the adapter and a manifest to --out."""
    import torch

    started_at = sft._now()
    args.out.mkdir(parents=True, exist_ok=True)
    log = log or sft.EventLog(args.out / "metrics.jsonl")
    cuda, dtype, args.bf16 = lora_tools.device_and_dtype(args.bf16)
    total_steps = _total_steps(args, len(train_rows))
    log.emit("start", method="dpo", base_model=args.base, init_adapter=str(args.init_adapter or ""),
             n_train=len(train_rows), n_val=len(val_rows), total_steps=total_steps,
             cuda=cuda, bf16=args.bf16, beta=args.beta, stats=stats)  # fmt: skip

    model = lora_tools.load_policy(
        args.base, adapter_dir=args.init_adapter, dtype=dtype, cuda=cuda,
        gradient_checkpointing=args.gradient_checkpointing, lora_r=args.lora_r,
        lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout, target_modules=args.target_modules,
    )  # fmt: skip
    n_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in lora_tools.trainable_parameters(model))
    log.emit(
        "model", params=n_params, trainable_params=trainable, trainable_frac=trainable / n_params
    )

    def logp(row: dict, side: str) -> Any:
        return lora_tools.label_logprobs(model, row[f"{side}_ids"], row[f"{side}_labels"]).sum()

    # The reference: the starting adapter's log-probabilities, frozen before any update.
    model.eval()
    with torch.no_grad():
        for row in [*train_rows, *val_rows]:
            row["ref_chosen"] = float(logp(row, "chosen"))
            row["ref_rejected"] = float(logp(row, "rejected"))

    def evaluate(step: int, epoch: float) -> None:
        if not val_rows:
            return
        model.eval()
        losses, correct = [], 0
        with torch.no_grad():
            for row in val_rows:
                loss, margin, _, _ = dpo_loss(logp(row, "chosen"), logp(row, "rejected"),
                                              row["ref_chosen"], row["ref_rejected"], args.beta)  # fmt: skip
                losses.append(float(loss))
                correct += float(margin) > 0
        log.emit("eval", loss=sum(losses) / len(losses), accuracy=correct / len(val_rows),
                 step=step, epoch=round(epoch, 4))  # fmt: skip
        model.train()

    opt, sched = lora_tools.make_optimizer(model, args.lr, total_steps, args.warmup_ratio)
    rng = random.Random(args.seed)
    model.train()
    step, seen, window = 0, 0, []
    log_every = max(1, min(10, total_steps // 20))
    n_epochs = math.ceil(args.epochs)
    for epoch in range(n_epochs):
        order = list(range(len(train_rows)))
        rng.shuffle(order)
        if epoch == n_epochs - 1 and args.epochs % 1:  # a fractional last epoch
            order = order[: max(1, round(len(order) * (args.epochs % 1)))]
        for k, i in enumerate(order):
            row = train_rows[i]
            loss, margin, cr, rr = dpo_loss(logp(row, "chosen"), logp(row, "rejected"),
                                            row["ref_chosen"], row["ref_rejected"], args.beta)  # fmt: skip
            (loss / args.grad_accum).backward()
            window.append(tuple(float(x.detach()) for x in (loss, margin, cr, rr)))
            seen += 1
            if seen % args.grad_accum and k != len(order) - 1:
                continue
            grad_norm = lora_tools.optimizer_step(model, opt, sched)
            step += 1
            if step % log_every == 0 or step == total_steps:
                n = len(window)
                log.emit("log", loss=sum(w[0] for w in window) / n,
                         accuracy=sum(w[1] > 0 for w in window) / n,
                         margin=sum(w[1] for w in window) / n,
                         chosen_reward=sum(w[2] for w in window) / n,
                         rejected_reward=sum(w[3] for w in window) / n,
                         grad_norm=grad_norm, learning_rate=sched.get_last_lr()[0],
                         step=step, epoch=round(epoch + (k + 1) / len(order), 4))  # fmt: skip
                window = []
            if step >= total_steps:
                break
        evaluate(step, epoch + 1)
        if step >= total_steps:
            break

    model.save_pretrained(str(args.out))
    if tokenizer is not None:
        tokenizer.save_pretrained(str(args.out))

    def last(event: str, key: str) -> float | None:
        values = [e[key] for e in log.events if e["event"] == event and key in e]
        return values[-1] if values else None

    manifest = {
        "format_version": FORMAT_VERSION, "method": "dpo", "base_model": args.base,
        "init_adapter": str(args.init_adapter or ""), "beta": args.beta,
        "data_files": [str(p) for p in [args.data, args.val] if p is not None],
        "dataset_sha256": sft.dataset_sha256(p for p in [args.data, args.val] if p is not None),
        "n_train": len(train_rows), "n_val": len(val_rows), "stats": stats,
        "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "started_at": started_at, "finished_at": sft._now(), "global_step": step,
        "final_train_loss": last("log", "loss"), "final_eval_loss": last("eval", "loss"),
        "final_train_accuracy": last("log", "accuracy"), "final_eval_accuracy": last("eval", "accuracy"),
        "trainable_params": trainable, "library_versions": sft._library_versions(),
    }  # fmt: skip
    (args.out / "envforge_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    log.emit("done", out=str(args.out), global_step=step, final_train_loss=manifest["final_train_loss"],
             final_eval_loss=manifest["final_eval_loss"])  # fmt: skip
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.val is not None and not args.val.is_file():
            raise sft.SFTError(f"validation file not found: {args.val}")
        tokenizer = sft.load_tokenizer(args.base)

        def make_render(tools: list[dict[str, Any]] | None) -> sft.Render:
            return sft.make_renderer(tokenizer, tools)

        stats: dict[str, Any] = {}
        train_rows = tokenize_pairs(read_pairs(args.data), make_render, args.max_len, stats)
        val_stats: dict[str, Any] = {}
        val_rows = (tokenize_pairs(read_pairs(args.val), make_render, args.max_len, val_stats)
                    if args.val else [])  # fmt: skip
        stats = {**stats, "n_train": len(train_rows), "n_val": len(val_rows)}
        if not train_rows:
            raise sft.SFTError(f"no training pairs left after filtering ({json.dumps(stats)})")
        if args.dry_run:
            print(json.dumps(stats, indent=2))
            return 0
        return train(args, tokenizer, train_rows, val_rows, stats)
    except sft.SFTError as e:
        print(sft.EVENT_PREFIX + json.dumps({"event": "error", "message": str(e)}), flush=True)
        print(f"dpo.py: error: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())

"""Supervised fine-tuning (SFT) with LoRA on envforge chat data. Standalone script.

This file is copied to a rented GPU box and run there:

    pip install torch transformers peft datasets accelerate
    python sft.py --data train.jsonl --val-frac 0.05 \\
        --base Qwen/Qwen2.5-7B-Instruct --out runs/sft-1

It deliberately imports nothing from envforge (only the standard library at module
level; torch / transformers / peft / datasets inside functions), so `--help` and the
pure helpers work anywhere, and `--dry-run` needs only `transformers` for the tokenizer.

Input is the chat-format JSONL that `envforge filter --format chat` writes: one
conversation per line, `{"messages": [...], "tools": [...], "meta": {...}}`, where the
messages are OpenAI-style (system, user, assistant with optional `tool_calls`, tool).

The ML in one paragraph
-----------------------
SFT on rejection-sampled trajectories ("keep the episodes the verifier liked, then
imitate them") is plain next-token prediction on those conversations. Two details matter:

* **Loss only on assistant tokens.** The system prompt, the task and the tool outputs
  were written by the environment, not the model. Training on them would teach the model
  to predict simulator output instead of how to act. So every non-assistant token gets
  the label -100, which PyTorch's cross-entropy ignores; only what the model itself said
  (its text, its tool calls, its end-of-turn token so it learns to stop) is learned.
* **LoRA (Low-Rank Adaptation).** Instead of updating all ~7B weights, we freeze them and
  learn a small correction for chosen linear layers: W' = W + (alpha / r) * B @ A, where
  A is r x d_in and B is d_out x r with r (e.g. 16) far smaller than d. That is well
  under 1% of the parameters, so optimizer state fits on one GPU, and the result is a
  ~100 MB "adapter" that vLLM can serve on top of the unchanged base model.

Progress is printed as machine-readable lines, `ENVFORGE {"event": ...}`, so an
orchestrator tailing the SSH stream can parse them; the same events go to metrics.jsonl.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import sys
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any

FORMAT_VERSION = 1
IGNORE_INDEX = -100  # label value that torch.nn.CrossEntropyLoss skips
EVENT_PREFIX = "ENVFORGE "

# A renderer turns a list of chat messages into token ids. `add_generation_prompt=True`
# appends the header that opens a new assistant turn (e.g. "<|im_start|>assistant\n").
Render = Callable[..., list[int]]


class SFTError(Exception):
    """A user-facing error: bad input, empty dataset, unusable environment."""


# --------------------------------------------------------------------------- CLI


def build_parser() -> argparse.ArgumentParser:
    """The command-line interface (kept separate from `main` so tests can parse args)."""
    p = argparse.ArgumentParser(
        prog="sft.py",
        description="LoRA supervised fine-tuning on envforge chat-format JSONL.",
    )
    p.add_argument("--data", required=True, type=Path, help="training JSONL (chat format)")
    val = p.add_mutually_exclusive_group()
    val.add_argument("--val", type=Path, default=None, help="validation JSONL")
    val.add_argument(
        "--val-frac",
        type=float,
        default=0.0,
        help="hold out this fraction of --data for validation (default 0: no validation)",
    )
    p.add_argument("--base", required=True, help="base model on the HF Hub or a local path")
    p.add_argument("--out", required=True, type=Path, help="output directory for the adapter")
    p.add_argument("--epochs", type=float, default=2.0)
    p.add_argument(
        "--lr", type=float, default=2e-4, help="peak learning rate (LoRA likes ~1e-4..2e-4)"
    )
    p.add_argument("--lora-r", type=int, default=16, help="LoRA rank r")
    p.add_argument(
        "--lora-alpha", type=int, default=32, help="LoRA scale; update is scaled by alpha/r"
    )
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument(
        "--target-modules",
        default="all-linear",
        help="'all-linear' or a comma-separated list such as q_proj,k_proj,v_proj,o_proj",
    )
    p.add_argument("--max-len", type=int, default=8192, help="drop (not truncate) longer examples")
    p.add_argument("--batch-size", type=int, default=1, help="per-device batch size")
    p.add_argument("--grad-accum", type=int, default=8, help="gradient accumulation steps")
    p.add_argument("--warmup-ratio", type=float, default=0.03)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--bf16",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="train in bfloat16 (default: on if the GPU supports it)",
    )
    p.add_argument(
        "--max-steps", type=int, default=0, help="stop after N optimizer steps (overrides --epochs)"
    )
    p.add_argument(
        "--gradient-checkpointing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="recompute activations in the backward pass to save memory (default on)",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="tokenise only and print dataset stats as JSON; no model weights, no GPU",
    )
    return p


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse and sanity-check command-line arguments."""
    parser = build_parser()
    args = parser.parse_args(argv)
    if not 0.0 <= args.val_frac < 1.0:
        parser.error("--val-frac must be in [0, 1)")
    if args.max_len <= 0 or args.batch_size <= 0 or args.grad_accum <= 0:
        parser.error("--max-len, --batch-size and --grad-accum must be positive")
    return args


# --------------------------------------------------------------------------- data


def read_examples(path: Path) -> list[dict[str, Any]]:
    """Read chat-format JSONL: one `{"messages": [...], "tools"?: [...]}` object per line."""
    if not path.is_file():
        raise SFTError(f"data file not found: {path}")
    examples: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise SFTError(f"{path}:{lineno}: invalid JSON ({e})") from e
            if not isinstance(obj, dict) or not isinstance(obj.get("messages"), list):
                raise SFTError(f'{path}:{lineno}: expected an object with a "messages" list')
            examples.append(obj)
    return examples


def dataset_sha256(paths: Iterable[Path]) -> str:
    """One SHA-256 over the bytes of the data file(s), in order: identifies the exact data."""
    h = hashlib.sha256()
    for path in paths:
        h.update(path.read_bytes())
    return h.hexdigest()


def normalize_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Make OpenAI-format messages acceptable to Hugging Face chat templates.

    - OpenAI sends tool-call arguments as a JSON *string*; HF templates expect a dict
      (they `tojson` it themselves), so a string would come out double-encoded.
    - Tool messages keep `tool_call_id`, and gain `name` (looked up from the assistant
      call they answer) because some templates print the tool's name.
    - `content: None` becomes "" so templates don't print "None".
    """
    call_names: dict[str, str] = {}
    out: list[dict[str, Any]] = []
    for msg in messages:
        m = dict(msg)
        if m.get("content") is None:
            m["content"] = ""
        if m.get("role") == "assistant" and m.get("tool_calls"):
            calls = []
            for call in m["tool_calls"]:
                call = dict(call)
                fn = dict(call.get("function", {}))
                args = fn.get("arguments")
                if isinstance(args, str):
                    try:
                        fn["arguments"] = json.loads(args) if args.strip() else {}
                    except json.JSONDecodeError:
                        pass  # leave malformed arguments as the model wrote them
                call["function"] = fn
                if call.get("id"):
                    call_names[call["id"]] = fn.get("name", "")
                calls.append(call)
            m["tool_calls"] = calls
        elif m.get("role") == "tool" and "name" not in m:
            name = call_names.get(m.get("tool_call_id", ""))
            if name:
                m["name"] = name
        out.append(m)
    return out


def split_examples(
    examples: list[dict[str, Any]], val_frac: float, seed: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Seeded random train/validation split. Keeps at least one training example."""
    if val_frac <= 0 or len(examples) < 2:
        return examples, []
    order = list(range(len(examples)))
    random.Random(seed).shuffle(order)
    n_val = min(len(examples) - 1, max(1, round(len(examples) * val_frac)))
    val_idx = set(order[:n_val])
    train = [e for i, e in enumerate(examples) if i not in val_idx]
    val = [e for i, e in enumerate(examples) if i in val_idx]
    return train, val


# --------------------------------------------------------------------------- masking


@dataclass
class Tokenized:
    """One conversation as model inputs: token ids plus labels (-100 = not trained on)."""

    input_ids: list[int]
    labels: list[int]
    mismatch: bool = False  # the template rewrote history; the fallback locator was used
    unlocated_turns: int = 0  # assistant turns the fallback could not find (left masked)

    @property
    def n_assistant_tokens(self) -> int:
        return sum(1 for x in self.labels if x != IGNORE_INDEX)


def _is_prefix(short: list[int], long: list[int]) -> bool:
    return len(short) <= len(long) and long[: len(short)] == short


def _common_prefix_len(a: list[int], b: list[int]) -> int:
    n = 0
    for x, y in zip(a, b, strict=False):
        if x != y:
            break
        n += 1
    return n


def _find_sublist(haystack: list[int], needle: list[int], start: int) -> int:
    """Index of the first occurrence of `needle` in `haystack` at or after `start`, else -1."""
    if not needle:
        return -1
    first = needle[0]
    for i in range(start, len(haystack) - len(needle) + 1):
        if haystack[i] == first and haystack[i : i + len(needle)] == needle:
            return i
    return -1


def _rfind_sublist(haystack: list[int], needle: list[int]) -> int:
    """Index of the last occurrence of `needle` in `haystack`, else -1."""
    for i in range(len(haystack) - len(needle), -1, -1):
        if haystack[i : i + len(needle)] == needle:
            return i
    return -1


def _assistant_segment(
    messages: list[dict[str, Any]], i: int, before: list[int], through: list[int], render: Render
) -> list[int]:
    """The tokens assistant turn i adds, found without assuming renderings share prefixes.

    The generation header ("<|im_start|>assistant\n" for ChatML) is what
    add_generation_prompt=True appends; the turn's own tokens are whatever follows the
    last copy of that header in `through`.
    """
    plain = render(messages[:i], add_generation_prompt=False) if i > 0 else []
    header = before[_common_prefix_len(plain, before) :]
    pos = _rfind_sublist(through, header) if header else -1
    if pos >= 0:
        return through[pos + len(header) :]
    return through[_common_prefix_len(before, through) :]


def build_labels(messages: list[dict[str, Any]], render: Render) -> Tokenized:
    """Tokenise a conversation and label only the tokens each assistant turn contributes.

    Works for any chat template because it never parses the template; it only diffs
    renderings. For the assistant message at index i:

        before  = render(messages[:i], add_generation_prompt=True)   # ends with the
                                                                      # assistant header
        through = render(messages[:i + 1], add_generation_prompt=False)

    The tokens `through` adds after `before` are exactly what the model generated in
    that turn (content, tool calls, end-of-turn token), so those positions of the full
    rendering keep their token id as the label; everything else becomes -100.

    That relies on each rendering being a token-prefix of the next. Some templates
    rewrite history (e.g. put the message count in the system prompt, or strip
    reasoning from earlier assistant turns), which breaks it. Then we fall back to
    `_assistant_segment`: cut the turn's tokens out of `through` (everything after the
    last assistant header) and search for them in the full rendering. Turns that still
    can't be found (the template renders them differently once they are history) stay
    masked and are counted in `unlocated_turns`.
    """
    full = render(messages, add_generation_prompt=False)
    labels = [IGNORE_INDEX] * len(full)
    mismatch = False
    unlocated = 0
    cursor = 0  # labelled spans must appear in order
    for i, msg in enumerate(messages):
        if msg.get("role") != "assistant":
            continue
        before = render(messages[:i], add_generation_prompt=True) if i > 0 else []
        through = render(messages[: i + 1], add_generation_prompt=False)
        if (
            _is_prefix(before, through)
            and _is_prefix(through, full)
            and len(before) >= cursor
            and len(through) > len(before)
        ):
            start, end = len(before), len(through)
        else:
            mismatch = True
            segment = _assistant_segment(messages, i, before, through, render)
            start = _find_sublist(full, segment, cursor)
            if start < 0:
                unlocated += 1
                continue
            end = start + len(segment)
        labels[start:end] = full[start:end]
        cursor = end
    return Tokenized(full, labels, mismatch=mismatch, unlocated_turns=unlocated)


def make_renderer(tokenizer: Any, tools: list[dict[str, Any]] | None) -> Render:
    """A `render(messages, add_generation_prompt=False) -> token ids` for a HF tokenizer."""

    def render(messages: list[dict[str, Any]], add_generation_prompt: bool = False) -> list[int]:
        out = tokenizer.apply_chat_template(
            messages,
            tools=tools or None,
            tokenize=True,
            return_dict=False,  # transformers 5 defaults to a dict; we want the id list
            add_generation_prompt=add_generation_prompt,
        )
        if isinstance(out, dict) or hasattr(out, "input_ids"):
            out = out["input_ids"]
        return list(out)

    return render


@dataclass
class DatasetStats:
    """Counters gathered while tokenising; printed by --dry-run and saved in the manifest."""

    examples_in: int = 0
    kept: int = 0
    dropped_too_long: int = 0
    dropped_no_assistant_tokens: int = 0
    template_mismatches: int = 0
    unlocated_turns: int = 0
    tokens_total: int = 0
    tokens_max: int = 0
    assistant_tokens: int = 0


def tokenize_examples(
    examples: list[dict[str, Any]],
    make_render: Callable[[list[dict[str, Any]] | None], Render],
    max_len: int,
    stats: DatasetStats,
) -> list[dict[str, list[int]]]:
    """Normalise, tokenise and mask every example; drop over-long or label-less ones.

    Over-long examples are dropped rather than truncated: cutting a trajectory mid-way
    would teach the model episodes that just stop, and could cut off the final turn.
    """
    rows: list[dict[str, list[int]]] = []
    for ex in examples:
        stats.examples_in += 1
        tok = build_labels(normalize_messages(ex["messages"]), make_render(ex.get("tools")))
        if tok.mismatch:
            stats.template_mismatches += 1
        stats.unlocated_turns += tok.unlocated_turns
        if len(tok.input_ids) > max_len:
            stats.dropped_too_long += 1
            continue
        if tok.n_assistant_tokens == 0:
            stats.dropped_no_assistant_tokens += 1
            continue
        stats.kept += 1
        stats.tokens_total += len(tok.input_ids)
        stats.tokens_max = max(stats.tokens_max, len(tok.input_ids))
        stats.assistant_tokens += tok.n_assistant_tokens
        rows.append({"input_ids": tok.input_ids, "labels": tok.labels})
    return rows


def summarize(train: DatasetStats, val: DatasetStats) -> dict[str, Any]:
    """The JSON stats object --dry-run prints."""
    total = train.tokens_total + val.tokens_total
    kept = train.kept + val.kept
    return {
        "n_train": train.kept,
        "n_val": val.kept,
        "dropped_too_long": train.dropped_too_long + val.dropped_too_long,
        "dropped_no_assistant_tokens": train.dropped_no_assistant_tokens
        + val.dropped_no_assistant_tokens,
        "template_mismatches": train.template_mismatches + val.template_mismatches,
        "unlocated_assistant_turns": train.unlocated_turns + val.unlocated_turns,
        "tokens_total": total,
        "tokens_mean": round(total / kept, 1) if kept else 0,
        "tokens_max": max(train.tokens_max, val.tokens_max),
        "assistant_token_fraction": round(
            (train.assistant_tokens + val.assistant_tokens) / total, 4
        )
        if total
        else 0.0,
    }


# --------------------------------------------------------------------------- events


class EventLog:
    """Prints `ENVFORGE {json}` lines to stdout and appends the same JSON to metrics.jsonl."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = path
        self.events: list[dict[str, Any]] = []
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("")

    def emit(self, event: str, **fields: Any) -> dict[str, Any]:
        record = {"event": event, "time": round(time.time(), 3), **fields}
        self.events.append(record)
        line = json.dumps(record, default=str)
        print(EVENT_PREFIX + line, flush=True)
        if self.path is not None:
            with self.path.open("a", encoding="utf-8") as f:
                f.write(line + "\n")
        return record


def _now() -> str:
    """Current UTC time, ISO 8601. (`timezone.utc`, not `datetime.UTC`: runs on Python 3.10.)"""
    return datetime.now(timezone.utc).isoformat()  # noqa: UP017


def _library_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in ("torch", "transformers", "peft", "datasets", "accelerate"):
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def _params_from_name(model_name: str) -> float | None:
    """Guess a model's size in billions from its name ("Qwen2.5-7B" -> 7.0), if it says."""
    m = re.search(r"(\d+(?:\.\d+)?)\s*[bB](?![a-zA-Z])", model_name)
    return float(m.group(1)) if m else None


def _warn_cpu(what: str) -> None:
    """Shout on stderr: training a real model without a GPU is almost always a mistake."""
    bar = "!" * 72
    print(
        f"\n{bar}\n!! WARNING: no CUDA GPU found and {what}.\n"
        "!! CPU training is fine for tiny test models; a multi-billion-parameter model\n"
        f"!! will be extremely slow and may run out of RAM.\n{bar}\n",
        file=sys.stderr,
        flush=True,
    )


# --------------------------------------------------------------------------- pipeline


def load_and_tokenize(args: argparse.Namespace, tokenizer: Any) -> tuple[list, list, dict]:
    """Read, split and tokenise the data. Returns (train_rows, val_rows, stats)."""
    train_ex = read_examples(args.data)
    if args.val is not None:
        val_ex = read_examples(args.val)
    else:
        train_ex, val_ex = split_examples(train_ex, args.val_frac, args.seed)

    def make_render(tools: list[dict[str, Any]] | None) -> Render:
        return make_renderer(tokenizer, tools)

    train_stats, val_stats = DatasetStats(), DatasetStats()
    train_rows = tokenize_examples(train_ex, make_render, args.max_len, train_stats)
    val_rows = tokenize_examples(val_ex, make_render, args.max_len, val_stats)
    stats = summarize(train_stats, val_stats)
    if not train_rows:
        raise SFTError(
            f"no training examples left after filtering ({json.dumps(stats)}); "
            "check --max-len and the data file"
        )
    return train_rows, val_rows, stats


def load_tokenizer(base: str) -> Any:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(base)
    if tokenizer.chat_template is None:
        raise SFTError(f"{base} has no chat template; use an -Instruct/-Chat model")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token  # padding is masked out anyway
    return tokenizer


def _total_steps(args: argparse.Namespace, n_train: int) -> int:
    """Optimizer steps the run will take: one step per batch_size * grad_accum examples."""
    if args.max_steps > 0:
        return args.max_steps
    batches = math.ceil(n_train / args.batch_size)
    return max(1, math.ceil(math.ceil(batches / args.grad_accum) * args.epochs))


def linear_module_names(model: Any) -> list[str]:
    """Short names of every linear layer except the output head, e.g. ["q_proj", ..., "up_proj"].

    This is what peft's "all-linear" means, spelled out. We resolve it ourselves so the
    saved adapter_config.json lists plain suffixes (which is what vLLM and older peft
    versions expect) instead of one full path per layer.
    """
    import torch

    head = model.get_output_embeddings()
    names = {
        name.rsplit(".", 1)[-1]
        for name, module in model.named_modules()
        if isinstance(module, torch.nn.Linear) and module is not head
    }
    return sorted(names)


def train(
    args: argparse.Namespace, tokenizer: Any, train_rows: list, val_rows: list, stats: dict
) -> int:
    """Load the base model, wrap it with LoRA, run the Trainer, save adapter + manifest."""
    import torch
    import transformers
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model
    from transformers import (
        AutoModelForCausalLM,
        DataCollatorForSeq2Seq,
        Trainer,
        TrainerCallback,
        TrainingArguments,
    )

    started_at = _now()
    args.out.mkdir(parents=True, exist_ok=True)
    log = EventLog(args.out / "metrics.jsonl")
    cuda = torch.cuda.is_available()
    if args.bf16 is None:
        args.bf16 = bool(cuda and torch.cuda.is_bf16_supported())
    dtype = torch.bfloat16 if args.bf16 else torch.float32

    size_b = _params_from_name(args.base)
    if not cuda and size_b is not None and size_b >= 1:
        _warn_cpu(f"{args.base} looks like a {size_b:g}B model")

    total_steps = _total_steps(args, len(train_rows))
    log.emit(
        "start",
        base_model=args.base,
        n_train=len(train_rows),
        n_val=len(val_rows),
        total_steps=total_steps,
        cuda=cuda,
        bf16=args.bf16,
        stats=stats,
    )

    # transformers renamed `torch_dtype` to `dtype` in 4.56.
    major, minor = (int(x) for x in transformers.__version__.split(".")[:2])
    dtype_kw = "dtype" if (major, minor) >= (4, 56) else "torch_dtype"
    model = AutoModelForCausalLM.from_pretrained(
        args.base, **{dtype_kw: dtype}, device_map="auto" if cuda else None
    )
    n_params = sum(p.numel() for p in model.parameters())
    if not cuda and n_params > 1e9 and not (size_b and size_b >= 1):
        _warn_cpu(f"{args.base} has {n_params / 1e9:.1f}B parameters")

    # Gradient checkpointing: keep only some activations in the forward pass and
    # recompute the rest during backward. ~30% slower, but activation memory for an
    # 8k-token sequence drops from tens of GB to a few. `enable_input_require_grads`
    # is needed because the frozen embeddings would otherwise produce outputs with no
    # grad, and checkpointed blocks then have nothing to backpropagate through.
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
        model.config.use_cache = False  # the KV cache is for generation, not training

    if args.target_modules == "all-linear":
        targets = linear_module_names(model)
    else:
        targets = [t.strip() for t in args.target_modules.split(",") if t.strip()]
    lora = LoraConfig(
        task_type="CAUSAL_LM",
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=targets,
    )
    model = get_peft_model(model, lora)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.emit(
        "model", params=n_params, trainable_params=trainable, trainable_frac=trainable / n_params
    )

    class ProgressCallback(TrainerCallback):
        """Turns the Trainer's log dicts into ENVFORGE events."""

        def on_log(
            self, _args: Any, state: Any, control: Any, logs: dict | None = None, **kw: Any
        ) -> None:
            logs = dict(logs or {})
            fields = {"step": state.global_step, "epoch": round(state.epoch or 0.0, 4)}
            if "eval_loss" in logs:
                metrics = {k.removeprefix("eval_"): v for k, v in logs.items()}
                log.emit("eval", **{**metrics, **fields})
            elif "loss" in logs:
                log.emit("log", **{**logs, **fields})
            elif "train_runtime" in logs:
                log.emit("train_summary", **{**logs, **fields})

    # Effective batch = batch_size * grad_accum sequences per optimizer step. Gradient
    # accumulation sums gradients over several small forward/backward passes before one
    # update, so a 1-sequence-at-a-time GPU still gets a stable batch of 8.
    # Cosine schedule with warmup: ramp the LR up from 0 for the first few % of steps
    # (avoids wrecking the model with big early updates), then decay it smoothly to 0.
    training_args = TrainingArguments(
        output_dir=str(args.out / "trainer"),
        num_train_epochs=args.epochs,
        max_steps=args.max_steps if args.max_steps > 0 else -1,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_steps=math.ceil(args.warmup_ratio * total_steps),
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        bf16=args.bf16,
        logging_steps=max(1, min(10, total_steps // 20)),
        eval_strategy="epoch" if val_rows else "no",
        save_strategy="no",  # we save the adapter ourselves at the end
        report_to="none",
        seed=args.seed,
        remove_unused_columns=False,
    )
    collator = DataCollatorForSeq2Seq(
        tokenizer,
        padding=True,
        label_pad_token_id=IGNORE_INDEX,  # padding must never contribute to the loss
        pad_to_multiple_of=8 if cuda else None,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=collator,
        train_dataset=Dataset.from_list(train_rows),
        eval_dataset=Dataset.from_list(val_rows) if val_rows else None,
        callbacks=[ProgressCallback()],
    )
    result = trainer.train()

    def last(event: str, key: str) -> float | None:
        values = [e[key] for e in log.events if e["event"] == event and key in e]
        return values[-1] if values else None

    if val_rows and last("eval", "loss") is None:
        trainer.evaluate()  # --max-steps ended before the first epoch-end evaluation

    # Standard PEFT layout (adapter_config.json + adapter_model.safetensors): this is what
    # `vllm serve BASE --enable-lora --lora-modules name=OUTDIR` loads.
    model.save_pretrained(str(args.out))
    tokenizer.save_pretrained(str(args.out))
    trainer_dir = args.out / "trainer"
    if trainer_dir.is_dir() and not any(trainer_dir.iterdir()):
        trainer_dir.rmdir()  # nothing was checkpointed there (save_strategy="no")

    manifest = {
        "format_version": FORMAT_VERSION,
        "base_model": args.base,
        "data_files": [str(p) for p in [args.data, args.val] if p is not None],
        "dataset_sha256": dataset_sha256(p for p in [args.data, args.val] if p is not None),
        "n_train": len(train_rows),
        "n_val": len(val_rows),
        "dropped_too_long": stats["dropped_too_long"],
        "stats": stats,
        "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "started_at": started_at,
        "finished_at": _now(),
        "global_step": result.global_step,
        "final_train_loss": last("log", "loss"),
        "mean_train_loss": result.training_loss,
        "final_eval_loss": last("eval", "loss"),
        "trainable_params": trainable,
        "library_versions": _library_versions(),
    }
    (args.out / "envforge_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    log.emit(
        "done",
        out=str(args.out),
        global_step=result.global_step,
        final_train_loss=manifest["final_train_loss"],
        final_eval_loss=manifest["final_eval_loss"],
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point: parse args, tokenise, then either print stats (--dry-run) or train."""
    args = parse_args(argv)
    try:
        if not args.data.is_file():
            raise SFTError(f"data file not found: {args.data}")
        if args.val is not None and not args.val.is_file():
            raise SFTError(f"validation file not found: {args.val}")
        tokenizer = load_tokenizer(args.base)
        train_rows, val_rows, stats = load_and_tokenize(args, tokenizer)
        if args.dry_run:
            print(json.dumps(stats, indent=2))
            return 0
        return train(args, tokenizer, train_rows, val_rows, stats)
    except SFTError as e:
        print(EVENT_PREFIX + json.dumps({"event": "error", "message": str(e)}), flush=True)
        print(f"sft.py: error: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())

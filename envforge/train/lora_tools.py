"""PyTorch helpers shared by dpo.py and grpo_step.py. Standalone, like sft.py.

Copied to the GPU box next to sft.py, dpo.py and grpo_step.py. Only the standard library
at module level (torch / transformers / peft are imported inside functions), so the
scripts' `--help` and their pure helpers work without a GPU.

The one idea worth understanding here is `label_logprobs`: preference and RL training
need the model's log-probability of *the tokens the agent actually wrote*, one number
per token. Position t of the model's output predicts token t+1, so the log-probability
of labelled token t+1 is log_softmax(logits[t])[token t+1].
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

IGNORE_INDEX = -100  # same convention as sft.py: labels of -100 are not trained on


def device_and_dtype(bf16: bool | None) -> tuple[bool, Any, bool]:
    """(cuda available, dtype to load in, bf16 on?). bf16 defaults to on if the GPU can."""
    import torch

    cuda = torch.cuda.is_available()
    if bf16 is None:
        bf16 = bool(cuda and torch.cuda.is_bf16_supported())
    return cuda, (torch.bfloat16 if bf16 else torch.float32), bf16


def load_policy(
    base: str,
    *,
    adapter_dir: str | Path | None,
    dtype: Any,
    cuda: bool,
    gradient_checkpointing: bool = True,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    target_modules: str = "all-linear",
) -> Any:
    """The base model with a *trainable* LoRA adapter on top, dropout switched off.

    With `adapter_dir`, training continues from that adapter (an SFT adapter for DPO,
    the latest round's for GRPO): same rank, same target layers. Without it, a fresh
    adapter is created as sft.py does.

    Dropout is off because DPO and GRPO compare the policy's log-probabilities with ones
    computed in eval mode (the reference, the old policy): with dropout on in training
    the two differ before any update, so the GRPO ratio isn't 1 at the start and clipping
    fires on noise. TRL disables it for the same reason.
    """
    import transformers
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoModelForCausalLM

    major, minor = (int(x) for x in transformers.__version__.split(".")[:2])
    dtype_kw = "dtype" if (major, minor) >= (4, 56) else "torch_dtype"
    model = AutoModelForCausalLM.from_pretrained(
        base, **{dtype_kw: dtype}, device_map="auto" if cuda else None
    )
    if gradient_checkpointing:  # see sft.py: trade ~30% speed for far less activation memory
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
        model.config.use_cache = False
    if adapter_dir is not None:
        return disable_dropout(
            PeftModel.from_pretrained(model, str(adapter_dir), is_trainable=True)
        )
    if target_modules == "all-linear":
        from sft import linear_module_names  # next to this file on the box

        targets = linear_module_names(model)
    else:
        targets = [t.strip() for t in target_modules.split(",") if t.strip()]
    config = LoraConfig(task_type="CAUSAL_LM", r=lora_r, lora_alpha=lora_alpha,
                        lora_dropout=lora_dropout, target_modules=targets)  # fmt: skip
    return disable_dropout(get_peft_model(model, config))


def disable_dropout(model: Any) -> Any:
    """Set every dropout layer's probability to 0 (LoRA's included); returns the model."""
    import torch

    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0.0
    return model


def label_logprobs(model: Any, input_ids: list[int], labels: list[int]) -> Any:
    """Log-probability of every labelled token, as a 1-D tensor (with grad if enabled).

    Only the positions that predict a labelled token are pushed through the output layer
    (`logits_to_keep`): an agent episode is ~95% prompt and tool output, and full logits
    for 7k positions x a 150k-token vocabulary would cost gigabytes for nothing.
    """
    import torch

    device = next(model.parameters()).device
    lab = torch.tensor(labels)
    keep = (lab[1:] != IGNORE_INDEX).nonzero().squeeze(-1)  # positions t whose next token counts
    if keep.numel() == 0:
        return torch.zeros(0, device=device)
    ids = torch.tensor([input_ids], device=device)
    try:
        logits = model(input_ids=ids, logits_to_keep=keep.to(device)).logits[0]
    except TypeError:  # older transformers: no logits_to_keep, slice the full output
        logits = model(input_ids=ids).logits[0][keep.to(device)]
    targets = lab[keep + 1].to(device)
    return torch.log_softmax(logits.float(), dim=-1).gather(-1, targets[:, None]).squeeze(-1)


def trainable_parameters(model: Any) -> list[Any]:
    return [p for p in model.parameters() if p.requires_grad]


def make_optimizer(model: Any, lr: float, total_steps: int, warmup_ratio: float) -> tuple[Any, Any]:
    """AdamW over the LoRA weights, with the same warmup + cosine schedule as sft.py."""
    import torch
    from transformers import get_cosine_schedule_with_warmup

    opt = torch.optim.AdamW(trainable_parameters(model), lr=lr, weight_decay=0.0)
    warmup = math.ceil(warmup_ratio * total_steps)
    return opt, get_cosine_schedule_with_warmup(opt, warmup, max(1, total_steps))


def optimizer_step(model: Any, opt: Any, sched: Any, max_grad_norm: float = 1.0) -> float:
    """Clip, step, schedule, zero; returns the gradient norm before clipping."""
    import torch

    norm = torch.nn.utils.clip_grad_norm_(trainable_parameters(model), max_grad_norm)
    opt.step()
    sched.step()
    opt.zero_grad(set_to_none=True)
    return float(norm)

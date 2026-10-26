"""Tests for envforge/train/grpo_step.py: args, the clipped loss, and a real update of a
tiny random Qwen2 model on CPU (needs torch + peft: `--extra train`)."""

from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path

import pytest
from test_train_dpo import TRAIN_DIR, WordTemplate, _load, _tiny_base

grpo = _load("grpo_step")
sft = grpo.sft


def _rollout(answer: str, advantage: float) -> dict:
    return {"messages": [{"role": "system", "content": "fix it"}, {"role": "user", "content": "R1 down"},
                         {"role": "assistant", "content": answer}], "tools": None, "advantage": advantage}  # fmt: skip


def test_parse_args():
    a = grpo.parse_args(["--data", "d", "--base", "b", "--policy-adapter", "p", "--out", "o"])
    assert a.kl_beta == 0.04 and a.clip_eps == 0.2 and a.epochs == 1.0 and a.ref_adapter is None
    with pytest.raises(SystemExit):
        grpo.parse_args(
            [
                "--data",
                "d",
                "--base",
                "b",
                "--policy-adapter",
                "p",
                "--out",
                "o",
                "--clip-eps",
                "1.5",
            ]
        )


def test_read_rollouts_needs_an_advantage(tmp_path):
    p = tmp_path / "r.jsonl"
    p.write_text(json.dumps(_rollout("ok", 0.5)) + "\n")
    assert grpo.read_rollouts(p)[0]["advantage"] == 0.5
    p.write_text(json.dumps({"messages": []}) + "\n")
    with pytest.raises(sft.SFTError, match="advantage"):
        grpo.read_rollouts(p)


def test_grpo_loss_math():
    torch = pytest.importorskip("torch")
    lp = torch.tensor([-1.0, -2.0])
    # on-policy, at the reference: ratio 1 and KL 0, so the loss is just -advantage
    loss, kl, clip = grpo.grpo_loss(lp, lp, lp, 0.7, clip_eps=0.2, kl_beta=0.1)
    assert math.isclose(float(loss), -0.7, rel_tol=1e-6) and float(kl) == 0 and float(clip) == 0
    # a token already 2x more likely than when sampled: a positive advantage is clipped at 1.2
    loss, _, clip = grpo.grpo_loss(
        lp + math.log(2), lp, lp + math.log(2), 1.0, clip_eps=0.2, kl_beta=0.0
    )
    assert math.isclose(float(loss), -1.2, rel_tol=1e-5) and float(clip) == 1.0
    # drifting from the reference costs KL (k3 >= 0)
    _, kl, _ = grpo.grpo_loss(lp, lp, lp - 0.5, 0.0, clip_eps=0.2, kl_beta=1.0)
    assert float(kl) > 0


def test_script_runs_standalone():
    out = subprocess.run([sys.executable, str(TRAIN_DIR / "grpo_step.py"), "--help"],
                         capture_output=True, text=True, cwd="/")  # fmt: skip
    assert out.returncode == 0 and "--policy-adapter" in out.stdout
    assert "envforge" not in "".join(
        ln
        for ln in (TRAIN_DIR / "grpo_step.py").read_text().splitlines()
        if ln.startswith(("import", "from"))
    )


def _adapter(base: Path, path: Path) -> Path:
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM

    model = get_peft_model(AutoModelForCausalLM.from_pretrained(base),
                           LoraConfig(task_type="CAUSAL_LM", r=4, lora_alpha=8,
                                      target_modules=["q_proj", "v_proj", "o_proj"]))  # fmt: skip
    model.save_pretrained(path)
    return path


def _answer_logprob(base: Path, adapter: Path, row: dict) -> float:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    lora_tools = grpo.lora_tools
    model = PeftModel.from_pretrained(
        AutoModelForCausalLM.from_pretrained(base), str(adapter)
    ).eval()
    with torch.no_grad():
        return float(lora_tools.label_logprobs(model, row["ids"], row["labels"]).sum())


@pytest.mark.parametrize("separate_ref", [False, True])
def test_grpo_step_makes_better_attempts_more_likely(tmp_path, separate_ref):
    base = _tiny_base(tmp_path)
    policy = _adapter(base, tmp_path / "policy")
    t = WordTemplate()
    rollouts = [_rollout("good answer", 1.0), _rollout("bad answer", -1.0)] * 2
    rows = grpo.tokenize_rollouts(rollouts, lambda tools: t.render, 1000, {})
    good, bad = rows[0], rows[1]
    before = _answer_logprob(base, policy, good) - _answer_logprob(base, policy, bad)

    data = tmp_path / "rollouts.jsonl"
    data.write_text("".join(json.dumps(r) + "\n" for r in rollouts))
    argv = ["--data", str(data), "--base", str(base), "--policy-adapter", str(policy),
            "--out", str(tmp_path / "out"), "--lr", "5e-3", "--grad-accum", "2", "--epochs", "6",
            "--no-gradient-checkpointing", "--kl-beta", "0.01"]  # fmt: skip
    if separate_ref:
        argv += ["--ref-adapter", str(_adapter(base, tmp_path / "ref"))]
    log = sft.EventLog(None)
    assert grpo.train(grpo.parse_args(argv), None, [dict(r) for r in rows], {}, log=log) == 0

    after = _answer_logprob(base, tmp_path / "out", good) - _answer_logprob(
        base, tmp_path / "out", bad
    )
    assert after > before + 0.5  # the above-average attempt gained on the below-average one
    logs = [e for e in log.events if e["event"] == "log"]
    assert logs and all(e["kl"] >= 0 for e in logs) and log.events[-1]["event"] == "done"
    manifest = json.loads((tmp_path / "out" / "envforge_manifest.json").read_text())
    assert manifest["method"] == "grpo" and manifest["n_train"] == 4


def test_policy_loads_without_dropout_so_training_matches_the_reference(tmp_path):
    """With LoRA dropout left on, train-mode log-probs differ from eval-mode ones before
    any update, so GRPO's ratio isn't 1 at the start (audit finding C8)."""
    torch = pytest.importorskip("torch")
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM

    base = _tiny_base(tmp_path)
    adapter = tmp_path / "noisy"
    model = get_peft_model(AutoModelForCausalLM.from_pretrained(base),
                           LoraConfig(task_type="CAUSAL_LM", r=4, lora_alpha=8, lora_dropout=0.5,
                                      target_modules=["q_proj", "v_proj"]))  # fmt: skip
    model.save_pretrained(adapter)
    lora_tools = grpo.lora_tools
    policy = lora_tools.load_policy(str(base), adapter_dir=adapter, dtype=torch.float32, cuda=False,
                                    gradient_checkpointing=False)  # fmt: skip
    t = WordTemplate()
    row = grpo.tokenize_rollouts(
        [_rollout("good answer here", 1.0)], lambda tools: t.render, 1000, {}
    )[0]
    policy.eval()
    with torch.no_grad():
        ref = lora_tools.label_logprobs(policy, row["ids"], row["labels"])
    policy.train()
    with torch.no_grad():
        now = lora_tools.label_logprobs(policy, row["ids"], row["labels"])
    assert torch.allclose(ref, now)  # ratio exactly 1 before any update

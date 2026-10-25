"""Tests for envforge/train/dpo.py (and lora_tools.py): args, pair masking, the loss, and a
real training run on a tiny random Qwen2 model on CPU (needs torch + peft: `--extra train`).

Like sft.py, dpo.py is a standalone script copied to GPU boxes, so it is loaded from its
file; it imports sft.py and lora_tools.py from its own directory.
"""

from __future__ import annotations

import importlib.util
import json
import math
import subprocess
import sys
from pathlib import Path

import pytest

TRAIN_DIR = Path(__file__).resolve().parents[1] / "envforge" / "train"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(
        f"envforge_{name}_standalone", TRAIN_DIR / f"{name}.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


dpo = _load("dpo")
sft = dpo.sft
IGNORE = sft.IGNORE_INDEX


class WordTemplate:
    """Each message renders as <s> ROLE words... </s>; ids index a growing word vocabulary."""

    def __init__(self):
        self.vocab: dict[str, int] = {}

    def render(self, messages, add_generation_prompt=False):
        words = []
        for m in messages:
            words += ["<s>", m["role"], *(m.get("content") or "").split(), "</s>"]
        if add_generation_prompt:
            words += ["<s>", "assistant"]
        return [self.vocab.setdefault(w, len(self.vocab)) for w in words]


def _pair(good: str = "fixed it", bad: str = "gave up") -> dict:
    prompt = [
        {"role": "system", "content": "you fix networks"},
        {"role": "user", "content": "R1 down"},
    ]
    return {"prompt": prompt, "chosen": [{"role": "assistant", "content": good}],
            "rejected": [{"role": "assistant", "content": bad}], "tools": None}  # fmt: skip


def test_parse_args_defaults_and_errors():
    a = dpo.parse_args(["--data", "d", "--base", "b", "--out", "o"])
    assert a.beta == 0.1 and a.epochs == 1.0 and a.lr == 2e-5 and a.init_adapter is None
    for bad in (["--beta", "0"], ["--grad-accum", "0"], ["--epochs", "-1"]):
        with pytest.raises(SystemExit):
            dpo.parse_args(["--data", "d", "--base", "b", "--out", "o", *bad])


def test_read_pairs_validates(tmp_path):
    p = tmp_path / "p.jsonl"
    p.write_text(json.dumps(_pair()) + "\n\n")
    assert len(dpo.read_pairs(p)) == 1
    p.write_text(json.dumps({"prompt": [], "chosen": []}) + "\n")
    with pytest.raises(sft.SFTError, match="rejected"):
        dpo.read_pairs(p)


def test_tokenize_pairs_labels_only_the_answers():
    t = WordTemplate()
    stats: dict = {}
    rows = dpo.tokenize_pairs([_pair()], lambda tools: t.render, 1000, stats)
    assert stats["kept"] == 1 and len(rows) == 1
    inv = {v: k for k, v in t.vocab.items()}
    for side, words in (("chosen", ["fixed", "it", "</s>"]), ("rejected", ["gave", "up", "</s>"])):
        labelled = [inv[x] for x in rows[0][f"{side}_labels"] if x != IGNORE]
        assert labelled == words  # the shared prompt is never trained on
    stats = {}
    assert dpo.tokenize_pairs([_pair()], lambda tools: t.render, 5, stats) == []
    assert stats["dropped_too_long"] == 1


def test_dpo_loss_math():
    torch = pytest.importorskip("torch")
    zero = torch.tensor(0.0)
    loss, margin, cr, rr = dpo.dpo_loss(zero, zero, zero, zero, beta=0.1)
    assert math.isclose(float(loss), math.log(2), rel_tol=1e-6) and float(margin) == 0  # undecided
    # making the chosen answer more likely than the reference did lowers the loss
    better, _, cr, rr = dpo.dpo_loss(torch.tensor(-1.0), torch.tensor(-5.0), torch.tensor(-3.0),
                                     torch.tensor(-3.0), beta=0.5)  # fmt: skip
    assert float(better) < math.log(2) and float(cr) == 1.0 and float(rr) == -1.0


def test_script_runs_standalone():
    out = subprocess.run([sys.executable, str(TRAIN_DIR / "dpo.py"), "--help"],
                         capture_output=True, text=True, cwd="/")  # fmt: skip
    assert out.returncode == 0 and "--init-adapter" in out.stdout
    source = (TRAIN_DIR / "dpo.py").read_text() + (TRAIN_DIR / "lora_tools.py").read_text()
    assert "import envforge" not in source and "from envforge" not in source
    top_level = [ln for ln in source.splitlines() if ln.startswith(("import ", "from "))]
    assert not any(m in ln for ln in top_level for m in ("torch", "transformers", "peft"))


# ---- a real (tiny) model on CPU --------------------------------------------------------


def _tiny_base(tmp_path: Path, vocab: int = 128) -> Path:
    transformers = pytest.importorskip("transformers")
    pytest.importorskip("peft")
    import torch

    torch.manual_seed(0)
    cfg = transformers.Qwen2Config(vocab_size=vocab, hidden_size=32, intermediate_size=64,
                                   num_hidden_layers=2, num_attention_heads=4,
                                   num_key_value_heads=2, max_position_embeddings=256)  # fmt: skip
    path = tmp_path / "tiny-base"
    transformers.Qwen2ForCausalLM(cfg).save_pretrained(path)
    return path


def _args(tmp_path: Path, base: Path, **kw) -> object:
    (tmp_path / "x.jsonl").write_text(json.dumps(_pair()) + "\n")  # hashed into the manifest
    argv = ["--data", str(tmp_path / "x.jsonl"), "--base", str(base), "--out", str(tmp_path / "out"),
            "--lr", "5e-3", "--grad-accum", "2", "--epochs", "8", "--no-gradient-checkpointing",
            "--beta", "0.5"]  # fmt: skip
    for k, v in kw.items():
        argv += [f"--{k.replace('_', '-')}", str(v)]
    return dpo.parse_args(argv)


def test_dpo_trains_a_tiny_model_to_prefer_the_chosen_answers(tmp_path):
    base = _tiny_base(tmp_path)
    t = WordTemplate()
    pairs = [_pair(f"good{i} answer", f"bad{i} answer") for i in range(4)]
    rows = dpo.tokenize_pairs(pairs, lambda tools: t.render, 1000, {})
    val = dpo.tokenize_pairs(
        [_pair("good0 answer", "bad1 answer")], lambda tools: t.render, 1000, {}
    )
    log = sft.EventLog(None)
    args = _args(tmp_path, base)
    assert dpo.train(args, None, rows, val, {"n_train": 4}, log=log) == 0

    logs = [e for e in log.events if e["event"] == "log"]
    evals = [e for e in log.events if e["event"] == "eval"]
    assert logs and evals and log.events[-1]["event"] == "done"
    assert logs[-1]["loss"] < math.log(2) and logs[-1]["accuracy"] == 1.0  # learned the preference
    assert logs[-1]["margin"] > 0 and logs[-1]["chosen_reward"] > logs[-1]["rejected_reward"]
    out = tmp_path / "out"
    assert (out / "adapter_config.json").is_file() and (out / "adapter_model.safetensors").is_file()
    manifest = json.loads((out / "envforge_manifest.json").read_text())
    assert manifest["method"] == "dpo" and manifest["final_train_accuracy"] == 1.0


def test_dpo_continues_from_an_existing_adapter(tmp_path):
    base = _tiny_base(tmp_path)
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM

    init = tmp_path / "sft-adapter"
    model = get_peft_model(AutoModelForCausalLM.from_pretrained(base),
                           LoraConfig(task_type="CAUSAL_LM", r=4, lora_alpha=8,
                                      target_modules=["q_proj", "v_proj"]))  # fmt: skip
    model.save_pretrained(init)

    t = WordTemplate()
    rows = dpo.tokenize_pairs([_pair(), _pair("fixed again", "gave up again")],
                              lambda tools: t.render, 1000, {})  # fmt: skip
    log = sft.EventLog(None)
    args = _args(tmp_path, base, init_adapter=init, epochs=2)
    assert dpo.train(args, None, rows, [], {}, log=log) == 0
    cfg = json.loads((tmp_path / "out" / "adapter_config.json").read_text())
    assert cfg["r"] == 4 and sorted(cfg["target_modules"]) == [
        "q_proj",
        "v_proj",
    ]  # same adapter shape
    # the first update starts exactly at the reference: undecided pairs
    first = next(e for e in log.events if e["event"] == "log")
    assert first["step"] >= 1

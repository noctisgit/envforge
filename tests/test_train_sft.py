"""Tests for envforge/train/sft.py: the offline parts (args, data, masking). No torch needed.

sft.py is a standalone script copied to GPU boxes, so it is loaded from its file path
rather than imported as `envforge.train.sft`; that also proves it has no envforge imports.
"""

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest

SFT_PATH = Path(__file__).resolve().parents[1] / "envforge" / "train" / "sft.py"


def _load_sft():
    spec = importlib.util.spec_from_file_location("envforge_sft_standalone", SFT_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses look their module up here
    spec.loader.exec_module(module)
    return module


sft = _load_sft()
IGNORE = sft.IGNORE_INDEX


# --------------------------------------------------------------------------- fake template


class FakeTemplate:
    """A ChatML-like template over a word vocabulary, so token ids are easy to reason about.

    Each message renders as: <s> ROLE word word ... [CALL name {args}] </s>.
    add_generation_prompt appends "<s> assistant". Options simulate templates that
    rewrite history, which is what the fallback path exists for.
    """

    def __init__(self, count_in_system: bool = False, strip_old_thoughts: bool = False):
        self.vocab: dict[str, int] = {}
        self.count_in_system = count_in_system
        self.strip_old_thoughts = strip_old_thoughts

    def ids(self, words):
        return [self.vocab.setdefault(w, len(self.vocab)) for w in words]

    def words_for(self, msg, is_last):
        words = ["<s>", msg["role"]]
        content = msg.get("content") or ""
        if self.strip_old_thoughts and msg["role"] == "assistant" and not is_last:
            content = content.split("</think>")[-1]
        words += content.split()
        for call in msg.get("tool_calls") or []:
            fn = call["function"]
            assert isinstance(fn["arguments"], dict), "renderer must receive dict arguments"
            words += ["CALL", fn["name"], json.dumps(fn["arguments"], sort_keys=True)]
        return [*words, "</s>"]

    def words(self, messages, add_generation_prompt=False):
        out = []
        if self.count_in_system:
            out += ["<s>", "system", f"n={len(messages)}", "</s>"]
        for k, msg in enumerate(messages):
            out += self.words_for(msg, is_last=k == len(messages) - 1)
        if add_generation_prompt:
            out += ["<s>", "assistant"]
        return out

    def render(self, messages, add_generation_prompt=False):
        return self.ids(self.words(messages, add_generation_prompt))

    def labelled_words(self, tok):
        """The words that carry a label, as lists of contiguous spans."""
        inv = {v: k for k, v in self.vocab.items()}
        spans, cur = [], []
        for t, label in zip(tok.input_ids, tok.labels, strict=True):
            if label != IGNORE:
                assert label == t
                cur.append(inv[t])
            elif cur:
                spans.append(cur)
                cur = []
        if cur:
            spans.append(cur)
        return spans


def _conversation():
    return sft.normalize_messages(
        [
            {"role": "system", "content": "you fix networks"},
            {"role": "user", "content": "R1 cannot reach R4"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "c1",
                        "type": "function",
                        "function": {"name": "show_links", "arguments": "{}"},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "c1", "content": "R3-R4 down"},
            {
                "role": "assistant",
                "content": "bring it up",
                "tool_calls": [
                    {
                        "id": "c2",
                        "type": "function",
                        "function": {"name": "set_link", "arguments": '{"a": "R3", "b": "R4"}'},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "c2", "content": "ok"},
            {"role": "assistant", "content": "fixed"},
        ]
    )


EXPECTED_SPANS = [
    ["CALL", "show_links", "{}", "</s>"],
    ["bring", "it", "up", "CALL", "set_link", '{"a": "R3", "b": "R4"}', "</s>"],
    ["fixed", "</s>"],
]


# --------------------------------------------------------------------------- args


def test_parse_args_defaults_and_flags(tmp_path):
    args = sft.parse_args(["--data", "d.jsonl", "--base", "Qwen/Qwen2.5-7B-Instruct", "--out", "o"])
    assert args.data == Path("d.jsonl") and args.out == Path("o")
    assert args.epochs == 2 and args.lr == 2e-4 and args.lora_r == 16 and args.lora_alpha == 32
    assert args.target_modules == "all-linear" and args.max_len == 8192
    assert args.batch_size == 1 and args.grad_accum == 8 and args.warmup_ratio == 0.03
    assert args.bf16 is None  # decided at train time from the GPU
    assert args.gradient_checkpointing is True and args.dry_run is False
    assert args.val is None and args.val_frac == 0.0 and args.max_steps == 0

    args = sft.parse_args(
        ["--data", "d", "--base", "b", "--out", "o", "--no-bf16", "--no-gradient-checkpointing"]
        + ["--max-steps", "4", "--val", "v.jsonl", "--dry-run", "--target-modules", "q_proj,v_proj"]
    )
    assert args.bf16 is False and args.gradient_checkpointing is False
    assert args.max_steps == 4 and args.val == Path("v.jsonl") and args.dry_run
    assert sft.parse_args(["--data", "d", "--base", "b", "--out", "o", "--bf16"]).bf16 is True


@pytest.mark.parametrize(
    "extra",
    [
        ["--val", "v", "--val-frac", "0.1"],  # mutually exclusive
        ["--val-frac", "1.5"],
        ["--batch-size", "0"],
    ],
)
def test_parse_args_rejects_bad_combinations(extra):
    with pytest.raises(SystemExit):
        sft.parse_args(["--data", "d", "--base", "b", "--out", "o", *extra])


# --------------------------------------------------------------------------- data


def test_normalize_parses_arguments_and_names_tool_messages():
    msgs = _conversation()
    assert msgs[2]["tool_calls"][0]["function"]["arguments"] == {}
    assert msgs[4]["tool_calls"][0]["function"]["arguments"] == {"a": "R3", "b": "R4"}
    assert msgs[3]["tool_call_id"] == "c1" and msgs[3]["name"] == "show_links"
    assert msgs[5]["name"] == "set_link"


def test_normalize_keeps_bad_json_and_does_not_mutate_input():
    raw = [
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "x", "type": "function", "function": {"name": "f", "arguments": "{oops"}}
        ]},
    ]  # fmt: skip
    out = sft.normalize_messages(raw)
    assert out[0]["tool_calls"][0]["function"]["arguments"] == "{oops"
    assert out[0]["content"] == ""
    assert raw[0]["content"] is None  # the caller's data is untouched


def test_read_examples(tmp_path):
    p = tmp_path / "train.jsonl"
    p.write_text(json.dumps({"messages": [{"role": "user", "content": "hi"}]}) + "\n\n")
    assert sft.read_examples(p) == [{"messages": [{"role": "user", "content": "hi"}]}]

    with pytest.raises(sft.SFTError, match="not found"):
        sft.read_examples(tmp_path / "missing.jsonl")
    bad = tmp_path / "bad.jsonl"
    bad.write_text('{"messages": []}\nnot json\n')
    with pytest.raises(sft.SFTError, match=":2: invalid JSON"):
        sft.read_examples(bad)
    bad.write_text('{"prompt": "x"}\n')
    with pytest.raises(sft.SFTError, match="messages"):
        sft.read_examples(bad)


def test_dataset_sha256(tmp_path):
    a, b = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    a.write_bytes(b"one\n")
    b.write_bytes(b"two\n")
    assert sft.dataset_sha256([a]) == hashlib.sha256(b"one\n").hexdigest()
    assert sft.dataset_sha256([a, b]) == hashlib.sha256(b"one\ntwo\n").hexdigest()
    assert sft.dataset_sha256([a, b]) != sft.dataset_sha256([b, a])


def test_split_is_seeded_and_keeps_a_training_example():
    ex = [{"messages": [], "i": i} for i in range(20)]
    train, val = sft.split_examples(ex, 0.25, seed=1)
    assert len(val) == 5 and len(train) == 15
    assert sft.split_examples(ex, 0.25, seed=1) == (train, val)
    assert sft.split_examples(ex, 0.0, seed=1) == (ex, [])
    train, val = sft.split_examples(ex[:2], 0.9, seed=0)
    assert len(train) == 1 and len(val) == 1


# --------------------------------------------------------------------------- masking


def test_build_labels_marks_only_assistant_turns():
    tmpl = FakeTemplate()
    tok = sft.build_labels(_conversation(), tmpl.render)
    assert tok.input_ids == tmpl.render(_conversation())
    assert len(tok.labels) == len(tok.input_ids)
    assert tmpl.labelled_words(tok) == EXPECTED_SPANS
    assert not tok.mismatch and tok.unlocated_turns == 0
    # The assistant header itself is prompt, not something the model should learn to emit.
    assert tok.n_assistant_tokens == sum(len(s) for s in EXPECTED_SPANS)


def test_build_labels_without_assistant_turns_masks_everything():
    msgs = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]
    tok = sft.build_labels(msgs, FakeTemplate().render)
    assert tok.n_assistant_tokens == 0


def test_build_labels_falls_back_when_template_rewrites_history():
    # The system prompt depends on the message count, so no rendering is a prefix of the
    # next. The fallback must still find every assistant turn.
    tmpl = FakeTemplate(count_in_system=True)
    tok = sft.build_labels(_conversation(), tmpl.render)
    assert tok.mismatch and tok.unlocated_turns == 0
    assert tmpl.labelled_words(tok) == EXPECTED_SPANS


def test_build_labels_counts_turns_it_cannot_locate():
    # Earlier assistant turns lose their reasoning once they are history (Qwen3-style), so
    # the turn as generated never appears in the full rendering: leave it masked, count it.
    msgs = [
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "<think> hmm </think> a1"},
        {"role": "user", "content": "q2"},
        {"role": "assistant", "content": "<think> ok </think> a2"},
    ]
    tmpl = FakeTemplate(strip_old_thoughts=True)
    tok = sft.build_labels(msgs, tmpl.render)
    assert tok.mismatch and tok.unlocated_turns == 1
    assert tmpl.labelled_words(tok) == [["<think>", "ok", "</think>", "a2", "</s>"]]


# --------------------------------------------------------------------------- tokenize + stats


def _example(n_words: int) -> dict:
    return {
        "messages": [
            {"role": "user", "content": " ".join(f"w{i}" for i in range(n_words))},
            {"role": "assistant", "content": "done"},
        ]
    }


def test_tokenize_examples_drops_too_long_and_counts():
    tmpl = FakeTemplate()
    stats = sft.DatasetStats()
    examples = [_example(3), _example(50), {"messages": [{"role": "user", "content": "x"}]}]
    rows = sft.tokenize_examples(examples, lambda tools: tmpl.render, max_len=20, stats=stats)
    assert len(rows) == 1
    assert stats.examples_in == 3 and stats.kept == 1
    assert stats.dropped_too_long == 1 and stats.dropped_no_assistant_tokens == 1
    row = rows[0]
    assert len(row["input_ids"]) == len(row["labels"]) <= 20  # dropped, never truncated
    assert stats.assistant_tokens == 2  # "done" + "</s>"

    summary = sft.summarize(stats, sft.DatasetStats())
    assert summary["n_train"] == 1 and summary["n_val"] == 0 and summary["dropped_too_long"] == 1
    assert summary["tokens_total"] == summary["tokens_max"] == len(row["input_ids"])
    assert summary["assistant_token_fraction"] == round(2 / len(row["input_ids"]), 4)


def test_tokenize_examples_passes_tools_to_renderer_factory():
    seen = []
    tmpl = FakeTemplate()

    def make_render(tools):
        seen.append(tools)
        return tmpl.render

    tools = [{"type": "function", "function": {"name": "finish", "parameters": {}}}]
    sft.tokenize_examples(
        [{**_example(2), "tools": tools}, _example(2)], make_render, 100, sft.DatasetStats()
    )
    assert seen == [tools, None]


def test_event_log_prints_prefixed_json_and_writes_metrics(tmp_path, capsys):
    log = sft.EventLog(tmp_path / "out" / "metrics.jsonl")
    log.emit("log", step=10, loss=1.23, epoch=0.4)
    line = capsys.readouterr().out.strip()
    assert line.startswith("ENVFORGE ")
    event = json.loads(line.removeprefix("ENVFORGE "))
    assert event["event"] == "log" and event["step"] == 10 and event["loss"] == 1.23
    assert json.loads((tmp_path / "out" / "metrics.jsonl").read_text()) == event


def test_params_from_name():
    assert sft._params_from_name("Qwen/Qwen2.5-7B-Instruct") == 7.0
    assert sft._params_from_name("Qwen/Qwen2.5-0.5B-Instruct") == 0.5
    assert sft._params_from_name("trl-internal-testing/tiny-Qwen2ForCausalLM-2.5") is None


def test_main_reports_missing_data_file(tmp_path, capsys):
    code = sft.main(["--data", str(tmp_path / "nope.jsonl"), "--base", "b", "--out", "o"])
    assert code == 2
    captured = capsys.readouterr()
    assert '"event": "error"' in captured.out and "not found" in captured.err


def test_script_has_no_envforge_or_heavy_top_level_imports():
    top_level = [
        line for line in SFT_PATH.read_text().splitlines() if line.startswith(("import ", "from "))
    ]
    assert not [line for line in top_level if "envforge" in line]
    for heavy in ("torch", "transformers", "peft", "datasets"):
        assert not [line for line in top_level if heavy in line.split()[1]]


# --------------------------------------------------------------------------- real tokenizer


def test_real_chatml_template_masks_assistant_turns():
    """With an actual Qwen-style template (needs transformers + a cached/online tokenizer)."""
    transformers = pytest.importorskip("transformers")
    try:
        tok = transformers.AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
    except Exception as e:  # offline and not cached
        pytest.skip(f"tokenizer unavailable: {e}")
    msgs = _conversation()
    out = sft.build_labels(msgs, sft.make_renderer(tok, None))
    assert not out.mismatch and out.unlocated_turns == 0
    text = tok.decode([t for t in out.labels if t != IGNORE])
    assert '"name": "set_link"' in text and "fixed<|im_end|>" in text
    assert "R1 cannot reach R4" not in text and "R3-R4 down" not in text

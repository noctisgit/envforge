"""Measure how much personal data redaction lets through — on data with a known answer.

Redaction is only as good as its measured leak rate. This module computes that rate on
a dataset whose personal values are known in advance, so "did a name survive?" is a
lookup, not a judgement call.

ABCD (Action-Based Conversations Dataset, ASAPP, MIT licence) is such a dataset: 10,042
customer-service chats between crowd workers playing customers and agents. Each
conversation comes with the fictional customer it was written about
(`scenario.personal`: name, email, phone, username), and the customers type those
values into the chat in every style people use — "Crystal Minh", "crystal minh",
"cminh730", "(977) 625-2661". All customers are fictional, which is what makes it safe
to benchmark on.

    from envforge.builder import pii_bench
    convs = pii_bench.abcd_conversations("abcd_v1.1.json", n=500)
    truth = pii_bench.abcd_ground_truth("abcd_v1.1.json")
    redacted, _ = redact_conversations(convs, profile="standard", tier2="auto")
    report = pii_bench.leak_report(redacted, truth)    # leak rate per type + overall

    python -m envforge.builder.pii_bench --abcd abcd_v1.1.json --n 500 \\
        --profile standard --tier2 auto                  # comparison table across modes

The human-review gate
---------------------

A measured leak rate on a benchmark is not a guarantee for *your* data. Before a
redacted dataset is shared, a person reviews a random sample (`sample_for_review`) and
counts conversations with a leak. With k leaks in n reviewed conversations, the Wilson
score interval (`wilson_interval`) bounds the true leak rate: 0 leaks in 300 reviewed
conversations means the rate is below ~1.3 % with 95 % confidence — "we saw none" is
not "there are none", and the interval says how many more you would need to look at.
"""

from __future__ import annotations

import argparse
import difflib
import gzip
import json
import math
import random
import re
import time
from collections import Counter
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

from envforge.builder import ner as ner_layer
from envforge.builder import pii_model
from envforge.builder.ingest import Conversation, Turn
from envforge.builder.redact import PLACEHOLDER_RE, redact_batch

GroundTruth = dict[str, dict[str, list[str]]]  # conversation id -> type -> values

ABCD_ROLES = {"customer": "customer", "agent": "agent", "action": "system"}


# ---- loading ABCD --------------------------------------------------------------------------


def _load_json(path: str | Path) -> Any:
    """Read a .json or .json.gz file."""
    p = Path(path)
    opener = gzip.open if p.suffix == ".gz" else open
    with opener(p, "rt", encoding="utf-8") as fh:
        return json.load(fh)


def _abcd_records(data: Any) -> list[dict[str, Any]]:
    """ABCD ships as {"train": [...], "dev": [...], "test": [...]}; samples as a list."""
    if isinstance(data, dict):
        return [rec for split in ("train", "dev", "test") for rec in data.get(split, [])]
    return list(data)


def abcd_ground_truth(path: str | Path) -> GroundTruth:
    """The personal values of every ABCD conversation, keyed by conversation id.

    Types: "name" (each part of `customer_name`, e.g. ["crystal", "minh"]), "email",
    "phone", "username". Handles .json and .json.gz, full dataset or a list sample.
    """
    truth: GroundTruth = {}
    for rec in _abcd_records(_load_json(path)):
        personal = (rec.get("scenario") or {}).get("personal") or {}
        values: dict[str, list[str]] = {}
        parts = [p for p in str(personal.get("customer_name") or "").split() if len(p) >= 2]
        if parts:
            values["name"] = parts
        for key, kind in (("email", "email"), ("phone", "phone"), ("username", "username")):
            if personal.get(key):
                values[kind] = [str(personal[key])]
        truth[str(rec.get("convo_id"))] = values
    return truth


def abcd_conversations(
    path: str | Path, *, n: int | None = None, seed: int = 0
) -> list[Conversation]:
    """ABCD conversations as `Conversation`s (role "action" -> system), optionally a
    deterministic random sample of `n` (see `sample_for_review`)."""
    records = _abcd_records(_load_json(path))
    by_id = {str(rec.get("convo_id")): rec for rec in records}
    ids = list(by_id)
    if n is not None and n < len(ids):
        ids = sample_for_review(ids, n, seed)
    convs = []
    for cid in ids:
        turns = [
            Turn(role=ABCD_ROLES.get(str(speaker), "unknown"), text=str(text))
            for speaker, text in by_id[cid].get("original", [])
        ]
        convs.append(Conversation(id=cid, turns=turns, meta={"source": "abcd"}))
    return convs


# ---- leak measurement ----------------------------------------------------------------------


def _value_regex(kind: str, value: str) -> re.Pattern[str]:
    """How a known value is found in text: phones by their digits (any separators),
    everything else case-insensitively and not inside a longer word."""
    if kind == "phone":
        digits = re.sub(r"\D", "", value)
        return re.compile(r"(?<!\d)" + r"\D{0,3}".join(digits) + r"(?!\d)")
    return re.compile(rf"(?i)(?<![A-Za-z0-9]){re.escape(value)}(?![A-Za-z0-9])")


def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a proportion k/n (default: 95 % confidence).

    Unlike the textbook p ± z·sqrt(p(1-p)/n), it behaves at the edges: 0 of 300 gives
    (0.0, ~0.0126) rather than the useless (0, 0).
    """
    if n <= 0:
        return 0.0, 1.0
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def sample_for_review(ids: Iterable[str], n: int, seed: int = 0) -> list[str]:
    """A deterministic random sample of `n` ids (all of them if fewer) for a human to
    review. Same ids + seed -> same sample, so a review can be repeated and audited."""
    pool = sorted(set(ids))
    rng = random.Random(seed)
    return rng.sample(pool, min(n, len(pool)))


def _rate(k: int, n: int) -> dict[str, Any]:
    lo, hi = wilson_interval(k, n)
    return {
        "values": n,
        "leaked": k,
        "rate": round(k / n, 4) if n else 0.0,
        "ci95": [round(lo, 4), round(hi, 4)],
    }


def leak_report(
    redacted_convs: Sequence[Conversation],
    ground_truth: GroundTruth,
    *,
    max_examples: int = 10,
    context_chars: int = 40,
) -> dict[str, Any]:
    """How many known personal values survived redaction.

    Each (conversation, type, value) counts once: leaked if it appears anywhere in the
    conversation's redacted text. Returns the leak rate per type and overall (with
    Wilson 95 % intervals), the share of conversations with any leak, and up to
    `max_examples` leaked snippets — taken from the REDACTED text (they show the value
    that leaked, in its context, and nothing else of the original).
    """
    per_type: Counter[str] = Counter()
    leaked_type: Counter[str] = Counter()
    convs_with_leak = 0
    examples: list[dict[str, str]] = []
    scored = 0
    for conv in redacted_convs:
        truth = ground_truth.get(conv.id)
        if truth is None:
            continue
        scored += 1
        text = "\n".join(t.text for t in conv.turns)
        any_leak = False
        for kind, values in truth.items():
            for value in values:
                per_type[kind] += 1
                m = _value_regex(kind, value).search(text)
                if m:
                    leaked_type[kind] += 1
                    any_leak = True
                    if len(examples) < max_examples:
                        a = max(0, m.start() - context_chars)
                        b = min(len(text), m.end() + context_chars)
                        snippet = text[a:b].replace("\n", " / ")
                        examples.append({"conversation": conv.id, "type": kind, "snippet": snippet})
        convs_with_leak += any_leak
    total, leaked = sum(per_type.values()), sum(leaked_type.values())
    return {
        "conversations": scored,
        "conversations_with_leak": _rate(convs_with_leak, scored),
        "overall": _rate(leaked, total),
        "by_type": {k: _rate(leaked_type[k], per_type[k]) for k in sorted(per_type)},
        "examples": examples,
    }


def over_redaction(
    originals: Sequence[Conversation],
    redacted: Sequence[Conversation],
    ground_truth: GroundTruth,
    *,
    max_examples: int = 12,
) -> dict[str, Any]:
    """What was replaced that is NOT a known personal value (benchmark-only: it reads
    the originals, which is fine for a fictional dataset and never done in production).

    Aligns each original turn with its redacted version word by word; every replaced
    stretch is either known PII (overlaps a ground-truth value) or "other". "Other"
    includes real catches outside the ground truth (street addresses, ZIPs) as well as
    false positives — the examples show which.
    """
    pii_segments = other_segments = 0
    examples: Counter[str] = Counter()
    for before, after in zip(originals, redacted, strict=True):
        truth = ground_truth.get(before.id, {})
        patterns = [_value_regex(k, v) for k, vs in truth.items() for v in vs]
        for tb, ta in zip(before.turns, after.turns, strict=False):
            a_words, b_words = tb.text.split(), ta.text.split()
            matcher = difflib.SequenceMatcher(a=a_words, b=b_words, autojunk=False)
            for op, i1, i2, _, _ in matcher.get_opcodes():
                if op not in ("replace", "delete"):
                    continue
                segment = " ".join(a_words[i1:i2])
                if any(p.search(segment) for p in patterns):
                    pii_segments += 1
                else:
                    other_segments += 1
                    examples[segment.strip(".,!?;:")] += 1
    total = pii_segments + other_segments
    return {
        "replaced_segments": total,
        "known_pii_segments": pii_segments,
        "other_segments": other_segments,
        "other_share": round(other_segments / total, 4) if total else 0.0,
        "other_per_conversation": round(other_segments / max(1, len(originals)), 3),
        "examples": [seg for seg, _ in examples.most_common(max_examples)],
    }


# ---- the comparison CLI --------------------------------------------------------------------


def parse_mode(mode: str) -> dict[str, Any]:
    """ "standard+ner+tier2=auto+surrogates" -> redact_batch keyword arguments.

    Parts: a profile (off|standard|strict), then any of `ner`, `tier2` (= auto),
    `tier2=auto|all`, `surrogates`.
    """
    parts = mode.split("+")
    kwargs: dict[str, Any] = {
        "profile": parts[0],
        "ner": False,
        "tier2": "off",
        "surrogates": False,
    }
    for part in parts[1:]:
        if part == "ner":
            kwargs["ner"] = True
        elif part.startswith("tier2"):
            kwargs["tier2"] = part.partition("=")[2] or "auto"
        elif part in ("surrogates", "sur"):
            kwargs["surrogates"] = True
        else:
            raise ValueError(f"unknown mode part {part!r} in {mode!r}")
    return kwargs


def run_mode(
    convs: Sequence[Conversation], truth: GroundTruth, mode: str, *, salt: bytes = b"bench"
) -> dict[str, Any]:
    """Redact `convs` in one mode and measure leaks, over-redaction, time and tier 2."""
    kwargs = parse_mode(mode)
    started = time.perf_counter()
    batch = redact_batch(convs, salt=salt, **kwargs)
    seconds = time.perf_counter() - started
    # over-redaction needs placeholders (surrogates are measured the same way: a
    # replaced word is a replaced word)
    return {
        "mode": mode,
        "leaks": leak_report(batch.conversations, truth),
        "over_redaction": over_redaction(convs, batch.conversations, truth),
        "ms_per_conversation": round(1000 * seconds / max(1, len(convs)), 1),
        "tier2_sent": batch.tier2_sent,
        "tier2_share": round(batch.tier2_sent / max(1, len(convs)), 3),
        "risk_reasons": batch.risk_reasons,
        "seconds": batch.seconds,
        "placeholders": sum(
            len(PLACEHOLDER_RE.findall(t.text)) for c in batch.conversations for t in c.turns
        ),
    }


def _pct(x: float) -> str:
    return f"{100 * x:5.1f}%"


def format_table(results: Sequence[dict[str, Any]]) -> str:
    """A plain-text comparison table (leak rates are per known value)."""
    types = sorted({t for r in results for t in r["leaks"]["by_type"]})
    head = ["mode", "leak overall [95% CI]", *types, "convs w/ leak", "other repl./conv",
            "tier2 share", "ms/conv"]  # fmt: skip
    rows = []
    for r in results:
        overall = r["leaks"]["overall"]
        lo, hi = overall["ci95"]
        rows.append(
            [
                r["mode"],
                f"{_pct(overall['rate'])} [{100 * lo:.1f}-{100 * hi:.1f}]",
                *[_pct(r["leaks"]["by_type"].get(t, {}).get("rate", 0.0)) for t in types],
                _pct(r["leaks"]["conversations_with_leak"]["rate"]),
                f"{r['over_redaction']['other_per_conversation']:.2f}",
                _pct(r["tier2_share"]),
                f"{r['ms_per_conversation']:.0f}",
            ]
        )
    widths = [max(len(str(x)) for x in col) for col in zip(head, *rows, strict=True)]
    lines = [
        " | ".join(str(x).ljust(w) for x, w in zip(row, widths, strict=True))
        for row in [head, *rows]
    ]
    lines.insert(1, "-+-".join("-" * w for w in widths))
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> None:
    """`python -m envforge.builder.pii_bench --abcd <path> [--n 500] [--profile standard]
    [--tier2 auto] [--modes a,b,c] [--json out.json]` — print a comparison table."""
    ap = argparse.ArgumentParser(
        prog="python -m envforge.builder.pii_bench", description=__doc__.split("\n")[0]
    )
    ap.add_argument("--abcd", required=True, help="ABCD file (.json or .json.gz)")
    ap.add_argument("--n", type=int, default=500, help="conversations to sample")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--profile", default="standard", choices=["standard", "strict"])
    ap.add_argument("--tier2", default="auto", choices=["off", "auto", "all"])
    ap.add_argument(
        "--modes", default="", help="comma-separated, e.g. standard,standard+ner+tier2=auto"
    )
    ap.add_argument("--surrogates", action="store_true", help="add +surrogates to every mode")
    ap.add_argument("--json", default="", help="also write the full results here")
    ap.add_argument(
        "--examples", type=int, default=6, help="leak / over-redaction examples per mode"
    )
    args = ap.parse_args(argv)

    if args.modes:
        modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    else:
        p, t2 = args.profile, args.tier2
        modes = [p, f"{p}+ner"]
        if t2 != "off":
            modes += [f"{p}+tier2={t2}", f"{p}+ner+tier2={t2}"]
    if args.surrogates:
        modes = [m if "surrogates" in m else f"{m}+surrogates" for m in modes]

    convs = abcd_conversations(args.abcd, n=args.n, seed=args.seed)
    truth = abcd_ground_truth(args.abcd)
    print(f"{len(convs)} ABCD conversations, {sum(len(c.turns) for c in convs)} turns")
    # load the models before timing anything
    if any("+ner" in m for m in modes):
        status = ner_layer.ner_available()
        print(f"NER: {'available' if status else 'unavailable'} ({status.model}) {status.reason}")
    if any("tier2" in m for m in modes):
        t = time.perf_counter()
        status2 = pii_model.pii_model_available()
        print(
            f"tier 2: {'available' if status2 else 'unavailable'} ({status2.model}) "
            f"{status2.reason} — loaded in {time.perf_counter() - t:.1f}s"
        )
    results = []
    for mode in modes:
        print(f"running {mode} ...", flush=True)
        results.append(run_mode(convs, truth, mode))
    print()
    print(format_table(results))
    for r in results:
        print(
            f"\n[{r['mode']}] risk reasons: {r['risk_reasons'] or '-'}  stage seconds: {r['seconds']}"
        )
        print("  over-redacted (not known PII):", r["over_redaction"]["examples"][: args.examples])
        for ex in r["leaks"]["examples"][: args.examples]:
            print(f"  leak {ex['type']}: …{ex['snippet']}…")
    if args.json:
        Path(args.json).write_text(json.dumps(results, indent=2))
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()

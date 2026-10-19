"""Tier 2 of PII redaction: a local span model (GLiNER) that runs on the CPU.

Tier 1 (`envforge.builder.redact`) is rules: regexes for things with a *shape* (emails,
phone numbers, cards) and cues for things without one ("my name is …", the reply to
"may I have your name?"). Rules miss what arrives with no cue and no shape — a lower-case
name typed on its own, a username dropped mid-sentence. A *span model* reads the words
around a phrase and labels it ("person", "username", …) without needing a cue.

GLiNER is a small (~300M parameter, mDeBERTa-based) model that takes the label names as
plain text, so one model covers any label set. `urchade/gliner_multi_pii-v1` is a
GLiNER fine-tuned for PII (Apache-2.0, ~1.1 GB download). Everything runs locally: no
text leaves the machine, and no LLM is involved.

    from envforge.builder import pii_model

    status = pii_model.pii_model_available()   # truthy when usable; status.reason if not
    spans = pii_model.detect(["sure, it's crystal minh", "Username: cminh730"])
    # [[Span(start=10, end=22, label='person', kind='NAME', ...)],
    #  [Span(start=10, end=18, label='username', kind='USERNAME', ...)]]

Why a filter is needed
----------------------

Measured on the ABCD dataset (6-core AVX2 CPU, ~112 ms per turn), the model caught 95 %
of the names and usernames tier 1 missed — and flagged plenty that is not personal data:
pronouns and roles as people ("I", "you", "Agent"), the *field names* themselves
("full name", "username", "email address", "phone number", "zip code", "account ID"),
placeholders a previous pass wrote ("NAME_1") and ordinary words ("silver" as an account
number: a membership tier). Over-redaction is not harmless — it destroys the text an
environment is built from — so `filter_spans` drops those. Its rules, in order:

    1. too short (< 2 characters), or inside / made of a placeholder (<NAME_1>, NAME_1)
    2. a span overlapping a value the caller wants kept (keep_patterns)
    3. trim leading/trailing filler words ("I'm Crystal" -> "Crystal"); drop if nothing
       is left: pronouns, greetings, role words, field labels (name, username, email,
       phone, address, account, id, zip, …) with or without my/your/the/full/new/…
    4. a single lower-case common word ("silver", "bronze", "order")
    5. per label: a person must contain a name-like token — capitalised and not a
       common word, or a name already registered in this conversation, or (any case)
       when the turn answers a name question; a username or account number must
       contain a digit or have a handle shape (j.doe, john_doe); an email needs "@";
       a phone number 7+ digits; an address a digit or a street word. A "person"
       that is one handle-shaped token with digits ("davidwilliams617" — the model
       often calls those people) becomes a username.

Setup (optional; redaction works without it):

    uv sync --extra pii      # gliner + CPU-only torch (+ protobuf for the tokenizer)

The model id comes from `ENVFORGE_PII_MODEL` (default `urchade/gliner_multi_pii-v1`). It
is downloaded from the Hugging Face Hub on first use into `$HF_HOME` (default
`~/.cache/huggingface`) and reused afterwards — point HF_HOME at a persistent volume in
containers. Set `HF_HUB_OFFLINE=1` to forbid downloads.

Graceful degradation: if gliner/torch or the model is missing, `pii_model_available()`
returns a falsy `Status` with a `reason`, `detect` returns empty lists, and redaction
carries on with tier 1 alone. Nothing here raises because the model is unavailable.

Cost: loading takes 10-30 s and ~1.2 GB of RAM per process (cached per process, so a
process pool loads it once per worker). Inference is ~100 ms per short turn on 6 cores —
about 100x tier 1 — which is why `redact.redact_conversations(tier2="auto")` only sends
conversations that tier-1 triage marks risky.
"""

from __future__ import annotations

import functools
import importlib.util
import logging
import os
import re
from collections.abc import Collection, Sequence
from dataclasses import dataclass, replace
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "urchade/gliner_multi_pii-v1"
MODEL_ENV_VAR = "ENVFORGE_PII_MODEL"
DEFAULT_THRESHOLD = 0.45
DEFAULT_BATCH_SIZE = 32
# GLiNER reads at most ~384 tokens; longer texts are cut into windows of this many
# characters (at whitespace) so nothing at the end of a long email is silently skipped.
MAX_CHARS = 1000

# label (what the model is asked for) -> placeholder kind used by redaction
LABEL_KINDS: dict[str, str] = {
    "person": "NAME",
    "username": "USERNAME",
    "email": "EMAIL",
    "phone number": "PHONE",
    "address": "ADDRESS",
    "account number": "ACCOUNT",
}
DEFAULT_LABELS: tuple[str, ...] = tuple(LABEL_KINDS)


@dataclass(frozen=True)
class Span:
    """One model finding: `text[start:end]` looks like a `label` (placeholder `kind`)."""

    start: int
    end: int
    label: str
    kind: str
    score: float
    text: str


@dataclass(frozen=True)
class Status:
    """Whether tier 2 can run. Truthy when available; `reason` explains a False."""

    available: bool
    model: str
    reason: str = ""

    def __bool__(self) -> bool:
        return self.available

    def to_dict(self) -> dict[str, Any]:
        """Plain dict (for reports)."""
        return {"available": self.available, "model": self.model, "reason": self.reason}


def pii_model_name(model: str | None = None) -> str:
    """The model to use: the argument, else `$ENVFORGE_PII_MODEL`, else the default."""
    return model or os.environ.get(MODEL_ENV_VAR) or DEFAULT_MODEL


@functools.lru_cache(maxsize=2)
def _load_model(name: str) -> tuple[Any | None, str]:
    """Load (once per process and model) a GLiNER model, or explain why not.

    Returns `(model, "")` or `(None, reason)`. Cached, so a missing model is reported
    once per process; `reset_pii_model_cache()` forgets it.
    """
    for module, hint in (("torch", "torch"), ("gliner", "gliner")):
        if importlib.util.find_spec(module) is None:
            return None, f"{hint} is not installed (uv sync --extra pii)"
    try:
        # transformers and gliner are chatty about tokenizer details we cannot change
        for noisy in ("transformers", "gliner", "huggingface_hub"):
            logging.getLogger(noisy).setLevel(logging.ERROR)
        import warnings

        from gliner import GLiNER

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = GLiNER.from_pretrained(name)
        model.eval()
        return model, ""
    except Exception as exc:  # noqa: BLE001 — any failure means "tier 1 only"
        return None, f"failed to load PII model {name!r}: {type(exc).__name__}: {exc}"


def reset_pii_model_cache() -> None:
    """Forget loaded models and cached failures (e.g. after installing the extra)."""
    _load_model.cache_clear()


def pii_model_installed() -> bool:
    """Cheap check (no model load): are gliner and torch importable?"""
    return all(importlib.util.find_spec(m) is not None for m in ("torch", "gliner"))


def pii_model_available(model: str | None = None) -> Status:
    """Check (and load, once per process) the tier-2 model. Never raises.

    The first call may take tens of seconds (and download the model if it is not in
    `$HF_HOME` yet); later calls are instant.
    """
    name = pii_model_name(model)
    loaded, reason = _load_model(name)
    return Status(available=loaded is not None, model=name, reason=reason)


def _windows(text: str, max_chars: int = MAX_CHARS) -> list[tuple[int, str]]:
    """Split `text` into (offset, piece) windows of at most ~max_chars, at whitespace."""
    if len(text) <= max_chars:
        return [(0, text)]
    out: list[tuple[int, str]] = []
    start = 0
    while start < len(text):
        end = min(len(text), start + max_chars)
        if end < len(text):
            cut = text.rfind(" ", start + max_chars // 2, end)
            end = cut if cut > start else end
        out.append((start, text[start:end]))
        start = end
    return out


def detect(
    texts: Sequence[str],
    *,
    labels: Sequence[str] = DEFAULT_LABELS,
    threshold: float = DEFAULT_THRESHOLD,
    batch_size: int = DEFAULT_BATCH_SIZE,
    model: str | None = None,
) -> list[list[Span]]:
    """Run the span model over many texts; one list of spans per text, same order.

    Raw model output — callers should pass each list through `filter_spans`.
    Texts without letters are skipped; long texts are windowed (see `MAX_CHARS`).
    Labels not in `LABEL_KINDS` get their upper-cased label as the kind. Returns
    empty lists when the model is unavailable or fails.
    """
    results: list[list[Span]] = [[] for _ in texts]
    if not texts:
        return results
    status = pii_model_available(model)
    if not status:
        logger.debug("tier-2 PII model unavailable: %s", status.reason)
        return results
    gliner, _ = _load_model(status.model)
    pieces: list[tuple[int, int, str]] = []  # (text index, offset, piece)
    for i, text in enumerate(texts):
        if text and any(ch.isalpha() for ch in text):
            pieces.extend((i, off, piece) for off, piece in _windows(text))
    if not pieces:
        return results
    # Batches are padded to their longest text, so similar lengths together roughly
    # halve the time on chat data (measured: 240 -> 118 ms per turn on 6 cores).
    pieces.sort(key=lambda p: len(p[2]))
    try:
        import torch

        with torch.inference_mode():
            found = gliner.inference(
                [p for _, _, p in pieces],
                list(labels),
                threshold=threshold,
                batch_size=batch_size,
            )
    except Exception as exc:  # noqa: BLE001 — never let tier 2 break redaction
        logger.warning("tier-2 PII model failed, tier 1 only: %s", exc)
        return results
    for (i, off, _), ents in zip(pieces, found, strict=True):
        text = texts[i]
        for e in ents:
            start, end = off + int(e["start"]), off + int(e["end"])
            label = str(e["label"])
            results[i].append(
                Span(
                    start=start,
                    end=end,
                    label=label,
                    kind=LABEL_KINDS.get(label, re.sub(r"\W+", "_", label.upper())),
                    score=float(e["score"]),
                    text=text[start:end],
                )
            )
    for spans in results:
        spans.sort(key=lambda s: (s.start, -s.end))
    return results


# ---- false-positive filter -----------------------------------------------------------------

PRONOUNS: frozenset[str] = frozenset(
    """i i'm im i've ive i'd id i'll ill me my mine myself you you're youre your yours
    yourself u ur he him his himself she her hers herself it its itself we us our ours
    ourselves they them their theirs themselves someone somebody anyone anybody everyone
    nobody who whom whose this that these those one""".split()
)
GREETINGS: frozenset[str] = frozenset(
    """hi hello hey hiya yo dear thanks thank thx ty cheers bye goodbye welcome greetings
    good morning afternoon evening night day regards sincerely best please pls sorry
    yes yeah yep no nope ok okay sure great alright""".split()
)
ROLE_WORDS: frozenset[str] = frozenset(
    """agent customer client caller user member guest rep representative support team
    staff operator manager supervisor admin administrator assistant bot sir madam maam
    ma'am mr mrs ms miss mx dr doctor officer""".split()
)
# Field labels: what a form or an agent *asks for*, not the value itself.
LABEL_WORDS: frozenset[str] = frozenset(
    """name names first last middle full given family surname nickname username usernames
    user login logon handle screen id ids identifier email emails e-mail mail address
    addresses phone phones telephone mobile cell number numbers no no. num # contact
    details detail info information account accounts acct profile membership member
    level tier status order orders zip zipcode postcode postal code codes street city
    state country province county apartment apt suite unit pin password passcode ssn
    social security card credit debit billing shipping delivery mailing home work
    residential correspondence verification""".split()
)
# Words that may precede a label: "my account", "your full name", "the new address".
LABEL_MODIFIERS: frozenset[str] = frozenset(
    """my your the a an his her their our its this that full new old current correct
    right same updated previous primary registered preferred verified valid
    first last""".split()
)
# Lower-case words often mislabelled as values (membership tiers, colours, statuses).
COMMON_VALUE_WORDS: frozenset[str] = frozenset(
    """gold silver bronze platinum guest premium basic standard free pro plus vip
    red blue green black white grey gray yellow pink purple orange brown
    yes no none null unknown n/a na pending active inactive cancelled canceled
    small medium large xl shirt jeans boots jacket item product order refund
    today tomorrow yesterday now later soon here there home""".split()
)
STREET_WORDS: frozenset[str] = frozenset(
    """street st avenue ave road rd boulevard blvd lane ln drive dr court ct way place
    pl terrace ter close crescent parkway pkwy highway hwy square sq circle cir trail
    row mews gardens grove apt apartment suite unit""".split()
)

_PLACEHOLDER_LIKE = re.compile(r"<[^<>]*>|\b[A-Z][A-Z0-9]*(?:_[A-Z0-9]+)*_\d+\b")
_TOKEN = re.compile(r"[\w'’.@+-]+")
_HANDLE_SHAPE = re.compile(r"^[A-Za-z0-9]+(?:[._-][A-Za-z0-9]+)+$")
_HANDLE_WITH_DIGITS = re.compile(r"^(?=[\w.-]*\d)[A-Za-z][\w.-]{2,39}$")


def _norm(word: str) -> str:
    return word.lower().strip(".,;:!?\"()[]{}'’").replace("’", "'")


def _filler(word: str, common: Collection[str]) -> bool:
    """A word that cannot be (part of) a value: pronoun, greeting, role, label…"""
    w = _norm(word)
    return (
        not w
        or w in PRONOUNS
        or w in GREETINGS
        or w in ROLE_WORDS
        or w in LABEL_WORDS
        or w in LABEL_MODIFIERS
        or (w in common and not word[:1].isupper())
        or w in COMMON_VALUE_WORDS
        or w in {"is", "am", "are", "was", "it's", "its", "and", "or", "of", "for", "at"}
    )


def _name_like(word: str, known: Collection[str], loose: bool, common: Collection[str]) -> bool:
    """Could this token be part of a person's name?"""
    w = _norm(word)
    if len(w) < 2 or any(ch.isdigit() for ch in w) or "@" in w:
        return False
    if w in known:
        return True
    if w in common or w in COMMON_VALUE_WORDS or _filler(word, ()):
        return False
    return word[:1].isupper() or loose


def _trim(text: str, span: Span, common: Collection[str]) -> Span | None:
    """Drop leading/trailing filler tokens; None if nothing is left. Where nothing is
    dropped the model's boundary stays ("(977) 625-2661" keeps its bracket)."""
    tokens = [(m.start(), m.end()) for m in _TOKEN.finditer(text, span.start, span.end)]
    n = len(tokens)
    while tokens and _filler(text[tokens[0][0] : tokens[0][1]], common):
        tokens.pop(0)
    front_trimmed = len(tokens) < n
    n = len(tokens)
    while tokens and _filler(text[tokens[-1][0] : tokens[-1][1]], common):
        tokens.pop()
    if not tokens:
        return None
    start = tokens[0][0] if front_trimmed else span.start
    end = tokens[-1][1] if len(tokens) < n else span.end
    while start < end and text[start].isspace():
        start += 1
    # trailing sentence punctuation is not part of a value ("cminh730." -> "cminh730")
    while end > start and text[end - 1] in ".,;:!?'’\" \t\n":
        end -= 1
    if end - start < 2:
        return None
    return replace(span, start=start, end=end, text=text[start:end])


def filter_spans(
    text: str,
    spans: Sequence[Span],
    *,
    known_names: Collection[str] = (),
    after_name_question: bool = False,
    keep: Sequence[re.Pattern[str]] = (),
    common_words: Collection[str] = (),
) -> list[Span]:
    """Drop model spans that are not personal data (see the module docstring's rules).

    `text` is the string the spans were found in. `known_names` are lower-case names
    (or name parts) already registered for this conversation; `after_name_question`
    says this turn answers "may I have your name?" (then lower-case names count);
    `keep` are compiled keep patterns; `common_words` extends the built-in list of
    words that are never a person's name on their own (redaction passes its
    `COMMON_WORDS`). Returns the surviving spans, trimmed of filler words.
    """
    known = {n.lower() for n in known_names}
    common = frozenset(w.lower() for w in common_words)
    placeholders = [m.span() for m in _PLACEHOLDER_LIKE.finditer(text)]
    kept_spans = [m.span() for rx in keep for m in rx.finditer(text) if m.end() > m.start()]
    out: list[Span] = []
    for span in spans:
        raw = span.text.strip()
        if len(raw) < 2:
            continue
        if any(s < span.end and span.start < e for s, e in placeholders):
            continue
        if any(s < span.end and span.start < e for s, e in kept_spans):
            continue
        trimmed = _trim(text, span, common)
        if trimmed is None:
            continue
        value = trimmed.text
        words = value.split()
        if trimmed.kind == "NAME" and len(words) == 1 and _HANDLE_WITH_DIGITS.match(value):
            trimmed = replace(trimmed, kind="USERNAME", label="username")
        if len(words) == 1 and value.islower() and (_norm(value) in common):
            continue
        if not _label_ok(trimmed.kind, value, words, known, after_name_question, common):
            continue
        out.append(trimmed)
    return out


def _label_ok(
    kind: str,
    value: str,
    words: list[str],
    known: Collection[str],
    loose: bool,
    common: Collection[str],
) -> bool:
    """Per-label shape checks for a trimmed span value."""
    digits = sum(ch.isdigit() for ch in value)
    if kind == "NAME":
        if digits or "@" in value or len(words) > 4:
            return False
        return any(_name_like(w, known, loose, common) for w in words)
    if kind in ("USERNAME", "ACCOUNT"):
        if len(words) != 1 or "@" in value:
            return False
        if kind == "ACCOUNT":
            return digits >= 3
        return digits > 0 or bool(_HANDLE_SHAPE.match(value))
    if kind == "EMAIL":
        return "@" in value and "." in value.split("@")[-1]
    if kind == "PHONE":
        return digits >= 7
    if kind == "ADDRESS":
        return digits > 0 or any(_norm(w) in STREET_WORDS for w in words[1:])
    return True

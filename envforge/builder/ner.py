"""Optional named-entity layer for redaction: Microsoft Presidio + a local spaCy model.

The regex rules in `envforge.builder.redact` find things with a recognisable *shape*
(emails, phone numbers, dates). They cannot find a name that arrives without a cue —
"Sarah said it was fine" — or a city mentioned in passing. A named-entity recogniser
(NER) can: it is a small statistical model that labels spans of text as PERSON,
LOCATION, DATE and so on from the words around them.

This module wraps Presidio's `AnalyzerEngine`, which runs a spaCy pipeline plus a set
of pattern recognisers, and turns its results into plain `NerSpan`s. Everything runs
locally on the CPU: no text leaves the machine, and no LLM is involved.

    from envforge.builder import ner

    status = ner.ner_available()       # truthy when usable; status.reason says why not
    spans = ner.analyze_texts(["Sarah lives in Boston."], profile="strict")
    # [[NerSpan(start=0, end=5, entity='PERSON', kind='NAME', ...),
    #   NerSpan(start=15, end=21, entity='LOCATION', kind='LOCATION', ...)]]

Setup (optional; redaction works without it):

    uv sync --extra privacy                    # presidio-analyzer + presidio-anonymizer
    uv pip install <spaCy model wheel>         # e.g. en_core_web_sm-3.8.0 for spaCy 3.8

The model name comes from the `ENVFORGE_NER_MODEL` environment variable (default
`en_core_web_sm`, ~12 MB, fast). `en_core_web_lg` (~600 MB) is noticeably more accurate
on names and places and is the better choice for real data if you have the memory;
`en_core_web_trf` is more accurate still but needs far more CPU time.

Graceful degradation: if Presidio or the model is missing, `ner_available()` returns a
falsy `NerStatus` with a human-readable `reason`, `analyze_texts` returns empty span
lists, and redaction carries on with the regex rules alone. Nothing here downloads a
model at runtime (Presidio would otherwise try to `pip install` one on first use).

Speed: texts are processed as a batch through spaCy's `nlp.pipe`, which is several
times faster than one call per text. Expect a few milliseconds per short message with
the small model on one CPU core.

Accuracy caveat: NER is statistical. It misses names (especially lower-case, unusual
or non-English ones) and mislabels product names or record keys as people or places.
Redaction errs towards over-redacting what the model flags, and the regex layer and
`keep_patterns` still apply — but NER is a second net, not a guarantee.
"""

from __future__ import annotations

import functools
import importlib.util
import logging
import os
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "en_core_web_sm"
MODEL_ENV_VAR = "ENVFORGE_NER_MODEL"
DEFAULT_SCORE_THRESHOLD = 0.5
DEFAULT_BATCH_SIZE = 64

# Presidio entity type -> the placeholder kind redaction uses for it. Kinds match the
# regex layer's names where they overlap (PERSON -> NAME, so a name found by NER and the
# same name found after "my name is" share one placeholder).
ENTITY_KINDS: dict[str, str] = {
    "PERSON": "NAME",
    "LOCATION": "LOCATION",
    "NRP": "NRP",  # nationality, religious or political group
    "DATE_TIME": "DATE",
    "MEDICAL_LICENSE": "LICENSE",
    "US_DRIVER_LICENSE": "LICENSE",
    "US_PASSPORT": "PASSPORT",
    "UK_NHS": "HEALTH_ID",
    "US_ITIN": "TAX_ID",
    "US_BANK_NUMBER": "ACCOUNT",
}

# Which entities each redaction profile asks for. Emails, phones, cards, IPs, ... are
# left to the regex layer (faster, validated). DATE_TIME is strict-only: the standard
# profile keeps dates. Driver's licence and bank-number recognisers are pattern-based and
# noisy without context (they flag tokens like "KB5034441"), so they are strict-only too.
PROFILE_ENTITIES: dict[str, tuple[str, ...]] = {
    "standard": (
        "PERSON",
        "LOCATION",
        "NRP",
        "MEDICAL_LICENSE",
        "US_PASSPORT",
        "UK_NHS",
        "US_ITIN",
    ),
    "strict": (
        "PERSON",
        "LOCATION",
        "NRP",
        "DATE_TIME",
        "MEDICAL_LICENSE",
        "US_DRIVER_LICENSE",
        "US_PASSPORT",
        "UK_NHS",
        "US_ITIN",
        "US_BANK_NUMBER",
    ),
}


@dataclass(frozen=True)
class NerSpan:
    """One entity found by the NER layer: `text[start:end]` is `entity` (Presidio's
    type), which redaction replaces with a `<kind_n>` placeholder."""

    start: int
    end: int
    entity: str
    kind: str
    score: float
    text: str


@dataclass(frozen=True)
class NerStatus:
    """Whether the NER layer can run. Truthy when available, so `if ner_available():`
    reads naturally; `reason` explains a False."""

    available: bool
    model: str
    reason: str = ""

    def __bool__(self) -> bool:
        return self.available

    def to_dict(self) -> dict[str, Any]:
        """Plain dict (for reports)."""
        return {"available": self.available, "model": self.model, "reason": self.reason}


def ner_model_name(model: str | None = None) -> str:
    """The spaCy model to use: the argument, else `$ENVFORGE_NER_MODEL`, else the default."""
    return model or os.environ.get(MODEL_ENV_VAR) or DEFAULT_MODEL


@functools.lru_cache(maxsize=4)
def _load_engine(model: str) -> tuple[Any | None, str]:
    """Build (once per model) a Presidio `BatchAnalyzerEngine`, or explain why not.

    Returns `(engine, "")` or `(None, reason)`. Cached, so a missing model is reported
    once per process rather than on every call; `reset_ner_cache()` forgets it.
    """
    if importlib.util.find_spec("presidio_analyzer") is None:
        return None, "presidio-analyzer is not installed (uv sync --extra privacy)"
    try:
        import spacy
    except ImportError as exc:  # presidio installed without spaCy: unusual, but possible
        return None, f"spaCy is not importable: {exc}"
    if not (spacy.util.is_package(model) or Path(model).exists()):
        return None, (
            f"spaCy model {model!r} is not installed — install its wheel into the venv "
            f"(e.g. uv pip install <{model} wheel URL>) or set {MODEL_ENV_VAR}"
        )
    try:
        logging.getLogger("presidio-analyzer").setLevel(logging.ERROR)  # very chatty
        from presidio_analyzer import AnalyzerEngine, BatchAnalyzerEngine
        from presidio_analyzer.nlp_engine import NlpEngineProvider

        config = {
            "nlp_engine_name": "spacy",
            "models": [{"lang_code": "en", "model_name": model}],
        }
        nlp_engine = NlpEngineProvider(nlp_configuration=config).create_engine()
        analyzer = AnalyzerEngine(nlp_engine=nlp_engine, supported_languages=["en"])
        # Presidio reads entities, tokens and lemmas; the dependency parser only adds
        # time. Disabling it roughly halves the cost of the small model.
        nlp = analyzer.nlp_engine.nlp["en"]
        if "parser" in nlp.pipe_names:
            nlp.disable_pipe("parser")
        return BatchAnalyzerEngine(analyzer), ""
    except Exception as exc:  # noqa: BLE001 — any failure means "run regex-only"
        return None, f"failed to load the NER engine with {model!r}: {exc}"


def reset_ner_cache() -> None:
    """Forget loaded engines and cached failures (e.g. after installing a model)."""
    _load_engine.cache_clear()


def ner_available(model: str | None = None) -> NerStatus:
    """Check (and load, once) the NER layer. Never raises."""
    name = ner_model_name(model)
    engine, reason = _load_engine(name)
    return NerStatus(available=engine is not None, model=name, reason=reason)


def analyze_texts(
    texts: Sequence[str],
    *,
    profile: Literal["standard", "strict"] = "standard",
    model: str | None = None,
    score_threshold: float = DEFAULT_SCORE_THRESHOLD,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> list[list[NerSpan]]:
    """Run NER over many texts at once; one list of spans per input text, same order.

    If the layer is unavailable (or fails mid-way), returns empty lists — callers treat
    that as "NER found nothing" and the regex layer still runs. Spans scoring below
    `score_threshold` are dropped (spaCy entities score 0.85; Presidio's pattern
    recognisers score lower unless context words such as "license" appear nearby).
    """
    empty: list[list[NerSpan]] = [[] for _ in texts]
    if not texts:
        return empty
    status = ner_available(model)
    if not status:
        logger.debug("NER unavailable, regex-only redaction: %s", status.reason)
        return empty
    engine, _ = _load_engine(status.model)
    entities = list(PROFILE_ENTITIES[profile])
    # Only texts with letters are worth a model pass; others keep an empty result.
    wanted = [i for i, t in enumerate(texts) if t and any(ch.isalpha() for ch in t)]
    try:
        max_len = engine.analyzer_engine.nlp_engine.nlp["en"].max_length
        batch = [texts[i][:max_len] for i in wanted]
        results = engine.analyze_iterator(
            batch,
            language="en",
            entities=entities,
            score_threshold=score_threshold,
            batch_size=batch_size,
        )
    except Exception as exc:  # noqa: BLE001 — never let NER break redaction
        logger.warning("NER failed, falling back to regex-only redaction: %s", exc)
        return empty
    for i, found in zip(wanted, results, strict=True):
        text = texts[i]
        empty[i] = sorted(
            (
                NerSpan(
                    start=r.start,
                    end=r.end,
                    entity=r.entity_type,
                    kind=ENTITY_KINDS.get(r.entity_type, r.entity_type),
                    score=float(r.score),
                    text=text[r.start : r.end],
                )
                for r in found
            ),
            key=lambda s: (s.start, -s.end),
        )
    return empty

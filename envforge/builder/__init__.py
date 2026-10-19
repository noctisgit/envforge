"""The data-ingestion half of the environment builder.

Users upload support chat logs; the builder agent turns them into a declarative
support_desk environment (`envforge.declarative.spec.EnvSpec` + `TaskRecord`s). These
modules are the pipeline steps the builder agent calls as tools:

    upload    a CSV / TXT / JSON / JSONL file saved by the web layer
      |
    parse     ingest.detect_format(path) -> FormatGuess (format, mapping, confidence)
              ingest.parse_file(path, mapping=None) -> ParseResult(conversations, ...)
      |
    redact    redact.redact_conversations(conversations) -> (redacted, counts)
              Best-effort PII removal with consistent placeholders (<EMAIL_1>, <NAME_2>).
              Must run BEFORE any text reaches an LLM or the builder chat.
      |
    profile   ingest.profile(redacted) -> dict (sizes, roles, language, intent words)
      |
    (spec)    the builder agent drafts an EnvSpec from the profile and samples
      |
    estimate  extract.estimate_cost(spec, redacted, "anthropic:claude-opus-5-5")
      |
    extract   extract.extract_batch(model, spec, redacted, limit=..., on_progress=...)
              One LLM call (plus at most one retry) per conversation -> TaskRecord | skip
      |
    validate  envforge.declarative.validate (runs reference solutions against checks)

Nothing in this package stores original PII values or sends unredacted text anywhere:
`extract` refuses conversations without `meta["redacted"] = True`.
"""

from envforge.builder.extract import (
    BatchResult,
    ExtractionResult,
    UnredactedConversationError,
    build_extraction_request,
    estimate_cost,
    extract_batch,
    extract_task,
)
from envforge.builder.ingest import (
    Conversation,
    FormatGuess,
    Mapping,
    ParseResult,
    Turn,
    detect_format,
    parse_file,
    profile,
)
from envforge.builder.redact import (
    RedactionResult,
    redact_conversation,
    redact_conversations,
    redact_text,
)

__all__ = [
    "BatchResult",
    "Conversation",
    "ExtractionResult",
    "FormatGuess",
    "Mapping",
    "ParseResult",
    "RedactionResult",
    "Turn",
    "UnredactedConversationError",
    "build_extraction_request",
    "detect_format",
    "estimate_cost",
    "extract_batch",
    "extract_task",
    "parse_file",
    "profile",
    "redact_conversation",
    "redact_conversations",
    "redact_text",
]

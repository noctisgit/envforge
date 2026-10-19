"""Parse uploaded support chat logs (CSV, JSON, JSONL, TXT) into one common shape.

Everything downstream (redaction, profiling, LLM extraction) works on:

    Conversation(id, turns=[Turn(role, text, ts, speaker), ...], meta={...})

where `role` is one of "customer" | "agent" | "system" | "unknown".

Three entry points:

    guess = detect_format(path)              # what we think the file is, and why
    result = parse_file(path)                # or parse_file(path, mapping=guess.mapping)
    stream = iter_conversations(path)        # the same, one conversation at a time
    for conv in stream: ...                  # (bounded memory, for multi-GB files)

`detect_format` returns a `FormatGuess` with a confidence and the `Mapping` it chose
(which column holds the conversation id, which holds the text, ...). If the guess is
wrong, the user (or the builder agent) edits the mapping and passes it back to
`parse_file` explicitly.

Supported shapes (best effort, auto-detected):

    CSV   one row per message: conversation id + role/sender + text (+ timestamp)
          one row per conversation: a transcript column ("Customer: ...\\nAgent: ...")
    JSON  a list of conversations ({"id", "messages": [{"role", "content"}]}, ...),
          a dict wrapping such a list ({"conversations": [...]}, {"tickets": [...]}),
          a dict of id -> message list, Zendesk-like tickets (comments + requester_id),
          Intercom-like conversations (source + conversation_parts),
          or a flat list of messages that each carry a conversation id
    JSONL the same objects, one per line
    TXT   transcripts with speaker prefixes ("Customer:", "[10:01] Jane:"), split into
          conversations by "---" lines, "Conversation 12" headers, or blank-line runs

Large files are read as a stream (line by line, or item by item for JSON arrays), and
each conversation is capped (`max_turns`, `max_turn_chars`); truncation is reported.
`parse_file` still returns every conversation as one list; `iter_conversations` hands
them out as soon as they are complete, so memory stays flat however big the file is
(see `ConversationStream` for how "complete" is decided).
Parsing never sends anything anywhere — the text may contain personal data, so run
`envforge.builder.redact` before showing it to a model or a person.
"""

from __future__ import annotations

import codecs
import csv
import html
import io
import json
import re
import statistics
import sys
from collections import Counter, deque
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal, TextIO

Role = Literal["customer", "agent", "system", "unknown"]
FileFormat = Literal["csv", "json", "jsonl", "txt"]
Layout = Literal["message_rows", "transcript_rows", "conversations", "transcript_text"]
TxtSplit = Literal["markers", "blank_runs", "blank_line"]

SAMPLE_BYTES = 64 * 1024  # how much of the file detect_format looks at
DEFAULT_MAX_TURNS = 200  # per conversation; longer threads are cut (and reported)
DEFAULT_MAX_TURN_CHARS = 4000  # per message
JSON_OBJECT_WARN_BYTES = 256 * 1024 * 1024  # a top-level JSON *object* must be loaded whole
# Streaming (iter_conversations): how many conversations may be "open" (still collecting
# messages) at once, and how much text they may hold in total, before the least
# recently touched one is considered complete and handed out.
DEFAULT_STREAM_WINDOW = 2000
DEFAULT_STREAM_OPEN_CHARS = 20_000_000
MAX_KEPT_WARNINGS = 200  # per parse; further warnings are only counted


# ---- the common shape ----------------------------------------------------------------


@dataclass
class Turn:
    """One message in a conversation."""

    role: Role
    text: str
    ts: str | None = None
    # the raw speaker label from the file ("Jane", "Agent Mike"), if any. It can be a
    # person's name, so redaction rewrites it.
    speaker: str | None = None


@dataclass
class Conversation:
    """One support conversation (ticket, chat, call transcript)."""

    id: str
    turns: list[Turn]
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Plain JSON-able dict (for persisting parsed or redacted conversations)."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Conversation:
        """Inverse of `to_dict`."""
        turns = [Turn(**t) for t in data.get("turns", [])]
        return cls(id=str(data["id"]), turns=turns, meta=dict(data.get("meta", {})))


@dataclass
class Mapping:
    """How to read a file: which layout, and which columns / fields mean what.

    Column and field names are the file's own names (e.g. "Ticket ID"). Leave a field
    as None to let the parser look for the usual synonyms itself (JSON only).
    """

    layout: Layout
    format: FileFormat | None = None  # None: use the detected format
    conversation_id: str | None = None
    role: str | None = None
    text: str | None = None
    timestamp: str | None = None
    transcript: str | None = None  # transcript_rows: the column holding the whole chat
    messages: str | None = None  # JSON conversations: the key holding the message list
    delimiter: str = ","  # CSV only
    txt_split: TxtSplit = "blank_runs"  # TXT only: how conversations are separated
    # extra role labels, lower-case label -> role, e.g. {"jane (tier 2)": "agent"}
    role_map: dict[str, Role] = field(default_factory=dict)


@dataclass
class FormatGuess:
    """What `detect_format` thinks a file is."""

    format: FileFormat
    mapping: Mapping
    confidence: float  # 0..1
    encoding: str
    reasons: list[str] = field(default_factory=list)


@dataclass
class ParseResult:
    """The output of `parse_file`."""

    conversations: list[Conversation]
    format: FormatGuess
    mapping: Mapping
    warnings: list[str] = field(default_factory=list)
    skipped_rows: int = 0  # rows / items / lines we could not use
    truncated: int = 0  # conversations cut by max_turns or max_turn_chars


# ---- roles -----------------------------------------------------------------------------

CUSTOMER_WORDS = {
    "customer", "user", "client", "caller", "requester", "visitor", "consumer", "guest",
    "member", "subscriber", "enduser", "human", "buyer", "shopper", "patient", "lead",
    "contact", "inbound",
}  # fmt: skip
AGENT_WORDS = {
    "agent", "support", "rep", "representative", "operator", "staff", "bot", "assistant",
    "admin", "advisor", "adviser", "technician", "tech", "csr", "helpdesk", "service",
    "team", "employee", "specialist", "associate", "gpt", "ai", "chatbot", "outbound",
    "care", "success",
}  # fmt: skip
SYSTEM_WORDS = {"system", "note", "internal", "automated", "event", "notification", "action"}

# phrases that give away the support side of a conversation
AGENT_GREETING = re.compile(
    r"(?i)\b(?:thanks?(?: you)? for (?:contacting|reaching out|calling|getting in touch|"
    r"chatting|your patience)|how (?:can|may|might) i (?:help|assist)|"
    r"what can i (?:do|help)|i(?:'| a)?m \w+ from|this is \w+ (?:from|with)|"
    r"welcome to|is there anything else i can)"
)


def normalize_role(label: Any, role_map: dict[str, Role] | None = None) -> Role:
    """Map a raw role / sender label to customer | agent | system | unknown.

    "Customer", "end-user", "Caller (Jane)" -> customer; "Support Agent", "bot", "admin"
    -> agent; "system", "internal note" -> system. A bare name ("Jane") is unknown;
    `assign_unknown_roles` guesses those from the conversation.
    """
    if label is None:
        return "unknown"
    s = str(label).strip().lower()
    if not s:
        return "unknown"
    if role_map and s in role_map:
        return role_map[s]
    tokens = set(re.findall(r"[a-z]+", s))
    # agent first: "Customer Support" / "customer service" are the support side
    if tokens & AGENT_WORDS:
        return "agent"
    if tokens & CUSTOMER_WORDS:
        return "customer"
    if tokens & SYSTEM_WORDS:
        return "system"
    return "unknown"


def assign_unknown_roles(turns: list[Turn]) -> int:
    """Guess roles for turns whose speaker label was a name (in place).

    Heuristics, in order:
      1. a speaker who greets like support ("thank you for contacting ...") is the agent;
      2. if the conversation already has a known customer (or agent), the unknown
         speakers are the other side;
      3. otherwise the first speaker is the customer and everyone else is the agent.
    Turns with no speaker label at all alternate customer / agent.

    Returns the number of turns whose role was guessed.
    """
    unknown = [t for t in turns if t.role == "unknown"]
    if not unknown:
        return 0
    speakers: list[str] = []
    for t in unknown:
        key = (t.speaker or "").strip()
        if key and key not in speakers:
            speakers.append(key)
    if not speakers:  # no labels at all: alternate, starting with the customer
        for i, t in enumerate(unknown):
            t.role = "customer" if i % 2 == 0 else "agent"
        return len(unknown)

    assigned: dict[str, Role] = {}
    greeters = {
        s for s in speakers if any(AGENT_GREETING.search(t.text) for t in unknown if t.speaker == s)
    }
    known = {t.role for t in turns if t.role in ("customer", "agent")}
    if greeters:
        for s in speakers:
            assigned[s] = "agent" if s in greeters else "customer"
    elif known == {"customer"}:
        assigned = dict.fromkeys(speakers, "agent")
    elif known == {"agent"}:
        assigned = dict.fromkeys(speakers, "customer")
    else:
        first = next((t.speaker for t in turns if t.role == "unknown"), None)
        for s in speakers:
            assigned[s] = "customer" if s == first else "agent"
    for t in unknown:
        t.role = assigned.get((t.speaker or "").strip(), "unknown")
    return len(unknown)


# ---- encodings and samples -------------------------------------------------------------


def detect_encoding(path: str | Path) -> str:
    """utf-8 (or utf-8-sig / utf-16 when there is a BOM), else latin-1.

    latin-1 decodes any byte sequence, so it is the last-resort fallback: accented
    characters from cp1252 exports come out right; true mojibake is still readable.
    """
    with open(path, "rb") as f:
        head = f.read(SAMPLE_BYTES)
    if head.startswith(codecs.BOM_UTF8):
        return "utf-8-sig"
    if head.startswith((codecs.BOM_UTF16_LE, codecs.BOM_UTF16_BE)):
        return "utf-16"
    try:
        codecs.getincrementaldecoder("utf-8")().decode(head, final=False)
        return "utf-8"
    except UnicodeDecodeError:
        return "latin-1"


def _open_text(path: str | Path, encoding: str) -> TextIO:
    # utf-8 is checked on the first 64 KB only; replace stray bad bytes further on
    # rather than failing a multi-GB parse at 90%.
    errors = "replace" if encoding.startswith("utf") else "strict"
    return open(path, encoding=encoding, errors=errors, newline="")


def read_sample(path: str | Path) -> tuple[str, str]:
    """(first ~64 KB of text, encoding)."""
    encoding = detect_encoding(path)
    with _open_text(path, encoding) as f:
        return f.read(SAMPLE_BYTES), encoding


# ---- column / field synonyms -------------------------------------------------------------

CONV_ID_NAMES = [
    "conversation_id", "conversationid", "conv_id", "ticket_id", "ticketid", "chat_id",
    "chatid", "thread_id", "threadid", "session_id", "sessionid", "case_id", "caseid",
    "dialog_id", "dialogue_id", "ticket", "conversation", "chat", "thread", "case",
    "ticket_number", "id",
]  # fmt: skip
ROLE_NAMES = [
    "role", "sender", "author", "from", "speaker", "sender_type", "author_type",
    "user_type", "participant", "direction", "who", "sender_role", "author_role", "type",
    "name", "user",
]  # fmt: skip
TEXT_NAMES = [
    "text", "message", "body", "content", "msg", "utterance", "comment", "message_text",
    "message_body", "plain_body", "value", "html_body",
]  # fmt: skip
TS_NAMES = [
    "timestamp", "time", "created_at", "createdat", "datetime", "date", "sent_at",
    "ts", "created", "time_sent",
]  # fmt: skip
TRANSCRIPT_NAMES = [
    "transcript", "conversation_text", "chat_log", "chatlog", "dialogue", "dialog",
    "full_text", "conversation", "chat", "log", "messages", "thread", "text", "body",
]  # fmt: skip
MESSAGE_LIST_KEYS = [
    "messages", "turns", "conversation", "dialog", "dialogue", "utterances", "comments",
    "original",  # ABCD (ASAPP): "original": [["customer", "text"], ...]
    "conversation_parts", "parts", "chat", "thread", "events", "transcript", "entries",
]  # fmt: skip
WRAPPER_KEYS = [
    "conversations", "tickets", "chats", "threads", "data", "items", "results", "records",
    "dialogs", "dialogues", "sessions", "cases",
]  # fmt: skip


def _norm(name: str) -> str:
    """'Ticket ID' -> 'ticket_id'."""
    return re.sub(r"[^a-z0-9]+", "_", name.strip().lower()).strip("_")


def _pick(columns: list[str], synonyms: list[str], exclude: set[str] = frozenset()) -> str | None:
    """First column whose normalised name matches a synonym (synonym order wins)."""
    normed = {_norm(c): c for c in columns if c not in exclude}
    for syn in synonyms:
        if syn in normed:
            return normed[syn]
    return None


# ---- TXT transcripts ---------------------------------------------------------------------

_TS = r"(?:\d{4}-\d{2}-\d{2}[ T])?\d{1,2}:\d{2}(?::\d{2})?(?:\s?[AaPp]\.?[Mm]\.?)?"
_SPEAKER_LINE = re.compile(
    rf"^\s*(?:[\[(](?P<ts1>{_TS})[\])]\s*[-–]?\s*|(?P<ts2>{_TS})\s*[-–|]?\s+)?"
    r"(?P<speaker>[A-Za-z][\w .'’@&/()-]{0,40}?)\s*"
    rf"(?:[\[(](?P<ts3>{_TS})[\])])?\s*:\s*(?P<text>.*)$"
)
_SEPARATOR = re.compile(r"^\s*(?:-{3,}|={3,}|\*{3,}|_{3,}|#{3,}|~{3,})\s*$")
_HEADER = re.compile(
    r"(?i)^\s*[=#*\-\s]*(?:conversation|chat|ticket|transcript|session|case|thread|"
    r"dialog(?:ue)?)\s*(?:id|no\.?|number)?\s*[#:]?\s*(?P<id>[\w-]*\d[\w-]*)?\s*[:.]?[=#*\-\s]*$"
)
# "Subject: ..." style header lines inside a transcript are metadata, not speakers
_META_LABELS = {
    "date", "time", "subject", "ticket", "ticket id", "status", "priority", "channel",
    "tags", "created", "updated", "email", "phone", "category", "rating", "duration",
    "started", "ended", "re", "to", "cc", "sent", "http", "https",
}  # fmt: skip


def _speaker_turn(line: str, known: set[str]) -> tuple[str, str, str | None] | None:
    """('Jane', 'my wifi is down', '10:01') if the line starts a new turn, else None."""
    m = _SPEAKER_LINE.match(line)
    if not m:
        return None
    speaker = m.group("speaker").strip()
    text = m.group("text").strip()
    ts = m.group("ts1") or m.group("ts2") or m.group("ts3")
    low = speaker.lower()
    if low in _META_LABELS or text.startswith("//"):
        return None
    words = speaker.split()
    if len(words) > 4:
        return None
    # Accept the label if it is clearly a speaker: a role word, a timestamp, a label we
    # have already seen, or a short capitalised name. "order status: shipped" is not.
    name_like = all(w[:1].isupper() or w[:1] in "(@" for w in words) and len(words) <= 3
    if normalize_role(speaker) != "unknown" or ts or low in known or name_like:
        return speaker, text, ts
    return None


def _txt_split_mode(sample: str) -> TxtSplit:
    """Pick how a TXT file separates conversations, from a sample."""
    lines = sample.splitlines()
    if any(_SEPARATOR.match(ln) or _HEADER.match(ln) for ln in lines if ln.strip()):
        return "markers"
    if re.search(r"\n[ \t]*\r?\n[ \t]*\r?\n", sample):
        return "blank_runs"
    blank = sum(1 for ln in lines if not ln.strip())
    speaker_lines = sum(1 for ln in lines if _speaker_turn(ln, set()))
    # single blank lines between conversations only if they are rarer than turns
    if blank and blank * 2 <= speaker_lines:
        return "blank_line"
    return "blank_runs"


def iter_transcript_blocks(
    lines: Iterator[str], split: TxtSplit
) -> Iterator[tuple[str | None, list[tuple[str, str, str | None]], int]]:
    """Split transcript lines into conversations.

    Yields (header id or None, [(speaker, text, ts), ...], skipped line count). Lines
    without a speaker prefix continue the previous turn.
    """
    header_id: str | None = None
    turns: list[tuple[str, str, str | None]] = []
    skipped = 0
    blanks = 0
    known: set[str] = set()

    def flush() -> Iterator[tuple[str | None, list[tuple[str, str, str | None]], int]]:
        nonlocal header_id, turns, skipped
        if turns:
            yield header_id, turns, skipped
            skipped = 0
        header_id, turns = None, []

    for raw in lines:
        line = raw.rstrip("\r\n")
        if not line.strip():
            blanks += 1
            if (split == "blank_line" and blanks == 1) or (split == "blank_runs" and blanks == 2):
                yield from flush()
            continue
        blanks = 0
        if _SEPARATOR.match(line):
            yield from flush()
            continue
        header = _HEADER.match(line)
        if header:
            yield from flush()
            header_id = header.group("id")
            continue
        turn = _speaker_turn(line, known)
        if turn:
            known.add(turn[0].lower())
            turns.append(turn)
        elif turns:
            speaker, text, ts = turns[-1]
            turns[-1] = (speaker, f"{text}\n{line.strip()}".strip(), ts)
        else:
            skipped += 1  # preamble before the first speaker line
    yield from flush()


def parse_transcript(text: str) -> list[tuple[str, str, str | None]]:
    """Turns of one transcript string ("Customer: hi\\nAgent: hello"). Empty if the text
    has no speaker prefixes at all."""
    turns: list[tuple[str, str, str | None]] = []
    for _, block, _ in iter_transcript_blocks(iter(text.splitlines()), "markers"):
        turns.extend(block)
    return turns


# ---- collecting conversations ----------------------------------------------------------


def _clean_text(value: Any) -> str:
    """Message text from a JSON/CSV value: joins content-block lists, strips HTML."""
    if value is None:
        return ""
    if isinstance(value, list):  # [{"type": "text", "text": "..."}] or ["a", "b"]
        parts = [
            v.get("text") or v.get("content") or "" if isinstance(v, dict) else str(v)
            for v in value
        ]
        value = "\n".join(p for p in parts if p)
    text = str(value)
    if re.search(r"</\w+>|<br\s*/?>|<p>", text, re.I):
        text = re.sub(r"(?i)<br\s*/?>|</p>|</div>", "\n", text)
        text = re.sub(r"<[^>]+>", "", text)
        text = html.unescape(text)
    return text.strip()


@dataclass
class _Draft:
    id: str
    turns: list[Turn] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)
    truncated: bool = False
    chars: int = 0  # text held so far (for the streaming memory cap)


class _Collector:
    """Accumulates turns per conversation id while enforcing the size caps.

    Two modes:

    - collect everything (`window=None`, used by `parse_file`): every draft stays open
      until `results()`, so messages of one conversation may be anywhere in the file;
    - streaming (`window=N`, used by `iter_conversations`): at most N drafts (and
      `max_open_chars` of text) stay open. When a new one would exceed that, the least
      recently touched draft is finished and queued in `ready`; the parser hands those
      out via `drain()` after every row. Memory is bounded by the window, not the file.
    """

    def __init__(
        self,
        mapping: Mapping,
        source: str,
        max_turns: int,
        max_turn_chars: int,
        max_conversations: int | None,
        *,
        window: int | None = None,
        max_open_chars: int | None = None,
    ) -> None:
        self.mapping = mapping
        self.source = source
        self.max_turns = max_turns
        self.max_turn_chars = max_turn_chars
        self.max_conversations = max_conversations
        self.window = window
        self.max_open_chars = max_open_chars
        self.drafts: dict[str, _Draft] = {}  # open drafts; LRU order when streaming
        self.ready: deque[Conversation] = deque()  # finished, waiting for drain()
        self.created = 0  # drafts ever opened (ids "conv-N" and max_conversations use it)
        self.open_chars = 0
        self.skipped = 0
        self.truncated = 0  # finished conversations that were cut by the caps
        self.empty = 0  # finished drafts with no usable message (dropped)
        self.full = False  # max_conversations reached
        self.warnings: list[str] = []  # the first MAX_KEPT_WARNINGS
        self.n_warnings = 0

    def warn(self, message: str) -> None:
        """Record a warning; only the first MAX_KEPT_WARNINGS are kept (a multi-GB file
        with a bad row every few lines must not fill memory with warnings)."""
        self.n_warnings += 1
        if len(self.warnings) < MAX_KEPT_WARNINGS:
            self.warnings.append(message)

    def draft(self, conv_id: Any) -> _Draft | None:
        cid = str(conv_id).strip() or f"conv-{self.created + 1}"
        d = self.drafts.get(cid)
        if d is None:
            if self.max_conversations is not None and self.created >= self.max_conversations:
                self.full = True
                return None
            d = self.drafts[cid] = _Draft(id=cid)
            self.created += 1
        elif self.window is not None:  # streaming: mark as most recently used
            self.drafts[cid] = self.drafts.pop(cid)
        return d

    def add(
        self,
        conv_id: Any,
        role_label: Any,
        text: Any,
        ts: Any = None,
        *,
        role: Role | None = None,
    ) -> None:
        """Add one message. `role` overrides label-based normalisation."""
        body = _clean_text(text)
        if not body:
            self.skipped += 1
            return
        d = self.draft(conv_id)
        if d is None:
            return
        if len(d.turns) >= self.max_turns:
            d.truncated = True
            return
        if len(body) > self.max_turn_chars:
            body = body[: self.max_turn_chars] + " [...]"
            d.truncated = True
        label = None if role_label is None else str(role_label).strip() or None
        d.turns.append(
            Turn(
                role=role or normalize_role(label, self.mapping.role_map),
                text=body,
                ts=None if ts in (None, "") else str(ts),
                speaker=label,
            )
        )
        d.chars += len(body)
        self.open_chars += len(body)
        if self.window is not None:
            self._evict()

    def _evict(self) -> None:
        """Streaming: finish least-recently-touched drafts until within the limits.

        The newest draft (the one just touched, last in LRU order) is never evicted.
        """
        while len(self.drafts) > 1 and (
            len(self.drafts) > (self.window or 0)
            or (self.max_open_chars is not None and self.open_chars > self.max_open_chars)
        ):
            oldest = next(iter(self.drafts))
            self._ready(self.drafts.pop(oldest))

    def _ready(self, d: _Draft) -> None:
        self.open_chars -= d.chars
        conv = self._finish(d)
        if conv is not None:
            self.ready.append(conv)

    def _finish(self, d: _Draft) -> Conversation | None:
        """Turn a draft into a Conversation (guess roles, add meta); None if empty."""
        if not d.turns:
            self.empty += 1
            return None
        guessed = assign_unknown_roles(d.turns)
        meta = {"source_file": self.source, "n_turns": len(d.turns), **d.meta}
        if guessed:
            meta["roles_guessed"] = guessed
        if d.truncated:
            meta["truncated"] = True
            self.truncated += 1
        return Conversation(id=d.id, turns=d.turns, meta=meta)

    def drain(self) -> Iterator[Conversation]:
        """Hand out (and forget) the conversations finished so far."""
        while self.ready:
            yield self.ready.popleft()

    def flush_all(self) -> None:
        """End of file: every open draft is complete."""
        for d in list(self.drafts.values()):
            self._ready(d)
        self.drafts.clear()

    def results(self) -> tuple[list[Conversation], int]:
        """Collect-everything mode: finish all drafts, in first-seen order."""
        self.flush_all()
        return list(self.drain()), self.truncated


# ---- CSV ---------------------------------------------------------------------------------


def _csv_mapping(
    header: list[str], rows: list[list[str]], delimiter: str
) -> tuple[Mapping, float, list[str]]:
    """Choose a CSV layout and columns from the header and a few sample rows."""
    reasons: list[str] = []
    text = _pick(header, TEXT_NAMES[:-2])  # not "value"/"html_body" for CSV
    role = _pick(header, ROLE_NAMES, exclude={text} if text else set())
    used = {c for c in (text, role) if c}
    conv = _pick(header, CONV_ID_NAMES, exclude=used)
    ts = _pick(header, TS_NAMES, exclude=used | ({conv} if conv else set()))
    transcript = _pick(header, TRANSCRIPT_NAMES)

    def col_values(col: str | None) -> list[str]:
        if col is None:
            return []
        i = header.index(col)
        return [r[i] for r in rows if i < len(r)]

    def looks_like_transcripts(col: str | None) -> bool:
        vals = col_values(col)
        hits = sum(1 for v in vals if len(parse_transcript(v)) >= 2)
        return bool(vals) and hits >= max(1, len(vals) // 2)

    # A "conversation" column full of multi-turn text is a transcript, not an id.
    if transcript and looks_like_transcripts(transcript) and not (text and role):
        conv_for_t = _pick(header, CONV_ID_NAMES, exclude={transcript})
        reasons.append(f"column '{transcript}' holds whole transcripts (one row per conversation)")
        if conv_for_t:
            reasons.append(f"conversation id from '{conv_for_t}'")
        m = Mapping(
            layout="transcript_rows",
            format="csv",
            conversation_id=conv_for_t,
            transcript=transcript,
            timestamp=_pick(header, TS_NAMES, exclude={transcript}),
            delimiter=delimiter,
        )
        return m, 0.85 if conv_for_t else 0.75, reasons

    if text:
        m = Mapping(
            layout="message_rows",
            format="csv",
            conversation_id=conv,
            role=role,
            text=text,
            timestamp=ts,
            delimiter=delimiter,
        )
        reasons.append(f"text column '{text}'")
        confidence = 0.5
        if conv:
            reasons.append(f"conversation id column '{conv}'")
            confidence += 0.25
        else:
            reasons.append("no conversation id column: the whole file is one conversation")
        if role:
            reasons.append(f"role/sender column '{role}'")
            confidence += 0.2
        else:
            reasons.append("no role column: roles will be guessed")
        return m, round(confidence, 2), reasons

    # Nothing matched by name: fall back to the column with the longest values.
    longest = max(header, key=lambda c: sum(len(v) for v in col_values(c)), default=None)
    reasons.append(f"no known column names; using the longest column '{longest}' as text")
    m = Mapping(layout="message_rows", format="csv", text=longest, delimiter=delimiter)
    return m, 0.2, reasons


def _sniff_delimiter(sample: str, path: Path) -> str:
    if path.suffix.lower() == ".tsv":
        return "\t"
    try:
        return csv.Sniffer().sniff(sample[:8192], delimiters=",;\t|").delimiter
    except csv.Error:
        return ","


def _parse_csv(f: TextIO, mapping: Mapping, col: _Collector) -> Iterator[Conversation]:
    """Read CSV rows into `col`, yielding conversations as the collector finishes them."""
    csv.field_size_limit(min(sys.maxsize, 2**31 - 1))  # transcripts can be long
    reader = csv.reader(f, delimiter=mapping.delimiter)
    header = next(reader, None)
    if header is None:
        return
    header = [h.strip() for h in header]

    def index(name: str | None) -> int | None:
        if name is None:
            return None
        if name in header:
            return header.index(name)
        normed = [_norm(h) for h in header]
        if _norm(name) in normed:
            return normed.index(_norm(name))
        raise ValueError(f"column '{name}' not found in CSV header {header}")

    ci, ri, ti = index(mapping.conversation_id), index(mapping.role), index(mapping.text)
    tsi, tri = index(mapping.timestamp), index(mapping.transcript)

    def cell(row: list[str], i: int | None) -> str | None:
        return row[i] if i is not None and i < len(row) else None

    for rownum, row in enumerate(reader, start=2):
        yield from col.drain()
        if col.full:
            break
        if not any(c.strip() for c in row):
            continue
        if mapping.layout == "transcript_rows":
            transcript = cell(row, tri) or ""
            cid = cell(row, ci) or f"row-{rownum}"
            turns = parse_transcript(transcript)
            if not turns:
                if transcript.strip():
                    col.add(cid, None, transcript, cell(row, tsi))
                    col.warn(f"row {rownum}: no speaker prefixes found in transcript")
                else:
                    col.skipped += 1
                continue
            for speaker, text, ts in turns:
                col.add(cid, speaker, text, ts)
        else:
            cid = cell(row, ci) if ci is not None else "conv-1"
            if ci is not None and not (cid or "").strip():
                col.skipped += 1
                continue
            col.add(cid, cell(row, ri), cell(row, ti), cell(row, tsi))


# ---- JSON / JSONL ------------------------------------------------------------------------

ID_KEYS = [
    "id",
    "conversation_id",
    "convo_id",
    "ticket_id",
    "chat_id",
    "thread_id",
    "session_id",
    "uuid",
]


def _first_key(d: dict[str, Any], keys: list[str]) -> str | None:
    lower = {k.lower(): k for k in d}
    for k in keys:
        if k in lower:
            return lower[k]
    return None


def _messages_value(conv: dict[str, Any], key: str | None) -> Any:
    """The message list (or transcript string) of a conversation object."""
    k = key if key and key in conv else _first_key(conv, MESSAGE_LIST_KEYS)
    if k is None:
        return None
    value = conv[k]
    if isinstance(value, dict):  # Intercom: {"conversation_parts": {"conversation_parts": [...]}}
        inner = _first_key(value, MESSAGE_LIST_KEYS)
        value = value.get(inner) if inner else None
    return value


def _message_fields(msg: Any, mapping: Mapping) -> tuple[Any, Any, Any]:
    """(role label, text, timestamp) of one JSON message, whatever its shape."""
    if isinstance(msg, str):
        return None, msg, None
    if isinstance(msg, (list, tuple)) and len(msg) == 2 and all(isinstance(x, str) for x in msg):
        return msg[0], msg[1], None  # a [speaker, text] pair (ABCD and similar corpora)
    if not isinstance(msg, dict):
        return None, None, None
    rk = mapping.role if mapping.role in msg else _first_key(msg, ROLE_NAMES)
    tk = mapping.text if mapping.text in msg else _first_key(msg, TEXT_NAMES)
    sk = mapping.timestamp if mapping.timestamp in msg else _first_key(msg, TS_NAMES)
    role: Any = msg.get(rk) if rk else None
    if isinstance(role, dict):  # {"author": {"type": "admin", "name": "Mike"}}
        role = role.get("type") or role.get("role") or role.get("name")
    if rk == tk:
        role = None
    return role, msg.get(tk) if tk else None, msg.get(sk) if sk else None


def _add_json_conversation(conv: Any, index: int, mapping: Mapping, col: _Collector) -> None:
    """Add one conversation-shaped JSON item (dict with messages, list, or string)."""
    if isinstance(conv, str):
        cid = f"conv-{index}"
        turns = parse_transcript(conv)
        for speaker, text, ts in turns or [(None, conv, None)]:
            col.add(cid, speaker, text, ts)
        return
    if isinstance(conv, list):
        cid = f"conv-{index}"
        for msg in conv:
            role, text, ts = _message_fields(msg, mapping)
            col.add(cid, role, text, ts)
        return
    if not isinstance(conv, dict):
        col.skipped += 1
        return

    idk = mapping.conversation_id if mapping.conversation_id in conv else _first_key(conv, ID_KEYS)
    cid = conv.get(idk) if idk else None
    cid = f"conv-{index}" if cid in (None, "") else cid
    messages = _messages_value(conv, mapping.messages)
    requester = conv.get("requester_id")  # Zendesk
    source = conv.get("source") if isinstance(conv.get("source"), dict) else None  # Intercom

    if source and (source.get("body") or source.get("text")):
        role, text, ts = _message_fields(source, mapping)
        col.add(cid, role or "customer", text, ts or conv.get("created_at"))
    if isinstance(messages, str):
        turns = parse_transcript(messages)
        for speaker, text, ts in turns or [(None, messages, None)]:
            col.add(cid, speaker, text, ts)
        return
    if not isinstance(messages, list):
        if conv.get("description"):  # a Zendesk ticket with no comments
            col.add(cid, "customer", conv["description"], conv.get("created_at"))
        else:
            col.skipped += 1
        return
    for msg in messages:
        if isinstance(msg, dict) and requester is not None and "author_id" in msg:
            if msg.get("public") is False:
                col.add(cid, None, msg.get("body"), msg.get("created_at"), role="system")
            else:
                role = "customer" if msg["author_id"] == requester else "agent"
                col.add(cid, None, msg.get("body"), msg.get("created_at"), role=role)
            continue
        if isinstance(msg, dict) and msg.get("part_type") == "note":  # Intercom internal note
            col.add(cid, None, msg.get("body"), msg.get("created_at"), role="system")
            continue
        role, text, ts = _message_fields(msg, mapping)
        col.add(cid, role, text, ts)


def _iter_json_array(f: TextIO, chunk_size: int = 1 << 16) -> Iterator[Any]:
    """Yield the items of a top-level JSON array without loading the whole file."""
    decoder = json.JSONDecoder()
    buf = ""
    pos = 0
    started = False
    eof = False
    while True:
        # skip whitespace, the opening bracket and commas between items
        while True:
            while pos < len(buf) and buf[pos] in " \t\r\n,":
                pos += 1
            if not started and pos < len(buf):
                if buf[pos] != "[":
                    raise ValueError("not a JSON array")
                started = True
                pos += 1
                continue
            break
        if pos < len(buf) and buf[pos] == "]":
            return
        try:
            item, end = decoder.raw_decode(buf, pos)
        except json.JSONDecodeError:
            if eof:
                if buf[pos:].strip():
                    raise
                return
            chunk = f.read(chunk_size)
            if not chunk:
                eof = True
            buf = buf[pos:] + chunk
            pos = 0
            continue
        yield item
        pos = end
        if pos > chunk_size:  # drop consumed text
            buf, pos = buf[pos:], 0


def _unwrap_json_object(obj: dict[str, Any]) -> list[Any]:
    """Find the list of conversations inside a top-level JSON object."""
    key = _first_key(obj, WRAPPER_KEYS)
    if key and isinstance(obj[key], list):
        return obj[key]
    if _messages_value(obj, None) is not None:  # the object is itself one conversation
        return [obj]
    # dataset splits: {"train": [conv, ...], "dev": [...], "test": [...]} -> all of them
    lists = [v for v in obj.values() if isinstance(v, list) and v]
    if lists and all(
        isinstance(v[0], dict) and _messages_value(v[0], None) is not None for v in lists
    ):
        return [
            {**c, "split": k} if isinstance(c, dict) else c
            for k, v in obj.items()
            if isinstance(v, list)
            for c in v
        ]
    # dict of id -> conversation / message list
    items: list[Any] = []
    for k, v in obj.items():
        if isinstance(v, list):
            items.append({"id": k, "messages": v})
        elif isinstance(v, dict):
            items.append({"id": k, **v})
    return items


def _is_message_row(item: Any, mapping: Mapping) -> bool:
    """A flat message carrying its conversation id (no nested message list)."""
    if not isinstance(item, dict) or _messages_value(item, mapping.messages) is not None:
        return False
    has_conv = _first_key(item, CONV_ID_NAMES[:-1]) is not None  # not plain "id"
    has_text = _first_key(item, TEXT_NAMES) is not None
    return has_conv and has_text


def _json_mapping(items: list[Any], fmt: FileFormat) -> tuple[Mapping, float, list[str]]:
    """Choose a JSON layout from the first few items."""
    reasons: list[str] = []
    if not items:
        return Mapping(layout="conversations", format=fmt), 0.1, ["no items found in sample"]
    first = items[0]
    if all(
        _is_message_row(i, Mapping(layout="message_rows")) for i in items if isinstance(i, dict)
    ):
        conv = _first_key(first, CONV_ID_NAMES[:-1])
        role = _first_key(first, ROLE_NAMES)
        text = _first_key(first, TEXT_NAMES)
        reasons.append(f"one message per item, grouped by '{conv}'")
        m = Mapping(
            layout="message_rows",
            format=fmt,
            conversation_id=conv,
            role=role,
            text=text,
            timestamp=_first_key(first, TS_NAMES),
        )
        return m, 0.9 if role else 0.7, reasons
    if isinstance(first, dict):
        msg_key = None
        for k in MESSAGE_LIST_KEYS:
            mk = _first_key(first, [k])
            if mk is not None:
                msg_key = mk
                break
        if msg_key is None:
            reasons.append("conversation objects without a recognised message list")
            return Mapping(layout="conversations", format=fmt), 0.3, reasons
        idk = _first_key(first, ID_KEYS)
        value = _messages_value(first, msg_key)
        sample_msg = value[0] if isinstance(value, list) and value else None
        role, text, _ = _message_fields(sample_msg, Mapping(layout="conversations"))
        reasons.append(f"conversation objects with messages under '{msg_key}'")
        if "requester_id" in first:
            reasons.append("Zendesk-like: roles from requester_id vs author_id")
        if isinstance(first.get("source"), dict):
            reasons.append("Intercom-like: first message from 'source'")
        m = Mapping(layout="conversations", format=fmt, conversation_id=idk, messages=msg_key)
        confidence = 0.9 if (text is not None or isinstance(value, str)) else 0.6
        if role is None and "requester_id" not in first and not isinstance(value, str):
            reasons.append("messages have no role field: roles will be guessed")
            confidence -= 0.2
        return m, round(confidence, 2), reasons
    if isinstance(first, list):
        reasons.append("each item is a list of messages")
        return Mapping(layout="conversations", format=fmt), 0.75, reasons
    if isinstance(first, str):
        reasons.append("each item is a transcript string")
        return Mapping(layout="conversations", format=fmt), 0.7, reasons
    return Mapping(layout="conversations", format=fmt), 0.2, ["unrecognised JSON items"]


def _sample_json_items(sample: str) -> list[Any]:
    """Up to 5 items from the (possibly truncated) start of a JSON document."""
    stripped = sample.lstrip()
    if stripped.startswith("["):
        items: list[Any] = []
        try:
            for item in _iter_json_array(io.StringIO(stripped)):
                items.append(item)
                if len(items) >= 5:
                    break
        except (ValueError, json.JSONDecodeError):
            pass
        return items
    try:
        return _unwrap_json_object(json.loads(stripped))[:5]
    except json.JSONDecodeError:
        pass
    # truncated object: decode the first element of the first array we can find
    decoder = json.JSONDecoder()
    for m in re.finditer(r"\[\s*", stripped):
        try:
            item, _ = decoder.raw_decode(stripped, m.end())
            if isinstance(item, (dict, list)):
                return [item]
        except json.JSONDecodeError:
            continue
    return []


def _sample_jsonl_items(sample: str) -> list[Any]:
    items: list[Any] = []
    for line in sample.splitlines()[:20]:
        if line.strip():
            try:
                items.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        if len(items) >= 5:
            break
    return items


def _parse_json_items(
    items: Iterator[Any], mapping: Mapping, col: _Collector
) -> Iterator[Conversation]:
    """Read JSON items into `col`, yielding conversations as the collector finishes them."""
    for index, item in enumerate(items, start=1):
        yield from col.drain()
        if col.full:
            break
        if mapping.layout == "message_rows":
            if not isinstance(item, dict):
                col.skipped += 1
                continue
            ck = mapping.conversation_id or _first_key(item, CONV_ID_NAMES)
            if not ck or item.get(ck) in (None, ""):
                col.skipped += 1
                continue
            role, text, ts = _message_fields(item, mapping)
            col.add(item[ck], role, text, ts)
        else:
            _add_json_conversation(item, index, mapping, col)


def _iter_jsonl(f: TextIO, col: _Collector) -> Iterator[Any]:
    for line in f:
        if not line.strip():
            continue
        try:
            yield json.loads(line)
        except json.JSONDecodeError:
            col.skipped += 1


class _PrefixedReader:
    """A `.read(n)` source that returns `prefix` first, then continues from `f`.

    Lets `_iter_json_array` start in the middle of text we have already read.
    """

    def __init__(self, prefix: str, f: TextIO) -> None:
        self.prefix = prefix
        self.f = f

    def read(self, n: int = -1) -> str:
        if self.prefix:
            out, self.prefix = self.prefix, ""
            return out
        return self.f.read(n)


def _wrapped_array_start(head: str) -> int | None:
    """Where the conversation list starts in `{"conversations": [ ...`, if we can tell.

    Walks the top-level keys of a JSON object at the start of `head`, decoding (small)
    values that come before the list. Returns the offset of the "[" of the first key
    that is a wrapper key ("conversations", "tickets", ...), or None when the object
    has another shape or the list does not start within `head`. This is what lets a
    multi-GB `{"conversations": [...]}` file stream instead of being loaded whole.
    """
    decoder = json.JSONDecoder()
    ws = " \t\r\n"
    pos = len(head) - len(head.lstrip(ws))
    if head[pos : pos + 1] != "{":
        return None
    pos += 1
    while True:
        while pos < len(head) and head[pos] in ws + ",":
            pos += 1
        if head[pos : pos + 1] != '"':
            return None
        try:
            key, pos = decoder.raw_decode(head, pos)
        except json.JSONDecodeError:
            return None
        while pos < len(head) and head[pos] in ws:
            pos += 1
        if head[pos : pos + 1] != ":":
            return None
        pos += 1
        while pos < len(head) and head[pos] in ws:
            pos += 1
        if str(key).lower() in WRAPPER_KEYS:
            return pos if head[pos : pos + 1] == "[" else None
        try:
            _, pos = decoder.raw_decode(head, pos)
        except json.JSONDecodeError:
            return None


def _iter_parsed(
    f: TextIO, path: Path, fmt: FileFormat, mapping: Mapping, col: _Collector, warnings: list[str]
) -> Iterator[Conversation]:
    """Dispatch on the format and feed the whole file through `col`.

    Yields conversations as the collector finishes them (nothing until the end in
    collect-everything mode). Shared by `parse_file` and `iter_conversations`.
    """
    if fmt == "csv":
        yield from _parse_csv(f, mapping, col)
    elif fmt == "jsonl":
        yield from _parse_json_items(_iter_jsonl(f, col), mapping, col)
    elif fmt == "json":
        head = f.read(1)
        while head and head.isspace():
            head = f.read(1)
        f.seek(0)
        if head == "[":
            yield from _parse_json_items(_iter_json_array(f), mapping, col)
            return
        sample = f.read(SAMPLE_BYTES)
        start = _wrapped_array_start(sample)
        if start is not None:  # {"conversations": [ ... ]}: stream the list
            items = _iter_json_array(_PrefixedReader(sample[start:], f))  # type: ignore[arg-type]
            yield from _parse_json_items(items, mapping, col)
            return
        f.seek(0)
        if path.stat().st_size > JSON_OBJECT_WARN_BYTES:
            warnings.append(
                "large JSON object loaded into memory; a top-level array or JSONL streams instead"
            )
        obj = json.load(f)
        items_list = _unwrap_json_object(obj) if isinstance(obj, dict) else [obj]
        yield from _parse_json_items(iter(items_list), mapping, col)
    else:
        blocks = iter_transcript_blocks(iter(f), mapping.txt_split)
        for n, (header_id, turns, skipped) in enumerate(blocks, start=1):
            yield from col.drain()
            col.skipped += skipped
            cid = header_id or f"conv-{n}"
            while cid in col.drafts:  # repeated header ids
                cid = f"{cid}-{n}"
            for speaker, text, ts in turns:
                col.add(cid, speaker, text, ts)
            if col.full:
                break


def _summary_warnings(
    col: _Collector, max_turns: int, max_turn_chars: int, max_conversations: int | None
) -> list[str]:
    """The end-of-parse warnings (shared by `parse_file` and `ConversationStream`)."""
    warnings = list(col.warnings[:20])
    if col.n_warnings > 20:
        warnings.append(f"... and {col.n_warnings - 20} more warnings")
    if col.empty:
        warnings.append(f"{col.empty} conversation(s) had no usable messages and were dropped")
    if col.truncated:
        warnings.append(
            f"{col.truncated} conversation(s) truncated (max {max_turns} turns, "
            f"{max_turn_chars} chars per turn)"
        )
    if col.full:
        warnings.append(f"stopped after {max_conversations} conversations (max_conversations)")
    if col.skipped:
        warnings.append(f"{col.skipped} row(s)/item(s) skipped (empty or unrecognised)")
    return warnings


# ---- detection -----------------------------------------------------------------------------


def detect_format(path: str | Path, sample: str | None = None) -> FormatGuess:
    """Guess a file's format and mapping from its extension and first ~64 KB.

    Pass `sample` to skip reading the file (e.g. when the upload is still in memory).
    """
    path = Path(path)
    encoding = detect_encoding(path) if sample is None else "utf-8"
    if sample is None:
        with _open_text(path, encoding) as f:
            sample = f.read(SAMPLE_BYTES)
    ext = path.suffix.lower()
    head = sample.lstrip()

    fmt: FileFormat
    if ext in (".jsonl", ".ndjson"):
        fmt = "jsonl"
    elif ext == ".json" or head[:1] in ("[", "{"):
        lines = [ln for ln in head.splitlines() if ln.strip()]
        one_object_per_line = (
            len(lines) >= 2
            and head[:1] == "{"
            and all(ln.strip().startswith("{") for ln in lines[:3])
            and _sample_jsonl_items("\n".join(lines[:2]))
        )
        fmt = "jsonl" if one_object_per_line else "json"
    elif ext in (".csv", ".tsv"):
        fmt = "csv"
    else:
        fmt = "txt"
        first_line = head.splitlines()[0] if head else ""
        delim = _sniff_delimiter(sample, path)
        cols = next(csv.reader([first_line], delimiter=delim), [])
        if len(cols) >= 2 and (_pick(cols, TEXT_NAMES) or _pick(cols, TRANSCRIPT_NAMES)):
            fmt = "csv"

    if fmt == "csv":
        delimiter = _sniff_delimiter(sample, path)
        rows = list(csv.reader(io.StringIO(sample), delimiter=delimiter))
        header, body = (rows[0], rows[1:-1] or rows[1:]) if rows else ([], [])
        mapping, confidence, reasons = _csv_mapping(
            [h.strip() for h in header], body[:50], delimiter
        )
    elif fmt in ("json", "jsonl"):
        items = _sample_jsonl_items(sample) if fmt == "jsonl" else _sample_json_items(sample)
        mapping, confidence, reasons = _json_mapping(items, fmt)
    else:
        split = _txt_split_mode(sample)
        n_speaker = sum(1 for ln in sample.splitlines() if _speaker_turn(ln, set()))
        mapping = Mapping(layout="transcript_text", format="txt", txt_split=split)
        confidence = 0.8 if n_speaker >= 2 else 0.3
        reasons = [
            f"plain-text transcripts ({n_speaker} speaker lines in sample)",
            f"conversations split by {split.replace('_', ' ')}",
        ]
    reasons.insert(0, f"format {fmt} (extension '{ext or 'none'}')")
    return FormatGuess(
        format=fmt, mapping=mapping, confidence=confidence, encoding=encoding, reasons=reasons
    )


# ---- parsing -------------------------------------------------------------------------------


def parse_file(
    path: str | Path,
    mapping: Mapping | None = None,
    *,
    max_turns: int = DEFAULT_MAX_TURNS,
    max_turn_chars: int = DEFAULT_MAX_TURN_CHARS,
    max_conversations: int | None = None,
) -> ParseResult:
    """Parse an uploaded file into conversations.

    `mapping` overrides the detected layout/columns (start from
    `detect_format(path).mapping` and change what is wrong). `max_conversations` stops
    early, which is handy for a quick preview of a huge file.

    Every conversation is held in memory at once; for files bigger than a few hundred
    MB use `iter_conversations`, which yields them one at a time.
    """
    path = Path(path)
    guess = detect_format(path)
    if mapping is None:
        mapping = guess.mapping
    fmt: FileFormat = mapping.format or guess.format
    col = _Collector(mapping, path.name, max_turns, max_turn_chars, max_conversations)
    warnings: list[str] = []
    if guess.encoding == "latin-1":
        warnings.append("file is not valid UTF-8; decoded as latin-1")

    with _open_text(path, guess.encoding) as f:
        for _ in _iter_parsed(f, path, fmt, mapping, col, warnings):
            pass  # collect-everything mode: nothing is yielded before the end

    conversations, truncated = col.results()
    warnings.extend(_summary_warnings(col, max_turns, max_turn_chars, max_conversations))
    if not conversations:
        warnings.append("no conversations found; check the mapping")
    return ParseResult(
        conversations=conversations,
        format=guess,
        mapping=mapping,
        warnings=warnings,
        skipped_rows=col.skipped,
        truncated=truncated,
    )


class ConversationStream:
    """The conversations of a file, parsed lazily: iterate over it once.

        stream = iter_conversations("chats.jsonl")
        for conv in stream:
            ...                              # stream.bytes_read / stream.bytes_total
        stream.warnings, stream.skipped_rows, stream.truncated   # final once done

    Same parser and mapping as `parse_file`, but a conversation is yielded as soon as
    it is complete instead of collected into a list, so memory does not grow with the
    file. "Complete" is easy when each item is a whole conversation (a JSON object, a
    transcript row, a TXT block). For one-message-per-row files it is a guess: a
    conversation is finished once `window` newer conversations have been touched since
    its last message (or the open ones hold more than `max_open_chars` of text). Rows
    are almost always grouped or time-ordered, so this rarely matters; if messages of
    one conversation are spread further apart, it is yielded in several parts that
    share an id (callers that need unique ids must handle that — `CorpusStore` renames
    the later parts to "<id>#2", ...). Conversation ids generated by the parser
    ("conv-12") and the `max_conversations` limit behave as in `parse_file`.

    Limits: a top-level JSON *array* streams, and so does an object whose first wrapper
    key holds the list (`{"conversations": [...]}`, possibly after small keys such as
    "version"). Any other top-level JSON object is loaded whole (a warning says so).
    """

    def __init__(
        self,
        path: str | Path,
        mapping: Mapping | None = None,
        *,
        max_turns: int = DEFAULT_MAX_TURNS,
        max_turn_chars: int = DEFAULT_MAX_TURN_CHARS,
        max_conversations: int | None = None,
        window: int = DEFAULT_STREAM_WINDOW,
        max_open_chars: int = DEFAULT_STREAM_OPEN_CHARS,
    ) -> None:
        self.path = Path(path)
        self.format = detect_format(self.path)
        self.mapping = mapping or self.format.mapping
        self.max_turns = max_turns
        self.max_turn_chars = max_turn_chars
        self.max_conversations = max_conversations
        self.bytes_total = self.path.stat().st_size
        self.conversations = 0  # yielded so far
        self.warnings: list[str] = []  # complete once iteration has finished
        self.done = False
        self._col = _Collector(
            self.mapping,
            self.path.name,
            max_turns,
            max_turn_chars,
            max_conversations,
            window=max(1, window),
            max_open_chars=max_open_chars,
        )
        self._file: TextIO | None = None
        self._started = False

    @property
    def bytes_read(self) -> int:
        """How far into the file the parser has read (approximate: reads are buffered)."""
        if self.done:
            return self.bytes_total
        if self._file is None:
            return 0
        try:
            return min(self.bytes_total, self._file.buffer.tell())  # type: ignore[attr-defined]
        except (AttributeError, OSError, ValueError):
            return 0

    @property
    def skipped_rows(self) -> int:
        return self._col.skipped

    @property
    def truncated(self) -> int:
        return self._col.truncated

    def __iter__(self) -> Iterator[Conversation]:
        if self._started:
            raise RuntimeError("a ConversationStream can only be iterated once")
        self._started = True
        warnings: list[str] = []
        if self.format.encoding == "latin-1":
            warnings.append("file is not valid UTF-8; decoded as latin-1")
        fmt: FileFormat = self.mapping.format or self.format.format
        col = self._col
        with _open_text(self.path, self.format.encoding) as f:
            self._file = f
            for conv in _iter_parsed(f, self.path, fmt, self.mapping, col, warnings):
                self.conversations += 1
                yield conv
            col.flush_all()
            for conv in col.drain():
                self.conversations += 1
                yield conv
        self._file = None
        self.done = True
        warnings.extend(
            _summary_warnings(col, self.max_turns, self.max_turn_chars, self.max_conversations)
        )
        if not self.conversations:
            warnings.append("no conversations found; check the mapping")
        self.warnings = warnings


def iter_conversations(
    path: str | Path,
    mapping: Mapping | None = None,
    *,
    max_turns: int = DEFAULT_MAX_TURNS,
    max_turn_chars: int = DEFAULT_MAX_TURN_CHARS,
    max_conversations: int | None = None,
    window: int = DEFAULT_STREAM_WINDOW,
    max_open_chars: int = DEFAULT_STREAM_OPEN_CHARS,
) -> ConversationStream:
    """Parse a file lazily: returns a `ConversationStream` to iterate over once.

    The streaming twin of `parse_file`, for files too big to hold in memory.
    """
    return ConversationStream(
        path,
        mapping,
        max_turns=max_turns,
        max_turn_chars=max_turn_chars,
        max_conversations=max_conversations,
        window=window,
        max_open_chars=max_open_chars,
    )


# ---- profiling -----------------------------------------------------------------------------

_STOPWORDS = {
    "en": {"the", "and", "is", "to", "my", "i", "it", "a", "of", "for", "you", "not", "this",
           "that", "have", "with", "can", "on", "in", "me", "be", "but", "was", "are", "your",
           "what", "how", "do", "am", "at", "so", "just", "get", "there", "an", "or", "if",
           "please", "hi", "hello", "thanks", "thank", "hey", "im", "i'm", "it's", "its",
           "we", "our", "they", "from", "has", "had", "been", "would", "could", "will",
           "since", "any", "all", "no", "yes", "when", "why", "out", "up", "now", "again",
           "still", "dont", "don't", "cant", "can't", "need", "help", "one", "some", "about",
           "know", "like", "also", "very", "really", "keeps", "keep", "got", "did", "does"},
    "es": {"el", "la", "de", "que", "y", "en", "los", "por", "con", "para", "una", "mi", "no",
           "es", "hola", "gracias"},
    "fr": {"le", "la", "les", "de", "et", "est", "je", "pas", "pour", "une", "mon", "bonjour",
           "merci", "vous"},
    "de": {"der", "die", "das", "und", "ist", "ich", "nicht", "mit", "ein", "eine", "mein",
           "hallo", "danke", "sie"},
    "pt": {"o", "a", "de", "que", "e", "não", "para", "com", "uma", "meu", "olá", "obrigado"},
    "it": {"il", "la", "di", "che", "e", "non", "per", "con", "una", "mio", "ciao", "grazie"},
    "nl": {"de", "het", "een", "en", "is", "ik", "niet", "met", "mijn", "hallo", "bedankt"},
}  # fmt: skip
_ALL_STOPWORDS = set().union(*_STOPWORDS.values())


def _guess_language(words: list[str]) -> str | None:
    """Very rough stopword vote; None if there is too little text to tell."""
    if len(words) < 20:
        return None
    votes = {lang: sum(1 for w in words if w in sw) for lang, sw in _STOPWORDS.items()}
    lang, best = max(votes.items(), key=lambda kv: kv[1])
    return lang if best >= max(3, len(words) * 0.05) else None


def _distribution(values: list[int]) -> dict[str, float]:
    if not values:
        return {"min": 0, "median": 0, "p90": 0, "max": 0, "mean": 0}
    ordered = sorted(values)
    return {
        "min": ordered[0],
        "median": statistics.median(ordered),
        "p90": ordered[min(len(ordered) - 1, int(len(ordered) * 0.9))],
        "max": ordered[-1],
        "mean": round(statistics.fmean(ordered), 1),
    }


def profile(conversations: list[Conversation], *, top_words: int = 25) -> dict[str, Any]:
    """Summary statistics for the builder agent and the UI.

    Intent words come from each conversation's first customer turn, minus stopwords
    and redaction placeholders. Run this on *redacted* conversations if the result
    will be shown to a model.
    """
    turns_per = [len(c.turns) for c in conversations]
    chars_per = [sum(len(t.text) for t in c.turns) for c in conversations]
    roles: Counter[str] = Counter(t.role for c in conversations for t in c.turns)
    intent: Counter[str] = Counter()
    lang_words: list[str] = []
    for c in conversations:
        first = next((t.text for t in c.turns if t.role == "customer"), "")
        text = re.sub(r"<[A-Z_]+_\d+>", " ", first.lower())
        words = re.findall(r"[^\W\d_][\w'’-]*", text)
        if len(lang_words) < 5000:
            lang_words.extend(words)
        intent.update(w for w in set(words) if w not in _ALL_STOPWORDS and len(w) > 2)
    return {
        "conversations": len(conversations),
        "turns": sum(turns_per),
        "turns_per_conversation": _distribution(turns_per),
        "chars_per_conversation": _distribution(chars_per),
        "total_chars": sum(chars_per),
        "roles": dict(roles),
        "roles_guessed": sum(1 for c in conversations if c.meta.get("roles_guessed")),
        "truncated": sum(1 for c in conversations if c.meta.get("truncated")),
        "redacted": sum(1 for c in conversations if c.meta.get("redacted")),
        "no_customer_turn": sum(
            1 for c in conversations if not any(t.role == "customer" for t in c.turns)
        ),
        "language": _guess_language(lang_words),
        "top_intent_words": intent.most_common(top_words),
        "sample_ids": [c.id for c in conversations[:5]],
    }

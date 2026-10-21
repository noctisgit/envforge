"""A per-project conversation store for large uploads (SQLite, stdlib `sqlite3`).

The principle: deterministic code reads 100% of the data; the builder LLM reads a small,
well-chosen sample. A multi-GB chat archive never fits in a prompt (or in memory), so
the pipeline is:

    ingest_file_to_store(path, store, upload_id)   stream: parse -> redact -> insert
          |                                        (exact duplicates found on the way)
    dedupe_near(store)                             MinHash + LSH: near-duplicate threads
          |
    cluster(store)                                 TF-IDF + k-means: what topics exist
          |
    overview(store)            -> dict             a compact corpus summary for the LLM
    stratified_sample(store, n) -> ids             a diverse, proportional sample
    store.get(ids)              -> conversations   the few the LLM actually reads

One SQLite file per project, in WAL mode (readers don't block the writer):

    conversations(id TEXT PK          "<upload_id>:<conversation id>" — stable across runs
                  seq INTEGER UNIQUE   insertion counter: the deterministic iteration order
                  upload_id, ext_id    where it came from (ext_id is the redacted file id)
                  n_turns, n_chars     sizes, for statistics without parsing `data`
                  content_hash         sha256 of the normalised text (exact duplicates)
                  near_dup_of          NULL, or the id of the earliest copy it duplicates
                  cluster              NULL until `cluster()` has run
                  data                 JSON: the *redacted* Conversation.to_dict())
    clusters(cluster PK, size, label, top_terms JSON, examples JSON)
    meta(key PK, value JSON)           upload reports, clustering parameters, staleness

Only redacted text is ever written: `add_conversations` refuses a conversation whose
`meta["redacted"]` is not True, and `ingest_file_to_store` redacts each conversation
before it reaches the store.

scikit-learn and numpy are needed for `dedupe_near` / `cluster` (numpy only for the
former); they are imported inside those functions, so the store itself is stdlib-only.
"""

from __future__ import annotations

import hashlib
import heapq
import json
import math
import re
import sqlite3
import tempfile
import time
import zlib
from collections import Counter, deque
from collections.abc import Callable, Iterable, Iterator, Sequence
from concurrent.futures import Future, ProcessPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from envforge.builder.ingest import (
    _ALL_STOPWORDS,
    DEFAULT_MAX_TURN_CHARS,
    DEFAULT_MAX_TURNS,
    DEFAULT_STREAM_WINDOW,
    Conversation,
    Mapping,
    iter_conversations,
)
from envforge.builder.redact import redact_conversation

SCHEMA_VERSION = 1
SQL_CHUNK = 500  # ids per "IN (...)" query (well under SQLite's parameter limit)

# A redactor takes a conversation and returns (redacted copy, counts by PII type) —
# the signature of `redact.redact_conversation`. Pass a different one (e.g. a
# `functools.partial` with a stricter profile) via `redact=`.
Redactor = Callable[[Conversation], tuple[Conversation, dict[str, int]]]

_SCHEMA = """
CREATE TABLE IF NOT EXISTS conversations (
    id           TEXT PRIMARY KEY,
    seq          INTEGER NOT NULL UNIQUE,
    upload_id    TEXT NOT NULL,
    ext_id       TEXT NOT NULL,
    n_turns      INTEGER NOT NULL,
    n_chars      INTEGER NOT NULL,
    content_hash TEXT NOT NULL,
    near_dup_of  TEXT,
    cluster      INTEGER,
    data         TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_conv_upload  ON conversations(upload_id);
CREATE INDEX IF NOT EXISTS ix_conv_cluster ON conversations(cluster);
CREATE INDEX IF NOT EXISTS ix_conv_neardup ON conversations(near_dup_of);
CREATE INDEX IF NOT EXISTS ix_conv_hash    ON conversations(content_hash);
CREATE TABLE IF NOT EXISTS clusters (
    cluster   INTEGER PRIMARY KEY,
    size      INTEGER NOT NULL,
    label     TEXT NOT NULL,
    top_terms TEXT NOT NULL,
    examples  TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""

_PLACEHOLDER = re.compile(r"<[A-Z][A-Z0-9_]*_\d+>")  # <EMAIL_1>, <NAME_2>, ...
_WORD = re.compile(r"\w+")


# ---- small helpers ---------------------------------------------------------------------------


def normalized_text(conv: Conversation) -> str:
    """The text that defines a conversation's identity: roles + lower-cased words.

    Ids, timestamps, speaker labels and whitespace differences are ignored, so the
    same thread exported twice (or by two systems) hashes the same.
    """
    return "\n".join(f"{t.role}: {' '.join(t.text.lower().split())}" for t in conv.turns)


def content_hash(conv: Conversation) -> str:
    """sha256 of `normalized_text` — equal hashes mean exact duplicates."""
    return hashlib.sha256(normalized_text(conv).encode("utf-8")).hexdigest()


def _rank(conv_id: str, seed: int) -> int:
    """A pseudo-random but deterministic sort key for (id, seed).

    Registered as the SQL function `ef_rank`, so `ORDER BY ef_rank(id, 7)` shuffles
    rows the same way every time for seed 7 (SQLite's RANDOM() can't be seeded).
    7 bytes keep the value inside SQLite's signed 64-bit integers.
    """
    digest = hashlib.blake2b(f"{seed}:{conv_id}".encode(), digest_size=7).digest()
    return int.from_bytes(digest, "big")


def _chunks(items: Sequence[Any], size: int = SQL_CHUNK) -> Iterator[Sequence[Any]]:
    for i in range(0, len(items), size):
        yield items[i : i + size]


def _distribution(values: Any) -> dict[str, float]:
    """min / median / p90 / max / mean of a numpy array of ints."""
    import numpy as np

    if len(values) == 0:
        return {"min": 0, "median": 0, "p90": 0, "max": 0, "mean": 0}
    return {
        "min": int(values.min()),
        "median": float(np.median(values)),
        "p90": float(np.percentile(values, 90)),
        "max": int(values.max()),
        "mean": round(float(values.mean()), 1),
    }


# ---- the store ---------------------------------------------------------------------------


class CorpusStore:
    """One project's conversations in one SQLite file.

        with CorpusStore("data/projects/p1/corpus.sqlite") as store:
            ingest_file_to_store("chats.jsonl", store, upload_id="u1")
            store.count()

    Iteration is always in `seq` (insertion) order, so every pass over the data — and
    therefore dedupe, clustering and sampling — is deterministic. Large scans page
    through the table by `seq` ("keyset pagination") instead of holding a cursor open,
    so it is safe to write while scanning.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.path)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=NORMAL")  # safe with WAL; much faster commits
        self.db.execute("PRAGMA cache_size=-65536")  # 64 MB page cache
        self.db.create_function("ef_rank", 2, _rank, deterministic=True)
        self.db.executescript(_SCHEMA)
        if self.get_meta("schema_version") is None:
            self.set_meta("schema_version", SCHEMA_VERSION)
        self.db.commit()

    # -- lifecycle

    def close(self) -> None:
        """Commit and close (the WAL is checkpointed into the main file)."""
        if self.db is not None:
            self.db.commit()
            self.db.close()
            self.db = None  # type: ignore[assignment]

    def __enter__(self) -> CorpusStore:
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        if exc_type is not None and self.db is not None:
            self.db.rollback()
        self.close()

    # -- meta

    def get_meta(self, key: str, default: Any = None) -> Any:
        row = self.db.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        return default if row is None else json.loads(row[0])

    def set_meta(self, key: str, value: Any) -> None:
        self.db.execute(
            "INSERT INTO meta(key, value) VALUES (?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (key, json.dumps(value)),
        )

    def _mark_stale(self, *steps: str) -> None:
        """Record that a derived result ("dedupe", "clusters") no longer matches the data."""
        stale = set(self.get_meta("stale", []))
        stale.update(steps)
        self.set_meta("stale", sorted(stale))

    def _mark_fresh(self, step: str) -> None:
        self.set_meta("stale", sorted(set(self.get_meta("stale", [])) - {step}))

    def stale(self) -> list[str]:
        """Derived results that should be recomputed (e.g. ["clusters", "dedupe"])."""
        return list(self.get_meta("stale", []))

    # -- counts and reads

    def count(self, *, unique_only: bool = False, upload_id: str | None = None) -> int:
        """Number of stored conversations (`unique_only`: skip duplicates)."""
        sql, params = "SELECT COUNT(*) FROM conversations WHERE 1=1", []
        if unique_only:
            sql += " AND near_dup_of IS NULL"
        if upload_id is not None:
            sql += " AND upload_id = ?"
            params.append(upload_id)
        return int(self.db.execute(sql, params).fetchone()[0])

    def get(self, ids: Iterable[str]) -> list[Conversation]:
        """The conversations with these ids, in the order asked (unknown ids skipped).

        Each comes back with `id` = the store id and `meta["upload_id"]`,
        `meta["ext_id"]`, `meta["cluster"]` filled in.
        """
        wanted = list(dict.fromkeys(ids))
        found: dict[str, Conversation] = {}
        for chunk in _chunks(wanted):
            marks = ",".join("?" * len(chunk))
            rows = self.db.execute(
                f"SELECT id, data, cluster FROM conversations WHERE id IN ({marks})", chunk
            )
            for cid, data, cluster in rows:
                conv = Conversation.from_dict(json.loads(data))
                conv.meta["cluster"] = cluster
                found[cid] = conv
        return [found[i] for i in wanted if i in found]

    def resolve_ids(self, refs: Iterable[str]) -> list[str]:
        """Store ids for conversation references, in the order given. A reference is a
        store id ("<upload>:<id>") or the conversation's id in the user's file (`ext_id`,
        which may match one conversation per upload). Unknown references are skipped."""
        wanted = [str(r) for r in dict.fromkeys(refs)]
        by_ref: dict[str, list[str]] = {}
        for chunk in _chunks(wanted):
            marks = ",".join("?" * len(chunk))
            rows = self.db.execute(
                f"SELECT id, ext_id FROM conversations WHERE id IN ({marks}) "
                f"OR ext_id IN ({marks}) ORDER BY seq",
                (*chunk, *chunk),
            )
            for cid, ext in rows:
                for ref in {cid, ext}:
                    by_ref.setdefault(ref, []).append(cid)
        out: list[str] = []
        for ref in wanted:
            out.extend(i for i in by_ref.get(ref, []) if i not in out)
        return out

    def cluster_ids(self, cluster: int) -> list[str]:
        """Every unique conversation id in one topic cluster, in seq order."""
        rows = self.db.execute(
            "SELECT id FROM conversations WHERE cluster = ? AND near_dup_of IS NULL ORDER BY seq",
            (cluster,),
        )
        return [r[0] for r in rows]

    def iter_rows(
        self,
        columns: str,
        *,
        where: str = "1=1",
        params: Sequence[Any] = (),
        batch_size: int = 2000,
    ) -> Iterator[list[tuple[Any, ...]]]:
        """Yield batches of rows (`seq` first, then `columns`) in seq order.

        Keyset pagination: each query asks for "seq > last seen", so no cursor stays
        open between batches and the caller may write to the table in between.
        """
        last = -1
        while True:
            rows = self.db.execute(
                f"SELECT seq, {columns} FROM conversations WHERE seq > ? AND ({where}) "
                f"ORDER BY seq LIMIT ?",
                (last, *params, batch_size),
            ).fetchall()
            if not rows:
                return
            yield rows
            last = rows[-1][0]

    def iter_ids(self, *, unique_only: bool = True) -> Iterator[str]:
        """Every conversation id in seq order."""
        where = "near_dup_of IS NULL" if unique_only else "1=1"
        for rows in self.iter_rows("id", where=where, batch_size=5000):
            for _, cid in rows:
                yield cid

    def examples(self, cluster: int, n: int = 3, *, seed: int = 0) -> list[Conversation]:
        """Up to n conversations from one cluster: the most central ones first, then a
        deterministic shuffle of the rest (duplicates excluded)."""
        row = self.db.execute(
            "SELECT examples FROM clusters WHERE cluster = ?", (cluster,)
        ).fetchone()
        ids: list[str] = json.loads(row[0])[:n] if row else []
        if len(ids) < n:
            more = self.db.execute(
                "SELECT id FROM conversations WHERE cluster = ? AND near_dup_of IS NULL "
                "ORDER BY ef_rank(id, ?) LIMIT ?",
                (cluster, seed, n + len(ids)),
            )
            ids += [r[0] for r in more if r[0] not in ids]
        return self.get(ids[:n])

    def clusters(self) -> list[ClusterInfo]:
        """The stored cluster table, largest first."""
        rows = self.db.execute(
            "SELECT cluster, size, label, top_terms, examples FROM clusters ORDER BY cluster"
        )
        return [
            ClusterInfo(c, size, label, json.loads(terms), json.loads(ex))
            for c, size, label, terms, ex in rows
        ]

    def uploads(self) -> list[dict[str, Any]]:
        """The ingest report of every upload in the store, in ingest order."""
        rows = self.db.execute("SELECT value FROM meta WHERE key LIKE 'upload/%' ORDER BY rowid")
        return [json.loads(r[0]) for r in rows]

    # -- writes

    def add_conversations(
        self, upload_id: str, conversations: Sequence[Conversation]
    ) -> tuple[int, int, int]:
        """Insert redacted conversations of one upload. Returns (inserted, exact
        duplicates, renamed ids). Does not commit (callers batch commits).

        - refuses anything not marked `meta["redacted"] = True` (nothing unredacted is
          ever written to disk);
        - the store id is "<upload_id>:<conversation id>"; if that is taken (the same
          id twice in one file), the later one becomes "<...>#2", "#3", ...;
        - a conversation whose content hash is already in the store (any upload) is an
          exact duplicate: `near_dup_of` points at the earliest copy.
        """
        if not conversations:
            return 0, 0, 0
        for conv in conversations:
            if not conv.meta.get("redacted"):
                raise ValueError(f"conversation {conv.id!r} is not redacted; redact before storing")
        seq = int(self.db.execute("SELECT COALESCE(MAX(seq), 0) FROM conversations").fetchone()[0])
        prepared = [(conv, f"{upload_id}:{conv.id}", content_hash(conv)) for conv in conversations]

        # which ids / hashes already exist? (one query per chunk, not per row)
        taken: set[str] = set()
        first_copy: dict[str, str] = {}
        ids = [p[1] for p in prepared]
        hashes = list({p[2] for p in prepared})
        for chunk in _chunks(ids):
            marks = ",".join("?" * len(chunk))
            taken.update(
                r[0]
                for r in self.db.execute(
                    f"SELECT id FROM conversations WHERE id IN ({marks})", chunk
                )
            )
        for chunk in _chunks(hashes):
            marks = ",".join("?" * len(chunk))
            # MIN(seq) makes SQLite return the other columns from the earliest row
            for h, cid, dup_of, _ in self.db.execute(
                f"SELECT content_hash, id, near_dup_of, MIN(seq) FROM conversations "
                f"WHERE content_hash IN ({marks}) GROUP BY content_hash",
                chunk,
            ):
                first_copy[h] = dup_of or cid

        rows = []
        dups = renamed = 0
        for conv, cid, h in prepared:
            if cid in taken:
                n = 2
                while f"{cid}#{n}" in taken or self._exists(f"{cid}#{n}"):
                    n += 1
                cid = f"{cid}#{n}"
                renamed += 1
            taken.add(cid)
            dup_of = first_copy.get(h)
            if dup_of is None:
                first_copy[h] = cid
            else:
                dups += 1
            seq += 1
            data = conv.to_dict()
            data["id"] = cid
            data["meta"] = {**data.get("meta", {}), "upload_id": upload_id, "ext_id": conv.id}
            n_chars = sum(len(t.text) for t in conv.turns)
            rows.append(
                (
                    cid,
                    seq,
                    upload_id,
                    conv.id,
                    len(conv.turns),
                    n_chars,
                    h,
                    dup_of,
                    json.dumps(data),
                )
            )
        self.db.executemany(
            "INSERT INTO conversations(id, seq, upload_id, ext_id, n_turns, n_chars, "
            "content_hash, near_dup_of, data) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            rows,
        )
        self._mark_stale("dedupe", "clusters")
        return len(rows), dups, renamed

    def _exists(self, cid: str) -> bool:
        return (
            self.db.execute("SELECT 1 FROM conversations WHERE id = ?", (cid,)).fetchone()
            is not None
        )

    def delete_upload(self, upload_id: str) -> int:
        """Remove one upload's conversations (and its report). Returns how many.

        Conversations in other uploads that pointed at a deleted one as their original
        lose that mark (re-run `dedupe_near` and `cluster` afterwards — `stale()` says so).
        """
        n = self.db.execute("DELETE FROM conversations WHERE upload_id = ?", (upload_id,)).rowcount
        # ids of this upload are "<upload_id>:..." — a range scan on the index:
        # ';' is the character right after ':'
        self.db.execute(
            "UPDATE conversations SET near_dup_of = NULL WHERE near_dup_of >= ? AND near_dup_of < ?",
            (f"{upload_id}:", f"{upload_id};"),
        )
        self.db.execute("DELETE FROM meta WHERE key = ?", (f"upload/{upload_id}",))
        if n:
            self.db.execute(
                "UPDATE clusters SET size = (SELECT COUNT(*) FROM conversations c "
                "WHERE c.cluster = clusters.cluster AND c.near_dup_of IS NULL)"
            )
            for info in self.clusters():
                kept = [e for e in info.examples if self._exists(e)]
                self.db.execute(
                    "UPDATE clusters SET examples = ? WHERE cluster = ?",
                    (json.dumps(kept), info.cluster),
                )
            self._mark_stale("dedupe", "clusters")
        self.db.commit()
        return n


def get(store: CorpusStore, ids: Iterable[str]) -> list[Conversation]:
    """Shorthand for `store.get(ids)`."""
    return store.get(ids)


def examples(store: CorpusStore, cluster: int, n: int = 3) -> list[Conversation]:
    """Shorthand for `store.examples(cluster, n)`."""
    return store.examples(cluster, n)


# ---- streaming ingest ------------------------------------------------------------------------


@dataclass
class IngestProgress:
    """Passed to `on_progress` during `ingest_file_to_store`."""

    bytes_read: int
    bytes_total: int
    conversations: int  # stored so far
    exact_duplicates: int
    elapsed_s: float

    @property
    def fraction(self) -> float:
        """0..1, by bytes read (the only measure known before the end)."""
        return min(1.0, self.bytes_read / self.bytes_total) if self.bytes_total else 1.0


@dataclass
class IngestReport:
    """What one `ingest_file_to_store` call did. Also saved in the store's meta."""

    upload_id: str
    source_file: str
    format: str
    mapping: dict[str, Any]
    confidence: float
    reasons: list[str]
    conversations: int = 0  # stored (exact duplicates included)
    exact_duplicates: int = 0  # of conversations already in the store (any upload)
    renamed: int = 0  # repeated or split ids stored as "<id>#2", ...
    skipped_rows: int = 0
    truncated: int = 0
    pii_counts: dict[str, int] = field(default_factory=dict)
    bytes_total: int = 0
    seconds: float = 0.0
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _redact_many(redact: Redactor, convs: list[Conversation]) -> list[tuple[Conversation, dict]]:
    """Worker-process entry point: redact a chunk of conversations."""
    return [redact(c) for c in convs]


def _redacted(
    convs: Iterable[Conversation], redact: Redactor, workers: int, chunk: int = 200
) -> Iterator[tuple[Conversation, dict[str, int]]]:
    """Redact a stream of conversations, in order, optionally in worker processes.

    Regex redaction costs ~1 ms per conversation, the slowest step of ingest. With
    `workers > 1`, chunks go to a process pool; at most `2 * workers` chunks are in
    flight, so a fast parser cannot pile up the whole file in memory, and results are
    taken in submission order, so the output order (and the store) is deterministic.
    """
    if workers <= 1:
        for conv in convs:
            yield redact(conv)
        return
    with ProcessPoolExecutor(max_workers=workers) as pool:
        inflight: deque[Future[list[tuple[Conversation, dict]]]] = deque()
        buf: list[Conversation] = []
        for conv in convs:
            buf.append(conv)
            if len(buf) >= chunk:
                inflight.append(pool.submit(_redact_many, redact, buf))
                buf = []
                while len(inflight) >= 2 * workers:
                    yield from inflight.popleft().result()
        if buf:
            inflight.append(pool.submit(_redact_many, redact, buf))
        while inflight:
            yield from inflight.popleft().result()


def ingest_file_to_store(
    path: str | Path,
    store: CorpusStore,
    upload_id: str,
    *,
    mapping: Mapping | None = None,
    redact: Redactor = redact_conversation,
    on_progress: Callable[[IngestProgress], None] | None = None,
    batch_size: int = 1000,
    workers: int = 1,
    progress_every_s: float = 2.0,
    max_turns: int = DEFAULT_MAX_TURNS,
    max_turn_chars: int = DEFAULT_MAX_TURN_CHARS,
    stream_window: int = DEFAULT_STREAM_WINDOW,
) -> IngestReport:
    """Stream one uploaded file into the store: parse -> redact -> insert, in batches.

    Memory stays bounded whatever the file size: the parser yields conversations one at
    a time (`ingest.iter_conversations`), each is redacted *before* it is stored, and
    rows are written `batch_size` at a time, one commit per batch. If anything fails,
    the upload's partial rows are removed again, and re-ingesting an `upload_id` that
    is already in the store replaces it — so a retry is always safe.

    `redact` must return (redacted conversation, counts) and mark the conversation
    `meta["redacted"] = True`. With `workers > 1` it must be picklable (a module-level
    function or a `functools.partial` of one). `on_progress` is called after every
    batch and at least every `progress_every_s` seconds.
    """
    started = time.monotonic()
    if store.count(upload_id=upload_id):
        store.delete_upload(upload_id)
    stream = iter_conversations(
        path, mapping, max_turns=max_turns, max_turn_chars=max_turn_chars, window=stream_window
    )
    guess = stream.format
    report = IngestReport(
        upload_id=upload_id,
        source_file=Path(path).name,
        format=stream.mapping.format or guess.format,
        mapping=asdict(stream.mapping),
        confidence=guess.confidence,
        reasons=list(guess.reasons),
        bytes_total=stream.bytes_total,
    )
    pii: Counter[str] = Counter()
    batch: list[Conversation] = []
    last_progress = started

    def progress() -> None:
        nonlocal last_progress
        last_progress = time.monotonic()
        if on_progress is not None:
            on_progress(
                IngestProgress(
                    bytes_read=stream.bytes_read,
                    bytes_total=stream.bytes_total,
                    conversations=report.conversations,
                    exact_duplicates=report.exact_duplicates,
                    elapsed_s=round(last_progress - started, 2),
                )
            )

    def flush() -> None:
        inserted, dups, renamed = store.add_conversations(upload_id, batch)
        report.conversations += inserted
        report.exact_duplicates += dups
        report.renamed += renamed
        store.db.commit()
        batch.clear()
        progress()

    try:
        for clean, counts in _redacted(stream, redact, workers):
            pii.update(counts)
            batch.append(clean)
            if len(batch) >= batch_size:
                flush()
            elif time.monotonic() - last_progress >= progress_every_s:
                progress()
        if batch:
            flush()
    except BaseException:
        store.db.rollback()
        store.delete_upload(upload_id)  # all or nothing: no half-ingested uploads
        raise

    report.skipped_rows = stream.skipped_rows
    report.truncated = stream.truncated
    report.warnings = stream.warnings[:30]
    if report.renamed:
        report.warnings.append(
            f"{report.renamed} conversation id(s) appeared more than once (or their messages "
            "were far apart in the file) and were stored as '<id>#2', ..."
        )
    report.pii_counts = dict(sorted(pii.items()))
    report.seconds = round(time.monotonic() - started, 2)
    store.set_meta(f"upload/{upload_id}", report.to_dict())
    store.db.commit()
    progress()
    return report


# ---- near-duplicates: MinHash + LSH ------------------------------------------------------------

DEDUPE_MAX_CHARS = 20_000  # of normalised text per conversation used for the signature


@dataclass
class DedupeReport:
    """What `dedupe_near` found."""

    conversations: int
    duplicates: int  # rows now marked near_dup_of (exact duplicates included)
    exact_duplicates: int
    groups: int  # sets of 2+ conversations that are copies of each other
    bands: int
    rows_per_band: int
    candidate_checks: int
    seconds: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def lsh_params(threshold: float, num_perm: int, *, recall: float = 0.99) -> tuple[int, int]:
    """Choose (bands, rows per band) for LSH over `num_perm` MinHash values.

    Two conversations with Jaccard similarity s agree on one MinHash value with
    probability s, on a whole band of r values with probability s**r, and become a
    candidate pair if any of b bands agree: P = 1 - (1 - s**r)**b. We take the
    largest r (fewest false candidates) that still catches pairs at the threshold with
    probability >= `recall`. Candidates are then verified, so false positives only
    cost time.
    """
    best = (num_perm, 1)
    for r in range(1, num_perm + 1):
        b = num_perm // r
        if b < 1:
            break
        if 1 - (1 - threshold**r) ** b >= recall:
            best = (b, r)
    return best


class _HashCache(dict):
    """word -> crc32, computed on first use (`map(cache.__getitem__, words)` is fast)."""

    def __missing__(self, word: str) -> int:
        value = self[word] = zlib.crc32(word.encode("utf-8"))
        return value


def _signatures(docs: list[str], a: Any, b: Any, shingle: int, cache: _HashCache) -> Any:
    """MinHash signatures (len(docs) x num_perm, uint32) of word-shingle sets.

    1. each word -> a 32-bit crc; each run of `shingle` consecutive words -> one 64-bit
       shingle hash (combined with numpy for the whole batch at once);
    2. num_perm hash functions h_i(x) = (a_i * x + b_i) mod 2**64 >> 32 ("multiply-
       shift": cheap, and good enough for MinHash);
    3. the signature is, per function, the minimum over the document's shingles —
       two documents agree on a value with probability = their Jaccard similarity.
    Duplicate shingles in a document don't change a minimum, so no `set()` is needed.
    """
    import numpy as np

    words_h: list[int] = []
    lengths: list[int] = []
    for doc in docs:
        words = _WORD.findall(doc[:DEDUPE_MAX_CHARS]) or [""]
        if len(words) < shingle:  # very short text: one shingle of whatever is there
            words += [""] * (shingle - len(words))
        words_h.extend(map(cache.__getitem__, words))
        lengths.append(len(words))
    h = np.asarray(words_h, dtype=np.uint64)
    lens = np.asarray(lengths, dtype=np.int64)
    starts = np.concatenate(([0], np.cumsum(lens)[:-1]))
    counts = lens - shingle + 1  # shingles per document (>= 1)
    offsets = np.concatenate(([0], np.cumsum(counts)[:-1]))
    # position of every shingle's first word in `h`, skipping runs that cross documents
    idx = np.arange(int(counts.sum()), dtype=np.int64) + np.repeat(starts - offsets, counts)
    mults = (0x9E3779B97F4A7C15, 0xC2B2AE3D27D4EB4F, 0x165667B19E3779F9)
    sh = np.zeros(len(idx), dtype=np.uint64)
    for j in range(shingle):
        sh ^= h[idx + j] * np.uint64(mults[j % 3] + 2 * j)
    sh ^= sh >> np.uint64(29)

    num_perm = len(a)
    out = np.empty((len(docs), num_perm), dtype=np.uint32)
    # bound the (shingles x num_perm) temporary to ~16 MB by processing docs in groups
    max_rows = max(1, (1 << 21) // num_perm)
    d = 0
    while d < len(docs):
        e, total = d, 0
        while e < len(docs) and (e == d or total + counts[e] <= max_rows):
            total += int(counts[e])
            e += 1
        lo, hi = int(offsets[d]), int(offsets[d]) + total
        # (num_perm x shingles), so each document's shingles are contiguous in memory for
        # the per-document minimum (`reduceat`); in-place ops avoid extra temporaries
        block = a[:, None] * sh[None, lo:hi]
        block += b[:, None]
        block >>= np.uint64(32)
        out[d:e] = np.minimum.reduceat(block, offsets[d:e] - lo, axis=1).T
        d = e
    return out


def dedupe_near(
    store: CorpusStore,
    *,
    threshold: float = 0.9,
    num_perm: int = 128,
    shingle: int = 3,
    seed: int = 0,
    batch_size: int = 2000,
    on_progress: Callable[[int, int], None] | None = None,
) -> DedupeReport:
    """Mark near-duplicate conversations (`near_dup_of` = id of the earliest copy).

    Method: MinHash signatures over word 3-gram shingles of the normalised text, then
    locality-sensitive hashing — split each signature into b bands of r values and
    bucket conversations by each band; only conversations sharing a bucket are
    compared (estimated Jaccard >= `threshold`), so the cost grows ~linearly with the
    corpus instead of with its square. Matches are merged with union-find; each group
    keeps its earliest conversation (lowest seq) as the original.

    Memory: signatures go to a temporary memory-mapped file next to the store
    (n x num_perm x 4 bytes: ~0.5 GB on disk for 1M conversations); RAM holds a few
    numbers per conversation (seq, union-find parent, one band's keys and their sort
    order: ~40 MB for 1M). Recomputes all marks, including
    exact duplicates, so it is safe to re-run after every upload.
    `on_progress(done, total)` is called after each batch of signatures.
    """
    import numpy as np

    started = time.monotonic()
    n = store.count()
    bands, rows = lsh_params(threshold, num_perm)
    report = DedupeReport(n, 0, 0, 0, bands, rows, 0, 0.0)
    if n == 0:
        store._mark_fresh("dedupe")
        store.db.commit()
        return report
    rng = np.random.default_rng(seed)
    a = rng.integers(1, 2**63, size=num_perm, dtype=np.uint64) * np.uint64(2) + np.uint64(1)
    b = rng.integers(0, 2**63, size=num_perm, dtype=np.uint64)
    cache = _HashCache()

    with tempfile.TemporaryDirectory(dir=store.path.parent, prefix=".dedupe-") as tmp:
        sig = np.lib.format.open_memmap(
            Path(tmp) / "sig.npy", mode="w+", dtype=np.uint32, shape=(n, num_perm)
        )
        seqs = np.empty(n, dtype=np.int64)
        i = 0
        for batch in store.iter_rows("data", batch_size=batch_size):
            texts = [normalized_text(Conversation.from_dict(json.loads(d))) for _, d in batch]
            sig[i : i + len(batch)] = _signatures(texts, a, b, shingle, cache)
            seqs[i : i + len(batch)] = [s for s, _ in batch]
            i += len(batch)
            if len(cache) > 2_000_000:
                cache.clear()
            if on_progress:
                on_progress(i, n)
        n = i  # (in case rows changed under us)

        parent = np.arange(n, dtype=np.int64)

        def find(x: int) -> int:
            root = x
            while parent[root] != root:
                root = int(parent[root])
            while parent[x] != root:  # path compression
                parent[x], x = root, int(parent[x])
            return root

        def similar(x: int, y: int) -> bool:
            report.candidate_checks += 1
            return float(np.mean(sig[x] == sig[y])) >= threshold

        def union(x: int, y: int) -> None:
            rx, ry = find(x), find(y)
            if rx != ry:  # the smaller index (earlier conversation) stays the root
                parent[max(rx, ry)] = min(rx, ry)

        mix = np.uint64(0x100000001B3)
        for band in range(bands):
            keys = np.full(n, 0xCBF29CE484222325, dtype=np.uint64)
            for j in range(band * rows, (band + 1) * rows):  # FNV-style mix into one key
                keys = (keys ^ sig[:n, j].astype(np.uint64)) * mix
            order = np.argsort(keys, kind="stable")  # stable: members stay in seq order
            sorted_keys = keys[order]
            bounds = np.flatnonzero(sorted_keys[1:] != sorted_keys[:-1]) + 1
            starts = np.concatenate(([0], bounds))
            ends = np.concatenate((bounds, [n]))
            for s, e in zip(starts[ends - starts > 1], ends[ends - starts > 1], strict=True):
                members = order[s:e]
                first = int(members[0])
                prev = first
                for m in members[1:]:
                    m = int(m)
                    if find(m) == find(first):
                        prev = m
                        continue
                    # compare with the bucket's first member, else with the previous one:
                    # 2 checks per member instead of all pairs in a big bucket
                    if similar(m, first):
                        union(m, first)
                    elif find(m) != find(prev) and similar(m, prev):
                        union(m, prev)
                    prev = m

    # every parent pointer points to a smaller index, so "pointer jumping" (replace each
    # pointer by its parent's pointer until nothing changes) finds all roots at once
    roots = parent.copy()
    while True:
        jumped = roots[roots]
        if np.array_equal(jumped, roots):
            break
        roots = jumped
    dup_idx = np.flatnonzero(roots != np.arange(n))
    store.db.execute("UPDATE conversations SET near_dup_of = NULL WHERE near_dup_of IS NOT NULL")
    store.db.executemany(
        "UPDATE conversations SET near_dup_of = (SELECT id FROM conversations WHERE seq = ?) "
        "WHERE seq = ?",
        ((int(seqs[roots[x]]), int(seqs[x])) for x in dup_idx),
    )
    store._mark_fresh("dedupe")
    store._mark_stale("clusters")
    store.db.commit()
    report.duplicates = len(dup_idx)
    report.groups = len(np.unique(roots[dup_idx]))
    report.exact_duplicates = n - int(
        store.db.execute("SELECT COUNT(DISTINCT content_hash) FROM conversations").fetchone()[0]
    )
    report.seconds = round(time.monotonic() - started, 2)
    store.set_meta("dedupe", report.to_dict())
    store.db.commit()
    return report


# ---- topic clustering ------------------------------------------------------------------------


@dataclass
class ClusterInfo:
    """One topic cluster: id (0 = largest), size (unique conversations), label (from
    the top terms), top terms, and example ids closest to the centroid."""

    cluster: int
    size: int
    label: str
    top_terms: list[str]
    examples: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


FULL_KMEANS_MAX = 50_000  # samples up to this size are fitted with exact k-means


def auto_k(n: int) -> int:
    """How many clusters for n conversations: ~sqrt(n/2), kept within 8..60, but never
    more than one cluster per 5 conversations (tiny corpora get few clusters)."""
    k = min(60, max(8, round(math.sqrt(n / 2))))
    return max(1, min(k, n // 5))


def _sides(conv: Conversation) -> tuple[str, str]:
    """(customer text, everyone else's text), with redaction placeholders removed."""
    cust = " ".join(t.text for t in conv.turns if t.role == "customer")
    other = " ".join(t.text for t in conv.turns if t.role != "customer")
    return _PLACEHOLDER.sub(" ", cust), _PLACEHOLDER.sub(" ", other)


def _load_sides(store: CorpusStore, seqs: Sequence[int]) -> tuple[list[str], list[str]]:
    """(customer texts, other texts) for these seq numbers, in seq order."""
    cust: list[str] = []
    other: list[str] = []
    # plain ints: sqlite3 silently binds a numpy int64 as a blob, matching nothing
    for chunk in _chunks([int(s) for s in seqs]):
        marks = ",".join("?" * len(chunk))
        rows = store.db.execute(
            f"SELECT data FROM conversations WHERE seq IN ({marks}) ORDER BY seq", chunk
        )
        for (data,) in rows:
            c, o = _sides(Conversation.from_dict(json.loads(data)))
            cust.append(c)
            other.append(o)
    return cust, other


def cluster(
    store: CorpusStore,
    *,
    k: int | None = None,
    sample_size: int = 20000,
    seed: int = 0,
    customer_weight: float = 3.0,
    n_examples: int = 5,
    batch_size: int = 5000,
    on_progress: Callable[[int, int], None] | None = None,
) -> list[ClusterInfo]:
    """Group the unique conversations by topic; store and return the clusters.

    1. Features: TF-IDF over words (stopwords, digits and redaction placeholders
       dropped). The customer's turns and the agent's turns are vectorised separately
       and added with the customer side weighted `customer_weight` times more — the
       customer's problem is what defines a task; agent boilerplate is noise.
    2. Fit: k-means with k clusters (`auto_k` if not given) on a deterministic random
       sample of `sample_size` conversations — fitting on 20k is as good as on 1M for
       finding topics, and much cheaper. Because the sample is bounded, exact k-means
       (several k-means++ restarts) is affordable and finds cleaner topics than
       MiniBatchKMeans, which is used only for samples above FULL_KMEANS_MAX.
    3. Assign: every unique conversation, in batches; duplicates inherit the cluster
       of their original.
    4. Describe: clusters are renumbered by size (0 = largest); each gets its centroid's
       top terms, a label (the first three), and the `n_examples` conversations
       closest to the centroid.

    Deterministic for a given seed and data. `on_progress(done, total)` reports the
    assignment pass.
    """
    import numpy as np
    from sklearn.cluster import KMeans, MiniBatchKMeans
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
    from sklearn.preprocessing import normalize

    seqs = np.fromiter(
        (s for rows in store.iter_rows("1", where="near_dup_of IS NULL", batch_size=20000)
         for s, _ in rows),
        dtype=np.int64,
    )  # fmt: skip
    n = len(seqs)
    store.db.execute("UPDATE conversations SET cluster = NULL WHERE cluster IS NOT NULL")
    store.db.execute("DELETE FROM clusters")
    if n == 0:
        store._mark_fresh("clusters")
        store.db.commit()
        return []
    k = auto_k(n) if k is None else max(1, min(int(k), n))
    rng = np.random.default_rng(seed)
    sample = np.sort(rng.choice(seqs, size=min(sample_size, n), replace=False))
    cust, other = _load_sides(store, sample)

    token_pattern = r"(?u)\b[^\W\d_]{2,}\b"  # words of 2+ letters, no digits
    # stopwords tokenised the same way as the text ("don't" -> "don")
    stop = sorted(
        {
            tok
            for w in set(ENGLISH_STOP_WORDS) | _ALL_STOPWORDS
            for tok in re.findall(token_pattern, w)
        }
    )
    vectorizer = TfidfVectorizer(
        stop_words=stop,
        token_pattern=token_pattern,
        lowercase=True,
        min_df=2 if len(sample) >= 100 else 1,
        max_df=0.7 if len(sample) >= 100 else 1.0,
        sublinear_tf=True,
        max_features=30000,
        dtype=np.float32,
    )

    def features(c: list[str], o: list[str]) -> Any:
        x = vectorizer.transform(c) * customer_weight + vectorizer.transform(o)
        return normalize(x)

    try:
        vectorizer.fit(cust + other)
        x_sample = features(cust, other)
        terms = vectorizer.get_feature_names_out()
    except ValueError:  # empty vocabulary: no usable words at all
        terms = None
    model = None
    if terms is not None and k > 1:
        if len(sample) <= FULL_KMEANS_MAX:  # exact k-means: more stable, affordable here
            model = KMeans(n_clusters=k, random_state=seed, n_init=4, max_iter=100)
        else:
            model = MiniBatchKMeans(
                n_clusters=k, random_state=seed, n_init=4, batch_size=4096, max_iter=100
            )
        model.fit(x_sample)
    del cust, other

    # assign everything, tracking per cluster the conversations closest to the centroid
    raw_k = k if model is not None else 1
    sizes = np.zeros(raw_k, dtype=np.int64)
    closest: list[list[tuple[float, int]]] = [[] for _ in range(raw_k)]
    done = 0
    for start in range(0, n, batch_size):
        part = seqs[start : start + batch_size]
        if model is None:
            labels = np.zeros(len(part), dtype=np.int64)
            dist = np.zeros(len(part))
        else:
            x = features(*_load_sides(store, part))
            labels = model.predict(x)
            dist = model.transform(x)[np.arange(len(part)), labels]
        sizes += np.bincount(labels, minlength=raw_k)
        for c in np.unique(labels):
            mine = np.flatnonzero(labels == c)
            best = mine[np.lexsort((part[mine], dist[mine]))][:n_examples]
            merged = closest[c] + [(float(dist[j]), int(part[j])) for j in best]
            closest[c] = heapq.nsmallest(n_examples, merged)
        store.db.executemany(
            "UPDATE conversations SET cluster = ? WHERE seq = ?",
            zip(labels.tolist(), part.tolist(), strict=True),
        )
        done += len(part)
        if on_progress:
            on_progress(done, n)

    # renumber: 0 = largest (ties by old id); empty clusters disappear
    order = [c for c in sorted(range(raw_k), key=lambda c: (-sizes[c], c)) if sizes[c] > 0]
    remap = {old: new for new, old in enumerate(order)}
    store.db.execute("CREATE TEMP TABLE IF NOT EXISTS _remap(old INTEGER PRIMARY KEY, new INTEGER)")
    store.db.execute("DELETE FROM _remap")
    store.db.executemany("INSERT INTO _remap VALUES (?, ?)", remap.items())
    store.db.execute(
        "UPDATE conversations SET cluster = (SELECT new FROM _remap WHERE old = conversations.cluster) "
        "WHERE cluster IS NOT NULL"
    )
    # duplicates join their original's cluster
    store.db.execute(
        "UPDATE conversations SET cluster = (SELECT o.cluster FROM conversations o "
        "WHERE o.id = conversations.near_dup_of) WHERE near_dup_of IS NOT NULL"
    )

    infos: list[ClusterInfo] = []
    for old in order:
        if model is not None and terms is not None:
            top = [str(terms[j]) for j in np.argsort(-model.cluster_centers_[old])[:8]]
        elif terms is not None:
            top = [str(t) for t in terms[np.argsort(-np.asarray(x_sample.sum(axis=0)).ravel())[:8]]]
        else:
            top = []
        ex_seqs = [s for _, s in closest[old]]
        ex_ids = _ids_for_seqs(store, ex_seqs)
        infos.append(
            ClusterInfo(
                cluster=remap[old],
                size=int(sizes[old]),
                label=" / ".join(top[:3]) or "(no text)",
                top_terms=top,
                examples=ex_ids,
            )
        )
    store.db.executemany(
        "INSERT INTO clusters(cluster, size, label, top_terms, examples) VALUES (?, ?, ?, ?, ?)",
        [
            (i.cluster, i.size, i.label, json.dumps(i.top_terms), json.dumps(i.examples))
            for i in infos
        ],
    )
    store.set_meta(
        "clustering",
        {"k": k, "clusters": len(infos), "n": n, "sample_size": len(sample), "seed": seed,
         "customer_weight": customer_weight},
    )  # fmt: skip
    store._mark_fresh("clusters")
    store.db.commit()
    return infos


def _ids_for_seqs(store: CorpusStore, seqs: Sequence[int]) -> list[str]:
    """Store ids for seq numbers, in the given order."""
    if not seqs:
        return []
    marks = ",".join("?" * len(seqs))
    found = dict(
        store.db.execute(f"SELECT seq, id FROM conversations WHERE seq IN ({marks})", list(seqs))
    )
    return [found[s] for s in seqs if s in found]


# ---- sampling ------------------------------------------------------------------------------


def allocate(n: int, sizes: dict[int, int], cap: int | None = None) -> dict[int, int]:
    """Split a sample of n across strata (clusters) of the given sizes.

    - at least 1 per stratum when n allows (otherwise 1 each for the n largest);
    - the rest proportional to size, rounded by largest remainder;
    - no stratum above `cap` (or its size); what a capped stratum can't take is
      re-spread over the others in proportion.
    Deterministic: ties go to the larger stratum, then the lower id.
    """
    order = sorted((c for c in sizes if sizes[c] > 0), key=lambda c: (-sizes[c], c))
    caps = {c: sizes[c] if cap is None else min(sizes[c], cap) for c in order}
    n = min(n, sum(caps.values()))
    quotas = dict.fromkeys(order, 0)
    if n <= 0:
        return quotas
    if n < len(order):
        for c in order[:n]:
            quotas[c] = 1
        return quotas
    for c in order:
        quotas[c] = 1
    remaining = n - len(order)
    rank = {c: i for i, c in enumerate(order)}
    while remaining > 0:
        open_ = [c for c in order if quotas[c] < caps[c]]
        if not open_:
            break
        weight = sum(sizes[c] for c in open_)
        exact = {c: remaining * sizes[c] / weight for c in open_}
        given = 0
        for c in open_:
            add = min(int(exact[c]), caps[c] - quotas[c])
            quotas[c] += add
            given += add
        left = remaining - given
        for c in sorted(open_, key=lambda c: (-(exact[c] - int(exact[c])), rank[c])):
            if left == 0:
                break
            if quotas[c] < caps[c]:
                quotas[c] += 1
                left -= 1
        remaining = left
    return quotas


def stratified_sample(
    store: CorpusStore,
    n: int,
    *,
    per_cluster_cap: int | None = None,
    exclude_ids: Iterable[str] = (),
    seed: int = 0,
) -> list[str]:
    """n conversation ids that represent the corpus: proportional to cluster sizes.

    Near-duplicates and `exclude_ids` (e.g. conversations already extracted into
    tasks) are skipped. Every cluster gets at least one when n allows;
    `per_cluster_cap` limits any one cluster (use it when showing a few examples to a
    model: diversity matters more than proportion there). Within a cluster the choice
    is a seeded shuffle, so the same call returns the same ids. The result is
    interleaved round-robin across clusters (largest first), so any prefix of it is
    diverse too. Before `cluster()` has run, the whole corpus is one stratum.
    """
    db = store.db
    db.execute("CREATE TEMP TABLE IF NOT EXISTS _exclude(id TEXT PRIMARY KEY)")
    db.execute("DELETE FROM _exclude")
    db.executemany("INSERT OR IGNORE INTO _exclude VALUES (?)", ((i,) for i in exclude_ids))
    base = "near_dup_of IS NULL AND id NOT IN (SELECT id FROM _exclude)"
    sizes = {
        (-1 if c is None else int(c)): int(cnt)
        for c, cnt in db.execute(
            f"SELECT cluster, COUNT(*) FROM conversations WHERE {base} GROUP BY cluster"
        )
    }
    quotas = allocate(n, sizes, per_cluster_cap)
    picked: dict[int, list[str]] = {}
    for c, q in quotas.items():
        if q <= 0:
            continue
        where = "cluster IS NULL" if c == -1 else "cluster = ?"
        params: tuple[Any, ...] = (seed, q) if c == -1 else (c, seed, q)
        picked[c] = [
            r[0]
            for r in db.execute(
                f"SELECT id FROM conversations WHERE {where} AND {base} "
                "ORDER BY ef_rank(id, ?), id LIMIT ?",
                params,
            )
        ]
    db.execute("DELETE FROM _exclude")
    out: list[str] = []
    lists = [picked[c] for c in sorted(picked, key=lambda c: (-sizes[c], c))]
    for i in range(max((len(x) for x in lists), default=0)):
        out.extend(x[i] for x in lists if i < len(x))
    return out


# ---- overview --------------------------------------------------------------------------------


def overview(store: CorpusStore, *, max_clusters: int = 60, n_terms: int = 2) -> dict[str, Any]:
    """Corpus statistics + the cluster table, compact enough for an LLM prompt.

    Kept under ~3k tokens even with 60 clusters: per cluster only its id, label (its
    top three terms), size, share of the unique conversations, the next `n_terms` top
    terms and one example id (the most central; fetch more with `store.examples`).
    """
    import numpy as np

    db = store.db
    n = store.count()
    unique = store.count(unique_only=True)
    distinct = int(
        db.execute("SELECT COUNT(DISTINCT content_hash) FROM conversations").fetchone()[0]
    )
    exact = n - distinct
    turns = np.fromiter((r[0] for r in db.execute("SELECT n_turns FROM conversations")), np.int64)
    chars = np.fromiter((r[0] for r in db.execute("SELECT n_chars FROM conversations")), np.int64)
    pii: Counter[str] = Counter()
    uploads = []
    for rep in store.uploads():
        pii.update(rep.get("pii_counts", {}))
        uploads.append(
            {
                "id": rep["upload_id"],
                "file": rep.get("source_file"),
                "format": rep.get("format"),
                "conversations": rep.get("conversations"),
            }
        )
    clusters = []
    for info in store.clusters()[:max_clusters]:
        clusters.append(
            {
                "id": info.cluster,
                "label": info.label,
                "size": info.size,
                "share": round(info.size / unique, 3) if unique else 0,
                "terms": info.top_terms[3 : 3 + n_terms],  # beyond the label's three
                "example": info.examples[0] if info.examples else None,
            }
        )
    return {
        "conversations": n,
        "unique": unique,
        "exact_duplicates": exact,
        "near_duplicates": max(0, n - unique - exact),
        "duplicate_rate": round((n - unique) / n, 3) if n else 0,
        "turns_per_conversation": _distribution(turns),
        "chars_per_conversation": _distribution(chars),
        "uploads": uploads[:20],
        "pii_redactions": dict(pii.most_common()),
        "stale": store.stale(),
        "clusters": clusters,
    }

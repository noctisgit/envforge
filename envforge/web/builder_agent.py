"""The environment builder agent: an LLM that interviews the user and edits a project.

It is a plain tool-use loop (the same `ChatModel` interface the runner uses, so any
provider works), executed by the builder worker because that's where provider keys
are decrypted. One *turn* = everything the model does in response to one user
message: it may call tools several times, then answers in prose (often a question).

Tools operate on the project's head revision; every change creates a new revision,
so the user can review, diff and roll back anything the builder did. Two rules are
enforced in code, not left to the prompt: extraction of more than a handful of
conversations needs the user's explicit OK (it costs money), and the model only ever
sees redacted conversations.
"""

from __future__ import annotations

import json
import traceback
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sqlalchemy.orm import Session

from envforge.core.env import ToolCall, ToolSpec
from envforge.core.trajectory import Usage
from envforge.models import load_model
from envforge.models.base import Message
from envforge.models.catalog import usage_cost
from envforge.web import builder as B
from envforge.web import services as svc
from envforge.web.models import EnvProject, EnvRevision, Job, User, Workspace

MAX_MODEL_CALLS = 16  # per turn
FREE_EXTRACTION = 5  # conversations per turn the builder may extract without the user's approval
MAX_EXTRACTION = 25  # per tool call
TOOL_RESULT_LIMIT = 8000

SYSTEM_PROMPT = """You are the envforge environment builder. You help a user turn their intent and \
their data (e.g. tech-support chat logs) into a reinforcement-learning environment that \
trains and evaluates AI agents. You work like a thoughtful product engineer: you ask \
focused questions, propose concrete designs, build them, check them, and explain briefly.

## What you are building
A declarative `support_desk` environment (an EnvSpec, schema below) plus task records. \
An agent under training receives a customer's ticket, investigates with tools over mock \
business records, may ask the customer for facts, takes actions, records a resolution \
category and finishes. The reward is computed by code: 1.0 if every success check of the \
task passes and no policy check fails, 0.3 if solved but a policy was broken, else 0.0. \
Because rewards must be verifiable, design success around *state changes and recorded \
resolutions*, never around the wording of replies.

## How to work
1. Understand the purpose: who the agent is, what "solved" means, what must never \
happen (policies), what systems/records exist, what actions agents take, how difficulty \
should vary. Ask at most two questions per message; propose sensible defaults the user \
can accept rather than interrogating.
2. Look at the data early. `get_state` shows the corpus: size, duplicates and topic \
clusters (similar conversations grouped, labelled by their top terms) — one cluster is \
often one kind of ticket. Read a spread with `sample_conversations`, then examples of the \
big or interesting clusters (`cluster=`): they tell you the real record types, actions, \
resolution categories and policies, and which topics the environment should cover. Data \
is already redacted: placeholders like <EMAIL_1> or <NAME_2> (or, if the project uses \
surrogates, realistic stand-in names and numbers) replace personal data — never try to \
recover it. Extraction picks conversations across clusters by default; to fill a gap \
target it with `conversation_ids` or `cluster`.
3. Draft the spec with `update_spec` as soon as you know enough, then refine. Every \
update creates a revision the user can review and roll back. Keep the spec valid; fix \
any problems `update_spec` reports.
4. Extract a small batch of tasks (`extract_tasks`, up to 5 conversations per turn without \
approval), check `validate`, look at failures, and improve the spec or prompt. To extract \
more, call `estimate_extraction`: it shows the user an approval card with the estimated \
cost. Tell them briefly and end your turn. Only their click on that card approves the \
spend (a "yes" typed in chat does not); once they approve, the approved number of \
conversations is available to `extract_tasks` and you'll be asked to continue.
5. Use `preview_task` to show what an agent would face. When the readiness checklist in \
`get_state` is green, suggest publishing (the user publishes from the UI).

Style: concise, concrete, friendly. Never paste the whole spec or long JSON into chat — \
summarise what changed. Refer to revisions by number ("revision 4").

## Filling record fields from tool arguments
In a tool's effect, a value in `set`, `values` or `match` is either a literal or a \
reference to one of the tool's parameters written as "$name", e.g. \
`"values": {{"customer_id": "$customer_id", "kind": "refund", "amount": "$amount"}}`. \
Only that form is filled in at run time: "{{customer_id}}", "<customer_id>" or \
"${{customer_id}}" would be stored as literal text (validation rejects them).

## Spec format (JSON Schema of EnvSpec)
{schema}

## Task record format (JSON Schema of TaskRecord)
{task_schema}

## A complete reference example (an ISP support desk)
{example}
"""


def _schemas() -> tuple[str, str, str]:
    from envforge.declarative.spec import EnvSpec, TaskRecord

    example_path = (
        Path(__file__).resolve().parents[1]
        / "declarative"
        / "examples"
        / "isp_support"
        / "spec.json"
    )
    # compact and whole: cutting it short once hid every "$param" reference from the model
    example = (
        json.dumps(json.loads(example_path.read_text()), separators=(",", ":"))
        if example_path.exists()
        else "(no example available)"
    )
    return (
        json.dumps(EnvSpec.model_json_schema(), separators=(",", ":")),
        json.dumps(TaskRecord.model_json_schema(), separators=(",", ":")),
        example,
    )


def _obj(props: dict[str, Any], required: list[str]) -> dict[str, Any]:
    return {"type": "object", "properties": props, "required": required}


# extraction can aim at specific conversations or one topic instead of a spread
TARGET_PARAMS: dict[str, Any] = {
    "conversation_ids": {
        "type": "array",
        "items": {"type": "string"},
        "description": "extract these conversations (ids from the user's file, or store ids "
        "'<upload>:<id>'); ones already extracted are skipped",
    },
    "cluster": {"type": "integer", "description": "extract from this topic cluster (get_state)"},
}

TOOLS: list[ToolSpec] = [
    ToolSpec(
        name="get_state",
        description="The project's purpose, head revision, spec (or null), task count, "
        "validation summary, uploads and the readiness checklist. Call this first in a new chat.",
        parameters=_obj({}, []),
    ),
    ToolSpec(
        name="update_spec",
        description="Replace the environment spec with a new complete EnvSpec JSON object. "
        "Creates a new revision (tasks carry over) and returns validation problems to fix.",
        parameters=_obj(
            {
                "spec": {"type": "object", "description": "the complete EnvSpec"},
                "message": {"type": "string", "description": "one line: what changed and why"},
            },
            ["spec", "message"],
        ),
    ),
    ToolSpec(
        name="sample_conversations",
        description="Read redacted conversations (text, with roles). By default a representative "
        "spread over the corpus's topic clusters (see corpus.clusters in get_state); pass `cluster` "
        "to read examples of one topic, or `upload_id` to read one upload in order.",
        parameters=_obj(
            {
                "cluster": {"type": "integer", "description": "a topic cluster id from get_state"},
                "upload_id": {"type": "string", "description": "default: all uploads"},
                "n": {"type": "integer", "minimum": 1, "maximum": 10},
                "offset": {"type": "integer", "minimum": 0},
            },
            ["n"],
        ),
    ),
    ToolSpec(
        name="estimate_extraction",
        description="Estimated cost of extracting task records from the next N not-yet-extracted "
        "conversations (or from N of the targeted ones; see extract_tasks). Beyond the free "
        f"{FREE_EXTRACTION} per turn, this also asks the user to approve the spend in the UI.",
        parameters=_obj(
            {"n": {"type": "integer", "minimum": 1, "maximum": 100000}, **TARGET_PARAMS}, ["n"]
        ),
    ),
    ToolSpec(
        name="extract_tasks",
        description=f"Turn the next N not-yet-extracted conversations into task records with the "
        f"current spec, validate each (its reference solution must earn reward 1.0), and save the "
        f"valid ones in a new revision. By default the conversations are a spread over the topic "
        f"clusters; pass `conversation_ids` or `cluster` to target topics the tasks don't cover yet. "
        f"N <= {MAX_EXTRACTION} per call. Up to {FREE_EXTRACTION} conversations per turn are free; "
        "more draw on what the user approved in the UI after estimate_extraction.",
        parameters=_obj(
            {"n": {"type": "integer", "minimum": 1, "maximum": MAX_EXTRACTION}, **TARGET_PARAMS},
            ["n"],
        ),
    ),
    ToolSpec(
        name="get_tasks",
        description="Read the head revision's task records. With `ids` or `conversations` "
        "(source conversation ids from the user's file): the full stored records (state, "
        "customer answers, reference, success checks), as many as fit in one result — read a "
        "record before replacing it with edit_tasks. Without them: a compact index (id, tier, source "
        "conversation, goal, expected resolution) paged by `offset`.",
        parameters=_obj(
            {
                "ids": {"type": "array", "items": {"type": "string"}},
                "conversations": {"type": "array", "items": {"type": "string"}},
                "offset": {"type": "integer", "minimum": 0},
            },
            [],
        ),
    ),
    ToolSpec(
        name="edit_tasks",
        description="Remove task records by id and/or add or replace task records (full TaskRecord "
        "objects). Creates a new revision.",
        parameters=_obj(
            {
                "remove_ids": {"type": "array", "items": {"type": "string"}},
                "upsert": {"type": "array", "items": {"type": "object"}},
                "message": {"type": "string"},
            },
            ["message"],
        ),
    ),
    ToolSpec(
        name="validate",
        description="Full validation of the head revision: spec problems and per-task problems.",
        parameters=_obj({}, []),
    ),
    ToolSpec(
        name="preview_task",
        description="Generate a task from the current spec + tasks as an agent would see it, with "
        "its success checks and whether the reference solution passes.",
        parameters=_obj(
            {"seed": {"type": "integer", "minimum": 0}, "tier": {"type": "integer"}}, []
        ),
    ),
]


# ---- context -------------------------------------------------------------------------


class Ctx:
    """What a tool needs: the DB session, workspace, project, user and the model."""

    def __init__(self, db: Session, ws: Workspace, project: EnvProject, user: User, model) -> None:
        self.db, self.ws, self.project, self.user, self.model = db, ws, project, user, model
        self.free_left = FREE_EXTRACTION  # extraction without approval, per turn
        self.usage = {
            "input_tokens": 0,
            "output_tokens": 0,
            "cache_read_tokens": 0,
            "cost_usd": 0.0,
        }

    @property
    def head(self) -> EnvRevision | None:
        return B.head(self.db, self.project)

    def tasks(self) -> list[dict[str, Any]]:
        h = self.head
        return B.load_tasks(self.ws, self.project, h.tasks_sha if h else None)

    def add_usage(self, usage: Usage) -> None:
        self.usage["input_tokens"] += usage.input_tokens
        self.usage["output_tokens"] += usage.output_tokens
        self.usage["cache_read_tokens"] += usage.cache_read_tokens
        c = usage_cost(self.model.model_id, usage)
        if c is not None and self.usage["cost_usd"] is not None:
            self.usage["cost_usd"] += c

    def spend_left(self) -> float | None:
        """USD left under the workspace's monthly limit, counting this turn so far."""
        return svc.spend_left(self.db, self.ws, pending=self.usage["cost_usd"] or 0.0)


def readiness(ctx: Ctx) -> list[dict[str, Any]]:
    h = ctx.head
    v = (h.validation or {}) if h else {}
    spec = (h.spec or {}) if h else {}
    uploads = [u for u in B.list_uploads(ctx.db, ctx.project) if u.status == "ready"]
    tiers = v.get("tiers") or {}
    n_ok = v.get("tasks_ok", 0)
    return [
        {"item": "purpose stated", "ok": bool(ctx.project.purpose.strip())},
        {"item": "data uploaded and redacted", "ok": bool(uploads)},
        {"item": "spec valid", "ok": bool(v.get("spec_ok"))},
        {"item": "at least one policy", "ok": bool(spec.get("policies"))},
        {"item": "10+ valid tasks", "ok": n_ok >= 10, "detail": f"{n_ok} valid"},
        {"item": "2+ difficulty tiers covered", "ok": len(tiers) >= 2, "detail": tiers},
    ]


def _store(ctx: Ctx):
    """The project's corpus store (redacted conversations), or None before any upload.

    Uploads processed before the store existed left one redacted JSONL file each; they
    are moved into the store the first time it is opened, then deleted.
    """
    from envforge.builder import corpus as C

    path = B.corpus_path(ctx.ws, ctx.project)
    legacy = [u for u in B.list_uploads(ctx.db, ctx.project) if u.status == "ready" and u.path]
    if not path.exists() and not legacy:
        return None
    store = C.CorpusStore(path)
    if legacy:
        from envforge.builder.ingest import Conversation

        for up in legacy:
            f = Path(up.path)
            if f.exists():
                convs = [Conversation.from_dict(json.loads(line))
                         for line in f.read_text().splitlines() if line.strip()]  # fmt: skip
                for c in convs:
                    c.meta["redacted"] = True  # these files only ever held redacted text
                store.add_conversations(up.id, convs)
                store.db.commit()
                f.unlink()
            up.path = None
        ctx.db.commit()
        C.dedupe_near(store)
        C.cluster(store)
    return store


def _render_conversation(conv: Any, limit: int = 3000) -> str:
    d = conv if isinstance(conv, dict) else conv.to_dict()
    cluster = (d.get("meta") or {}).get("cluster")
    head = f"[conversation {d.get('id')}" + (
        f" · topic cluster {cluster}]" if cluster is not None else "]"
    )
    lines = [head]
    for t in d.get("turns", []):
        lines.append(f"{t.get('role', '?')}: {t.get('text', '')}")
    text = "\n".join(lines)
    return text if len(text) <= limit else text[:limit] + "\n…[truncated]"


def _extracted_ids(ctx: Ctx) -> set[str]:
    """Store ids ("<upload>:<conversation>") of conversations already turned into tasks."""
    out = set()
    for t in ctx.tasks():
        src = t.get("source") or {}
        if src.get("corpus_id"):
            out.add(str(src["corpus_id"]))
            continue
        cid, upload = src.get("conversation"), src.get("upload")
        if cid is None:
            continue
        cid = str(cid)
        # tasks extracted before the corpus store recorded the bare conversation id
        out.add(cid if ":" in cid or not upload else f"{upload}:{cid}")
    return out


# ---- tool implementations --------------------------------------------------------------


def t_get_state(ctx: Ctx, args: dict[str, Any]) -> dict[str, Any]:
    h = ctx.head
    uploads = B.list_uploads(ctx.db, ctx.project)
    return {
        "project": {
            "title": ctx.project.title,
            "name": ctx.project.name,
            "purpose": ctx.project.purpose,
        },
        "head_revision": h.number if h else None,
        "spec": h.spec if h else None,
        "n_tasks": h.n_tasks if h else 0,
        "validation": {
            k: v for k, v in ((h.validation or {}) if h else {}).items() if k != "task_problems"
        },
        "task_problem_count": len((h.validation or {}).get("task_problems") or {}) if h else 0,
        "uploads": [
            {
                "id": u.id,
                "file": u.filename,
                "status": u.status,
                "conversations": u.n_conversations,
                "redactions": u.redactions,
                "error": u.error,
            }  # fmt: skip
            for u in uploads
        ],
        "corpus": _corpus_overview(ctx),
        "conversations_extracted": len(_extracted_ids(ctx)),
        "readiness": readiness(ctx),
    }


def _corpus_overview(ctx: Ctx) -> dict[str, Any] | None:
    """Corpus stats + topic clusters (compact): how to pick what to read and extract."""
    from envforge.builder import corpus as C

    store = _store(ctx)
    if store is None:
        return None
    with store:
        if not store.count():
            return None
        ov = C.overview(store)
    ov.pop("uploads", None)  # already listed under "uploads"
    return ov


def t_update_spec(ctx: Ctx, args: dict[str, Any]) -> dict[str, Any]:
    from pydantic import ValidationError

    from envforge.declarative.spec import EnvSpec

    spec = args.get("spec")
    if not isinstance(spec, dict):
        return {"error": "spec must be a JSON object"}
    try:
        EnvSpec.model_validate(spec)
    except ValidationError as e:
        errs = [f"{'.'.join(map(str, x['loc']))}: {x['msg']}" for x in e.errors()][:25]
        return {
            "error": "spec does not match the EnvSpec schema; no revision created",
            "problems": errs,
        }
    rev = B.create_revision(
        ctx.db, ctx.ws, ctx.project, spec=spec, message=str(args.get("message", "update spec"))
    )
    v = rev.validation or {}
    return {
        "revision": rev.number,
        "spec_problems": v.get("spec_problems", []),
        "tasks_ok": v.get("tasks_ok", 0),
        "tasks_with_problems": len(v.get("task_problems") or {}),
        "_meta": {"revision": rev.number},
    }


def t_sample_conversations(ctx: Ctx, args: dict[str, Any]) -> dict[str, Any]:
    from envforge.builder import corpus as C

    store = _store(ctx)
    if store is None:
        return {
            "error": "no processed uploads yet — ask the user to upload chat logs (CSV, TXT or JSON)"
        }
    off = int(args.get("offset", 0))
    n = min(int(args.get("n", 3)), 10)
    with store:
        total = store.count(unique_only=True)
        if not total:
            return {"error": "the uploads contain no conversations"}
        if args.get("cluster") is not None:
            convs = store.examples(int(args["cluster"]), off + n)[off:]
        elif args.get("upload_id"):
            ids = [r[0] for r in store.db.execute(
                "SELECT id FROM conversations WHERE upload_id = ? AND near_dup_of IS NULL "
                "ORDER BY seq LIMIT ? OFFSET ?", (args["upload_id"], n, off))]  # fmt: skip
            convs = store.get(ids)
        else:
            # a representative spread over the topic clusters; offset pages through it
            ids = C.stratified_sample(store, off + n, per_cluster_cap=max(2, (off + n) // 3))
            convs = store.get(ids[off:])
    return {"total": total, "conversations": [_render_conversation(c) for c in convs]}


def _next_unextracted(ctx: Ctx, n: int, args: dict[str, Any] | None = None) -> list[Any]:
    """The next n conversations to extract. By default a stratified sample of the ones not
    yet extracted, so even a small batch covers the corpus's topics; `conversation_ids` or
    `cluster` in `args` aim it at particular conversations or one topic instead."""
    from envforge.builder import corpus as C

    args = args or {}
    store = _store(ctx)
    if store is None:
        return []
    done = _extracted_ids(ctx)
    with store:
        if args.get("conversation_ids"):
            ids = store.resolve_ids(map(str, args["conversation_ids"]))
        elif args.get("cluster") is not None:
            ids = store.cluster_ids(int(args["cluster"]))
        else:
            return store.get(C.stratified_sample(store, n, exclude_ids=done))
        return store.get([i for i in ids if i not in done][:n])


def t_estimate_extraction(ctx: Ctx, args: dict[str, Any]) -> dict[str, Any]:
    from envforge.builder.extract import estimate_cost
    from envforge.declarative.spec import EnvSpec

    h = ctx.head
    if not h or not h.spec:
        return {"error": "draft a spec first"}
    want = int(args["n"])
    store = _store(ctx)
    if store is None:
        return {"error": "no processed uploads yet"}
    targeted = bool(args.get("conversation_ids")) or args.get("cluster") is not None
    if targeted:
        n = len(_next_unextracted(ctx, want, args))
    else:
        with store:
            left = max(0, store.count(unique_only=True) - len(_extracted_ids(ctx)))
        n = min(want, left)
    if not n:
        return {"error": "nothing left to extract there" if targeted
                else "every uploaded conversation has been extracted"}  # fmt: skip
    # price a stratified sample of up to 500 and scale it: reading 100k conversations
    # just to count characters would be slow and tell us little more
    sample = _next_unextracted(ctx, min(n, 500), args)
    est = estimate_cost(EnvSpec.model_validate(h.spec), sample, ctx.model.model_id, max_listed=5)
    if len(sample) < n:
        scale = n / len(sample)
        for k in ("input_tokens", "output_tokens", "cost_usd", "cost_usd_with_retries"):
            if isinstance(est.get(k), (int, float)):
                est[k] = (
                    round(est[k] * scale, 4) if isinstance(est[k], float) else round(est[k] * scale)
                )
        est["conversations"] = n
        est.setdefault("notes", []).append(f"scaled from a stratified sample of {len(sample)}")
    out: dict[str, Any] = {"conversations": n, "model": ctx.model.model_id, "estimate": est}
    if n > ctx.free_left and n > (ctx.project.extraction_allowance or 0):
        # the user approves this by clicking the card (POST .../extraction/approve), never
        # through the model: text in an uploaded conversation must not be able to spend money
        ctx.project.extraction_quote = {
            "id": uuid.uuid4().hex,
            "conversations": n,
            "model": ctx.model.model_id,
            "cost_usd": est.get("cost_usd"),
            "cost_usd_with_retries": est.get("cost_usd_with_retries"),
            "created_at": datetime.now(UTC).isoformat(),
        }
        ctx.db.commit()
        out["approval"] = ("the user now sees an approval card for this estimate; tell them "
                           "briefly and end your turn. Their click approves it, not a chat reply.")  # fmt: skip
    return out


def t_extract_tasks(ctx: Ctx, args: dict[str, Any]) -> dict[str, Any]:
    from envforge.builder.extract import extract_batch
    from envforge.declarative.spec import EnvSpec, TaskRecord
    from envforge.declarative.validate import validate_task

    n = min(int(args.get("n", 1)), MAX_EXTRACTION)
    review = ctx.project.privacy_review or {}
    if review.get("status") == "failed":
        return {"error": "the last privacy review found too much personal data left in the "
                "redacted conversations: tighten the privacy settings (Data tab), re-upload, "
                "and pass a new review before extracting"}  # fmt: skip
    if ctx.project.privacy_profile == "strict" and ctx.project.privacy_signed_off_at is None:
        return {"error": "this project uses the strict privacy profile: a person must pass a "
                "privacy review (Data tab: check a sample of redacted conversations) before "
                "any extraction"}  # fmt: skip
    h = ctx.head
    if not h or not h.spec or not (h.validation or {}).get("spec_ok"):
        return {"error": "the spec must be valid before extracting (see validate)"}
    spec = EnvSpec.model_validate(h.spec)
    todo = _next_unextracted(ctx, n, args)
    if not todo:
        if args.get("conversation_ids") or args.get("cluster") is not None:
            return {"error": "none of the targeted conversations are left to extract (unknown "
                    "ids, or already extracted)"}  # fmt: skip
        return {"error": "every uploaded conversation has been extracted"}

    # Spending: a few conversations per turn are free; more need the user's approval
    # (their click on an estimate, POST .../extraction/approve), never the model's word.
    free = len(todo) <= ctx.free_left
    if not free and len(todo) > (ctx.project.extraction_allowance or 0):
        return {"error": f"only {ctx.free_left} more conversation(s) can be extracted this turn without "
                f"approval, and {ctx.project.extraction_allowance or 0} are approved: call "
                "estimate_extraction, which shows the user an approval card, then end your turn"}  # fmt: skip
    left = ctx.spend_left()
    if left is not None and left <= 0:
        return {"error": svc.limit_reached_message(ctx.ws)}
    # use up the free quota or the approval before spending, so a failure can't be retried
    # for free
    if free:
        ctx.free_left -= len(todo)
    else:
        ctx.project.extraction_allowance -= len(todo)
        ctx.db.commit()
    batch = extract_batch(ctx.model, spec, todo)
    ctx.add_usage(batch.usage)
    existing = ctx.tasks()
    ids = {t.get("id") for t in existing}
    added, invalid, skipped = [], {}, {}
    for conv, res in zip(todo, batch.results, strict=False):
        cid = conv.id
        if res.record is None:
            skipped[cid] = res.skipped_reason or res.error or "no record"
            continue
        rec: TaskRecord = res.record
        if rec.id in ids:
            rec = rec.model_copy(update={"id": f"{rec.id}-{cid}"[:80]})
        # "conversation" stays the id from the user's file; "corpus_id" is the store's
        rec = rec.model_copy(update={"source": {
            **rec.source, "conversation": str(conv.meta.get("ext_id") or cid),
            "upload": conv.meta.get("upload_id"), "corpus_id": cid}})  # fmt: skip
        problems = validate_task(spec, rec)
        if problems:
            invalid[cid] = problems[:5]
        added.append(rec.model_dump(mode="json"))  # keep invalid ones too, so they can be fixed
        ids.add(rec.id)
    rev = B.create_revision(
        ctx.db, ctx.ws, ctx.project, spec=h.spec, tasks=existing + added,
        message=f"Extracted {len(added)} task(s) from {len(todo)} conversation(s)",
    )  # fmt: skip
    return {
        "revision": rev.number,
        "conversations": len(todo),
        "tasks_added": len(added),
        "valid": len(added) - len(invalid),
        "invalid": invalid,
        "skipped": skipped,
        "cost_usd": batch.cost_usd,
        "_meta": {"revision": rev.number},
    }


GET_TASKS_INDEX = 40  # index rows per page


def t_get_tasks(ctx: Ctx, args: dict[str, Any]) -> dict[str, Any]:
    tasks = ctx.tasks()
    ids = {str(i) for i in args.get("ids") or []}
    convs = {str(c) for c in args.get("conversations") or []}
    if ids or convs:
        hits = [
            t for t in tasks
            if str(t.get("id")) in ids
            or str((t.get("source") or {}).get("conversation")) in convs
        ]  # fmt: skip
        if not hits:
            return {"error": "no task with those ids or source conversations"}
        # whole records only, as many as fit under the tool-result limit (a cut record
        # couldn't be edited and passed back)
        out, size = [], 0
        for t in hits:
            size += len(json.dumps(t)) + 2
            if out and size > TOOL_RESULT_LIMIT - 500:
                break
            out.append(t)
        return {"tasks": out, "matched": len(hits), "returned": len(out)}
    off = int(args.get("offset", 0))

    def resolution(t: dict[str, Any]) -> str | None:
        return next((c.get("category") for c in t.get("success") or []
                     if c.get("check") == "resolution"), None)  # fmt: skip

    rows = [
        {"id": t.get("id"), "tier": t.get("tier"),
         "conversation": (t.get("source") or {}).get("conversation"),
         "goal": str(t.get("goal", ""))[:90], "resolution": resolution(t)}
        for t in tasks[off : off + GET_TASKS_INDEX]
    ]  # fmt: skip
    return {"total": len(tasks), "offset": off, "tasks": rows}


def t_edit_tasks(ctx: Ctx, args: dict[str, Any]) -> dict[str, Any]:
    h = ctx.head
    if not h:
        return {"error": "no revision yet"}
    remove = set(map(str, args.get("remove_ids") or []))
    upsert = [t for t in (args.get("upsert") or []) if isinstance(t, dict) and t.get("id")]
    by_id = {str(t["id"]): t for t in ctx.tasks() if str(t.get("id")) not in remove}
    for t in upsert:
        by_id[str(t["id"])] = t
    rev = B.create_revision(ctx.db, ctx.ws, ctx.project, spec=h.spec, tasks=list(by_id.values()),
                            message=str(args.get("message", "edit tasks")))  # fmt: skip
    v = rev.validation or {}
    return {"revision": rev.number, "n_tasks": rev.n_tasks, "tasks_ok": v.get("tasks_ok", 0),
            "problems": {k: p for k, p in list((v.get("task_problems") or {}).items())[:10]},
            "_meta": {"revision": rev.number}}  # fmt: skip


def t_validate(ctx: Ctx, args: dict[str, Any]) -> dict[str, Any]:
    h = ctx.head
    if not h:
        return {"error": "no revision yet"}
    v = h.validation or {}
    probs = v.get("task_problems") or {}
    return {**{k: x for k, x in v.items() if k != "task_problems"},
            "task_problems": dict(list(probs.items())[:15]), "task_problem_count": len(probs)}  # fmt: skip


def head_env_class(ws: Workspace, project: EnvProject, rev: EnvRevision):
    """An Environment class for a revision (valid tasks only), for previews."""
    from envforge.declarative.runtime import make_environment
    from envforge.declarative.spec import EnvSpec, TaskRecord

    problems = (rev.validation or {}).get("task_problems") or {}
    tasks = [TaskRecord.model_validate(t) for t in B.load_tasks(ws, project, rev.tasks_sha)
             if str(t.get("id")) not in problems]  # fmt: skip
    if not tasks:
        raise ValueError("no valid tasks yet")
    return make_environment(
        EnvSpec.model_validate(rev.spec), tasks, name=f"{project.name}@r{rev.number}"
    )


def t_preview_task(ctx: Ctx, args: dict[str, Any]) -> dict[str, Any]:
    from envforge.core.explore import preview_task

    h = ctx.head
    if not h or not (h.validation or {}).get("spec_ok"):
        return {"error": "needs a valid spec and at least one valid task"}
    try:
        env_cls = head_env_class(ctx.ws, ctx.project, h)
    except ValueError as e:
        return {"error": str(e)}
    tier = int(args.get("tier") or env_cls.tiers[0])
    if tier not in env_cls.tiers:
        tier = env_cls.tiers[0]
    p = preview_task(env_cls, int(args.get("seed", 0)), tier)
    return {
        "goal": p["task"]["goal"],
        "first_observation": (p["first_observation"] or "")[:1500],
        "reference_solution": [{"tool": s["name"], "args": s["args"]} for s in p["solution"]],
        "verdict_after_reference": p["verdict"],
    }


HANDLERS = {
    "get_state": t_get_state,
    "update_spec": t_update_spec,
    "sample_conversations": t_sample_conversations,
    "estimate_extraction": t_estimate_extraction,
    "extract_tasks": t_extract_tasks,
    "get_tasks": t_get_tasks,
    "edit_tasks": t_edit_tasks,
    "validate": t_validate,
    "preview_task": t_preview_task,
}

ACTIVITY = {
    "get_state": "Reading the project",
    "update_spec": "Updating the spec",
    "sample_conversations": "Reading conversations",
    "estimate_extraction": "Estimating extraction cost",
    "extract_tasks": "Extracting tasks",
    "get_tasks": "Reading tasks",
    "edit_tasks": "Editing tasks",
    "validate": "Validating",
    "preview_task": "Previewing a task",
}


def run_tool(ctx: Ctx, call: ToolCall) -> tuple[str, dict[str, Any]]:
    """Execute one tool call; returns (result text for the model, meta for the UI)."""
    fn = HANDLERS.get(call.name)
    if fn is None:
        return json.dumps({"error": f"unknown tool {call.name}"}), {}
    try:
        result = fn(ctx, call.args or {})
    except Exception as e:  # noqa: BLE001 - tool failures go back to the model
        traceback.print_exc()
        result = {"error": f"{type(e).__name__}: {e}"}
    meta = result.pop("_meta", {}) if isinstance(result, dict) else {}
    text = json.dumps(result, default=str)
    if len(text) > TOOL_RESULT_LIMIT:
        text = text[:TOOL_RESULT_LIMIT] + '…[truncated]"'
    return text, meta


# ---- the turn loop -----------------------------------------------------------------------


def _history(db: Session, project: EnvProject) -> list[Message]:
    """The chat as model messages, exactly as stored.

    Append-only on purpose: the prompt is cached up to the end of the previous call,
    so rewriting anything earlier (e.g. shortening old tool results) would make every
    later token a cache miss. Tool results are already capped when stored.
    """
    rows = B.messages(db, project)
    out: list[Message] = []
    for m in rows:
        if m.role == "user":
            out.append(Message(role="user", content=m.content))
        elif m.role == "assistant":
            calls = [ToolCall(**c) for c in (m.tool_calls or [])]
            out.append(Message(role="assistant", content=m.content, tool_calls=calls))
        elif m.role == "tool":
            out.append(
                Message(role="tool", content=m.content, tool_call_id=m.tool_call_id, name=m.name)
            )
    return out


def run_turn(db: Session, job: Job) -> None:
    """Process one `builder_turn` job: call the model, run tools, record everything."""
    project = db.get(EnvProject, job.project_id)
    ws = db.get(Workspace, project.workspace_id) if project else None
    user = db.get(User, job.created_by)
    assert project and ws and user
    spec_str = ws.builder_model
    provider = svc.provider_of(spec_str)
    if provider == "scripted":
        raise RuntimeError(
            "the builder needs a real model: set one in Workspace settings → Builder model"
        )
    api_key, base_url = svc.resolve_key(db, user.id, provider)
    model = load_model(spec_str, api_key=api_key, base_url=base_url, max_tokens=16000)
    ctx = Ctx(db, ws, project, user, model)

    schema, task_schema, example = _schemas()
    system = Message(
        role="system",
        content=SYSTEM_PROMPT.format(schema=schema, task_schema=task_schema, example=example),
    )
    for _ in range(MAX_MODEL_CALLS):
        left = ctx.spend_left()
        if left is not None and left <= 0:
            B.add_message(db, project, "assistant",
                          f"(Paused: {svc.limit_reached_message(ws)}.)")  # fmt: skip
            break
        job.progress = "Thinking"
        db.commit()
        turn = model.complete([system, *_history(db, project)], TOOLS)
        ctx.add_usage(turn.usage)
        B.add_message(
            db, project, "assistant", turn.text,
            tool_calls=[c.model_dump() for c in turn.tool_calls] or None,
        )  # fmt: skip
        if not turn.tool_calls:
            break
        for call in turn.tool_calls:
            job.progress = ACTIVITY.get(call.name, call.name)
            job.heartbeat_at = datetime.now(UTC)
            db.commit()
            text, meta = run_tool(ctx, call)
            B.add_message(
                db, project, "tool", text, tool_call_id=call.id, name=call.name, meta=meta or None
            )
    else:
        B.add_message(
            db,
            project,
            "assistant",
            "(I've paused here to keep this turn short — reply to continue.)",
        )
    job.input_tokens = ctx.usage["input_tokens"]
    job.output_tokens = ctx.usage["output_tokens"]
    job.cache_read_tokens = ctx.usage["cache_read_tokens"]
    job.cost_usd = ctx.usage["cost_usd"]
    db.commit()

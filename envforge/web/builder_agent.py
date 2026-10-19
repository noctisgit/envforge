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
FREE_EXTRACTION = 5  # conversations the builder may extract without asking
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
2. Look at the data (`sample_conversations`) early: it tells you the real record types, \
actions, resolution categories and policies. Data is already redacted: placeholders like \
<EMAIL_1> or <NAME_2> stand for removed personal data — never try to recover it.
3. Draft the spec with `update_spec` as soon as you know enough, then refine. Every \
update creates a revision the user can review and roll back. Keep the spec valid; fix \
any problems `update_spec` reports.
4. Extract a small batch of tasks (`extract_tasks`, up to 5 without asking), check \
`validate`, look at failures, and improve the spec or prompt. Before extracting more than \
5 conversations, call `estimate_extraction`, tell the user the estimated cost and wait \
for their explicit yes; then call `extract_tasks` with `confirmed=true`.
5. Use `preview_task` to show what an agent would face. When the readiness checklist in \
`get_state` is green, suggest publishing (the user publishes from the UI).

Style: concise, concrete, friendly. Never paste the whole spec or long JSON into chat — \
summarise what changed. Refer to revisions by number ("revision 4").

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
    example = example_path.read_text() if example_path.exists() else "(no example available)"
    return (
        json.dumps(EnvSpec.model_json_schema(), separators=(",", ":")),
        json.dumps(TaskRecord.model_json_schema(), separators=(",", ":")),
        example[:12000],
    )


def _obj(props: dict[str, Any], required: list[str]) -> dict[str, Any]:
    return {"type": "object", "properties": props, "required": required}


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
        description="Read redacted conversations from the uploads (text, with roles).",
        parameters=_obj(
            {
                "upload_id": {"type": "string", "description": "default: all uploads"},
                "n": {"type": "integer", "minimum": 1, "maximum": 10},
                "offset": {"type": "integer", "minimum": 0},
            },
            ["n"],
        ),
    ),
    ToolSpec(
        name="estimate_extraction",
        description="Estimated cost of extracting task records from the next N not-yet-extracted conversations.",
        parameters=_obj({"n": {"type": "integer", "minimum": 1, "maximum": 100000}}, ["n"]),
    ),
    ToolSpec(
        name="extract_tasks",
        description=f"Turn the next N not-yet-extracted conversations into task records with the "
        f"current spec, validate each (its reference solution must earn reward 1.0), and save the "
        f"valid ones in a new revision. N <= {MAX_EXTRACTION} per call. N > {FREE_EXTRACTION} requires "
        "confirmed=true, which you may only set after the user explicitly approved the estimated cost.",
        parameters=_obj(
            {
                "n": {"type": "integer", "minimum": 1, "maximum": MAX_EXTRACTION},
                "confirmed": {"type": "boolean"},
            },
            ["n"],
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


def _conversations(ctx: Ctx, upload_id: str | None = None) -> list[dict[str, Any]]:
    """All redacted conversations of the project (or one upload), in upload order."""
    rows = []
    for up in B.list_uploads(ctx.db, ctx.project):
        if up.status != "ready" or not up.path or (upload_id and up.id != upload_id):
            continue
        for line in Path(up.path).read_text().splitlines():
            if line.strip():
                conv = json.loads(line)
                conv.setdefault("meta", {})["upload_id"] = up.id
                rows.append(conv)
    return rows


def _render_conversation(conv: dict[str, Any], limit: int = 3000) -> str:
    lines = [f"[conversation {conv.get('id')}]"]
    for t in conv.get("turns", []):
        lines.append(f"{t.get('role', '?')}: {t.get('text', '')}")
    text = "\n".join(lines)
    return text if len(text) <= limit else text[:limit] + "\n…[truncated]"


def _extracted_ids(ctx: Ctx) -> set[str]:
    return {str((t.get("source") or {}).get("conversation")) for t in ctx.tasks()}


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
        "conversations_extracted": len(_extracted_ids(ctx) - {"None"}),
        "readiness": readiness(ctx),
    }


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
    convs = _conversations(ctx, args.get("upload_id"))
    if not convs:
        return {
            "error": "no processed uploads yet — ask the user to upload chat logs (CSV, TXT or JSON)"
        }
    off = int(args.get("offset", 0))
    n = min(int(args.get("n", 3)), 10)
    chosen = convs[off : off + n]
    return {"total": len(convs), "conversations": [_render_conversation(c) for c in chosen]}


def _next_unextracted(ctx: Ctx, n: int) -> list[dict[str, Any]]:
    done = _extracted_ids(ctx)
    return [c for c in _conversations(ctx) if str(c.get("id")) not in done][:n]


def t_estimate_extraction(ctx: Ctx, args: dict[str, Any]) -> dict[str, Any]:
    from envforge.builder.extract import estimate_cost
    from envforge.declarative.spec import EnvSpec

    h = ctx.head
    if not h or not h.spec:
        return {"error": "draft a spec first"}
    todo = _next_unextracted(ctx, int(args["n"]))
    if not todo:
        return {"error": "every uploaded conversation has been extracted"}
    est = estimate_cost(
        EnvSpec.model_validate(h.spec), [_as_conversation(c) for c in todo], ctx.model.model_id
    )
    return {"conversations": len(todo), "model": ctx.model.model_id, "estimate": est}


def _as_conversation(c: dict[str, Any]):
    from envforge.builder.ingest import Conversation

    return Conversation.from_dict(c)


def t_extract_tasks(ctx: Ctx, args: dict[str, Any]) -> dict[str, Any]:
    from envforge.builder.extract import extract_batch
    from envforge.declarative.spec import EnvSpec, TaskRecord
    from envforge.declarative.validate import validate_task

    n = min(int(args.get("n", 1)), MAX_EXTRACTION)
    if n > FREE_EXTRACTION and not args.get("confirmed"):
        return {"error": f"extracting more than {FREE_EXTRACTION} needs the user's explicit OK: call "
                "estimate_extraction, tell the user the cost, and wait for their yes"}  # fmt: skip
    h = ctx.head
    if not h or not h.spec or not (h.validation or {}).get("spec_ok"):
        return {"error": "the spec must be valid before extracting (see validate)"}
    spec = EnvSpec.model_validate(h.spec)
    todo = _next_unextracted(ctx, n)
    if not todo:
        return {"error": "every uploaded conversation has been extracted"}

    batch = extract_batch(ctx.model, spec, [_as_conversation(c) for c in todo])
    ctx.add_usage(batch.usage)
    existing = ctx.tasks()
    ids = {t.get("id") for t in existing}
    added, invalid, skipped = [], {}, {}
    for conv, res in zip(todo, batch.results, strict=False):
        cid = str(conv.get("id"))
        if res.record is None:
            skipped[cid] = res.skipped_reason or res.error or "no record"
            continue
        rec: TaskRecord = res.record
        if rec.id in ids:
            rec = rec.model_copy(update={"id": f"{rec.id}-{cid}"[:80]})
        rec = rec.model_copy(update={"source": {**rec.source, "conversation": cid,
                                                 "upload": conv["meta"].get("upload_id")}})  # fmt: skip
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

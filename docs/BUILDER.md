# Environment builder

Turn an intent and a pile of data (e.g. tech-support chat logs) into a verifiable
training environment by chatting with a builder agent — then run, evaluate and train on
it like any other pack.

```
you ──chat──► builder agent (your builder model, on your key; runs in the builder worker)
 │              │ tools: get_state · update_spec · sample_conversations · estimate_extraction
 │              │        extract_tasks · edit_tasks · validate · preview_task
 │              ▼
 │      project ── revisions (immutable DAG) ── publish ──► "<project>@v<n>" in this workspace
 │              │                                              │
 └─upload──► parse → redact → profile (raw file deleted)       └► Runs · Environments · replays · training
```

## What gets built

A **declarative environment** (`envforge/declarative/`): a spec plus task records,
turned into a normal `Environment` class at runtime. The first archetype is
`support_desk`:

| spec part | meaning |
|---|---|
| `collections` | record types in mock business systems (customers, orders, devices…), with a key and an owner field |
| `tools` | what the agent can do; each has a declarative effect: `get`, `find`, `set`, `add` a record, `ask_customer`, `resolve` |
| `policies` | rules told to the agent, each with a check that must still hold at the end (breaking one caps reward at 0.3) |
| `resolution_categories`, `answer_topics` | allowed diagnoses, and what the agent may ask the customer |
| task records | one per ticket: the customer's message, starting records, what the customer knows, success checks, and a reference solution |

Rewards are computed by code from the end state (1.0 / 0.3 / 0.0), never by a model, so
an agent can't talk its way to a high score. A task is only valid if its reference
solution actually earns 1.0 in the runtime — the builder checks this for every task.

See `envforge/declarative/examples/isp_support/` for a complete hand-written example,
and `uv run envforge env validate <dir>` / `envforge run <dir> --model scripted` to work
with environment directories from the CLI.

## Data and privacy

- Uploads: CSV, TXT, JSON/JSONL (up to 95 MB each). Formats are detected (per-message
  rows, transcript columns, speaker-prefixed text, common helpdesk exports).
- **Redaction happens first**, in the builder worker: emails, phone numbers, card numbers
  (Luhn-checked), IBANs, SSNs, public IPs, MAC addresses, secret-bearing URLs, addresses,
  labelled dates of birth, names after cues ("my name is …") and in speaker labels —
  replaced by consistent placeholders (`<EMAIL_1>`, `<NAME_2>`). The original file is then
  deleted; only redacted text is stored or ever sent to the builder model.
- Redaction is best-effort (names without a cue or free-form addresses can slip through):
  review samples in the Data tab before extracting at scale.

## Cost control

The builder uses the workspace's **Builder model** (Workspace settings; default
`anthropic:claude-opus-5-5`) with your own provider key. Each chat turn costs a little;
extraction costs per conversation. The builder may extract up to 5 conversations
without asking; beyond that it must show an estimate and get your explicit yes — this is
enforced in code. Builder spend appears on the Usage page.

## Revisions, rollback, branches, versions

- Every change (by the builder or by you editing the spec) is an **immutable revision**
  with a parent, a message, validation results and a content-addressed task set.
- **Restore** = new revision copying an old one (history kept). **Check out** = move the
  head; the next change becomes a child of that revision, i.e. a branch.
- **Changes** shows a unified diff of the spec and which tasks were added/removed.
- **Publish** freezes a valid revision (its valid tasks only) as `<project>@v<n>`, private
  to the workspace. Runs record that exact name, so results are reproducible.

## Running it

Workers: `envforge-worker --kind builder` handles chat turns and uploads; `--kind runs`
handles evaluation runs (docker-compose runs both as separate services).

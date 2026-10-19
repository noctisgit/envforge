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
- **Redaction happens first**, in the builder worker (`envforge/builder/redact.py`):
  emails, phone numbers, card numbers (Luhn-checked), IBANs, SSNs, public IPs, MAC
  addresses, secret-bearing URLs, addresses, labelled dates of birth, names after cues
  ("my name is …") and in speaker labels — replaced by consistent placeholders
  (`<EMAIL_1>`, `<NAME_2>`). The original file is then deleted; only redacted text is
  stored or ever sent to the builder model.
- Redaction is best-effort (names without a cue or free-form addresses can slip through):
  review samples in the Data tab before extracting at scale.

### Profiles

Every redaction function takes `profile="standard"` (the default, described above) or
`profile="strict"`. Strict aims at the 18 identifier categories of HIPAA's **Safe
Harbor** method and is always a superset of standard:

| Safe Harbor category | strict rule | coverage |
|---|---|---|
| Names | cues, speaker labels, titles ("Dr Lee", "Patient Smith"), NER `PERSON` | partial; good with NER |
| Geography below state | street addresses, UK postcodes, "City, ST", NER `LOCATION`; ZIP after a state or "zip" cue → `021XX` (`000XX` for the 17 low-population prefixes HHS lists) | partial |
| Dates (except year), ages > 89 | numeric / ISO / month-name dates → `<DATE_1> 2024`; DOB → `<DOB_1>`; years ≥ 90 years ago dropped; ages over 89 → `90+` | common formats; relative dates ("last Tuesday") kept |
| Phone, fax, email, SSN | as standard | good |
| Medical record, health plan, account, certificate/licence numbers | after cues (MRN, member id, policy number, account #, license, DEA, NPI…) + any token with 6+ digits | cue-based |
| Vehicle identifiers | 17-character VINs, plates after a cue | partial |
| Device identifiers / serials | after cues (serial, S/N, IMEI, service tag), MAC addresses, 6+ digit tokens | cue-based |
| URLs, IP addresses | all of them (private IPs too) | good |
| Biometrics, full-face photos | cannot be detected in text; attachments are not processed | none |
| Any other unique number or characteristic | 6+ digit tokens, `extra_patterns` | partial |

**keep_patterns.** Strict would otherwise destroy the record keys an environment needs,
so values matching `keep_patterns` are never touched. The strict default keeps an
upper-case prefix + dash + digits (`HD-1001`, `PRN-302`, `C-1101`) except health-ID
prefixes (`MRN-`, `SSN-`, `MBI-`, `NPI-`, `MD-` …). Pass your own list, or `()` to
keep nothing. Known strict false positives: error codes and KB numbers with 6+ digits
become `<ID_n>` (add a keep pattern), and dates the environment checks exactly become
`<DATE_n> YYYY` (use `skip_keys` when sanitizing task records).

### Optional NER layer

`ner=True` adds a local named-entity recogniser (Microsoft Presidio + a spaCy model; no
LLM, nothing leaves the machine) that finds names and places without cues — and, in
strict, dates. Its findings share the regex layer's placeholders. Setup:
`uv sync --extra privacy`, then install a spaCy model wheel matching the installed
spaCy into the venv (e.g. `en_core_web_sm-3.8.0`). The model is chosen with
`ENVFORGE_NER_MODEL` (default `en_core_web_sm`; `en_core_web_lg` is more accurate,
`en_core_web_trf` more accurate still but much slower). If Presidio or the model is
missing, `ner_available()` is falsy with a `reason`, redaction runs regex-only, and
`meta["redaction_ner"]` is `False` — it never crashes.

### Audit and exports

- `build_redaction_report(redacted, profile=…)` / `audit(conversations, …)` return a
  JSON-able report for a human sign-off: counts by type, profile, NER status, keep
  patterns, a sample of redacted snippets, and "suspicious remnants" from `scan_text`
  (a detector-only pass that lists what still looks like PII). It is built from
  redacted text only — never the originals — but remnants may themselves be missed
  personal data, so handle the report like the dataset.
- `sanitize_json_strings(obj, profile, keep_patterns=…, skip_keys=…)` re-redacts every
  string in a JSON-like structure (task records, chat-format training examples) with one
  consistent session, keeping keys and kept values. Apply it when publishing an
  environment and when exporting training data.

This is a technical safeguard, **not** a HIPAA or GDPR certification: Safe Harbor also
requires that you have no actual knowledge that what remains could identify someone,
which only a person reviewing the data can judge.

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

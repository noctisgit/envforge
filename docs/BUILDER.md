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

**Held-out tasks.** A declarative environment has a fixed list of tickets, so it can't
make a new task per seed the way netops does. Instead about a quarter of the task records
are held out for evaluation: seeds below 10000 (the evaluation seeds) pick only those,
and seeds from 10000 up (training data) pick only the rest. Which side a record is on
depends only on a hash of its id, so it stays put across versions: training on v2 and
evaluating on v3 is still clean. The Environments page shows the counts. A task set too
small to split (one side would be empty) uses every record for both. The dataset
manifest and the overlap check compare task records, not seeds, for these environments.

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

Every redaction function takes `profile="standard"` (the default, described above),
`profile="strict"` or `profile="off"`.

**off** is for public or synthetic datasets whose customers are fictional (e.g. ABCD):
text is returned unchanged, but conversations are still marked `meta["redacted"] =
True`, `meta["redaction_profile"] = "off"` — the corpus store refuses unmarked
conversations, and "off" is an explicit choice a person makes in the UI, recorded with
the data. `sanitize_json_strings(obj, "off")` is a no-op.

Strict aims at the 18 identifier categories of HIPAA's **Safe
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

### Surrogates (fake values instead of placeholders)

`surrogates=True` (on `redact_text`, `redact_conversation(s)`, `sanitize_json_strings`,
`RedactionSession`) writes consistent, realistic **fake** values instead of `<NAME_1>`:

| type | surrogate |
|---|---|
| names | from built-in, culturally diverse first/last-name lists (~160 each); one surrogate person per real person, so "Thanks, Crystal" becomes that person's surrogate first name; case kept ("crystal minh" → "imani okafor") |
| usernames | surrogate name + new digits (`cminh730` → `iokafor418`); an email built on the same handle gets the same handle |
| emails | `<handle>@example.com` / `.org` / `.net` (reserved for documentation) |
| phones | the fictional 555-01xx range, in the original's format (`(977) 625-2661` → `(415) 555-0137`) |
| addresses | fictional street + generic town (Springfield, Fairview, …) + state + ZIP; strict keeps ZIP3 (`123XX`) |
| cards | Luhn-valid test numbers (`4111 …`), same length and grouping |
| IPs | documentation ranges 192.0.2.x / 198.51.100.x / 203.0.113.x, `2001:db8::` |
| IDs, MACs, … | same shape: letters → letters, digits → digits |
| dates, DOB | placeholders as before (a made-up date can change a timeline's meaning) |

Why: training on text full of `<NAME_1>` teaches a model to write placeholders and never
shows it a real-looking identifier to handle; natural text keeps the conversation's
rhythm and still teaches the model to *use* names, handles and emails ("Thanks, Imani —
I found the account for iokafor418") without learning anyone's real ones. Surrogates
are deterministic (seeded by the session salt and the value's salted digest: the same
value always gets the same surrogate within a conversation, and — with the default
random salt — an unrelated one in the next), never equal the value they replace, and
two values of one type never share a surrogate. Counts are still reported by type.
They are fake: any resemblance to a real person is coincidence.

### Tier 2: a local PII span model, with risk triage

Tier 1 (rules + optional NER) misses identifiers that arrive with no cue and no shape —
a lower-case name typed on its own, a username mid-sentence. Tier 2
(`envforge/builder/pii_model.py`) is [GLiNER](https://github.com/urchade/GLiNER)
`urchade/gliner_multi_pii-v1` (Apache-2.0, ~1.1 GB), run locally on the CPU (no text
leaves the machine). Setup: `uv sync --extra pii` (gliner + CPU-only torch); the model
id comes from `ENVFORGE_PII_MODEL` and downloads once into `$HF_HOME` (the API/worker
image sets `HF_HOME=/data/hf-cache`, on the data volume). If anything is missing,
`pii_model_available()` is falsy with a `reason` and redaction stays tier 1 — it never
crashes.

The model is ~100x slower than tier 1 (~120 ms per turn on 6 cores) and flags field
labels and pronouns as PII ("full name", "username", "email address", "I", "Agent",
"silver" as an account number), so:

- **Triage** — `redact_conversations(..., tier2="auto")` runs tier 1 on everything, then
  `risk(before, after)` scores each conversation: an unredacted reply to "your name?"
  (1.0) or "username?" (1.0), a short unredacted reply to any identity question (0.5),
  a surviving "@" (0.6), capitalised word pair (0.5), letter+digit token (0.5), 7+ digit
  run (0.2), NER off (0.3). Conversations scoring ≥ `tier2_threshold` (default 1.0) go
  through the model in one batch (customer/system turns, and agent turns with
  something suspicious left). `tier2="all"` sends everything.
- **False-positive filter** (`pii_model.filter_spans`) — drops placeholders, keep-pattern
  values, pronouns, greetings, role words, field labels (with my/your/the/full/new…),
  single lower-case common words; a person must contain a capitalised non-common word,
  a name already registered in the conversation, or answer a name question; usernames
  and account numbers need a digit or a handle shape. A one-token "person" with digits
  (`davidwilliams617`) becomes a username.
- Surviving spans feed the same placeholder / surrogate machinery as NER (no double
  replacement). `meta["redaction_tier2"]`, `meta["redaction_risk"]` and
  `meta["redaction_risk_reasons"]` record what happened; `redact_batch()` returns the
  same plus counts of conversations and turns sent to the model.

For ingest, `Redactor(profile=…, ner=…, surrogates=…, tier2="auto")` is a picklable
callable (`redact(conv) -> (conv, counts)`, plus `.batch(convs)`) to pass as
`ingest_file_to_store(redact=…)`. Each process holds its own copy of the model
(~1.2 GB RAM), so keep worker processes few when tier 2 is on.

### Measuring leaks

`envforge/builder/pii_bench.py` measures the leak rate on ABCD, whose fictional
customers' name, email, phone and username are known for every conversation:
`python -m envforge.builder.pii_bench --abcd abcd_v1.1.json --n 500 --profile standard
--tier2 auto` prints a comparison table. `leak_report(redacted, ground_truth)` gives
leak rates per type and overall with Wilson 95 % intervals; `wilson_interval(k, n)` and
`sample_for_review(ids, n, seed)` support the human-review gate (0 leaks in 300
reviewed conversations bounds the leak rate below ~1.3 %).

Measured on 500 random ABCD conversations (11,031 turns, 6-core AVX2 CPU; leak rate per
known value, Wilson 95 % interval; "other repl." = replaced word spans per conversation
that are not a known value — real catches such as street addresses, plus false
positives like brand names from NER):

| mode | leak overall | name | username | email | phone | convs with a leak | other repl./conv | sent to tier 2 | ms/conv |
|---|---|---|---|---|---|---|---|---|---|
| standard | 5.0 % [4.1–6.1] | 8.8 % | 3.8 % | 0 % | 0 % | 11.0 % | 0.76 | – | 3 |
| standard + NER | 1.3 % [0.9–1.9] | 1.6 % | 3.8 % | 0 % | 0 % | 3.8 % | 1.41 | – | 42 |
| standard + tier 2 auto | 0.9 % [0.6–1.5] | 1.6 % | 0.9 % | 0 % | 0 % | 2.6 % | 0.90 | 46 % | 521 |
| standard + NER + tier 2 auto | 0.6 % [0.4–1.1] | 0.7 % | 2.1 % | 0 % | 0 % | 2.0 % | 1.46 | 21 % | 410 |
| strict + NER + tier 2 auto | 0.6 % [0.4–1.1] | 0.8 % | 1.7 % | 0 % | 0 % | 2.0 % | 2.24 | 12 % | 206 |

(With NER on, fewer conversations look risky, so fewer reach tier 2: names improve,
lone usernames such as a bare "sa098180" with no question nearby slip through more.)

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
extraction costs per conversation. The builder may extract up to 5 conversations per
turn on its own. For more, `estimate_extraction` puts an **approval card** in the chat
(conversations, model, estimated cost, cost with retries). Only your click on **Approve**
(`POST .../extraction/approve`) adds those conversations to what the builder may extract;
then it continues by itself. A "yes" typed in chat approves nothing, and neither does a
`confirmed` flag from the model, so text planted in an uploaded conversation can't make it
spend your money. Builder spend appears on the Usage page.

A workspace can also have a **monthly spending limit** (Workspace settings), covering
evaluation runs and the builder this calendar month (UTC). At the limit, new runs and
builder messages are refused, the builder pauses before its next model call, a running run
stops after its current episode, and an approval that would pass the limit is refused.
Self-hosted (vLLM) and scripted models don't count.

Extraction takes a spread of conversations over the topic clusters by default. To fill a
gap, the builder can aim it with `conversation_ids` (ids from your file) or `cluster`. Its
`get_tasks` tool reads the stored task records (a compact index, or whole records by id or
source conversation), so it can audit a task and edit one field without guessing the rest.

## Revisions, rollback, branches, versions

- Every change (by the builder or by you editing the spec) is an **immutable revision**
  with a parent, a message, validation results and a content-addressed task set.
- **Restore** = new revision copying an old one (history kept). **Check out** = move the
  head; the next change becomes a child of that revision, i.e. a branch.
- **Changes** shows a unified diff of the spec and which tasks were added/removed.
- **Publish** freezes a valid revision (its valid tasks only) as `<project>@v<n>`, private
  to the workspace. Runs record that exact name, so results are reproducible.
- **Delete** a published version from Environments, its page, or the project's Overview
  (the workspace owner or whoever published it; not while a run on it is queued or
  running). It leaves the list and can't start new runs, but its files stay, so runs
  made on it still open, replay and export. Version numbers are never reused. Deleting
  the project deletes all of its published versions the same way.

## Running it

Workers: `envforge-worker --kind builder` handles chat turns and uploads; `--kind runs`
handles evaluation runs (docker-compose runs both as separate services).

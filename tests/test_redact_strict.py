"""Offline tests for the strict (HIPAA Safe Harbor-oriented) redaction profile, keep
patterns, the optional NER layer, the audit report, scan_text and sanitize_json_strings.

NER tests skip themselves when Presidio or the spaCy model is not installed, so the
suite passes on a bare `uv sync --extra dev`.
"""

from __future__ import annotations

import copy
import json

import pytest

from envforge.builder import ner as ner_layer
from envforge.builder.ingest import Conversation, Turn
from envforge.builder.redact import (
    DEFAULT_KEEP_PATTERNS,
    PLACEHOLDER_RE,
    audit,
    build_redaction_report,
    redact_conversation,
    redact_conversations,
    redact_text,
    sanitize_json_strings,
    scan_text,
)
from envforge.declarative.spec import TaskRecord

CLINICAL = (
    "Patient: Margaret O'Neil, DOB 03/14/1961, MRN 00482913. Seen by Dr. Alan Whitfield "
    "on March 3, 2024. She is a 94-year-old widow (aged 94 per chart). Admitted "
    "2024-02-11T08:30:00Z, discharged 02/20/2024. Lives at 18 Elm Street; mail to "
    "Boston, MA 02139. Member ID: XWZ483920115, policy number 7781-2234-99, account # "
    "99812345, medical license MD-449201. Insulin pump serial number SN48213-77, "
    "IMEI 490154203237518. Car VIN 1HGCM82633A004352, license plate 7ABC123. "
    "Portal https://portal.example.org/patients, workstation 10.0.0.12, fax 555-201-3344, "
    "email m.oneil@example.com, SSN 123-45-6789."
)
CLINICAL_SECRETS = [
    "Margaret", "O'Neil", "03/14/1961", "00482913", "Whitfield", "March 3", "94-year",
    "aged 94", "02-11", "02/20", "Elm Street", "Boston", "02139", "XWZ483920115",
    "7781-2234-99", "99812345", "MD-449201", "SN48213-77", "490154203237518",
    "1HGCM82633A004352", "7ABC123", "portal.example.org", "10.0.0.12", "555-201-3344",
    "m.oneil@example.com", "123-45-6789",
]  # fmt: skip

HELPDESK = (
    "Hi, this is Tom Becker. Laptop HD-1001 (ticket PRN-302) lost Wi-Fi after the update "
    "on 2024-05-01. Router at 192.168.1.1, docs at https://intranet.example.com/kb. "
    "Error 0x80070005."
)


# ---- strict profile: Safe Harbor categories ---------------------------------------------


def test_strict_clinical_text_removes_every_category():
    r = redact_text(CLINICAL, profile="strict")
    out = r.text
    for secret in CLINICAL_SECRETS:
        assert secret not in out, secret
    for kind in ("NAME", "DOB", "MRN", "DATE", "ADDRESS", "LOCATION", "HEALTH_PLAN"):
        assert f"<{kind}_1>" in out, kind
    for kind in ("ACCOUNT", "LICENSE", "DEVICE", "VEHICLE", "URL", "IP", "PHONE", "EMAIL"):
        assert f"<{kind}_1>" in out, kind
    assert "<SSN_1>" in out
    assert "<DATE_1> 2024" in out  # year kept
    assert "90+-year-old" in out and "aged 90+" in out
    assert "MA 021XX" in out  # ZIP -> first three digits
    assert r.counts_by_type["AGE"] == 2 and r.counts_by_type["ZIP"] == 1
    assert scan_text(out, "strict") == []  # nothing obviously left


def test_standard_profile_is_unchanged_by_default():
    salt = b"fixed"
    default = redact_text(CLINICAL, salt=salt).text
    assert redact_text(CLINICAL, salt=salt, profile="standard").text == default
    # standard keeps dates, private IPs, ordinary URLs, ages, ZIPs and record numbers
    for kept in ("March 3, 2024", "10.0.0.12", "https://portal.example.org/patients",
                 "94-year-old", "02139", "XWZ483920115"):  # fmt: skip
        assert kept in default, kept


@pytest.mark.parametrize(
    "text",
    [
        "seen 2024-05-01", "seen 2024-05-01T10:00:00Z", "seen 05/01/2024", "seen 5-1-24",
        "seen 01.05.2024", "seen 2024/05/01", "seen May 1, 2024", "seen 1st of May, 2024",
        "seen 1 May 2024", "seen Sept. 3 2024", "seen May 2024", "seen 2024-05",
        "seen 05/2024",
    ],
)  # fmt: skip
def test_strict_dates_keep_only_the_year(text):
    out = redact_text(text, profile="strict").text
    assert out == "seen <DATE_1> 2024", out


def test_strict_dates_edge_cases():
    out = redact_text("Visit on May 3 and 3 June. Born in 1930, seen 1931-02-03.", profile="strict")
    assert out.text == "Visit on <DATE_1> and <DATE_2>. Born in <DATE_4>, seen <DATE_3>."
    keep = "Option 5 may work. Version 10.0.19045.3803, ratio 3/4, part 12-34."
    assert redact_text(keep, profile="strict").text == keep
    assert "<DATE" not in redact_text("part 12-34-56", profile="strict").text  # not a date
    # the same day in two formats is one placeholder
    same = redact_text("2024-05-01 or May 1, 2024", profile="strict").text
    assert same == "<DATE_1> 2024 or <DATE_1> 2024"


def test_zip_codes_and_cities():
    out = redact_text(
        "Ship to Springfield, IL 62704 or zip code 03601. Order 12345 units.", profile="strict"
    ).text
    assert "IL 627XX" in out and "000XX" in out  # 036 is a restricted 3-digit ZIP
    assert "Springfield" not in out and "<LOCATION_1>" in out
    assert "Order 12345 units" in out  # a bare 5-digit number is not a ZIP


def test_ages_over_89_only():
    out = redact_text(
        "A 92 year old, aged 101, age: 95, a 45-year-old and 30 yo.", profile="strict"
    ).text
    assert out == "A 90+ year old, aged 90+, age: 90+, a 45-year-old and 30 yo."
    assert redact_text(out, profile="strict").text == out  # idempotent


def test_record_vehicle_and_device_identifiers():
    text = (
        "MRN: 7734120, member number ABC12345, group no 55821, acct 00918273, "
        "DEA # AB1234563, NPI 1234567893, certificate 99812, plate ABC 1234, "
        "S/N C02XK1ZZJGH5, IMEI: 356938035643809, VIN 1M8GDM9AXKP042788."
    )
    r = redact_text(text, profile="strict")
    for kind, n in [("MRN", 1), ("HEALTH_PLAN", 2), ("ACCOUNT", 1), ("LICENSE", 3),
                    ("VEHICLE", 2), ("DEVICE", 2)]:  # fmt: skip
        assert r.counts_by_type.get(kind) == n, (kind, r.counts_by_type)
    assert not any(ch.isdigit() for ch in PLACEHOLDER_RE.sub("", r.text))  # no digits left


def test_urls_and_ips_all_in_strict_only():
    std = redact_text(HELPDESK).text
    strict = redact_text(HELPDESK, profile="strict").text
    assert "192.168.1.1" in std and "https://intranet.example.com/kb" in std
    assert "192.168.1.1" not in strict and "intranet" not in strict
    assert "<IP_1>" in strict and "<URL_1>" in strict


def test_long_id_tokens_and_their_exceptions():
    out = redact_text(
        "Ref 48213377, token A1B2C3D4E5F6, error 0x80070005, cost $1250000, v10.0.19045.",
        profile="strict",
    ).text
    assert out.startswith("Ref <ID_1>, token <ID_2>,")
    assert "0x80070005" in out and "$1250000" in out and "10.0.19045" in out


def test_unknown_profile_raises():
    with pytest.raises(ValueError):
        redact_text("x", profile="paranoid")  # type: ignore[arg-type]


# ---- keep patterns ---------------------------------------------------------------------


def test_default_keep_patterns_preserve_record_keys():
    out = redact_text(HELPDESK, profile="strict").text
    assert "HD-1001" in out and "PRN-302" in out and "0x80070005" in out
    assert "Tom" not in out and "<DATE_1> 2024" in out
    # health-identifier prefixes are not record keys
    assert "MRN-1234567" not in redact_text("see MRN-1234567", profile="strict").text
    assert "HD-100123" in redact_text("asset HD-100123", profile="strict").text
    assert DEFAULT_KEEP_PATTERNS  # documented default exists


def test_custom_and_empty_keep_patterns():
    text = "asset HD-100123 and KB5034441"
    none = redact_text(text, profile="strict", keep_patterns=()).text
    assert "HD-100123" not in none and "KB5034441" not in none
    custom = redact_text(text, profile="strict", keep_patterns=[r"\bKB\d+\b"]).text
    assert "KB5034441" in custom and "HD-100123" not in custom
    # keep patterns also apply in standard when given explicitly
    std = redact_text("order ORD-12345", pseudonymize_ids=True, keep_patterns=[r"ORD-\d+"])
    assert std.text == "order ORD-12345"


# ---- conversations ---------------------------------------------------------------------


def _conv() -> Conversation:
    return Conversation(
        id="T-1",
        turns=[
            Turn(
                "customer",
                "Hi, I'm Rosa Diaz, DOB 1/2/1950. My pump stopped on 3/4/2024.",
                speaker="Rosa",
            ),  # fmt: skip
            Turn(
                "agent",
                "Thanks Rosa. Is the serial number SN-48213 or 7731AB2209?",
                speaker="Agent Lee",
            ),  # fmt: skip
            Turn("customer", "The second. It broke March 4, 2024 at home in Austin, TX 78701."),
        ],
    )


def test_strict_conversation_placeholders_are_consistent():
    red, counts = redact_conversation(_conv(), profile="strict")
    texts = [t.text for t in red.turns]
    assert "Rosa" not in " ".join(texts) and red.turns[0].speaker == "<NAME_1>"
    assert "<DATE_1> 2024" in texts[0] and "<DATE_1> 2024" in texts[2]  # same day
    assert "SN-48213" in texts[1]  # matches the default keep pattern
    assert "7731AB2209" not in texts[1]
    assert "TX 787XX" in texts[2] and "Austin" not in texts[2]
    assert red.meta["redaction_profile"] == "strict" and red.meta["redaction_ner"] is False
    assert counts["DOB"] == 1


def test_redact_conversations_passes_profile():
    (red,), totals = redact_conversations([_conv()], profile="strict")
    assert totals["ZIP"] == 1 and red.meta["redaction_profile"] == "strict"
    (std,), _ = redact_conversations([_conv()])
    assert "3/4/2024" in std.turns[0].text and std.meta["redaction_profile"] == "standard"


# ---- scan_text ---------------------------------------------------------------------------


def test_scan_text_finds_planted_remnants():
    planted = (
        "<NAME_1> said Sarah Connor called from 555-201-3344 about acct 99812345; "
        "seen 03/14/2024, she is 93 years old, VIN 1HGCM82633A004352, Dr Patel, "
        "email a@b.co, IP 10.1.2.3, https://x.example.org/a, 221 Baker Street."
    )
    kinds = {(f.kind, f.text) for f in scan_text(planted, "strict")}
    for expected in [("NAME", "Sarah Connor"), ("PHONE", "555-201-3344"), ("DATE", "03/14/2024"),
                     ("AGE", "93"), ("VEHICLE", "1HGCM82633A004352"), ("NAME", "Patel"),
                     ("EMAIL", "a@b.co"), ("IP", "10.1.2.3"), ("URL", "https://x.example.org/a")]:  # fmt: skip
        assert expected in kinds, expected
    assert any(f.kind == "ACCOUNT" or f.text == "99812345" for f in scan_text(planted, "strict"))
    std_kinds = {f.kind for f in scan_text(planted, "standard")}
    assert "IP" not in std_kinds and "DATE" not in std_kinds  # private IPs/dates are fine there
    assert all("<NAME_1>" not in f.text for f in scan_text(planted, "strict"))


def test_scan_text_quiet_on_clean_text():
    clean = "Thanks for waiting. Please restart the router and check the Windows update."
    assert scan_text(clean, "strict") == []
    assert scan_text("Ticket PRN-302 for HD-1001.", "strict") == []


# ---- sanitize_json_strings -------------------------------------------------------------


def _task_record() -> dict:
    return {
        "id": "01-router-offline",
        "tier": 1,
        "customer_id": "C-1101",
        "goal": "Hi, Priya Shah here. Router D-3101 is offline since 2024-05-01, call 555-201-3344.",
        "state": {
            "customers": [
                {
                    "customer_id": "C-1101",
                    "name": "Priya Shah",
                    "email": "priya@example.net",
                    "status": "active",
                },  # fmt: skip
            ],
            "subscriptions": [
                {
                    "subscription_id": "S-2101",
                    "customer_id": "C-1101",
                    "address": "13 Hillcrest Road",
                    "started_on": "2024-05-01",
                },  # fmt: skip
            ],
            "devices": [{"device_id": "D-3101", "customer_id": "C-1101", "status": "offline"}],
        },
        "customer_answers": {"router_lights": "blinking orange"},
        "success": [
            {
                "check": "field_equals",
                "collection": "devices",
                "key": "D-3101",
                "field": "status",
                "value": "online",
            },
            {"check": "resolution", "category": "connectivity_restored"},
        ],  # fmt: skip
        "reference": [{"tool": "reboot_device", "args": {"device_id": "D-3101"}}],
        "source": {"conversation": "T-1"},
        "notes": "",
    }


def test_sanitize_json_strings_task_record():
    rec = _task_record()
    original = copy.deepcopy(rec)
    out, counts = sanitize_json_strings(rec, "strict")
    assert rec == original  # input not modified
    assert set(out) == set(rec) and set(out["state"]) == set(rec["state"])  # keys kept
    cust = out["state"]["customers"][0]
    assert cust["name"] == "<NAME_1>" and "Priya" not in out["goal"]
    assert out["goal"].startswith("Hi, <NAME_1> here.")  # one session: same placeholder
    assert cust["email"] == "<EMAIL_1>" and cust["customer_id"] == "C-1101"
    assert out["state"]["subscriptions"][0]["address"] == "<ADDRESS_1>"
    assert out["state"]["subscriptions"][0]["started_on"] == "<DATE_1> 2024"
    assert out["reference"] == rec["reference"] and out["success"] == rec["success"]
    assert out["tier"] == 1 and counts["NAME"] >= 2 and counts["PHONE"] == 1
    TaskRecord(**out)  # still a valid record
    # skip_keys leaves fields verbatim
    out2, _ = sanitize_json_strings(rec, "strict", skip_keys=("started_on",))
    assert out2["state"]["subscriptions"][0]["started_on"] == "2024-05-01"


def test_sanitize_json_strings_chat_example():
    chat = {
        "messages": [
            {"role": "system", "content": "You are a support agent."},
            {"role": "user", "content": "My name is Ana Souza, MRN 00482913, DOB 1/2/1950."},
            {"role": "assistant", "content": "Thanks Ana, I see ticket PRN-302."},
        ]
    }
    out, counts = sanitize_json_strings(chat, "strict")
    msgs = out["messages"]
    assert [m["role"] for m in msgs] == ["system", "user", "assistant"]
    assert msgs[0]["content"] == "You are a support agent."
    assert msgs[1]["content"] == "My name is <NAME_1>, MRN <MRN_1>, DOB <DOB_1>."
    assert msgs[2]["content"] == "Thanks <NAME_1>, I see ticket PRN-302."
    assert counts == {"NAME": 2, "MRN": 1, "DOB": 1}
    std, _ = sanitize_json_strings(chat)  # standard profile: MRN kept
    assert "00482913" in std["messages"][1]["content"] and "Ana" not in json.dumps(std)


# ---- audit report ----------------------------------------------------------------------


def test_report_never_contains_originals_and_lists_remnants():
    convs = [
        _conv(),
        Conversation("T-2", [Turn("customer", "Sarah Connor needs a reset, email s@c.com")]),
    ]
    redacted, report = audit(convs, profile="standard", sample_size=10)
    dumped = json.dumps(report)
    for secret in ("Rosa", "1/2/1950", "s@c.com"):
        assert secret not in dumped
    assert report["profile"] == "standard" and report["conversations"] == 2
    assert report["counts_by_type"]["EMAIL"] == 1 and report["total_redactions"] > 0
    assert report["ner"] == {"requested": False, "available": False, "model": None, "reason": ""}
    assert report["samples"] and all("<" in s["text"] for s in report["samples"])
    # "Sarah Connor" has no cue: standard regex misses it, the scan flags it
    assert any(r["text"] == "Sarah Connor" for r in report["remnants"])
    assert report["remnant_counts"]["NAME"] >= 1
    # the report can also be built later from stored redacted conversations
    again = build_redaction_report(redacted, profile="standard")
    assert again["counts_by_type"] == report["counts_by_type"]


# ---- NER layer -------------------------------------------------------------------------


def test_ner_fallback_when_model_missing(monkeypatch):
    monkeypatch.setenv(ner_layer.MODEL_ENV_VAR, "no_such_spacy_model_xyz")
    status = ner_layer.ner_available()
    assert not status and status.model == "no_such_spacy_model_xyz" and status.reason
    assert ner_layer.analyze_texts(["Sarah lives in Boston."]) == [[]]
    r = redact_text("my name is Jane, email j@x.com", ner=True, profile="strict")
    assert r.text == "my name is <NAME_1>, email <EMAIL_1>"  # regex-only, no crash
    red, _ = redact_conversation(_conv(), ner=True)
    assert red.meta["redaction_ner"] is False
    _, report = audit([_conv()], ner=True)
    assert report["ner"]["requested"] and not report["ner"]["available"]


@pytest.fixture
def ner_ready():
    pytest.importorskip("presidio_analyzer")
    status = ner_layer.ner_available()
    if not status:
        pytest.skip(f"NER unavailable: {status.reason}")
    return status


def test_ner_finds_names_and_places_without_cues(ner_ready):
    text = "Sarah said the pain started in Boston. Later Sarah called again."
    assert redact_text(text, profile="strict").text == text  # regex alone misses both
    r = redact_text(text, profile="strict", ner=True)
    assert "Sarah" not in r.text and "Boston" not in r.text
    assert r.text.count("<NAME_1>") == 2 and "<LOCATION_1>" in r.text


def test_ner_merges_with_regex_placeholders(ner_ready):
    conv = Conversation(
        "c",
        [
            Turn("customer", "My name is Maria Gonzalez and my router is broken."),
            Turn("agent", "Maria Gonzalez, I spoke with Alan Whitfield about ticket PRN-302."),
        ],
    )
    (red,), totals = redact_conversations([conv], profile="strict", ner=True)
    text = " ".join(t.text for t in red.turns)
    assert "Maria" not in text and "Whitfield" not in text
    assert red.turns[1].text.startswith("<NAME_1>, I spoke with <NAME_2>")
    assert "PRN-302" in text  # NER sometimes calls record keys places; keep patterns win
    assert red.meta["redaction_ner"] is True and totals["NAME"] >= 3


def test_ner_strict_dates_and_json(ner_ready):
    r = redact_text("It started the third of March 2024.", profile="strict", ner=True)
    assert "March" not in r.text and "2024" in r.text
    out, _ = sanitize_json_strings(
        {"goal": "Felix Braun from Docklands called."}, "strict", ner=True
    )
    assert "Felix" not in out["goal"] and "Docklands" not in out["goal"]

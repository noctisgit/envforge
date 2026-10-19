"""Offline tests for the privacy tiers: profile "off", surrogates, the tier-2 false-positive
filter, risk triage, tier-2 integration (with a fake model), and the leak benchmark maths.

Only the tests marked `slow` load the real GLiNER model (and skip if it is not installed):
    uv run pytest tests/test_privacy_tiers.py -m slow
"""

from __future__ import annotations

import gzip
import ipaddress
import json
import pickle
import re

import pytest

from envforge.builder import pii_bench, pii_model
from envforge.builder.ingest import Conversation, Turn
from envforge.builder.pii_model import Span, filter_spans
from envforge.builder.redact import (
    COMMON_WORDS,
    PLACEHOLDER_RE,
    SURROGATE_FIRST_NAMES,
    SURROGATE_LAST_NAMES,
    Redactor,
    luhn_ok,
    redact_batch,
    redact_conversation,
    redact_conversations,
    redact_text,
    risk,
    sanitize_json_strings,
)

SALT = b"test-salt"


def conv_of(*pairs: tuple[str, str], cid: str = "c1") -> Conversation:
    return Conversation(id=cid, turns=[Turn(role=r, text=t) for r, t in pairs])


ABCD_LIKE = conv_of(
    ("agent", "sure, may I have your name please?"),
    ("customer", "Crystal Minh"),
    ("agent", "Thanks Crystal! And your username and email?"),
    ("customer", "cminh730 cminh730@email.com"),
    ("agent", "Account found for crystal minh, phone (977) 625-2661 on file. Thanks, Crystal."),
    ("customer", "My card is 4242 4242 4242 4242 and my IP is 8.8.8.8"),
)
SECRETS = ["crystal", "minh", "cminh730", "email.com", "977", "625-2661", "4242", "8.8.8.8"]


def text_of(conv: Conversation) -> str:
    return "\n".join(t.text for t in conv.turns)


# ---- profile "off" ------------------------------------------------------------------------


def test_off_profile_returns_text_unchanged_but_marks_the_conversation():
    out, counts = redact_conversation(ABCD_LIKE, profile="off")
    assert [t.text for t in out.turns] == [t.text for t in ABCD_LIKE.turns]
    assert counts == {}
    assert out.meta["redacted"] is True and out.meta["redaction_profile"] == "off"
    assert out is not ABCD_LIKE and "redacted" not in ABCD_LIKE.meta  # input untouched
    assert redact_text("mail me at a@b.com", profile="off").text == "mail me at a@b.com"


def test_off_profile_in_batches_and_json():
    convs, counts = redact_conversations([ABCD_LIKE], profile="off", tier2="auto")
    assert counts == {} and convs[0].meta["redaction_profile"] == "off"
    assert convs[0].meta["redaction_tier2"] is False
    obj = {"customer_name": "Priya Shah", "email": "p@corp.com"}
    same, counts = sanitize_json_strings(obj, "off")
    assert same is obj and counts == {}


def test_unknown_profile_still_rejected():
    with pytest.raises(ValueError):
        redact_conversation(ABCD_LIKE, profile="loose")  # type: ignore[arg-type]


# ---- surrogates ----------------------------------------------------------------------------


def test_surrogates_replace_everything_with_consistent_fakes():
    out, counts = redact_conversation(ABCD_LIKE, surrogates=True, salt=SALT)
    text = text_of(out)
    for secret in SECRETS:
        assert secret not in text.lower(), secret
    assert not PLACEHOLDER_RE.search(text)  # natural text, no <NAME_1>
    assert out.meta["redaction_surrogates"] is True
    # counts are still by type, exactly as with placeholders
    plain, plain_counts = redact_conversation(ABCD_LIKE, salt=SALT)
    assert counts == plain_counts and counts["NAME"] >= 4 and counts["USERNAME"] >= 1

    full = out.turns[1].text  # "Crystal Minh" -> "<First> <Last>"
    first, last = full.split()
    assert first in SURROGATE_FIRST_NAMES and last in SURROGATE_LAST_NAMES
    assert out.turns[2].text.startswith(f"Thanks {first}!")  # first-name-only mention
    assert f"for {first.lower()} {last.lower()}," in out.turns[4].text  # case kept
    assert out.turns[4].text.endswith(f"Thanks, {first}.")


def test_surrogate_formats():
    out, _ = redact_conversation(ABCD_LIKE, surrogates=True, salt=SALT)
    first, last = out.turns[1].text.split()
    username, email = out.turns[3].text.split()
    # username: surrogate name + new digits, same number of digits as "cminh730"
    assert re.fullmatch(rf"{first[0].lower()}{last.lower()}\d{{3}}", username)
    assert username != "cminh730"
    # the email built on the same handle gets the same handle
    assert re.fullmatch(rf"{re.escape(username)}@example\.(com|org|net)", email)
    # phone keeps its format and is in the fictional 555-01xx range
    assert re.search(r"\(\d{3}\) 555-01\d\d", out.turns[4].text)
    card = re.search(r"\d{4} \d{4} \d{4} \d{4}", out.turns[5].text).group(0)
    assert card.startswith("4111") and luhn_ok(card.replace(" ", ""))
    ip = re.search(r"\d+\.\d+\.\d+\.\d+", out.turns[5].text).group(0)
    assert any(
        ipaddress.ip_address(ip) in ipaddress.ip_network(net)
        for net in ("192.0.2.0/24", "198.51.100.0/24", "203.0.113.0/24")
    )


def test_surrogates_are_deterministic_per_salt_and_differ_across_salts():
    a, _ = redact_conversation(ABCD_LIKE, surrogates=True, salt=SALT)
    b, _ = redact_conversation(ABCD_LIKE, surrogates=True, salt=SALT)
    assert text_of(a) == text_of(b)
    others = {
        text_of(redact_conversation(ABCD_LIKE, surrogates=True, salt=bytes([i]))[0])
        for i in range(5)
    }
    assert len(others) > 1  # a new salt (the default: random per conversation) re-rolls


def test_surrogates_never_equal_originals_and_never_collide():
    emails = [f"user{i}@corp.com" for i in range(30)]
    phones = [f"(212) 555-01{i:02d}" for i in range(30)]  # originals already in the fake range
    r = redact_text("; ".join(emails + phones), surrogates=True, salt=SALT)
    fakes = r.text.split("; ")
    fake_emails, fake_phones = fakes[:30], "; ".join(fakes[30:])
    assert len(set(fake_emails)) == 30 and not set(fake_emails) & set(emails)
    fake_phone_list = re.findall(r"\(\d{3}\) \d{3}-\d{4}", fake_phones)
    assert len(set(fake_phone_list)) == 30 and not set(fake_phone_list) & set(phones)
    assert r.counts_by_type == {"EMAIL": 30, "PHONE": 30}


def test_surrogate_ids_keep_their_shape_and_dates_keep_placeholders():
    r = redact_text(
        "order ABC-123456 for acct 99812345 on March 3, 2024",
        profile="strict",
        keep_patterns=(),
        surrogates=True,
        salt=SALT,
    )
    assert "ABC-123456" not in r.text and re.search(r"\b[A-Z]{3}-\d{6}\b", r.text)
    assert "99812345" not in r.text and re.search(r"acct \d{8}\b", r.text)
    assert re.search(r"<DATE_1> 2024", r.text)  # dates keep the year placeholder rule


def test_surrogates_respect_keep_patterns_and_strict_zip3():
    r = redact_text(
        "Ticket HD-1001: I live at 12 Baker Street, Boston, MA 02139.",
        profile="strict",
        surrogates=True,
        salt=SALT,
    )
    assert "HD-1001" in r.text  # strict default keep pattern
    assert "Baker" not in r.text and "02139" not in r.text
    assert re.search(r"\b[A-Z]{2} \d{3}XX\b", r.text)  # fake ZIP, still only 3 digits


def test_surrogates_with_speakers_and_the_redactor_wrapper():
    conv = Conversation(
        id="s1",
        turns=[
            Turn(role="customer", text="Hi, my name is Jane Doe", speaker="Jane"),
            Turn(role="agent", text="Hello Jane", speaker="Agent Mike"),
        ],
    )
    out, _ = Redactor(surrogates=True)(conv)
    first = out.turns[0].speaker
    assert first in SURROGATE_FIRST_NAMES and out.turns[1].text == f"Hello {first}"
    assert out.turns[1].speaker.startswith("Agent ") and "Mike" not in out.turns[1].speaker


def test_surrogate_name_lists_are_large_distinct_and_not_common_words():
    for names in (SURROGATE_FIRST_NAMES, SURROGATE_LAST_NAMES):
        assert len(names) >= 150 and len(set(names)) == len(names)
        assert not {n.lower() for n in names} & COMMON_WORDS


def test_sanitize_json_strings_with_surrogates():
    obj = {
        "customer": {"customer_name": "Priya Shah", "email": "p.shah@corp.com"},
        "goal": "Priya Shah wants a refund; reply to p.shah@corp.com",
        "messages": [{"role": "user", "content": "Thanks, Priya"}],
    }
    out, counts = sanitize_json_strings(obj, surrogates=True, salt=SALT)
    fake = out["customer"]["customer_name"]
    first, last = fake.split()
    assert first in SURROGATE_FIRST_NAMES and "Priya" not in json.dumps(out)
    assert out["goal"].startswith(f"{fake} wants a refund")
    assert out["messages"][0]["content"] == f"Thanks, {first}"
    assert out["customer"]["email"] == f"{first[0].lower()}.{last.lower()}@example.com" or (
        out["customer"]["email"].startswith(f"{first[0].lower()}.{last.lower()}@example.")
    )
    assert out["customer"]["email"] in out["goal"]  # one session, one surrogate
    assert counts["NAME"] == 3 and counts["EMAIL"] == 2
    again, _ = sanitize_json_strings(obj, surrogates=True, salt=SALT)
    assert again == out


# ---- tier-2 false-positive filter -----------------------------------------------------------


def span(text: str, value: str, label: str, score: float = 0.9) -> Span:
    start = text.index(value)
    return Span(start, start + len(value), label, pii_model.LABEL_KINDS[label], score, value)


# (sentence, flagged value, label) — false positives the model really produced on ABCD
FALSE_POSITIVES = [
    ("I need help with my order", "I", "person"),
    ("Can you check it for you?", "you", "person"),
    ("Agent here, how can I help?", "Agent", "person"),
    ("What is your full name?", "full name", "person"),
    ("What is your username?", "username", "username"),
    ("I can't get into my account", "my account", "username"),
    ("Hello NAME_1, welcome back", "NAME_1", "username"),
    ("Hello <NAME_1>, welcome back", "NAME_1", "person"),
    ("What's your email address?", "email address", "email"),
    ("And your phone number?", "phone number", "phone number"),
    ("I need your account ID", "account ID", "account number"),
    ("I'm a silver member", "silver", "account number"),
    ("What's the zip code?", "zip code", "address"),
    ("Please give me the new address", "new address", "address"),
    ("I'm a bronze", "I'm a bronze", "person"),
    ("ok", "ok", "person"),
]


@pytest.mark.parametrize(("text", "value", "label"), FALSE_POSITIVES)
def test_filter_drops_known_false_positives(text, value, label):
    assert filter_spans(text, [span(text, value, label)], common_words=COMMON_WORDS) == []


def test_filter_keeps_true_positives():
    cases = [
        ("Crystal Minh", "Crystal Minh", "person", {}),
        ("crystal minh", "crystal minh", "person", {"after_name_question": True}),
        ("thanks again crystal", "crystal", "person", {"known_names": {"crystal"}}),
        ("Username: cminh730", "cminh730", "username", {}),
        ("it's j.doe on the forum", "j.doe", "username", {}),
        ("mail cminh730@email.com", "cminh730@email.com", "email", {}),
        ("call (977) 625-2661", "(977) 625-2661", "phone number", {}),
        ("acct 3348917502", "3348917502", "account number", {}),
        ("ship to 6821 1st ave", "6821 1st ave", "address", {}),
    ]
    for text, value, label, kw in cases:
        kept = filter_spans(text, [span(text, value, label)], common_words=COMMON_WORDS, **kw)
        assert [s.text for s in kept] == [value], (text, value)


def test_filter_trims_filler_and_relabels_handles():
    text = "hi I'm Crystal, my login is davidwilliams617"
    kept = filter_spans(
        text,
        [span(text, "I'm Crystal", "person"), span(text, "davidwilliams617", "person")],
        common_words=COMMON_WORDS,
    )
    assert [(s.text, s.kind) for s in kept] == [
        ("Crystal", "NAME"),
        ("davidwilliams617", "USERNAME"),
    ]
    # a lower-case name with no context is not trusted
    assert filter_spans("yes crystal", [span("yes crystal", "crystal", "person")]) == []


def test_filter_respects_keep_patterns():
    text = "Ticket HD-1001 is open"
    keep = [re.compile(r"HD-\d+")]
    assert filter_spans(text, [span(text, "HD-1001", "account number")], keep=keep) == []


# ---- triage --------------------------------------------------------------------------------


def test_risk_flags_an_unredacted_name_answer():
    conv = conv_of(
        ("agent", "May I get your full name?"),
        ("customer", "albert sanders"),
        ("agent", "And your username?"),
        ("customer", "asanders1"),
    )
    score, reasons = risk(conv, conv, ner_used=True)  # "tier 1 changed nothing"
    assert "name_answer_unredacted" in reasons and "username_answer_unredacted" in reasons
    assert "letter_digit_token" in reasons and score >= 2.0


def test_risk_is_low_for_a_cleanly_redacted_conversation():
    out, _ = redact_conversation(ABCD_LIKE, salt=SALT)
    score, reasons = risk(ABCD_LIKE, out, ner_used=True)
    assert reasons == [] and score == 0.0
    score, reasons = risk(ABCD_LIKE, out, ner_used=False)
    assert reasons == ["ner_off"] and score < 1.0


def test_risk_ignores_surrogates_and_counts_remnants():
    out, _ = redact_conversation(ABCD_LIKE, surrogates=True, salt=SALT)
    assert risk(ABCD_LIKE, out, ner_used=True) == (0.0, [])  # fake names are not remnants
    conv = conv_of(("customer", "Joseph Banter here, reach me at jb@x.io, order 12345678"))
    score, reasons = risk(conv, conv, ner_used=True)
    assert {"capitalised_pair", "at_sign_remaining", "long_digit_run"} <= set(reasons)


# ---- tier 2 integration with a fake model ------------------------------------------------


@pytest.fixture
def fake_model(monkeypatch):
    """Replace GLiNER with a tiny stand-in that flags a few known strings."""
    flags = {"Albert Sanders": "person", "zx99": "username", "Gold": "account number"}
    calls: list[list[str]] = []

    def detect(texts, **_):
        calls.append(list(texts))
        out = []
        for text in texts:
            spans = []
            for value, label in flags.items():
                for m in re.finditer(re.escape(value), text):
                    spans.append(
                        Span(m.start(), m.end(), label, pii_model.LABEL_KINDS[label], 0.9, value)
                    )
            out.append(spans)
        return out

    monkeypatch.setattr(pii_model, "detect", detect)
    monkeypatch.setattr(
        pii_model, "pii_model_available", lambda model=None: pii_model.Status(True, "fake")
    )
    return calls


RISKY = conv_of(
    ("agent", "Hi! How can I help?"),
    ("customer", "it's for Albert Sanders, Gold member, I can't log in"),
    ("agent", "Sorry Albert Sanders, what's your login? It might be zx99"),
    cid="risky",
)
CLEAN = conv_of(("customer", "my router keeps dropping wifi"), cid="clean")


def test_tier2_auto_only_runs_on_risky_conversations(fake_model):
    batch = redact_batch([RISKY, CLEAN], tier2="auto", ner=False, tier2_threshold=0.5)
    risky, clean = batch.conversations
    assert risky.meta["redaction_tier2"] is True and clean.meta["redaction_tier2"] is False
    assert batch.tier2_sent == 1 and batch.tier2_turns >= 1
    assert "albert" not in text_of(risky).lower() and "zx99" not in text_of(risky)
    assert "Gold" in text_of(risky)  # account numbers need a digit, and standard keeps IDs
    assert risky.meta["redaction_risk"] >= 0.5 and risky.meta["redaction_risk_reasons"]
    # one placeholder for the name everywhere: no double replacement
    assert text_of(risky).count("<NAME_1>") == 2 and "<NAME_2>" not in text_of(risky)
    assert "<USERNAME_1>" in text_of(risky)
    # only the risky conversation's turns went to the model, in one call
    assert len(fake_model) == 1 and not any("router" in t for t in fake_model[0])


def test_tier2_all_and_surrogates(fake_model):
    convs, counts = redact_conversations([RISKY, CLEAN], tier2="all", surrogates=True)
    assert all(c.meta["redaction_tier2"] for c in convs)
    assert "albert" not in text_of(convs[0]).lower() and counts["NAME"] == 2
    assert not PLACEHOLDER_RE.search(text_of(convs[0]))


def test_tier2_unavailable_keeps_tier1(monkeypatch):
    monkeypatch.setattr(
        pii_model, "pii_model_available", lambda model=None: pii_model.Status(False, "x", "nope")
    )
    batch = redact_batch([RISKY], tier2="all")
    out = batch.conversations[0]
    assert (
        out.meta["redaction_tier2"] is False and out.meta["redaction_tier2_unavailable"] == "nope"
    )
    assert batch.tier2_status == {"available": False, "model": "x", "reason": "nope"}


def test_redactor_is_picklable_and_per_conversation(fake_model):
    redact = Redactor(profile="standard", surrogates=True, tier2="auto", tier2_threshold=0.5)
    clone = pickle.loads(pickle.dumps(redact))
    assert clone == redact
    out, counts = clone(RISKY)
    assert out.meta["redacted"] and out.meta["redaction_tier2"] and counts["NAME"] >= 2


def test_ingest_with_redactor_in_a_process_pool(tmp_path):
    from envforge.builder import corpus as C

    path = tmp_path / "chats.jsonl"
    rows = [
        {
            "conversation_id": f"c{i}",
            "role": "customer",
            "text": f"my name is Jane Doe{i} mail j{i}@x.com",
        }
        for i in range(6)
    ]
    path.write_text("\n".join(json.dumps(r) for r in rows))
    with C.CorpusStore(tmp_path / "corpus.db") as store:
        C.ingest_file_to_store(path, store, "u1", redact=Redactor(surrogates=True), workers=2)
        texts = [c.turns[0].text for c in store.get(store.iter_ids(unique_only=False))]
    assert len(texts) == 6 and not any("@x.com" in t or "Jane" in t for t in texts)


# ---- benchmark maths -----------------------------------------------------------------------


def test_wilson_interval():
    lo, hi = pii_bench.wilson_interval(0, 300)
    assert lo == 0.0 and 0.012 < hi < 0.013  # "none in 300" -> below ~1.3 %
    lo, hi = pii_bench.wilson_interval(5, 100)
    assert abs(lo - 0.0215) < 1e-3 and abs(hi - 0.1118) < 1e-3
    lo, hi = pii_bench.wilson_interval(10, 10)
    assert hi == 1.0 and 0.72 < lo < 0.73
    assert pii_bench.wilson_interval(0, 0) == (0.0, 1.0)


def test_sample_for_review_is_deterministic():
    ids = [f"c{i}" for i in range(100)]
    a = pii_bench.sample_for_review(ids, 10, seed=3)
    assert a == pii_bench.sample_for_review(reversed(ids), 10, seed=3)
    assert len(set(a)) == 10 and a != pii_bench.sample_for_review(ids, 10, seed=4)
    assert sorted(pii_bench.sample_for_review(ids[:5], 10)) == ids[:5]


def test_leak_report_math():
    truth = {
        "1": {"name": ["crystal", "minh"], "phone": ["(977) 625-2661"], "username": ["cminh730"]},
        "2": {"name": ["joseph", "banter"], "email": ["jb@x.io"]},
    }
    redacted = [
        conv_of(("customer", "<NAME_1> here, call 977.625.2661"), cid="1"),  # phone leaks
        conv_of(("customer", "Joseph <NAME_1> / JB@X.IO"), cid="2"),  # name part + email
        conv_of(("customer", "not in the ground truth"), cid="3"),
    ]
    rep = pii_bench.leak_report(redacted, truth)
    assert rep["conversations"] == 2
    assert rep["overall"]["values"] == 7 and rep["overall"]["leaked"] == 3
    assert rep["by_type"]["name"] == {**rep["by_type"]["name"], "values": 4, "leaked": 1}
    assert rep["by_type"]["phone"]["rate"] == 1.0 and rep["by_type"]["username"]["leaked"] == 0
    assert rep["conversations_with_leak"]["leaked"] == 2
    assert {e["type"] for e in rep["examples"]} == {"phone", "name", "email"}
    # "minh" inside a longer word is not a leak
    assert (
        pii_bench.leak_report([conv_of(("customer", "minhas"), cid="1")], truth)["overall"][
            "leaked"
        ]
        == 0
    )


def test_abcd_loader_json_and_gz(tmp_path):
    data = {
        "train": [
            {
                "convo_id": 7,
                "scenario": {
                    "personal": {
                        "customer_name": "crystal minh",
                        "email": "cminh730@email.com",
                        "phone": "(977) 625-2661",
                        "username": "cminh730",
                    }
                },
                "original": [
                    ["agent", "your name?"],
                    ["customer", "Crystal Minh"],
                    ["action", "x"],
                ],
            }
        ],
        "dev": [],
        "test": [],
    }
    plain = tmp_path / "abcd.json"
    plain.write_text(json.dumps(data))
    packed = tmp_path / "abcd.json.gz"
    with gzip.open(packed, "wt") as fh:
        json.dump(data, fh)
    for path in (plain, packed):
        truth = pii_bench.abcd_ground_truth(path)
        assert truth == {
            "7": {
                "name": ["crystal", "minh"],
                "email": ["cminh730@email.com"],
                "phone": ["(977) 625-2661"],
                "username": ["cminh730"],
            }
        }
        (conv,) = pii_bench.abcd_conversations(path)
        assert [t.role for t in conv.turns] == ["agent", "customer", "system"]
    redacted, _ = redact_conversations(pii_bench.abcd_conversations(plain))
    assert pii_bench.leak_report(redacted, truth)["overall"]["leaked"] == 0


def test_parse_mode():
    assert pii_bench.parse_mode("strict+ner+tier2=all+surrogates") == {
        "profile": "strict",
        "ner": True,
        "tier2": "all",
        "surrogates": True,
    }
    assert pii_bench.parse_mode("standard+tier2")["tier2"] == "auto"
    with pytest.raises(ValueError):
        pii_bench.parse_mode("standard+magic")


# ---- the real model (slow, opt-in) -----------------------------------------------------------


@pytest.mark.slow
def test_real_model_detects_and_filter_cleans():
    status = pii_model.pii_model_available()
    if not status:
        pytest.skip(f"tier-2 model unavailable: {status.reason}")
    texts = ["Crystal Minh", "Username: cminh730", "What is your full name and username?"]
    spans = pii_model.detect(texts)
    assert any(s.kind == "NAME" for s in spans[0])
    assert any(s.text == "cminh730" for s in spans[1])
    kept = [
        filter_spans(t, s, common_words=COMMON_WORDS) for t, s in zip(texts, spans, strict=True)
    ]
    assert kept[2] == []  # field labels are not personal data


@pytest.mark.slow
def test_real_tier2_on_a_risky_conversation():
    if not pii_model.pii_model_available():
        pytest.skip("tier-2 model unavailable")
    conv = conv_of(
        ("agent", "Also account id and order id"),
        ("customer", "Chloe Zhang"),
        ("system", "Account has been pulled up for Chloe Zhang."),
    )
    out, _ = redact_conversations([conv], tier2="auto")
    assert out[0].meta["redaction_tier2"] and "Zhang" not in text_of(out[0])

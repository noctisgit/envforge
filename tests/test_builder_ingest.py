"""Offline tests for the builder's ingestion pipeline: parse -> redact -> profile -> extract.

No network: extraction uses a fake ChatModel that replays canned assistant turns.
"""

from __future__ import annotations

import io
import json
from pathlib import Path

import pytest

from envforge.builder import extract as ex
from envforge.builder.ingest import (
    Conversation,
    Mapping,
    Turn,
    _iter_json_array,
    assign_unknown_roles,
    detect_format,
    normalize_role,
    parse_file,
    profile,
)
from envforge.builder.redact import (
    iban_ok,
    luhn_ok,
    redact_conversation,
    redact_conversations,
    redact_text,
)
from envforge.core.env import ToolCall
from envforge.core.trajectory import Usage
from envforge.declarative.spec import EnvSpec
from envforge.models import catalog
from envforge.models.base import AssistantTurn, ChatModel, Message

# ---- fixtures -------------------------------------------------------------------------------

MESSAGE_CSV = """\
Ticket ID,Sender,Message,Created At
T-100,Customer,"Hi, my name is Jane Doe. Order ORD-12345 never arrived. Email me at jane.doe@example.com",2024-05-01T10:00:00
T-100,Support Agent,"Thanks for contacting Acme! Sorry Jane, let me check ORD-12345.",2024-05-01T10:01:00
T-100,Customer,You can also call me on +44 20 7946 0958,2024-05-01T10:02:00
T-200,end-user,My card 4111 1111 1111 1111 was charged twice,2024-05-02T09:00:00
T-200,bot,I can help with that.,2024-05-02T09:00:05
"""

TRANSCRIPT_CSV = """\
chat_id,transcript
A1,"Customer: my internet is down
Agent: Thank you for contacting us. Is the router light red?
Customer: yes it is red"
A2,"User: how do I reset my password?
Support: Click 'forgot password' on the login page."
"""

TXT_LOG = """\
Conversation 1
[10:01] Jane: hi, my wifi keeps dropping
[10:02] Mike: Thank you for contacting NetCo, my name is Mike. Can you restart the router?
[10:05] Jane: done, still dropping
---
Conversation 2
Customer: I want to cancel my subscription SUB-7781
Agent: Sure, I have cancelled it.
It may take a day to show.
Customer: thanks
"""


@pytest.fixture
def tmp(tmp_path: Path):
    def write(name: str, content: str | bytes) -> Path:
        p = tmp_path / name
        if isinstance(content, bytes):
            p.write_bytes(content)
        else:
            p.write_text(content, encoding="utf-8")
        return p

    return write


def _redacted(conv: Conversation) -> Conversation:
    return redact_conversation(conv)[0]


# ---- detection and parsing ------------------------------------------------------------------


def test_csv_message_rows_detected_and_parsed(tmp):
    path = tmp("chats.csv", MESSAGE_CSV)
    guess = detect_format(path)
    assert guess.format == "csv"
    assert guess.mapping.layout == "message_rows"
    assert guess.mapping.conversation_id == "Ticket ID"
    assert guess.mapping.role == "Sender"
    assert guess.mapping.text == "Message"
    assert guess.mapping.timestamp == "Created At"
    assert guess.confidence >= 0.9

    result = parse_file(path)
    assert [c.id for c in result.conversations] == ["T-100", "T-200"]
    t100 = result.conversations[0]
    assert [t.role for t in t100.turns] == ["customer", "agent", "customer"]
    assert t100.turns[0].ts == "2024-05-01T10:00:00"
    assert [t.role for t in result.conversations[1].turns] == ["customer", "agent"]


def test_csv_transcript_rows(tmp):
    path = tmp("transcripts.csv", TRANSCRIPT_CSV)
    guess = detect_format(path)
    assert guess.mapping.layout == "transcript_rows"
    assert guess.mapping.transcript == "transcript"
    assert guess.mapping.conversation_id == "chat_id"
    result = parse_file(path)
    a1, a2 = result.conversations
    assert a1.id == "A1" and len(a1.turns) == 3
    assert [t.role for t in a1.turns] == ["customer", "agent", "customer"]
    assert [t.role for t in a2.turns] == ["customer", "agent"]


def test_mapping_override(tmp):
    path = tmp("odd.csv", "c,s,t\n1,customer,hello\n1,agent,hi there\n2,customer,yo\n")
    guess = detect_format(path)
    assert guess.confidence < 0.5  # nothing matched by name
    mapping = Mapping(layout="message_rows", conversation_id="c", role="s", text="t")
    result = parse_file(path, mapping=mapping)
    assert [c.id for c in result.conversations] == ["1", "2"]
    assert [t.role for t in result.conversations[0].turns] == ["customer", "agent"]
    assert result.mapping is mapping


def test_mapping_override_unknown_column_raises(tmp):
    path = tmp("odd.csv", "c,s,t\n1,customer,hello\n")
    with pytest.raises(ValueError, match="nope"):
        parse_file(path, mapping=Mapping(layout="message_rows", text="nope"))


def test_json_conversations_zendesk_and_intercom(tmp):
    data = [
        {
            "id": "c1",
            "messages": [
                {"role": "user", "content": "Router is blinking"},
                {"role": "assistant", "content": "Please unplug it for 30s"},
            ],
        },
        {  # Zendesk-like
            "id": 77,
            "requester_id": 5,
            "comments": [
                {"author_id": 5, "body": "<p>Refund please</p>", "public": True},
                {"author_id": 9, "body": "Internal: check fraud", "public": False},
                {"author_id": 9, "body": "Refund issued", "public": True},
            ],
        },
        {  # Intercom-like
            "id": "ic-1",
            "source": {"body": "<p>Can't log in</p>", "author": {"type": "user"}},
            "conversation_parts": {
                "conversation_parts": [
                    {"part_type": "comment", "body": "Try resetting", "author": {"type": "admin"}},
                    {"part_type": "assignment", "body": None, "author": {"type": "bot"}},
                ]
            },
        },
    ]
    path = tmp("export.json", json.dumps(data))
    guess = detect_format(path)
    assert guess.format == "json" and guess.mapping.layout == "conversations"
    convs = parse_file(path).conversations
    assert [c.id for c in convs] == ["c1", "77", "ic-1"]
    assert [t.role for t in convs[0].turns] == ["customer", "agent"]
    assert [t.role for t in convs[1].turns] == ["customer", "system", "agent"]
    assert convs[1].turns[0].text == "Refund please"  # HTML stripped
    assert [t.role for t in convs[2].turns] == ["customer", "agent"]


def test_json_wrapped_object(tmp):
    data = {"conversations": [{"chat_id": "x", "turns": [{"sender": "client", "text": "hi"}]}]}
    convs = parse_file(tmp("wrapped.json", json.dumps(data))).conversations
    assert convs[0].id == "x" and convs[0].turns[0].role == "customer"


def test_jsonl_message_rows(tmp):
    rows = [
        {"conversation_id": "a", "author": "customer", "body": "printer jammed", "ts": "1"},
        {"conversation_id": "b", "author": "customer", "body": "no sound"},
        {"conversation_id": "a", "author": "operator", "body": "open tray 2"},
    ]
    path = tmp("msgs.jsonl", "\n".join(json.dumps(r) for r in rows) + "\nnot json\n")
    guess = detect_format(path)
    assert guess.format == "jsonl" and guess.mapping.layout == "message_rows"
    result = parse_file(path)
    a, b = result.conversations
    assert a.id == "a" and [t.role for t in a.turns] == ["customer", "agent"]
    assert b.id == "b"
    assert result.skipped_rows == 1  # the bad line


def test_streaming_json_array_small_chunks():
    items = [{"id": i, "messages": [{"role": "user", "content": "x" * 50}]} for i in range(30)]
    got = list(_iter_json_array(io.StringIO(json.dumps(items)), chunk_size=17))
    assert got == items


def test_txt_multi_conversation_split_and_name_roles(tmp):
    path = tmp("log.txt", TXT_LOG)
    guess = detect_format(path)
    assert guess.format == "txt" and guess.mapping.txt_split == "markers"
    convs = parse_file(path).conversations
    assert [c.id for c in convs] == ["1", "2"]
    first, second = convs
    # Mike greets like support -> agent; Jane -> customer
    assert [(t.speaker, t.role) for t in first.turns] == [
        ("Jane", "customer"),
        ("Mike", "agent"),
        ("Jane", "customer"),
    ]
    assert first.turns[0].ts == "10:01"
    assert first.meta["roles_guessed"] == 3
    # continuation line folded into the previous turn
    assert second.turns[1].text == "Sure, I have cancelled it.\nIt may take a day to show."


def test_txt_blank_line_runs(tmp):
    text = "Customer: a\nAgent: b\n\n\nCustomer: c\nAgent: d\n"
    convs = parse_file(tmp("runs.txt", text)).conversations
    assert len(convs) == 2


def test_encoding_fallback_latin1(tmp):
    raw = "ticket_id,role,text\n1,customer,Caf\xe9 wifi is down\n".encode("latin-1")
    result = parse_file(tmp("latin.csv", raw))
    assert result.format.encoding == "latin-1"
    assert result.conversations[0].turns[0].text == "Café wifi is down"
    assert any("latin-1" in w for w in result.warnings)


def test_utf8_bom(tmp):
    raw = "﻿ticket_id,role,text\n1,customer,hello\n".encode()
    result = parse_file(tmp("bom.csv", raw))
    assert result.format.encoding == "utf-8-sig"
    assert result.mapping.conversation_id == "ticket_id"


def test_truncation_reported(tmp):
    rows = "".join(f"1,customer,msg {i}\n" for i in range(10))
    result = parse_file(tmp("long.csv", "ticket_id,role,text\n" + rows), max_turns=4)
    assert len(result.conversations[0].turns) == 4
    assert result.truncated == 1 and result.conversations[0].meta["truncated"]


def test_role_normalisation():
    assert normalize_role("Customer") == "customer"
    assert normalize_role("end-user") == "customer"
    assert normalize_role("Caller (Jane)") == "customer"
    assert normalize_role("Support Agent") == "agent"
    assert normalize_role("Customer Support") == "agent"
    assert normalize_role("rep_42") == "agent"
    assert normalize_role("BOT") == "agent"
    assert normalize_role("system") == "system"
    assert normalize_role("Jane") == "unknown"
    assert normalize_role("Jane", {"jane": "agent"}) == "agent"


def test_first_speaker_is_customer_without_greeting():
    turns = [
        Turn("unknown", "my bill is wrong", speaker="Ann"),
        Turn("unknown", "ok", speaker="Bo"),
    ]
    assert assign_unknown_roles(turns) == 2
    assert [t.role for t in turns] == ["customer", "agent"]


def test_profile(tmp):
    convs = parse_file(tmp("chats.csv", MESSAGE_CSV)).conversations
    p = profile(convs)
    assert p["conversations"] == 2 and p["turns"] == 5
    assert p["turns_per_conversation"]["max"] == 3
    assert p["roles"] == {"customer": 3, "agent": 2}
    assert p["sample_ids"] == ["T-100", "T-200"]
    words = dict(p["top_intent_words"])
    assert "charged" in words and "the" not in words


# ---- redaction --------------------------------------------------------------------------------


def test_redaction_types_and_consistency():
    text = (
        "Email jane.doe@example.com or JANE.DOE@example.com. Call +1 (555) 123-4567. "
        "Card 4111-1111-1111-1111. IBAN GB82 WEST 1234 5698 7654 32. SSN 123-45-6789. "
        "My IP is 8.8.8.8, router 192.168.1.1, mac 00:1A:2B:3C:4D:5E, v6 2001:4860:4860::8888. "
        "Reset link https://acme.com/reset?token=abc123 and docs https://acme.com/help. "
        "DOB: 12/03/1985. I live at 221 Baker Street, Apt 4."
    )
    r = redact_text(text)
    out = r.text
    assert "<EMAIL_1>" in out and "<EMAIL_2>" not in out  # same address, case-insensitive
    assert out.count("<EMAIL_1>") == 2
    assert "<PHONE_1>" in out and "555" not in out
    assert "<CARD_1>" in out and "4111" not in out
    assert "<IBAN_1>" in out and "WEST" not in out
    assert "<SSN_1>" in out
    assert "8.8.8.8" not in out and "192.168.1.1" in out  # private IP kept
    assert "<MAC_1>" in out
    assert "2001:4860" not in out
    assert "token=abc123" not in out and "https://acme.com/help" in out
    assert "<DOB_1>" in out and "1985" not in out
    assert "Baker" not in out and "<ADDRESS_1>" in out
    assert r.counts_by_type["EMAIL"] == 2
    assert set(r.mapping_hashes) >= {"<EMAIL_1>", "<CARD_1>"}
    assert "jane" not in json.dumps(r.mapping_hashes)  # hashes only


def test_international_phone_formats():
    for phone in ["+44 20 7946 0958", "0049 30 1234567", "(020) 7946 0958", "555-123-4567"]:
        assert "<PHONE_1>" in redact_text(f"reach me at {phone} please").text, phone
    cued = redact_text("call me on 5551234567").text
    assert "<PHONE_1>" in cued


def test_luhn_and_non_card_numbers_survive():
    assert luhn_ok("4111111111111111")
    assert not luhn_ok("4111111111111112")
    assert iban_ok("GB82WEST12345698765432") and not iban_ok("GB00WEST12345698765432")
    text = "Tracking 4111 1111 1111 1112, order 123456789, date 2024-05-01, ref 1234567890123"
    out = redact_text(text).text
    assert out == text  # no Luhn, no phone formatting, no cue: untouched


def test_order_ids_survive_unless_pseudonymised():
    text = "My order ORD-12345 and account AC998877 (order number 55512) are stuck. Account number is 88812345."
    assert redact_text(text).text == text
    pseudo = redact_text(text + " Again: ORD-12345.", pseudonymize_ids=True).text
    assert "ORD-12345" not in pseudo and "AC998877" not in pseudo
    assert pseudo.count("<ID_1>") == 2  # same id, same placeholder


def test_names_from_cues_and_speakers():
    conv = Conversation(
        id="c1",
        turns=[
            Turn("customer", "Hi, my name is Jane Doe and my router is broken.", speaker="Jane"),
            Turn("agent", "Hi Jane, this is Mike from NetCo. Let me help.", speaker="Agent Mike"),
            Turn("customer", "Thanks, Jane", speaker="Jane"),
            Turn("agent", "Glad to help Jane. Regards, Mike", speaker="Agent Mike"),
        ],
    )
    red, counts = redact_conversation(conv)
    texts = " ".join(t.text for t in red.turns)
    assert "Jane" not in texts and "Mike" not in texts and "Doe" not in texts
    assert red.turns[0].speaker == "<NAME_1>"
    assert red.turns[1].speaker == "Agent <NAME_2>"
    assert "my name is <NAME_1>" in red.turns[0].text
    assert red.turns[1].text.startswith("Hi <NAME_1>, this is <NAME_2>")
    assert red.meta["redacted"] is True and counts["NAME"] >= 6
    assert conv.turns[0].text.startswith("Hi, my name is Jane")  # input untouched


def test_placeholders_restart_per_conversation_and_extra_patterns():
    a = Conversation("a", [Turn("customer", "x@a.com passport X12345678")])
    b = Conversation("b", [Turn("customer", "y@b.com")])
    (ra, rb), totals = redact_conversations([a, b], extra_patterns={"passport": r"\b[A-Z]\d{8}\b"})
    assert ra.turns[0].text == "<EMAIL_1> passport <PASSPORT_1>"
    assert rb.turns[0].text == "<EMAIL_1>"
    assert totals == {"EMAIL": 2, "PASSPORT": 1}


def test_conversation_id_with_pii_is_replaced():
    red, _ = redact_conversation(Conversation("bob@example.com", [Turn("customer", "hi")]))
    assert "@" not in red.id and red.meta["id_redacted"]


# ---- extraction ---------------------------------------------------------------------------------

SPEC = EnvSpec.model_validate(
    {
        "name": "shop-support",
        "title": "Shop support",
        "description": "Order problems for an online shop",
        "agent_role": "You are a support agent.",
        "resolution_categories": ["refunded", "cancelled", "info_given"],
        "answer_topics": ["order_number"],
        "collections": [
            {
                "name": "orders",
                "key": "order_id",
                "owner_field": "customer_id",
                "fields": [
                    {"name": "order_id"},
                    {"name": "customer_id"},
                    {"name": "status", "type": "enum", "enum": ["placed", "shipped", "cancelled"]},
                    {"name": "amount", "type": "number"},
                ],
            },
            {
                "name": "refunds",
                "key": "refund_id",
                "owner_field": "customer_id",
                "fields": [
                    {"name": "refund_id"},
                    {"name": "order_id"},
                    {"name": "customer_id"},
                    {"name": "amount", "type": "number"},
                ],
            },
        ],
        "tools": [
            {
                "name": "get_order",
                "description": "Look up an order",
                "category": "observe",
                "params": [{"name": "order_id"}],
                "effect": {"op": "get", "collection": "orders", "key_param": "order_id"},
            },
            {
                "name": "cancel_order",
                "description": "Cancel an order",
                "category": "act",
                "params": [{"name": "order_id"}],
                "effect": {
                    "op": "set",
                    "collection": "orders",
                    "key_param": "order_id",
                    "set": {"status": "cancelled"},
                },
            },
            {
                "name": "ask_customer",
                "description": "Ask the customer",
                "category": "observe",
                "params": [{"name": "topic"}],
                "effect": {"op": "ask_customer", "topic_param": "topic"},
            },
            {
                "name": "resolve",
                "description": "Close the ticket",
                "category": "act",
                "params": [{"name": "category"}, {"name": "note", "required": False}],
                "effect": {"op": "resolve", "category_param": "category", "note_param": "note"},
            },
        ],
    }
)

GOOD_ARGS = {
    "tier": 1,
    "customer_id": "C-1",
    "goal": "Please cancel order ORD-1, I don't need it. <EMAIL_1>",
    "state": {
        "orders": [
            {"order_id": "ORD-1", "customer_id": "C-1", "status": "placed", "amount": "19.99"},
        ],
        "refunds": [],
    },
    "customer_answers": {"order_number": "ORD-1"},
    "success": [
        {
            "check": "field_equals",
            "collection": "orders",
            "key": "ORD-1",
            "field": "status",
            "value": "cancelled",
        },
        {"check": "resolution", "category": "cancelled"},
    ],
    "reference": [
        {"tool": "get_order", "args": {"order_id": "ORD-1"}},
        {"tool": "cancel_order", "args": {"order_id": "ORD-1"}},
        {
            "tool": "resolve",
            "args": {"category": "cancelled", "note": "cancelled at customer request"},
        },
    ],
    "notes": "amount invented",
}


class FakeModel(ChatModel):
    """Replays canned turns and records what it was sent."""

    def __init__(
        self, turns: list[AssistantTurn | Exception], model_id: str = "anthropic:claude-opus-5-5"
    ):
        self.turns = list(turns)
        self.calls: list[tuple[list[Message], list]] = []
        self.model_id = model_id

    def complete(self, messages, tools):
        self.calls.append((list(messages), tools))
        nxt = self.turns.pop(0)
        if isinstance(nxt, Exception):
            raise nxt
        return nxt


def _call(name: str, args: dict, cid: str = "tc1") -> AssistantTurn:
    return AssistantTurn(
        tool_calls=[ToolCall(id=cid, name=name, args=args)],
        usage=Usage(input_tokens=1000, output_tokens=500),
    )


def _conv(cid: str = "T-1", redacted: bool = True) -> Conversation:
    conv = Conversation(
        id=cid,
        turns=[
            Turn("customer", "Please cancel order ORD-1, I don't need it. jane@x.com"),
            Turn("agent", "Done, it is cancelled."),
        ],
        meta={"source_file": "chats.csv"},
    )
    return _redacted(conv) if redacted else conv


def test_extraction_request_shape():
    req = ex.build_extraction_request(SPEC, _conv())
    assert [m.role for m in req["messages"]] == ["system", "user"]
    assert [t.name for t in req["tools"]] == [ex.SUBMIT_TOOL, ex.SKIP_TOOL]
    schema = req["schema"]
    assert set(schema["required"]) >= {
        "tier",
        "customer_id",
        "goal",
        "state",
        "success",
        "reference",
    }
    orders = schema["properties"]["state"]["properties"]["orders"]["items"]
    assert orders["properties"]["status"]["enum"] == ["placed", "shipped", "cancelled"]
    assert orders["properties"]["amount"]["type"] == "number"
    ref_tool = schema["properties"]["reference"]["items"]["properties"]["tool"]
    assert ref_tool["enum"] == ["get_order", "cancel_order", "ask_customer", "resolve"]
    assert "<EMAIL_1>" in req["user"] and "jane@x.com" not in req["user"]
    assert "cancel_order" in req["system"] and "untrusted" in req["system"]
    # the system prompt depends only on the spec (cache-friendly across a batch)
    assert ex.build_extraction_request(SPEC, _conv("T-2"))["system"] == req["system"]


def test_unredacted_conversations_are_refused():
    model = FakeModel([])
    with pytest.raises(ex.UnredactedConversationError):
        ex.extract_task(model, SPEC, _conv(redacted=False), validator=None)
    with pytest.raises(ex.UnredactedConversationError):
        ex.extract_batch(model, SPEC, [_conv("a"), _conv("b", redacted=False)], validator=None)
    assert model.calls == []  # nothing was sent


def test_extract_success():
    model = FakeModel([_call(ex.SUBMIT_TOOL, GOOD_ARGS)])
    res = ex.extract_task(model, SPEC, _conv(), validator=None)
    assert res.status == "ok", res.error
    rec = res.record
    assert rec.id == "t-t-1" and rec.customer_id == "C-1"
    assert rec.state["orders"][0]["amount"] == "19.99"  # state kept as given
    assert rec.reference[1].tool == "cancel_order"
    assert rec.source == {"conversation": "T-1", "file": "chats.csv"}
    assert res.attempts == 1 and res.usage.output_tokens == 500


def test_extract_skip():
    model = FakeModel([_call(ex.SKIP_TOOL, {"reason": "spam"})])
    res = ex.extract_task(model, SPEC, _conv(), validator=None)
    assert res.status == "skipped" and res.skipped_reason == "spam" and res.record is None


def test_retry_on_invalid_record_feeds_errors_back():
    bad = json.loads(json.dumps(GOOD_ARGS))
    bad["reference"][1] = {"tool": "cancel_order", "args": {"order_id": "ORD-999"}}
    bad["success"][1]["category"] = "teleported"
    model = FakeModel([_call(ex.SUBMIT_TOOL, bad, "tc1"), _call(ex.SUBMIT_TOOL, GOOD_ARGS, "tc2")])
    res = ex.extract_task(model, SPEC, _conv(), validator=None)
    assert res.status == "ok" and res.attempts == 2
    retry_msgs = model.calls[1][0]
    assert retry_msgs[-2].role == "assistant" and retry_msgs[-2].tool_calls[0].id == "tc1"
    tool_result = retry_msgs[-1]
    assert tool_result.role == "tool" and tool_result.tool_call_id == "tc1"
    assert "ORD-999" in tool_result.content and "teleported" in tool_result.content
    assert res.usage.input_tokens == 2000


def test_retry_on_schema_mismatch_then_give_up():
    broken = {
        "tier": 1,
        "goal": "x",
        "state": {},
        "success": [{"check": "field_equals"}],
        "reference": [],
    }
    model = FakeModel([_call(ex.SUBMIT_TOOL, broken), _call(ex.SUBMIT_TOOL, broken)])
    res = ex.extract_task(model, SPEC, _conv(), validator=None)
    assert res.status == "error" and res.attempts == 2
    assert "customer_id" in res.error
    assert "customer_id" in model.calls[1][0][-1].content  # pydantic error sent back


def test_no_tool_call_gets_a_nudge():
    model = FakeModel(
        [AssistantTurn(text="Here is the record: ..."), _call(ex.SUBMIT_TOOL, GOOD_ARGS)]
    )
    res = ex.extract_task(model, SPEC, _conv(), validator=None)
    assert res.status == "ok"
    assert model.calls[1][0][-1].role == "user" and ex.SUBMIT_TOOL in model.calls[1][0][-1].content


def test_max_tokens_truncation_retries_with_compact_request():
    cut = _call(ex.SUBMIT_TOOL, {"tier": 1, "goal": "Please"})
    cut.raw = {"stop_reason": "max_tokens"}
    model = FakeModel([cut, _call(ex.SUBMIT_TOOL, GOOD_ARGS, "tc2")])
    res = ex.extract_task(model, SPEC, _conv(), validator=None)
    assert res.status == "ok" and res.raw[0]["stop_reason"] == "max_tokens"
    assert "cut off" in model.calls[1][0][-1].content


def test_refusal_is_not_retried():
    refused = AssistantTurn(text="", raw={"stop_reason": "refusal"})
    model = FakeModel([refused, _call(ex.SUBMIT_TOOL, GOOD_ARGS)])
    res = ex.extract_task(model, SPEC, _conv(), validator=None)
    assert res.status == "error" and "declined" in res.error
    assert len(model.calls) == 1


def test_model_exception_becomes_error():
    model = FakeModel([RuntimeError("401 bad key")])
    res = ex.extract_task(model, SPEC, _conv(), validator=None)
    assert res.status == "error" and "401" in res.error


def test_runtime_validator_hook():
    calls = []

    def validator(spec, record):
        calls.append(record.id)
        return ["reference does not make success[0] pass"] if len(calls) == 1 else []

    model = FakeModel([_call(ex.SUBMIT_TOOL, GOOD_ARGS), _call(ex.SUBMIT_TOOL, GOOD_ARGS, "tc2")])
    res = ex.extract_task(model, SPEC, _conv(), validator=validator)
    assert res.status == "ok" and len(calls) == 2


def test_batch_progress_usage_and_early_stop():
    convs = [_conv(f"T-{i}") for i in range(5)]
    seen = []
    model = FakeModel(
        [_call(ex.SUBMIT_TOOL, GOOD_ARGS), _call(ex.SKIP_TOOL, {"reason": "spam"})]
        + [RuntimeError("outage")] * 3
    )
    batch = ex.extract_batch(
        model,
        SPEC,
        convs,
        limit=5,
        validator=None,
        on_progress=lambda d, n, r: seen.append((d, n, r.status)),
    )
    assert seen[:2] == [(1, 5, "ok"), (2, 5, "skipped")]
    assert len(batch.results) == 5 and len(batch.records) == 1 and len(batch.errors) == 3
    assert batch.stopped_reason and "outage" in batch.stopped_reason
    assert batch.usage.input_tokens == 2000
    assert batch.summary()["records"] == 1


@pytest.fixture
def priced_catalog():
    saved = (catalog._cache, catalog._cache_at)
    catalog.set_gateway_models(
        [
            {
                "id": "anthropic/claude-opus-5.5",
                "type": "language",
                "tags": ["tool-use"],
                "pricing": {"input": "0.000004", "output": "0.00002"},
            }
        ]
    )
    yield
    catalog._cache, catalog._cache_at = saved


def test_cost_estimate_math(priced_catalog):
    convs = [_conv("T-1"), _conv("T-2", redacted=False)]  # estimating sends nothing
    est = ex.estimate_cost(
        SPEC, convs, "anthropic:claude-opus-5-5", output_tokens_per_conversation=1000
    )
    assert est["estimate"] is True and est["conversations"] == 2
    req = ex.build_extraction_request(SPEC, convs[0])
    schema_chars = len(json.dumps([t.model_dump() for t in req["tools"]]))
    expected_in = (
        -(-len(req["system"] + req["user"]) // 4) + -(-schema_chars // 4) + ex.TOOL_OVERHEAD_TOKENS
    )
    first = est["per_conversation"][0]
    assert first["input_tokens"] == expected_in
    assert first["cost_usd"] == pytest.approx(expected_in * 4e-6 + 1000 * 2e-5)
    assert est["output_tokens"] == 2000
    assert est["cost_usd"] == pytest.approx(est["input_tokens"] * 4e-6 + 2000 * 2e-5)
    assert est["cost_usd_with_retries"] == pytest.approx(2 * est["cost_usd"])
    assert "Estimate" in est["notes"][0]


def test_cost_estimate_unknown_price(priced_catalog):
    est = ex.estimate_cost(SPEC, [_conv()], "openai:some-unknown-model")
    assert est["cost_usd"] is None and est["input_tokens"] > 0
    assert any("No price" in n for n in est["notes"])


def test_batch_cost_uses_catalog(priced_catalog):
    model = FakeModel([_call(ex.SUBMIT_TOOL, GOOD_ARGS)])
    batch = ex.extract_batch(model, SPEC, [_conv()], validator=None)
    assert batch.cost_usd == pytest.approx(1000 * 4e-6 + 500 * 2e-5)
    assert batch.results[0].cost_usd == pytest.approx(batch.cost_usd)


def test_end_to_end_parse_redact_extract(tmp):
    convs = parse_file(tmp("chats.csv", MESSAGE_CSV)).conversations
    redacted, totals = redact_conversations(convs)
    assert totals["EMAIL"] == 1 and totals["CARD"] == 1 and totals["PHONE"] == 1
    req = ex.build_extraction_request(SPEC, redacted[0])
    assert "jane.doe@example.com" not in req["user"] and "Jane" not in req["user"]
    assert "ORD-12345" in req["user"]  # order ids survive for record keys


# ---- regressions found with the sample IT-helpdesk dataset ----------------------------


def test_dash_signoff_redacts_full_name_everywhere():
    from envforge.builder.ingest import Conversation, Turn
    from envforge.builder.redact import redact_conversation

    conv = Conversation(id="x", turns=[
        Turn(role="customer", text="My keyboard never arrived. — Tom Becker"),
        Turn(role="agent", text="Hi Tom, I've reshipped it. Thanks Mr Becker."),
    ])  # fmt: skip
    out, counts = redact_conversation(conv)
    text = " ".join(t.text for t in out.turns)
    assert "Tom" not in text and "Becker" not in text and counts["NAME"] >= 2


def test_name_here_and_local_phone():
    from envforge.builder.redact import redact_text

    r = redact_text("Laptop won't turn on. Ana Souza here, 555-0173 if you need me.")
    assert "Ana" not in r.text and "Souza" not in r.text and "555-0173" not in r.text
    assert r.counts_by_type.get("PHONE") == 1


def test_delivery_location_is_not_an_address_but_a_street_is():
    from envforge.builder.redact import redact_text

    r = redact_text("It shows delivered to the 4th floor mailroom on Monday.")
    assert "4th floor mailroom" in r.text and not r.counts_by_type.get("ADDRESS")
    r = redact_text("Please ship it to 221 Baker Street, London.")
    assert "Baker Street" not in r.text and r.counts_by_type.get("ADDRESS") == 1

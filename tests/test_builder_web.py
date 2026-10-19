"""Environment builder, end to end through the API, offline.

A fake ChatModel plays both roles: the builder agent (scripted tool calls) and the
extraction model (answers `submit_task_record` with a task from the ISP example).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from test_web_api import _signup, client  # noqa: F401 - shared fixture

from envforge.core.env import ToolCall
from envforge.core.trajectory import Usage
from envforge.models.base import AssistantTurn, ChatModel

EXAMPLE = (
    Path(__file__).resolve().parents[1] / "envforge" / "declarative" / "examples" / "isp_support"
)
SPEC = json.loads((EXAMPLE / "spec.json").read_text())
TASKS = [json.loads(x) for x in (EXAMPLE / "tasks.jsonl").read_text().splitlines() if x.strip()]

CSV = """conversation_id,role,message
c1,customer,"Hi, my name is Jane Doe, my internet is down. Email jane.doe@example.com, phone +1 415 555 0101"
c1,agent,"Thanks for contacting us Jane, let me check your router."
c2,customer,"I was double charged this month"
c2,agent,"Thank you for contacting support, I'll apply a credit."
"""


class FakeModel(ChatModel):
    """Builder turns follow `script`; extraction calls get example task records."""

    def __init__(self, script: list[list[tuple[str, dict]]]) -> None:
        self.model_id = "anthropic:claude-opus-5-5"
        self.script = list(script)
        self.extractions = 0

    def complete(self, messages, tools):
        names = {t.name for t in tools}
        usage = Usage(input_tokens=1000, output_tokens=100)
        if "submit_task_record" in names:  # extraction call
            task = dict(TASKS[self.extractions % len(TASKS)])
            self.extractions += 1
            task["id"] = f"extracted-{self.extractions}"
            call = ToolCall(id=f"x{self.extractions}", name="submit_task_record", args=task)
            return AssistantTurn(text="", tool_calls=[call], usage=usage)
        step = self.script.pop(0) if self.script else []
        calls = [
            ToolCall(id=f"c{len(self.script)}-{i}", name=n, args=a) for i, (n, a) in enumerate(step)
        ]
        return AssistantTurn(
            text="" if calls else "Here's what I built. Anything to change?",
            tool_calls=calls,
            usage=usage,
        )


@pytest.fixture()
def builder(client, monkeypatch):  # noqa: F811
    """Signed-in user with an Anthropic key and a project. Returns (client, ws_id, project_id, set_script)."""
    _signup(client)
    client.put("/api/me/keys", json={"provider": "anthropic", "secret": "sk-ant-test-0001"})
    ws = client.get("/api/workspaces").json()[0]
    p = client.post(
        f"/api/workspaces/{ws['id']}/builder/projects",
        json={"title": "ISP support", "purpose": "resolve ISP tickets"},
    ).json()
    holder: dict = {}
    monkeypatch.setattr("envforge.web.builder_agent.load_model", lambda *a, **k: holder["model"])

    def set_script(script):
        holder["model"] = FakeModel(script)
        return holder["model"]

    return client, ws["id"], p["id"], set_script


def _work():
    from envforge.web.worker import run_forever

    return run_forever(once=True, kind="builder")


def _base(ws: str, p: str) -> str:
    return f"/api/workspaces/{ws}/builder/projects/{p}"


def _upload(c, ws, p, text=CSV, name="chats.csv"):
    r = c.post(f"{_base(ws, p)}/uploads?filename={name}", content=text.encode(),
               headers={"content-type": "application/octet-stream"})  # fmt: skip
    assert r.status_code == 202, r.text
    assert _work() == 1
    return r.json()


def test_upload_is_parsed_redacted_and_raw_file_deleted(builder):
    c, ws, p, _ = builder
    up = _upload(c, ws, p)
    proj = c.get(_base(ws, p)).json()
    u = proj["uploads"][0]
    assert u["status"] == "ready" and u["n_conversations"] == 2
    assert u["redactions"].get("EMAIL") and u["redactions"].get("PHONE")
    from envforge.builder.corpus import CorpusStore
    from envforge.web import builder as B
    from envforge.web import db as dbmod
    from envforge.web.models import EnvProject, EnvUpload, Workspace

    with dbmod.session_factory()() as db:
        row = db.get(EnvUpload, up["id"])
        project = db.get(EnvProject, p)
        path = B.corpus_path(db.get(Workspace, ws), project)
    assert row.path is None  # nothing left on disk but the redacted store
    assert not list(path.parent.glob("*.raw"))
    with CorpusStore(path) as store:
        assert store.count(upload_id=up["id"]) == 2
        stored = "\n".join(
            c.render() if hasattr(c, "render") else str(c.to_dict())
            for c in store.get(store.iter_ids(unique_only=False))
        )
    assert (
        "jane.doe@example.com" not in stored
        and "555 0101" not in stored
        and "Jane Doe" not in stored
        and "<EMAIL_1>" in stored
    )


def test_deleting_an_upload_removes_it_from_the_corpus(builder):
    c, ws, p, _ = builder
    up = _upload(c, ws, p)
    assert c.delete(f"{_base(ws, p)}/uploads/{up['id']}").status_code == 204
    assert _work() == 1  # the recluster job
    from envforge.builder.corpus import CorpusStore
    from envforge.web import builder as B
    from envforge.web import db as dbmod
    from envforge.web.models import EnvProject, Workspace

    with dbmod.session_factory()() as db:
        path = B.corpus_path(db.get(Workspace, ws), db.get(EnvProject, p))
    with CorpusStore(path) as store:
        assert store.count() == 0


def test_bad_uploads_rejected(builder):
    c, ws, p, _ = builder
    r = c.post(f"{_base(ws, p)}/uploads?filename=x.exe", content=b"MZ")
    assert r.status_code == 400
    r = c.post(f"{_base(ws, p)}/uploads?filename=x.csv", content=b"")
    assert r.status_code == 400


def test_builder_turn_builds_spec_extracts_and_versions(builder):
    c, ws, p, set_script = builder
    _upload(c, ws, p)
    model = set_script(
        [
            [("get_state", {})],
            [("sample_conversations", {"n": 2})],
            [("update_spec", {"spec": SPEC, "message": "first draft"})],
            [("extract_tasks", {"n": 2})],
            [],  # final prose answer
        ]
    )
    assert c.post(f"{_base(ws, p)}/messages", json={"content": "build it"}).status_code == 202
    assert c.post(f"{_base(ws, p)}/messages", json={"content": "again"}).status_code == 409  # busy
    assert _work() == 1
    assert model.extractions == 2

    msgs = c.get(f"{_base(ws, p)}/messages").json()["messages"]
    roles = [m["role"] for m in msgs]
    assert roles[0] == "user" and roles[-1] == "assistant" and "tool" in roles
    assert any(
        m.get("meta", {}) and m["meta"].get("revision") == 2 for m in msgs if m["role"] == "tool"
    )

    revs = c.get(f"{_base(ws, p)}/revisions").json()
    assert [r["number"] for r in revs] == [1, 2]
    assert revs[0]["spec_ok"] and revs[1]["n_tasks"] == 2 and revs[1]["tasks_ok"] == 2
    tasks = c.get(f"{_base(ws, p)}/revisions/{revs[1]['id']}/tasks").json()
    assert {t["source"]["conversation"] for t in tasks} == {"c1", "c2"}

    # extraction above the free limit needs the user's confirmation
    set_script([[("extract_tasks", {"n": 10})], []])
    c.post(f"{_base(ws, p)}/messages", json={"content": "extract more"})
    _work()
    tool_msgs = [
        m for m in c.get(f"{_base(ws, p)}/messages").json()["messages"] if m["role"] == "tool"
    ]
    assert "explicit OK" in tool_msgs[-1]["content"]


def test_revisions_restore_checkout_branch_and_diff(builder):
    c, ws, p, set_script = builder
    set_script([[("update_spec", {"spec": SPEC, "message": "v1"})], []])
    c.post(f"{_base(ws, p)}/messages", json={"content": "draft"})
    _work()
    edited = {**SPEC, "title": "ISP support (edited)"}
    r = c.put(f"{_base(ws, p)}/spec", json={"spec": edited, "message": "rename"})
    assert r.status_code == 200 and r.json()["revision"] == 2
    revs = c.get(f"{_base(ws, p)}/revisions").json()
    d = c.get(f"{_base(ws, p)}/revisions/{revs[1]['id']}/diff").json()
    assert d["from"] == 1 and "ISP support (edited)" in d["spec_diff"]

    # restore r1 -> new r3 with r1's spec; nothing lost
    assert c.post(f"{_base(ws, p)}/revisions/{revs[0]['id']}/restore").json()["revision"] == 3
    r3 = c.get(f"{_base(ws, p)}/revisions").json()[2]
    assert r3["is_head"] and r3["parent_number"] == 2
    assert c.get(f"{_base(ws, p)}/revisions/{r3['id']}").json()["spec"]["title"] == SPEC["title"]

    # check out r1 and edit: the new revision branches from r1
    c.post(f"{_base(ws, p)}/revisions/{revs[0]['id']}/checkout")
    c.put(f"{_base(ws, p)}/spec", json={"spec": {**SPEC, "max_steps": 12}, "message": "branch"})
    r4 = c.get(f"{_base(ws, p)}/revisions").json()[3]
    assert r4["number"] == 4 and r4["parent_number"] == 1 and r4["is_head"]

    bad = c.put(f"{_base(ws, p)}/spec", json={"spec": {"name": "x"}, "message": "broken"})
    assert bad.status_code == 400


def test_publish_run_and_isolation(builder):
    c, ws, p, set_script = builder
    tasks = [dict(t) for t in TASKS[:6]]
    set_script([[("update_spec", {"spec": SPEC, "message": "spec"})],
                [("edit_tasks", {"upsert": tasks, "message": "seed tasks"})], []])  # fmt: skip
    c.post(f"{_base(ws, p)}/messages", json={"content": "go"})
    _work()
    proj = c.get(_base(ws, p)).json()
    assert proj["head"]["validation"]["tasks_ok"] == 6
    pub = c.post(f"{_base(ws, p)}/publish", json={}).json()
    assert pub["pack_name"].endswith("@v1")

    packs = {x["name"]: x for x in c.get(f"/api/workspaces/{ws}/packs").json()}
    assert packs[pub["pack_name"]]["custom"] is True and "netops" in packs
    detail = c.get(f"/api/workspaces/{ws}/packs/{pub['pack_name']}").json()
    assert detail["tools"] and detail["reward_rules"]

    r = c.post(
        f"/api/workspaces/{ws}/runs",
        json={"pack": pub["pack_name"], "model": "scripted", "episodes": 4},
    )
    assert r.status_code == 202, r.text
    from envforge.web.worker import run_forever

    assert run_forever(once=True, kind="runs") == 1
    run = c.get(f"/api/workspaces/{ws}/runs/{r.json()['id']}").json()
    assert run["status"] == "completed" and run["metrics"]["overall"]["solve_rate"] == 1.0

    # another user's workspace can't see or run it
    c.post("/api/auth/logout")
    _signup(c, email="bob@example.com", pw="bobs-long-password")
    bws = c.get("/api/workspaces").json()[0]["id"]
    assert pub["pack_name"] not in {x["name"] for x in c.get(f"/api/workspaces/{bws}/packs").json()}
    r = c.post(f"/api/workspaces/{bws}/runs", json={"pack": pub["pack_name"], "model": "scripted"})
    assert r.status_code == 400
    assert c.get(_base(ws, p)).status_code == 404
    assert c.get(f"/api/workspaces/{ws}/builder/projects").status_code == 404


def test_message_needs_a_key_for_the_builder_model(client):  # noqa: F811
    _signup(client)
    ws = client.get("/api/workspaces").json()[0]
    p = client.post(f"/api/workspaces/{ws['id']}/builder/projects", json={"title": "x"}).json()
    r = client.post(
        f"/api/workspaces/{ws['id']}/builder/projects/{p['id']}/messages", json={"content": "hi"}
    )
    assert r.status_code == 400 and "anthropic key" in r.json()["detail"]


def test_builder_history_is_append_only(builder):
    """Old messages must reach the model byte-identical on every call, or the prompt
    cache misses from the first changed message onwards."""
    c, ws, p, set_script = builder
    set_script([[("get_state", {})] for _ in range(14)] + [[]])
    c.post(f"{_base(ws, p)}/messages", json={"content": "look around"})
    _work()
    from envforge.web import builder_agent
    from envforge.web import db as dbmod
    from envforge.web.models import EnvProject

    with dbmod.session_factory()() as db:
        proj = db.get(EnvProject, p)
        stored = [m.content for m in builder_agent.B.messages(db, proj)]
        sent = [m.content for m in builder_agent._history(db, proj)]
    assert sent == stored and sum(1 for m in sent if m.startswith("{")) >= 14


def test_chunked_upload_resumes_and_completes(builder, monkeypatch):
    c, ws, p, _ = builder
    from envforge.web import builder as B

    monkeypatch.setattr(B, "CHUNK_BYTES", 64)  # tiny pieces so the test exercises several
    data = CSV.encode()
    up = c.post(
        f"{_base(ws, p)}/uploads/init", json={"filename": "big.csv", "size_bytes": len(data)}
    ).json()
    assert up["status"] == "receiving" and up["bytes_received"] == 0
    url = f"{_base(ws, p)}/uploads/{up['id']}"
    r = c.put(f"{url}/chunk?offset=0", content=data[:64])
    assert r.json()["bytes_received"] == 64
    # a client that lost track sends the wrong offset: the server says where it is
    r = c.put(f"{url}/chunk?offset=0", content=data[:64])
    assert r.status_code == 409 and r.json()["detail"]["bytes_received"] == 64
    assert c.post(f"{url}/complete").status_code == 409  # not all bytes yet
    off = 64
    while off < len(data):
        off = c.put(f"{url}/chunk?offset={off}", content=data[off : off + 64]).json()[
            "bytes_received"
        ]
    assert c.put(f"{url}/chunk?offset={off}", content=b"x").status_code == 413  # past declared size
    assert c.post(f"{url}/complete").status_code == 202
    assert _work() == 1
    u = next(x for x in c.get(_base(ws, p)).json()["uploads"] if x["id"] == up["id"])
    assert u["status"] == "ready" and u["n_conversations"] == 2


def test_chunked_upload_limits(builder):
    c, ws, p, _ = builder
    r = c.post(f"{_base(ws, p)}/uploads/init", json={"filename": "x.exe", "size_bytes": 10})
    assert r.status_code == 400
    r = c.post(f"{_base(ws, p)}/uploads/init", json={"filename": "x.csv", "size_bytes": 10**15})
    assert r.status_code == 413


@pytest.mark.parametrize(
    "url",
    [
        "http://example.com/a.csv",
        "https://127.0.0.1/a.csv",
        "https://10.0.0.5/a.csv",
        "https://localhost/a.csv",
    ],
)
def test_import_url_refuses_insecure_or_internal_targets(builder, url):
    c, ws, p, _ = builder
    assert c.post(f"{_base(ws, p)}/uploads/import", json={"url": url}).status_code == 400


def _extract_once(c, ws, p, set_script) -> str:
    """Ask the builder to extract one task; returns the extract tool's reply."""
    set_script(
        [[("update_spec", {"spec": SPEC, "message": "spec"})], [("extract_tasks", {"n": 1})], []]
    )
    c.post(f"{_base(ws, p)}/messages", json={"content": "extract"})
    _work()
    msgs = c.get(f"{_base(ws, p)}/messages").json()["messages"]
    return [m for m in msgs if m["role"] == "tool"][-1]["content"]


def test_strict_privacy_requires_a_passed_review_before_extraction(builder):
    c, ws, p, set_script = builder
    r = c.patch(f"{_base(ws, p)}/privacy", json={"profile": "strict", "ner": False})
    priv = r.json()["privacy"]
    assert priv["profile"] == "strict" and priv["needs_signoff"] and priv["review"] is None
    assert c.post(f"{_base(ws, p)}/privacy/review", json={}).status_code == 400  # no data yet
    _upload(c, ws, p)
    assert "privacy review" in _extract_once(c, ws, p, set_script)

    rv = c.post(f"{_base(ws, p)}/privacy/review", json={}).json()
    convs = rv["conversations"]
    assert len(convs) == 2  # the whole (tiny) corpus
    text = str(convs)
    assert "jane.doe@example.com" not in text and "<EMAIL_1>" in text
    assert c.get(f"{_base(ws, p)}/privacy/review").json()["conversations"] == convs

    rid = rv["review"]["id"]
    proj = c.post(f"{_base(ws, p)}/privacy/review/submit",
                  json={"review_id": rid, "flagged": []}).json()  # fmt: skip
    res = proj["privacy"]["review"]["result"]
    assert res["exact"] and res["passed"] and res["leak_rate"] == 0
    assert proj["privacy"]["needs_signoff"] is False
    # a submitted review can't be submitted again
    r = c.post(f"{_base(ws, p)}/privacy/review/submit", json={"review_id": rid, "flagged": []})
    assert r.status_code == 409
    # changing a redaction setting makes the review outdated
    proj = c.patch(f"{_base(ws, p)}/privacy", json={"surrogates": True}).json()
    assert proj["privacy"]["needs_signoff"] is True
    assert proj["privacy"]["review"]["status"] == "outdated"


def test_failed_review_blocks_extraction_on_any_profile(builder):
    c, ws, p, set_script = builder
    _upload(c, ws, p)
    rv = c.post(f"{_base(ws, p)}/privacy/review", json={}).json()
    one = rv["conversations"][0]["id"]
    proj = c.post(f"{_base(ws, p)}/privacy/review/submit",
                  json={"review_id": rv["review"]["id"], "flagged": [one]}).json()  # fmt: skip
    assert proj["privacy"]["review"]["status"] == "failed" and proj["privacy"]["blocked"]
    assert "too much personal data" in _extract_once(c, ws, p, set_script)
    bad = c.post(f"{_base(ws, p)}/privacy/review/submit",
                 json={"review_id": "nope", "flagged": []})  # fmt: skip
    assert bad.status_code == 409


def test_privacy_off_needs_confirmation(builder):
    c, ws, p, _ = builder
    assert c.patch(f"{_base(ws, p)}/privacy", json={"profile": "off"}).status_code == 400
    r = c.patch(f"{_base(ws, p)}/privacy", json={"profile": "off", "confirm_off": True})
    assert r.json()["privacy"]["profile"] == "off"
    assert c.patch(f"{_base(ws, p)}/privacy", json={"tier2": "bogus"}).status_code == 422


def test_corpus_overview_and_cluster_examples(builder):
    c, ws, p, _ = builder
    assert c.get(f"{_base(ws, p)}/corpus").json()["corpus"] is None
    _upload(c, ws, p)
    ov = c.get(f"{_base(ws, p)}/corpus").json()["corpus"]
    assert ov["conversations"] == 2 and ov["clusters"]
    cid = ov["clusters"][0]["id"]
    ex = c.get(f"{_base(ws, p)}/corpus/clusters/{cid}?n=2").json()["conversations"]
    assert ex and "jane.doe@example.com" not in str(ex)


def test_publish_and_export_resanitize_planted_pii(builder):
    c, ws, p, set_script = builder
    tasks = [dict(t) for t in TASKS[:3]]
    tasks[0] = {
        **tasks[0],
        "goal": tasks[0]["goal"] + " You can email me at leaky.person@example.com.",
    }
    set_script([[("update_spec", {"spec": SPEC, "message": "spec"})],
                [("edit_tasks", {"upsert": tasks, "message": "tasks"})], []])  # fmt: skip
    c.post(f"{_base(ws, p)}/messages", json={"content": "go"})
    _work()
    pub = c.post(f"{_base(ws, p)}/publish", json={}).json()
    assert pub["privacy"]["redactions"].get("EMAIL", 0) >= 1
    from envforge.web import db as dbmod
    from envforge.web.models import EnvVersion

    with dbmod.session_factory()() as db:
        env_dir = Path(db.query(EnvVersion).one().env_dir)
    assert "leaky.person@example.com" not in (env_dir / "tasks.jsonl").read_text()

    r = c.post(
        f"/api/workspaces/{ws}/runs",
        json={"pack": pub["pack_name"], "model": "scripted", "episodes": 3},
    )
    from envforge.web.worker import run_forever

    run_forever(once=True, kind="runs")
    out = c.get(f"/api/workspaces/{ws}/runs/export?runs={r.json()['id']}&format=chat").text
    assert out and "leaky.person@example.com" not in out and "@example.com" not in out

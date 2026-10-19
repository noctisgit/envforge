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
    from envforge.web import db as dbmod
    from envforge.web.models import EnvUpload

    with dbmod.session_factory()() as db:
        row = db.get(EnvUpload, up["id"])
        stored = Path(row.path).read_text()
    assert row.path.endswith(".jsonl") and not Path(row.path).with_suffix(".raw").exists()
    assert (
        "jane.doe@example.com" not in stored
        and "555 0101" not in stored
        and "Jane Doe" not in stored
    )


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

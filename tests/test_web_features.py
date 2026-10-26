"""Web layer: tenant isolation, usage stats, model catalog, export and hardening. Offline."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

import pytest
from test_web_api import _signup, client  # noqa: F401 - shared fixture

from envforge.models import catalog
from envforge.web import db as dbmod

GATEWAY = [
    {  # a tool-capable Anthropic model, also reachable natively
        "id": "anthropic/claude-sonnet-4.5",
        "name": "Claude Sonnet 4.5",
        "owned_by": "anthropic",
        "type": "language",
        "context_window": 200000,
        "tags": ["tool-use", "reasoning"],
        "pricing": {"input": "0.000003", "output": "0.000015"},
    },
    {  # same model's batch variant: hidden
        "id": "anthropic/claude-sonnet-4.5-batch",
        "name": "Claude Sonnet 4.5 (batch)",
        "owned_by": "anthropic",
        "type": "language",
        "tags": ["tool-use"],
        "pricing": {"input": "0.0000015", "output": "0.0000075"},
    },
    {  # no tool use: hidden
        "id": "acme/chat-only",
        "name": "Chat Only",
        "owned_by": "acme",
        "type": "language",
        "tags": [],
        "pricing": {"input": "0.000001", "output": "0.000001"},
    },
    {  # embeddings: hidden
        "id": "acme/embed",
        "name": "Embed",
        "owned_by": "acme",
        "type": "embedding",
        "tags": ["tool-use"],
        "pricing": {"input": "0.0000001"},
    },
    {  # open weights served by the gateway: vercel only, never "openai:"
        "id": "openai/gpt-oss-120b",
        "name": "gpt-oss-120b",
        "owned_by": "openai",
        "type": "language",
        "tags": ["tool-use", "open-weights"],
        "pricing": {"input": "0.0000001", "output": "0.0000005"},
    },
    {
        "id": "acme/tool-model",
        "name": "Acme Tool Model",
        "owned_by": "acme",
        "type": "language",
        "tags": ["tool-use"],
        "pricing": {"input": "0.000001", "output": "0.000002"},
    },
]


@pytest.fixture(autouse=True)
def _catalog():
    catalog.set_gateway_models(GATEWAY)
    yield
    catalog.set_gateway_models([])


def _ws(c) -> dict:
    return c.get("/api/workspaces").json()[0]


def _scripted_run(c, ws_id: str, episodes: int = 3) -> dict:
    r = c.post(f"/api/workspaces/{ws_id}/runs", json={"model": "scripted", "episodes": episodes})
    assert r.status_code == 202, r.text
    from envforge.web.worker import run_forever

    assert run_forever(once=True) == 1
    return c.get(f"/api/workspaces/{ws_id}/runs/{r.json()['id']}").json()


# ---- isolation ------------------------------------------------------------------


def test_workspaces_are_fully_isolated_between_users(client):  # noqa: F811
    _signup(client, email="alice@example.com")
    a_ws = _ws(client)
    client.put(
        "/api/me/keys",
        json={"provider": "vercel", "secret": "vck-alice-secret-0001"},
    )
    a_run = _scripted_run(client, a_ws["id"])
    client.post("/api/auth/logout")

    _signup(client, email="bob@example.com", pw="bobs-long-password")
    b_ws = _ws(client)
    assert [w["id"] for w in client.get("/api/workspaces").json()] == [b_ws["id"]]

    base = f"/api/workspaces/{a_ws['id']}"
    run = f"{base}/runs/{a_run['id']}"
    attempts = [
        ("get", base, None),
        ("patch", base, {"name": "pwned"}),
        ("get", f"{base}/models", None),
        ("get", f"{base}/usage", None),
        ("get", f"{base}/runs", None),
        ("post", f"{base}/runs", {"model": "scripted", "episodes": 1}),
        ("get", run, None),
        ("post", f"{run}/cancel", None),
        ("get", f"{run}/episodes", None),
        ("get", f"{run}/episodes/0", None),
        ("get", f"{base}/runs/export?runs={a_run['id']}", None),
    ]
    for method, url, body in attempts:
        kwargs = {"json": body} if body is not None else {}
        r = getattr(client, method)(url, **kwargs)
        assert r.status_code == 404, (method, url, r.status_code)

    # Alice's run id through Bob's own workspace path is not found either
    mine = f"/api/workspaces/{b_ws['id']}/runs/{a_run['id']}"
    for url in (mine, f"{mine}/episodes", f"{mine}/episodes/0"):
        assert client.get(url).status_code == 404, url
    r = client.get(f"/api/workspaces/{b_ws['id']}/runs/export?runs={a_run['id']}")
    assert r.status_code == 404

    # Bob's usage and models know nothing of Alice
    assert client.get("/api/auth/me/usage").json()["totals"]["runs"] == 0
    assert client.get(f"/api/workspaces/{b_ws['id']}/models").json()["providers"] == []
    # keys are per user: Bob sees none of Alice's and can't run on them
    assert client.get("/api/me/keys").json() == []
    r = client.post(
        f"/api/workspaces/{b_ws['id']}/runs",
        json={"model": "vercel:acme/tool-model", "episodes": 1},
    )
    assert r.status_code == 400 and "no vercel key" in r.json()["detail"]

    # and nothing Bob tried changed Alice's workspace
    client.post("/api/auth/logout")
    client.post(
        "/api/auth/login", json={"email": "alice@example.com", "password": "correct horse battery"}
    )
    ws = client.get(f"/api/workspaces/{a_ws['id']}").json()
    assert ws["name"] == a_ws["name"]
    assert [k["provider"] for k in client.get("/api/me/keys").json()] == ["vercel"]


def test_same_workspace_name_gets_separate_storage(client):  # noqa: F811
    _signup(client, email="alice@example.com")
    a = client.post("/api/workspaces", json={"name": "team"}).json()
    client.post("/api/auth/logout")
    _signup(client, email="bob@example.com", pw="bobs-long-password")
    b = client.post("/api/workspaces", json={"name": "team"}).json()
    assert a["slug"] != b["slug"]  # the slug is the on-disk directory
    run = _scripted_run(client, b["id"], episodes=1)
    from sqlalchemy import select

    from envforge.web.models import Run

    with dbmod.session_factory()() as db:
        row = db.scalar(select(Run).where(Run.id == run["id"]))
        assert f"/workspaces/{b['slug']}/runs/" in row.run_dir


# ---- usage ----------------------------------------------------------------------


def test_usage_totals_per_workspace_and_user(client):  # noqa: F811
    _signup(client)
    ws = _ws(client)
    _scripted_run(client, ws["id"], episodes=3)
    _scripted_run(client, ws["id"], episodes=2)
    u = client.get(f"/api/workspaces/{ws['id']}/usage?days=7").json()
    assert u["totals"]["runs"] == 2 and u["totals"]["episodes"] == 5
    assert u["totals"]["cost_usd"] == 0.0 and u["totals"]["unpriced_runs"] == 0
    assert len(u["by_day"]) == 7 and u["by_day"][-1]["runs"] == 2
    assert u["by_model"][0]["model"] == "scripted"
    assert u["groups"][0]["email"] == "nick@example.com"
    me = client.get("/api/auth/me/usage").json()
    assert me["totals"]["episodes"] == 5 and me["groups"][0]["workspace_id"] == ws["id"]


def test_worker_records_tokens_and_cost(client, monkeypatch):  # noqa: F811
    from envforge.core.trajectory import Usage
    from envforge.models.scripted import ScriptedModel

    class PricedScripted(ScriptedModel):
        """The reference solver, pretending to be a paid model that reports usage."""

        def __init__(self) -> None:
            super().__init__()
            self.model_id = "vercel:acme/tool-model"

        def complete(self, messages, tools):
            turn = super().complete(messages, tools)
            turn.usage = Usage(input_tokens=1000, output_tokens=100)
            return turn

    monkeypatch.setattr("envforge.web.worker.load_model", lambda *a, **k: PricedScripted())
    _signup(client)
    ws = _ws(client)
    client.put(
        "/api/me/keys",
        json={"provider": "vercel", "secret": "vck-test-secret-0001"},
    )
    r = client.post(
        f"/api/workspaces/{ws['id']}/runs",
        json={"model": "vercel:acme/tool-model", "episodes": 2},
    )
    from envforge.web.worker import run_forever

    run_forever(once=True)
    run = client.get(f"/api/workspaces/{ws['id']}/runs/{r.json()['id']}").json()
    assert run["status"] == "completed"
    assert run["input_tokens"] > 0 and run["output_tokens"] == run["input_tokens"] // 10
    expected = run["input_tokens"] * 1e-6 + run["output_tokens"] * 2e-6
    assert run["cost_usd"] == pytest.approx(expected)
    assert run["metrics"]["overall"]["cost_usd"] == pytest.approx(expected)
    usage = client.get(f"/api/workspaces/{ws['id']}/usage").json()
    assert usage["totals"]["cost_usd"] == pytest.approx(expected)


# ---- models ---------------------------------------------------------------------


def test_models_follow_configured_providers(client):  # noqa: F811
    _signup(client)
    ws = _ws(client)
    url = f"/api/workspaces/{ws['id']}/models"
    ids = [m["id"] for m in client.get(url).json()["models"]]
    assert ids == ["scripted"]  # no keys yet

    client.put(
        "/api/me/keys",
        json={"provider": "vercel", "secret": "vck-test-secret-0001"},
    )
    client.put(
        "/api/me/keys",
        json={"provider": "anthropic", "secret": "sk-ant-test-0001"},
    )
    body = client.get(url).json()
    ids = {m["id"] for m in body["models"]}
    assert body["providers"] == ["anthropic", "vercel"] and body["catalog_available"]
    assert {
        "scripted",
        "vercel:anthropic/claude-sonnet-4.5",
        "anthropic:claude-sonnet-4-5",
        "vercel:openai/gpt-oss-120b",
        "vercel:acme/tool-model",
    } == ids  # no batch, no chat-only, no embeddings, no openai: without a key
    sonnet = next(m for m in body["models"] if m["id"] == "anthropic:claude-sonnet-4-5")
    assert sonnet["input_per_mtok"] == pytest.approx(3.0)
    assert sonnet["output_per_mtok"] == pytest.approx(15.0)


def test_catalog_pricing_and_native_ids():
    assert catalog.price_per_token("scripted") == (0.0, 0.0)
    assert catalog.price_per_token("vllm:Qwen/Qwen2.5-3B-Instruct") == (0.0, 0.0)
    assert catalog.price_per_token("anthropic:claude-sonnet-4-5") == (3e-6, 15e-6)
    assert catalog.price_per_token("vercel:anthropic/claude-sonnet-4.5") == (3e-6, 15e-6)
    assert catalog.price_per_token("openai:gpt-oss-120b") is None  # open weights ≠ OpenAI API
    assert catalog.cost_usd("vercel:acme/tool-model", 1_000_000, 500_000) == pytest.approx(2.0)
    assert catalog.cost_usd("vercel:unknown/model", 10, 10) is None
    assert catalog.cost_usd("vercel:unknown/model", 0, 0) == 0.0


# ---- export ---------------------------------------------------------------------


def test_export_chat_and_trajectory_formats(client):  # noqa: F811
    _signup(client)
    ws = _ws(client)
    run = _scripted_run(client, ws["id"], episodes=4)
    url = f"/api/workspaces/{ws['id']}/runs/export?runs={run['id']}"

    r = client.get(url + "&format=chat&min_reward=1.0")
    assert r.status_code == 200 and r.headers["x-episodes"] == "4"
    assert "attachment" in r.headers["content-disposition"]
    lines = [json.loads(x) for x in r.text.splitlines()]
    assert len(lines) == 4
    ex = lines[0]
    roles = [m["role"] for m in ex["messages"]]
    assert roles[:3] == ["system", "user", "assistant"] and "tool" in roles
    assert "INITIAL STATE" in ex["messages"][1]["content"]
    assert any(t["function"]["name"] == "finish" for t in ex["tools"])
    tool_msgs = [m for m in ex["messages"] if m["role"] == "tool"]
    call_ids = {c["id"] for m in ex["messages"] for c in m.get("tool_calls", [])}
    assert {m["tool_call_id"] for m in tool_msgs} <= call_ids

    r = client.get(url + "&format=trajectory")
    assert len(r.text.splitlines()) == 4
    assert json.loads(r.text.splitlines()[0])["run_id"] == run["id"]


# ---- hardening ------------------------------------------------------------------


@pytest.mark.parametrize(
    "base_url",
    [
        "http://127.0.0.1:8000/v1",
        "http://localhost:8000/v1",
        "http://169.254.169.254/latest",
        "http://10.0.0.5:8000/v1",
        "http://[::1]:8000/v1",
        "ftp://example.com/v1",
        "http://user:pw@100.100.1.1:8000/v1",
    ],
)
def test_vllm_base_url_blocks_internal_targets(client, base_url):  # noqa: F811
    _signup(client)
    r = client.put("/api/me/keys", json={"provider": "vllm", "base_url": base_url})
    assert r.status_code == 400, r.text


def test_vllm_tailscale_url_allowed_and_hosted_providers_ignore_urls(client):  # noqa: F811
    _signup(client)
    r = client.put(
        "/api/me/keys",
        json={"provider": "vllm", "base_url": "http://100.101.102.103:8000/v1/"},
    )
    assert r.status_code == 200 and r.json()[0]["base_url"] == "http://100.101.102.103:8000/v1"
    r = client.put(
        "/api/me/keys",
        json={
            "provider": "openai",
            "secret": "sk-test-000001",
            "base_url": "http://127.0.0.1:9/v1",
        },
    )
    openai_key = next(k for k in r.json() if k["provider"] == "openai")
    assert openai_key["base_url"] is None


def test_login_is_rate_limited(client):  # noqa: F811
    _signup(client)
    client.post("/api/auth/logout")
    codes = [
        client.post(
            "/api/auth/login", json={"email": "nick@example.com", "password": "wrong"}
        ).status_code
        for _ in range(12)
    ]
    assert codes[:10] == [401] * 10 and codes[-1] == 429


def test_logout_all_revokes_existing_sessions(client):  # noqa: F811
    _signup(client)
    old_cookie = client.cookies.get("envforge_session")
    assert client.post("/api/auth/logout-all").status_code == 204
    client.cookies.set("envforge_session", old_cookie)
    assert client.get("/api/auth/me").status_code == 401


def test_active_run_cap_and_input_limits(client):  # noqa: F811
    _signup(client)
    ws = _ws(client)
    url = f"/api/workspaces/{ws['id']}/runs"
    for _ in range(3):
        assert client.post(url, json={"model": "scripted", "episodes": 1}).status_code == 202
    r = client.post(url, json={"model": "scripted", "episodes": 1})
    assert r.status_code == 429
    assert client.post(url, json={"model": "x" * 300}).status_code == 422
    assert client.post(url, json={"model": "scripted", "label": "l" * 300}).status_code == 422
    assert client.post(url, json={"model": "scripted", "seed_start": 2**40}).status_code == 422


def test_key_deleted_after_queueing_fails_the_run(client):  # noqa: F811
    _signup(client)
    ws = _ws(client)
    client.put(
        "/api/me/keys",
        json={"provider": "vercel", "secret": "vck-test-secret-0001"},
    )
    r = client.post(
        f"/api/workspaces/{ws['id']}/runs",
        json={"model": "vercel:acme/tool-model", "episodes": 1},
    )
    client.delete("/api/me/keys/vercel")
    from envforge.web.worker import run_forever

    run_forever(once=True)
    run = client.get(f"/api/workspaces/{ws['id']}/runs/{r.json()['id']}").json()
    assert run["status"] == "failed" and "no vercel key" in run["error"]
    assert "Traceback" not in run["error"]


def test_cancel_during_finish_is_not_overwritten(client):  # noqa: F811
    _signup(client)
    ws = _ws(client)
    r = client.post(f"/api/workspaces/{ws['id']}/runs", json={"model": "scripted"})
    from sqlalchemy import update

    from envforge.web.models import Run
    from envforge.web.worker import _finish, claim_next

    with dbmod.session_factory()() as db:
        run = claim_next(db)
        assert run is not None and run.id == r.json()["id"]
        db.execute(update(Run).where(Run.id == run.id).values(status="cancelled"))  # the API
        db.commit()
        _finish(db, run, "completed")
        assert run.status == "cancelled" and run.finished_at is not None


def test_stale_running_runs_are_reaped(client):  # noqa: F811
    _signup(client)
    ws = _ws(client)
    client.post(f"/api/workspaces/{ws['id']}/runs", json={"model": "scripted"})
    from envforge.web.models import Run
    from envforge.web.worker import claim_next, reap_stale

    with dbmod.session_factory()() as db:
        run = claim_next(db)
        assert reap_stale(db) == 0  # fresh heartbeat
        run.heartbeat_at = datetime.now(UTC) - timedelta(hours=1)
        db.commit()
        assert reap_stale(db) == 1
        db.refresh(run)
        assert run.status == "failed" and "heartbeat" in run.error
        assert db.get(Run, run.id).finished_at is not None


def test_gpu_and_api_lanes_take_different_runs(client):  # noqa: F811
    _signup(client)
    ws = _ws(client)
    api_run = client.post(f"/api/workspaces/{ws['id']}/runs", json={"model": "scripted"}).json()
    from envforge.web.models import Run
    from envforge.web.worker import claim_next

    with dbmod.session_factory()() as db:
        gpu_run = db.get(Run, api_run["id"])  # a queued vLLM run, older than the API one
        gpu_run = Run(**{c.name: getattr(gpu_run, c.name) for c in Run.__table__.columns
                         if c.name not in ("id", "model", "created_at")},
                      id="gpu-run", model="vllm:qwen-abcd-v1",
                      created_at=gpu_run.created_at - timedelta(minutes=1))  # fmt: skip
        db.add(gpu_run)
        db.commit()
        assert claim_next(db, "api").id == api_run["id"]  # skips the older vLLM run
        assert claim_next(db, "api") is None
        assert claim_next(db, "gpu").id == "gpu-run"  # while the API run is still running
        assert claim_next(db, "any") is None


# ---- key verification -------------------------------------------------------------


def test_key_is_verified_on_save_and_limits_native_models(client, monkeypatch):  # noqa: F811
    from envforge.web import services

    monkeypatch.setenv("ENVFORGE_OFFLINE", "0")  # let services._verify run (provider mocked)
    calls = []

    def fake_list(provider, secret, base_url=None):
        calls.append((provider, secret))
        if secret == "sk-ant-revoked-0001":
            raise services.KeyRejected("Anthropic rejected this key (401)")
        # Sonnet 4.5 retired on the native API; dated snapshot ids as the API returns them
        return ["claude-haiku-4-5-20251001", "claude-brand-new-9"]

    monkeypatch.setattr(services, "list_provider_models", fake_list)
    _signup(client)
    ws = _ws(client)
    url = "/api/me/keys"

    r = client.put(url, json={"provider": "anthropic", "secret": "sk-ant-revoked-0001"})
    assert r.status_code == 400 and "rejected" in r.json()["detail"]
    assert client.get("/api/me/keys").json() == []  # not saved

    r = client.put(url, json={"provider": "anthropic", "secret": "sk-ant-good-key-0001"})
    key = r.json()[0]
    assert key["verified_at"] and key["n_models"] == 2 and key["verify_note"] is None

    ids = {m["id"] for m in client.get(f"/api/workspaces/{ws['id']}/models").json()["models"]}
    assert "anthropic:claude-sonnet-4-5" not in ids  # in the gateway catalog, not on this key
    assert "anthropic:claude-brand-new-9" in ids  # on the key, not yet in the catalog
    assert catalog.price_per_token("anthropic:claude-haiku-4-5-20251001") is None  # not in GATEWAY


def test_key_saved_with_note_when_provider_unreachable(client, monkeypatch):  # noqa: F811
    from envforge.web import services

    monkeypatch.setenv("ENVFORGE_OFFLINE", "0")

    def boom(provider, secret):
        raise ConnectionError("network down")

    monkeypatch.setattr(services, "list_provider_models", boom)
    _signup(client)
    r = client.put(
        "/api/me/keys",
        json={"provider": "openai", "secret": "sk-test-000001"},
    )
    key = r.json()[0]
    assert r.status_code == 200 and key["verified_at"] is None
    assert "could not reach openai" in key["verify_note"]


# ---- environments + run deletion ------------------------------------------------


def test_environment_endpoints(client):  # noqa: F811
    _signup(client)
    d = client.get("/api/packs/netops").json()
    assert d["name"] == "netops" and d["replayable"] is True
    assert d["world_diagram"]["nodes"] and d["dimensions"] and d["reward_rules"]
    assert all(t["category"] in ("observe", "act") for t in d["tools"])
    p = client.get("/api/packs/netops/preview?seed=3&tier=3").json()
    assert p["task"]["hidden"]["faults"] and p["verdict"]["reward"] == 1.0
    assert any(n["state"] != "ok" for n in p["diagram_before"]["nodes"])
    assert client.get("/api/packs/netops/preview?tier=9").status_code == 400
    assert client.get("/api/packs/nope").status_code == 404
    client.post("/api/auth/logout")
    assert client.get("/api/packs/netops").status_code == 401


def test_replay_endpoint(client):  # noqa: F811
    _signup(client)
    ws = _ws(client)
    run = _scripted_run(client, ws["id"], episodes=2)
    r = client.get(f"/api/workspaces/{ws['id']}/runs/{run['id']}/episodes/1/replay").json()
    assert len(r["frames"]) >= 2 and r["verdict"]["success"] is True


def test_delete_run_removes_files_hides_it_and_keeps_usage(client):  # noqa: F811
    from pathlib import Path

    from sqlalchemy import select

    from envforge.web.models import Run

    _signup(client)
    ws = _ws(client)
    run = _scripted_run(client, ws["id"], episodes=2)
    with dbmod.session_factory()() as db:
        run_dir = Path(db.scalar(select(Run).where(Run.id == run["id"])).run_dir)
    assert run_dir.exists()

    url = f"/api/workspaces/{ws['id']}/runs/{run['id']}"
    assert client.delete(url).status_code == 204
    assert not run_dir.exists()
    assert client.get(url).status_code == 404
    assert client.get(f"/api/workspaces/{ws['id']}/runs").json() == []
    assert client.delete(url).status_code == 404
    usage = client.get(f"/api/workspaces/{ws['id']}/usage").json()
    assert usage["totals"]["runs"] == 1 and usage["totals"]["episodes"] == 2


def test_cannot_delete_active_or_someone_elses_run(client):  # noqa: F811
    _signup(client, email="alice@example.com")
    ws = _ws(client)
    queued = client.post(f"/api/workspaces/{ws['id']}/runs", json={"model": "scripted"}).json()
    url = f"/api/workspaces/{ws['id']}/runs/{queued['id']}"
    assert client.delete(url).status_code == 409  # cancel first
    client.post(f"{url}/cancel")
    client.post("/api/auth/logout")
    _signup(client, email="bob@example.com", pw="bobs-long-password")
    assert client.delete(url).status_code == 404  # not Bob's workspace
    client.post("/api/auth/logout")
    client.post(
        "/api/auth/login", json={"email": "alice@example.com", "password": "correct horse battery"}
    )
    assert client.delete(url).status_code == 204


def test_keys_belong_to_the_user_and_work_in_every_workspace(client):  # noqa: F811
    _signup(client)
    first = _ws(client)
    second = client.post("/api/workspaces", json={"name": "second"}).json()
    client.put("/api/me/keys", json={"provider": "vercel", "secret": "vck-test-secret-0001"})
    for ws in (first, second):
        body = client.get(f"/api/workspaces/{ws['id']}/models").json()
        assert body["providers"] == ["vercel"]
        r = client.post(
            f"/api/workspaces/{ws['id']}/runs",
            json={"model": "vercel:acme/tool-model", "episodes": 1},
        )
        assert r.status_code == 202, r.text


def test_profile_and_password_change(client):  # noqa: F811
    _signup(client)
    assert (
        client.patch("/api/me", json={"display_name": "Nick L"}).json()["display_name"] == "Nick L"
    )
    r = client.post(
        "/api/me/password",
        json={"current_password": "wrong", "new_password": "a-new-long-password"},
    )
    assert r.status_code == 400
    old_cookie = client.cookies.get("envforge_session")
    r = client.post(
        "/api/me/password",
        json={"current_password": "correct horse battery", "new_password": "a-new-long-password"},
    )
    assert r.status_code == 204
    assert client.get("/api/auth/me").status_code == 200  # this session got a fresh cookie
    client.cookies.set("envforge_session", old_cookie)
    assert client.get("/api/auth/me").status_code == 401  # other sessions are signed out
    client.cookies.clear()
    r = client.post(
        "/api/auth/login", json={"email": "nick@example.com", "password": "a-new-long-password"}
    )
    assert r.status_code == 200

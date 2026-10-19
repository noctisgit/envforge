"""API tests on an in-memory SQLite DB with local auth. No network."""

from __future__ import annotations

import os

import pytest

os.environ.update(
    {
        "DATABASE_URL": "sqlite://",
        "AUTH_PROVIDER": "local",
        "APP_SECRET_KEY": "test-secret-key-that-is-long-enough-0123456789",
        "ALLOW_SERVER_KEYS": "false",
        "ALLOW_SIGNUP": "true",
        "COOKIE_SECURE": "false",
        "ENVFORGE_OFFLINE": "1",  # model catalog: never fetch in tests
        "ENVFORGE_DATA_DIR": "",  # set per test below
    }
)

from fastapi.testclient import TestClient  # noqa: E402

from envforge.web import db as dbmod  # noqa: E402
from envforge.web.settings import get_settings  # noqa: E402


@pytest.fixture()
def client(tmp_path, monkeypatch):
    # fresh engine + schema per test
    get_settings.cache_clear()
    monkeypatch.setenv("ENVFORGE_DATA_DIR", str(tmp_path / "data"))
    import importlib

    from envforge import config

    importlib.reload(config)
    from envforge.web.security import limiter

    limiter.reset()
    engine = dbmod.make_engine("sqlite://")
    dbmod.set_engine(engine)
    from envforge.web import models  # noqa: F401

    dbmod.Base.metadata.create_all(engine)
    from envforge.web.app import create_app

    with TestClient(create_app()) as c:
        yield c


def _signup(c: TestClient, email="nick@example.com", pw="correct horse battery"):
    r = c.post("/api/auth/signup", json={"email": email, "password": pw, "display_name": "Nick"})
    assert r.status_code == 201, r.text
    return r.json()


def test_info_and_health(client):
    assert client.get("/api/health").json()["ok"] is True
    info = client.get("/api/auth/info").json()
    assert info["auth_provider"] == "local" and "anthropic" in info["providers"]


def test_signup_login_me_logout(client):
    u = _signup(client)
    assert u["is_admin"] is True  # first user administers
    assert client.get("/api/auth/me").json()["email"] == "nick@example.com"
    client.post("/api/auth/logout")
    assert client.get("/api/auth/me").status_code == 401
    r = client.post("/api/auth/login", json={"email": "nick@example.com", "password": "wrong"})
    assert r.status_code == 401
    r = client.post(
        "/api/auth/login", json={"email": "NICK@example.com", "password": "correct horse battery"}
    )
    assert r.status_code == 200
    assert client.get("/api/auth/me").status_code == 200


def test_duplicate_email_rejected(client):
    _signup(client)
    r = client.post(
        "/api/auth/signup", json={"email": "nick@example.com", "password": "another-long-pw"}
    )
    assert r.status_code == 409


def test_personal_workspace_and_settings(client):
    _signup(client)
    wss = client.get("/api/workspaces").json()
    assert len(wss) == 1 and wss[0]["role"] == "owner"
    ws = wss[0]
    r = client.patch(
        f"/api/workspaces/{ws['id']}",
        json={"default_model": "vercel:openai/gpt-4o-mini", "default_episodes": 5},
    )
    assert r.status_code == 200 and r.json()["default_model"] == "vercel:openai/gpt-4o-mini"


def test_keys_are_encrypted_and_masked(client):
    _signup(client)
    r = client.put(
        "/api/me/keys",
        json={"provider": "anthropic", "secret": "sk-ant-api03-SECRETSECRET-xy12"},
    )
    assert r.status_code == 200
    keys = r.json()
    assert keys[0]["provider"] == "anthropic" and keys[0]["masked"] == "…xy12"
    assert "SECRET" not in r.text  # never echoed

    # stored ciphertext is not the plaintext, and decrypts back
    from sqlalchemy import select

    from envforge.web import crypto
    from envforge.web.models import ProviderKey

    with dbmod.session_factory()() as db:
        row = db.scalar(select(ProviderKey))
        assert "SECRET" not in row.ciphertext
        assert crypto.decrypt(row.ciphertext) == "sk-ant-api03-SECRETSECRET-xy12"

    r = client.delete("/api/me/keys/anthropic")
    assert r.status_code == 200 and r.json() == []


def test_run_requires_key_for_api_provider(client):
    _signup(client)
    ws = client.get("/api/workspaces").json()[0]
    r = client.post(
        f"/api/workspaces/{ws['id']}/runs",
        json={"model": "anthropic:claude-sonnet-4-5", "episodes": 2},
    )
    assert r.status_code == 400 and "no anthropic key" in r.json()["detail"]


def test_enqueue_worker_and_read_results(client):
    _signup(client)
    ws = client.get("/api/workspaces").json()[0]
    r = client.post(
        f"/api/workspaces/{ws['id']}/runs",
        json={
            "pack": "netops",
            "model": "scripted",
            "episodes": 6,
            "tiers": [1, 2, 3],
            "label": "smoke",
        },
    )
    assert r.status_code == 202, r.text
    run = r.json()
    assert run["status"] == "queued"

    from envforge.web.worker import run_forever

    assert run_forever(once=True) == 1

    r = client.get(f"/api/workspaces/{ws['id']}/runs/{run['id']}").json()
    assert r["status"] == "completed" and r["progress"] == 6
    assert r["metrics"]["overall"]["solve_rate"] == 1.0

    eps = client.get(f"/api/workspaces/{ws['id']}/runs/{run['id']}/episodes").json()
    assert len(eps) == 6 and all(e["success"] for e in eps)
    ep = client.get(f"/api/workspaces/{ws['id']}/runs/{run['id']}/episodes/2").json()
    assert ep["task"]["tier"] == 3 and ep["steps"][-1]["tool_calls"][-1]["name"] == "finish"

    # listed, newest first
    assert client.get(f"/api/workspaces/{ws['id']}/runs").json()[0]["id"] == run["id"]


def test_other_users_cannot_see_my_workspace(client):
    _signup(client)
    ws = client.get("/api/workspaces").json()[0]
    client.post("/api/auth/logout")
    _signup(client, email="mallory@example.com", pw="another-long-password")
    assert client.get(f"/api/workspaces/{ws['id']}").status_code == 404
    assert client.get(f"/api/workspaces/{ws['id']}/runs").status_code == 404
    assert client.get(f"/api/workspaces/{ws['id']}/models").status_code == 404

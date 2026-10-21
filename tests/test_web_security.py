"""Regression tests for the security audit fixes: tailnet URLs are admin-only, outbound
requests to user-supplied hosts never follow redirects, the client IP comes from one
configured header, and run ids stay safe and short. Offline, SQLite, scripted model."""

from __future__ import annotations

import http.server
import json
import re
import threading
import urllib.error

import pytest
from test_web_api import _signup, client  # noqa: F401 - shared fixture

from envforge.web.settings import get_settings

TAILNET = "http://100.101.102.103:8000/v1"


def _second_user(c) -> dict:
    return _signup(c, email="other@example.com", pw="another good password")


# ---- Tailscale (100.64.0.0/10) base URLs ---------------------------------------------------


def test_tailnet_base_url_is_admin_only(client):  # noqa: F811
    _signup(client)  # the first account is the admin
    r = client.put("/api/me/keys", json={"provider": "vllm", "base_url": TAILNET})
    assert r.status_code == 200, r.text
    client.post("/api/auth/logout")

    _second_user(client)
    r = client.put("/api/me/keys", json={"provider": "vllm", "base_url": TAILNET})
    assert r.status_code == 400
    assert "admin" in r.json()["detail"]


def test_worker_rechecks_tailnet_url_against_the_owners_admin_status(client):  # noqa: F811
    from sqlalchemy import select

    from envforge.web import services as svc
    from envforge.web.db import session_factory
    from envforge.web.models import User

    _signup(client)
    assert (
        client.put("/api/me/keys", json={"provider": "vllm", "base_url": TAILNET}).status_code
        == 200
    )
    with session_factory()() as db:
        user = db.scalar(select(User))
        assert svc.resolve_key(db, user.id, "vllm")[1] == TAILNET
        user.is_admin = False  # demoted after saving the key: the worker must refuse it
        db.commit()
        with pytest.raises(ValueError, match="admin"):
            svc.resolve_key(db, user.id, "vllm")


def test_url_imports_never_accept_tailnet_hosts(client):  # noqa: F811
    _signup(client)  # even for the admin: imports run in the builder worker
    ws = client.get("/api/workspaces").json()[0]
    p = client.post(
        f"/api/workspaces/{ws['id']}/builder/projects", json={"title": "t", "purpose": "p"}
    ).json()
    r = client.post(
        f"/api/workspaces/{ws['id']}/builder/projects/{p['id']}/uploads/import",
        json={"url": "https://100.101.102.103/logs.jsonl"},
    )
    assert r.status_code == 400


# ---- redirects from user-supplied hosts --------------------------------------------------------


class _Redirecting(http.server.BaseHTTPRequestHandler):
    """/v1/models bounces to /secret, which would 'leak' if the redirect were followed."""

    def do_GET(self):  # noqa: N802
        if self.path == "/v1/models":
            self.send_response(302)
            self.send_header("Location", "/secret")
            self.end_headers()
        else:
            body = json.dumps({"data": [{"id": "leaked"}]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(body)

    def log_message(self, *a):  # quiet
        pass


@pytest.fixture()
def redirecting_server():
    srv = http.server.HTTPServer(("127.0.0.1", 0), _Redirecting)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    yield f"http://127.0.0.1:{srv.server_port}/v1"
    srv.shutdown()


def test_vllm_key_check_does_not_follow_redirects(redirecting_server):
    from envforge.web import services as svc

    with pytest.raises(urllib.error.HTTPError) as e:
        svc.list_provider_models("vllm", "", redirecting_server)
    assert e.value.code == 302


def test_vllm_model_client_does_not_follow_redirects():
    from envforge.models.openai_compat import OpenAICompatModel

    m = OpenAICompatModel("vllm", "m", base_url=TAILNET, api_key="x")
    assert m._client._client.follow_redirects is False
    hosted = OpenAICompatModel("openai", "gpt-4o-mini", api_key="sk-test")
    assert hosted._client._client.follow_redirects is True  # fixed, trusted URL: unchanged


# ---- client IP for rate limits --------------------------------------------------------------------


def _request(headers: dict[str, str]):
    from starlette.requests import Request

    scope = {
        "type": "http",
        "method": "POST",
        "path": "/",
        "headers": [(k.lower().encode(), v.encode()) for k, v in headers.items()],
        "client": ("10.0.0.9", 1234),
    }
    return Request(scope)


def test_client_ip_reads_only_the_configured_header(monkeypatch):
    from envforge.web.security import client_ip

    get_settings.cache_clear()
    spoof = {"cf-connecting-ip": "1.1.1.1", "x-real-ip": "2.2.2.2"}
    assert client_ip(_request(spoof)) == "10.0.0.9"  # headers a client may send: ignored
    assert client_ip(_request({**spoof, "x-forwarded-for": "3.3.3.3, 10.0.0.1"})) == "3.3.3.3"

    monkeypatch.setenv("CLIENT_IP_HEADER", "cf-connecting-ip")
    get_settings.cache_clear()
    try:
        assert client_ip(_request({**spoof, "x-forwarded-for": "3.3.3.3"})) == "1.1.1.1"
    finally:
        get_settings.cache_clear()


# ---- run ids ---------------------------------------------------------------------------------------


def test_run_id_is_short_and_filesystem_safe_for_any_model_string(client):  # noqa: F811
    _signup(client)
    client.put("/api/me/keys", json={"provider": "anthropic", "secret": "sk-ant-test-0001"})
    ws = client.get("/api/workspaces").json()[0]
    model = "anthropic:" + "../x y/" * 20  # 150 chars of path and whitespace characters
    r = client.post(f"/api/workspaces/{ws['id']}/runs", json={"model": model, "episodes": 1})
    assert r.status_code == 202, r.text
    run_id = r.json()["id"]
    assert len(run_id) <= 80
    assert re.fullmatch(r"[A-Za-z0-9._-]+", run_id), run_id


# ---- sign-out ------------------------------------------------------------------------------


def test_logout_revokes_the_session_token_everywhere(client):  # noqa: F811
    _signup(client)
    name = get_settings().COOKIE_NAME
    copied = client.cookies.get(name)  # e.g. a stolen cookie, or a second browser
    assert client.post("/api/auth/logout").status_code == 204
    client.cookies.set(name, copied)
    assert client.get("/api/auth/me").status_code == 401
    client.cookies.clear()
    assert client.post("/api/auth/logout").status_code == 204  # signed out already: still fine

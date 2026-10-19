"""`envforge admin create-user` / `users --count`: what install.sh relies on."""

from __future__ import annotations

from typer.testing import CliRunner

from envforge.web.admin_cli import admin_app
from tests.test_web_api import client  # noqa: F401 - fixture: fresh in-memory DB + app

runner = CliRunner()


def test_create_admin_then_log_in(client):  # noqa: F811
    assert runner.invoke(admin_app, ["users", "--count"]).output.strip() == "0"
    r = runner.invoke(
        admin_app,
        ["create-user", "Owner@Example.com", "--name", "Owner", "--admin", "--password-stdin"],
        input="a long enough password\n",
    )
    assert r.exit_code == 0, r.output
    assert "admin" in r.output
    assert runner.invoke(admin_app, ["users", "--count"]).output.strip() == "1"

    login = client.post(
        "/api/auth/login", json={"email": "owner@example.com", "password": "a long enough password"}
    )
    assert login.status_code == 200 and login.json()["is_admin"] is True
    assert len(client.get("/api/workspaces").json()) == 1  # a private workspace came with it


def test_create_user_refuses_bad_input(client):  # noqa: F811
    args = ["create-user", "a@example.com", "--password-stdin"]
    assert runner.invoke(admin_app, args, input="short\n").exit_code != 0
    assert runner.invoke(admin_app, args, input="long enough pass\n").exit_code == 0
    again = runner.invoke(admin_app, args, input="long enough pass\n")
    assert again.exit_code != 0 and "already" in again.output
    bad = runner.invoke(
        admin_app, ["create-user", "not-an-email", "--password-stdin"], input="x" * 12
    )
    assert bad.exit_code != 0

"""Deleting a workspace: owner only, never the last one, never while work is active,
and everything in it (rows and files) goes. Offline, SQLite."""

from __future__ import annotations

from pathlib import Path

from sqlalchemy import func, select
from test_web_api import _signup, client  # noqa: F401 - shared fixture

from envforge.web import db as dbmod


def _scripted_run(c, ws_id: str, episodes: int = 2) -> dict:
    r = c.post(f"/api/workspaces/{ws_id}/runs", json={"model": "scripted", "episodes": episodes})
    assert r.status_code == 202, r.text
    from envforge.web.worker import run_forever

    assert run_forever(once=True, kind="runs") == 1
    return c.get(f"/api/workspaces/{ws_id}/runs/{r.json()['id']}").json()


def _add_builder_data(ws_id: str, user_email: str) -> Path:
    """A builder project with a revision, published version, upload, message and a
    finished job, plus files on disk. Returns the project's directory."""
    from envforge.web import builder as B
    from envforge.web.models import (
        BuilderMessage,
        EnvProject,
        EnvRevision,
        EnvUpload,
        EnvVersion,
        Job,
        User,
        Workspace,
    )

    with dbmod.session_factory()() as db:
        ws = db.get(Workspace, ws_id)
        user = db.scalar(select(User).where(User.email == user_email))
        p = EnvProject(workspace_id=ws.id, name="isp", title="ISP", created_by=user.id)
        db.add(p)
        db.flush()
        rev = EnvRevision(project_id=p.id, number=1, message="spec")
        db.add(rev)
        db.flush()
        env_dir = B.environments_dir(ws) / "isp@v1"
        env_dir.mkdir(parents=True)
        (env_dir / "spec.json").write_text("{}")
        db.add(
            EnvVersion(
                project_id=p.id,
                workspace_id=ws.id,
                version=1,
                revision_id=rev.id,
                pack_name="isp@v1",
                env_dir=str(env_dir),
                created_by=user.id,
            )
        )
        pdir = B.project_dir(ws, p)
        (pdir / "tasks").mkdir(parents=True)
        (pdir / "tasks" / "abc.jsonl").write_text("{}\n")
        B.corpus_path(ws, p).write_text("not really sqlite")
        db.add(EnvUpload(project_id=p.id, filename="x.csv", status="ready", created_by=user.id))
        db.add(BuilderMessage(project_id=p.id, seq=1, role="user", content="hi"))
        db.add(Job(kind="builder_turn", project_id=p.id, created_by=user.id, status="completed"))
        db.commit()
        return pdir


def _count(model, **where) -> int:
    with dbmod.session_factory()() as db:
        q = select(func.count()).select_from(model)
        for k, v in where.items():
            q = q.where(getattr(model, k) == v)
        return db.scalar(q)


def test_owner_deletes_workspace_with_runs_builder_data_and_files(client):  # noqa: F811
    from envforge import config
    from envforge.web.models import (
        BuilderMessage,
        EnvProject,
        EnvRevision,
        EnvUpload,
        EnvVersion,
        Job,
        Run,
        Workspace,
        WorkspaceMember,
    )

    _signup(client)
    keep = client.get("/api/workspaces").json()[0]
    doomed = client.post("/api/workspaces", json={"name": "Scratch"}).json()
    run = _scripted_run(client, doomed["id"])
    kept_run = _scripted_run(client, keep["id"])
    pdir = _add_builder_data(doomed["id"], "nick@example.com")
    ws_dir = config.workspace_dir(doomed["slug"])
    assert pdir.exists() and (ws_dir / "runs" / run["id"]).exists()

    assert client.delete(f"/api/workspaces/{doomed['id']}").status_code == 204

    assert not ws_dir.exists()
    assert [w["id"] for w in client.get("/api/workspaces").json()] == [keep["id"]]
    assert client.get(f"/api/workspaces/{doomed['id']}").status_code == 404
    assert _count(Workspace, id=doomed["id"]) == 0
    assert _count(WorkspaceMember, workspace_id=doomed["id"]) == 0
    assert _count(Run, workspace_id=doomed["id"]) == 0
    assert _count(EnvProject) == 0 and _count(EnvVersion) == 0 and _count(EnvRevision) == 0
    assert _count(EnvUpload) == 0 and _count(BuilderMessage) == 0 and _count(Job) == 0
    # the other workspace is untouched
    assert config.workspace_dir(keep["slug"]).exists()
    r = client.get(f"/api/workspaces/{keep['id']}/runs/{kept_run['id']}")
    assert r.status_code == 200
    assert client.get(f"/api/workspaces/{keep['id']}/runs/{kept_run['id']}/episodes").json()


def test_cannot_delete_last_workspace(client):  # noqa: F811
    _signup(client)
    ws = client.get("/api/workspaces").json()[0]
    r = client.delete(f"/api/workspaces/{ws['id']}")
    assert r.status_code == 409 and "only workspace" in r.json()["detail"]
    assert client.get(f"/api/workspaces/{ws['id']}").status_code == 200


def test_non_member_gets_404_and_non_owner_member_403(client):  # noqa: F811
    from envforge.web.models import User, WorkspaceMember

    _signup(client, email="alice@example.com")
    client.get("/api/workspaces")
    alice_ws = client.post("/api/workspaces", json={"name": "Team"}).json()
    client.post("/api/auth/logout")

    _signup(client, email="bob@example.com", pw="bobs-long-password")
    client.post("/api/workspaces", json={"name": "Bob two"})  # so "last workspace" can't apply
    assert client.delete(f"/api/workspaces/{alice_ws['id']}").status_code == 404

    # make Bob a plain member: still not allowed to delete
    with dbmod.session_factory()() as db:
        bob = db.scalar(select(User).where(User.email == "bob@example.com"))
        db.add(WorkspaceMember(workspace_id=alice_ws["id"], user_id=bob.id, role="member"))
        db.commit()
    assert client.delete(f"/api/workspaces/{alice_ws['id']}").status_code == 403
    assert client.get(f"/api/workspaces/{alice_ws['id']}").status_code == 200


def test_active_run_or_builder_job_blocks_delete(client):  # noqa: F811
    from envforge.web.models import Job

    _signup(client)
    client.get("/api/workspaces")
    ws = client.post("/api/workspaces", json={"name": "Busy"}).json()
    url = f"/api/workspaces/{ws['id']}"

    queued = client.post(f"{url}/runs", json={"model": "scripted"}).json()
    r = client.delete(url)
    assert r.status_code == 409 and "queued or running" in r.json()["detail"]
    client.post(f"{url}/runs/{queued['id']}/cancel")

    _add_builder_data(ws["id"], "nick@example.com")
    with dbmod.session_factory()() as db:
        job = db.scalar(select(Job))
        job.status = "running"
        db.commit()
        job_id = job.id
    assert client.delete(url).status_code == 409

    with dbmod.session_factory()() as db:
        db.get(Job, job_id).status = "completed"
        db.commit()
    assert client.delete(url).status_code == 204
    assert client.get(url).status_code == 404

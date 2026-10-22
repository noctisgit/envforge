"""GPU boxes, datasets, training jobs and adapters through the web API. Offline.

No SSH and no GPU: `compute_service.set_box_factory` swaps the SSH connection for
`envforge.compute.fake.FakeGPUBox`, which answers the exact commands envforge sends.
"""

from __future__ import annotations

import io
import json
import tarfile
from contextlib import nullcontext
from pathlib import Path

import pytest
from test_web_api import _signup, client  # noqa: F401 - shared fixture

from envforge.compute.fake import FakeGPUBox
from envforge.web import compute_service as C
from envforge.web import db as dbmod
from envforge.web.settings import get_settings


def _ssh_key() -> str:
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric import ed25519

    return (
        ed25519.Ed25519PrivateKey.generate()
        .private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.OpenSSH,
            serialization.NoEncryption(),
        )
        .decode()
    )


KEY = _ssh_key()
KEY_BODY = KEY.splitlines()[1]  # a line of the base64 body: must never appear in a response


@pytest.fixture()
def fake_boxes():
    """One persistent FakeGPUBox per box row; hosts starting with 'down' are unreachable."""
    boxes: dict[str, FakeGPUBox] = {}

    def factory(db, row):
        if row.id not in boxes:
            boxes[row.id] = FakeGPUBox(row.host, unreachable=row.host.startswith("down"))
        return nullcontext(boxes[row.id])

    C.set_box_factory(factory)
    yield boxes
    C.set_box_factory(None)


@pytest.fixture()
def fast_polls(monkeypatch):
    monkeypatch.setenv("TRAIN_POLL_SECONDS", "0")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def _ws(c) -> dict:
    return c.get("/api/workspaces").json()[0]


def _work(n: int = 1) -> None:
    """Let the compute worker run `n` iterations (a job each, or a training poll)."""
    for _ in range(n):
        with dbmod.session_factory()() as db:
            C.work_once(db)


def _add_box(c, ws_id: str, **kw) -> dict:
    body = {"name": "gpu-1", "host": "100.64.0.7", "ssh_private_key": KEY, "hourly_usd": 0.8, **kw}
    r = c.post(f"/api/workspaces/{ws_id}/gpu-boxes", json=body)
    assert r.status_code == 201, r.text
    return r.json()


def _scripted_runs(c, ws_id: str, *specs: tuple[int, int]) -> list[str]:
    """Queue and execute scripted runs: (seed_start, episodes) each."""
    from envforge.web.worker import run_forever

    ids = []
    for seed, episodes in specs:
        r = c.post(
            f"/api/workspaces/{ws_id}/runs",
            json={"model": "scripted", "pack": "netops", "episodes": episodes, "seed_start": seed},
        )
        assert r.status_code == 202, r.text
        ids.append(r.json()["id"])
        assert run_forever(once=True, kind="runs") == 1
    return ids


def _second_user(c) -> dict:
    """Sign out and sign up user B (not an admin: only the first account is)."""
    c.post("/api/auth/logout")
    _signup(c, email="bee@example.com", pw="another long password")
    return _ws(c)


def _login_a(c) -> None:
    c.post("/api/auth/logout")
    r = c.post(
        "/api/auth/login", json={"email": "nick@example.com", "password": "correct horse battery"}
    )
    assert r.status_code == 200


# ---- boxes: keys, admin gate, checks, serving ------------------------------------------


def test_box_key_is_encrypted_and_never_returned(client, fake_boxes):  # noqa: F811
    _signup(client)
    ws = _ws(client)
    box = _add_box(client, ws["id"], vllm_api_key="my-vllm-key-1234", hf_token="hf_SECRETTOKEN9")
    listing = client.get(f"/api/workspaces/{ws['id']}/gpu-boxes")
    for text in (json.dumps(box), listing.text):
        assert KEY_BODY not in text and "PRIVATE KEY" not in text
        assert "my-vllm-key-1234" not in text and "hf_SECRETTOKEN9" not in text
    assert box["ssh_key_fingerprint"].startswith("SHA256:") and "…" in box["ssh_key_fingerprint"]
    assert box["vllm_key"] == "…1234" and box["hf_token"] == "…KEN9"

    from sqlalchemy import select

    from envforge.web import crypto
    from envforge.web.models import GpuBox

    with dbmod.session_factory()() as db:
        row = db.scalar(select(GpuBox))
        assert KEY_BODY not in row.ssh_key_ciphertext
        assert crypto.decrypt(row.ssh_key_ciphertext).strip() == KEY.strip()


def test_bad_keys_and_hostile_fields_rejected(client):  # noqa: F811
    _signup(client)
    ws = _ws(client)
    url = f"/api/workspaces/{ws['id']}/gpu-boxes"
    base = {"name": "gpu-1", "host": "100.64.0.7", "ssh_private_key": KEY}
    assert client.post(url, json={**base, "ssh_private_key": "x" * 64}).status_code == 400
    assert client.post(url, json={**base, "host": "-oProxyCommand=evil"}).status_code == 400
    assert client.post(url, json={**base, "user": "root;id"}).status_code == 400
    assert client.post(url, json={**base, "name": "Has Space"}).status_code == 400
    assert client.post(url, json={**base, "bind": "0.0.0.0; reboot"}).status_code == 400
    assert client.post(url, json=base).status_code == 201
    assert client.post(url, json=base).status_code == 409  # same name


def test_gpu_boxes_are_admin_only_by_default(client, monkeypatch):  # noqa: F811
    _signup(client)  # first user: admin
    ws_b = _second_user(client)  # not an admin
    url = f"/api/workspaces/{ws_b['id']}/gpu-boxes"
    body = {"name": "mine", "host": "100.64.0.9", "ssh_private_key": KEY}
    r = client.post(url, json=body)
    assert r.status_code == 403 and "admin-only" in r.json()["detail"]
    assert client.get(url).json()["can_manage"] is False

    monkeypatch.setenv("GPU_BOXES_ADMIN_ONLY", "false")
    get_settings.cache_clear()
    try:
        # non-admins may only point the worker at hosts the outbound-URL rules allow
        r = client.post(url, json={**body, "host": "10.1.2.3"})
        assert r.status_code == 400 and "not allowed" in r.json()["detail"]
        r = client.post(url, json=body)  # the tailnet is admin-only too
        assert r.status_code == 400 and "admin-only" in r.json()["detail"]
        assert client.post(url, json={**body, "host": "1.1.1.1"}).status_code == 201
    finally:
        get_settings.cache_clear()


def test_check_job_records_gpu_and_facts(client, fake_boxes):  # noqa: F811
    _signup(client)
    ws = _ws(client)
    box = _add_box(client, ws["id"])
    r = client.post(f"/api/workspaces/{ws['id']}/gpu-boxes/{box['id']}/check")
    assert r.status_code == 202 and r.json()["job"]["kind"] == "box_check"
    # one job per box at a time
    assert client.post(f"/api/workspaces/{ws['id']}/gpu-boxes/{box['id']}/check").status_code == 409
    _work()
    b = client.get(f"/api/workspaces/{ws['id']}/gpu-boxes").json()["boxes"][0]
    assert b["status"] == "ok" and b["job"] is None and b["checked_at"]
    res = b["check_result"]
    assert res["gpus"][0]["name"] == "NVIDIA L40S" and res["gpus"][0]["memory_total_mib"] == 46068
    assert res["driver"] == "550.54.15" and res["cuda"] == "12.4" and res["disk_free_gib"] == 100.0
    assert res["vllm_image"] is True and res["runtime"] == "docker"
    assert b["detected_runtime"] == "docker"


def test_unreachable_box_is_reported_not_hung(client, fake_boxes):  # noqa: F811
    _signup(client)
    ws = _ws(client)
    box = _add_box(client, ws["id"], host="down.example.com", name="down")
    client.post(f"/api/workspaces/{ws['id']}/gpu-boxes/{box['id']}/check")
    _work()
    b = client.get(f"/api/workspaces/{ws['id']}/gpu-boxes").json()["boxes"][0]
    assert b["status"] == "unreachable" and b["job"] is None
    assert b["check_result"]["checks"][0] == {"name": "ssh", "ok": False, "detail": "ssh: connect to host down.example.com port 22: timed out"}  # fmt: skip

    # serving on it fails with a readable error instead of staying "starting"
    client.post(
        f"/api/workspaces/{ws['id']}/gpu-boxes/{box['id']}/serve",
        json={"model": "Qwen/Qwen2.5-0.5B-Instruct"},
    )
    _work()
    b = client.get(f"/api/workspaces/{ws['id']}/gpu-boxes").json()["boxes"][0]
    assert b["serve_state"] == "failed" and "timed out" in b["serve_error"]


def test_serve_shows_in_model_picker_and_runs_use_the_box(client, fake_boxes):  # noqa: F811
    _signup(client)
    ws = _ws(client)
    box = _add_box(client, ws["id"], vllm_api_key="box-vllm-key-5678")
    base = f"/api/workspaces/{ws['id']}/gpu-boxes/{box['id']}"
    bad = client.post(f"{base}/serve", json={"model": "--trust-remote-code"})
    assert bad.status_code == 400
    assert client.post(f"{base}/serve", json={"model": "Qwen/x; rm -rf /"}).status_code == 400
    r = client.post(f"{base}/serve", json={"model": "Qwen/Qwen2.5-0.5B-Instruct"})
    assert r.status_code == 202 and r.json()["serve_state"] == "starting"
    _work()
    b = client.get(f"/api/workspaces/{ws['id']}/gpu-boxes").json()["boxes"][0]
    assert b["serve_state"] == "serving" and b["endpoint_url"] == "http://100.64.0.7:8000/v1"
    fake = fake_boxes[box["id"]]
    assert fake.serving[8000]["model"] == "Qwen/Qwen2.5-0.5B-Instruct"
    run_cmd = next(
        c for c in fake.commands if "docker run --detach" in c or "docker run --env-file" in c
    )
    assert "box-vllm-key-5678" not in run_cmd  # the vLLM key travels on stdin
    assert any(i and "VLLM_API_KEY=box-vllm-key-5678" in i for i in fake.inputs)

    models = client.get(f"/api/workspaces/{ws['id']}/models").json()
    ids = {m["id"] for m in models["models"]}
    assert "vllm:Qwen/Qwen2.5-0.5B-Instruct" in ids and "vllm" in models["providers"]

    # no vLLM key on the account: the run is accepted because the workspace box serves it
    r = client.post(
        f"/api/workspaces/{ws['id']}/runs",
        json={"model": "vllm:Qwen/Qwen2.5-0.5B-Instruct", "episodes": 1},
    )
    assert r.status_code == 202, r.text
    from envforge.web.models import Run

    with dbmod.session_factory()() as db:
        run = db.get(Run, r.json()["id"])
        assert run.gpu_box_id == box["id"]
        assert C.endpoint_for_run(db, run) == ("box-vllm-key-5678", "http://100.64.0.7:8000/v1")
    client.post(f"/api/workspaces/{ws['id']}/runs/{r.json()['id']}/cancel")

    client.post(f"{base}/stop")
    _work()
    b = client.get(f"/api/workspaces/{ws['id']}/gpu-boxes").json()["boxes"][0]
    assert b["serve_state"] == "stopped" and b["endpoint_url"] is None
    ids = {m["id"] for m in client.get(f"/api/workspaces/{ws['id']}/models").json()["models"]}
    assert "vllm:Qwen/Qwen2.5-0.5B-Instruct" not in ids


def test_worker_only_connects_to_the_stored_host(client, fake_boxes):  # noqa: F811
    """A payload naming another host is ignored: the box row decides where we connect."""
    _signup(client)
    ws = _ws(client)
    box = _add_box(client, ws["id"])
    from envforge.web.models import Job, User

    with dbmod.session_factory()() as db:
        user = db.query(User).first()
        db.add(Job(kind="box_check", workspace_id=ws["id"], created_by=user.id,
                   payload={"box_id": box["id"], "host": "evil.example.com"}))  # fmt: skip
        db.commit()
    _work()
    assert [b.host for b in fake_boxes.values()] == ["100.64.0.7"]


# ---- isolation -------------------------------------------------------------------------


def test_isolation_boxes_datasets_jobs_adapters(client, fake_boxes, fast_polls):  # noqa: F811
    """User B can't see or use A's boxes, datasets, training jobs or adapters."""
    _signup(client)
    ws_a = _ws(client)
    box = _add_box(client, ws_a["id"])
    runs = _scripted_runs(client, ws_a["id"], (10_000, 4))
    d = client.post(
        f"/api/workspaces/{ws_a['id']}/datasets", json={"name": "a-data", "run_ids": runs}
    ).json()
    tj = client.post(f"/api/workspaces/{ws_a['id']}/training", json={"dataset_id": d["id"], "box_id": box["id"],
                     "base_model": "Qwen/Qwen2.5-0.5B-Instruct", "max_steps": 4}).json()  # fmt: skip
    _work(5)
    adapter_id = client.get(f"/api/workspaces/{ws_a['id']}/training/{tj['id']}").json()[
        "adapter_id"
    ]
    assert adapter_id

    ws_b = _second_user(client)
    for ws_id in (ws_a["id"], ws_b["id"]):  # A's workspace: 404; B's own with A's ids: 404
        p = f"/api/workspaces/{ws_id}"
        checks = [
            client.get(f"{p}/gpu-boxes"),
            client.delete(f"{p}/gpu-boxes/{box['id']}"),
            client.post(f"{p}/gpu-boxes/{box['id']}/check"),
            client.post(
                f"{p}/gpu-boxes/{box['id']}/serve", json={"model": "Qwen/Qwen2.5-0.5B-Instruct"}
            ),
            client.post(f"{p}/gpu-boxes/{box['id']}/stop"),
            client.get(f"{p}/datasets/{d['id']}"),
            client.get(f"{p}/datasets/{d['id']}/download"),
            client.delete(f"{p}/datasets/{d['id']}"),
            client.post(f"{p}/datasets/preview", json={"run_ids": runs}),
            client.get(f"{p}/training/{tj['id']}"),
            client.post(f"{p}/training/{tj['id']}/cancel"),
            client.get(f"{p}/adapters/{adapter_id}"),
            client.get(f"{p}/adapters/{adapter_id}/download"),
            client.post(f"{p}/adapters/{adapter_id}/serve", json={"box_id": box["id"]}),
            client.post(f"{p}/adapters/{adapter_id}/test", json={"box_id": box["id"]}),
            client.delete(f"{p}/adapters/{adapter_id}"),
        ]
        if ws_id == ws_b["id"]:
            checks = checks[1:]  # B may list B's own (empty) boxes
        for r in checks:
            assert r.status_code in (403, 404), (r.request.method, r.request.url, r.status_code)
            assert r.status_code == 404 or "admin-only" in r.text
    p = f"/api/workspaces/{ws_b['id']}"
    assert client.get(f"{p}/gpu-boxes").json()["boxes"] == []
    assert client.get(f"{p}/datasets").json() == []
    assert client.get(f"{p}/training").json()["jobs"] == []
    assert client.get(f"{p}/adapters").json() == []
    assert not any(m["id"].startswith("vllm:") for m in client.get(f"{p}/models").json()["models"])


# ---- datasets ---------------------------------------------------------------------------


def test_dataset_preview_save_download_delete(client):  # noqa: F811
    _signup(client)
    ws = _ws(client)
    p = f"/api/workspaces/{ws['id']}"
    train_a, train_b, evaluation, crossing = _scripted_runs(
        client, ws["id"], (10_000, 6), (10_000, 6), (0, 3), (9_997, 5)
    )
    sources = {s["id"]: s for s in client.get(f"{p}/datasets/sources").json()}
    assert sources[train_a]["eligible"] and not sources[evaluation]["eligible"]
    assert "evaluation seeds" in sources[evaluation]["reason"]

    r = client.post(f"{p}/datasets/preview", json={"run_ids": [train_a, evaluation]})
    assert r.status_code == 400 and "evaluation seeds" in r.json()["detail"]

    best = client.post(
        f"{p}/datasets/preview", json={"run_ids": [train_a, train_b], "val_frac": 0.3}
    ).json()
    assert best["kept"] == 6 and best["duplicates_dropped"] == 6 and best["tasks"] == 6
    assert best["n_train"] + best["n_val"] == 6 and best["examples_per_task"]["max"] == 1
    # the run on seeds 9997-10001 is an evaluation run that shares tasks 10000-10001
    assert [o["run_id"] for o in best["overlap"]] == [crossing] and best["overlap"][0]["tasks"] == 2
    assert any("share tasks" in w for w in best["warnings"])

    every = client.post(
        f"{p}/datasets/preview", json={"run_ids": [train_a, train_b], "best_per_task": False}
    ).json()
    assert every["kept"] == 12 and every["examples_per_task"] == {"min": 2, "max": 2, "mean": 2.0}

    none = client.post(
        f"{p}/datasets/preview", json={"run_ids": [train_a], "min_reward": 1.0}
    ).json()
    assert none["kept"] == 6  # scripted solves everything
    assert (
        client.post(f"{p}/datasets", json={"name": "Bad Name", "run_ids": [train_a]}).status_code
        == 400
    )

    r = client.post(
        f"{p}/datasets", json={"name": "netops-v1", "run_ids": [train_a, train_b], "val_frac": 0.3}
    )
    assert r.status_code == 201, r.text
    d = r.json()
    assert (
        d["manifest"]["tasks"] == {"netops": [[10_000, 10_005]]} and d["n_train"] + d["n_val"] == 6
    )
    assert (
        client.post(f"{p}/datasets", json={"name": "netops-v1", "run_ids": [train_a]}).status_code
        == 409
    )
    from envforge import config

    folder = config.sft_dir(ws["slug"]) / "netops-v1"
    assert (folder / "train.jsonl").is_file() and (folder / "manifest.json").is_file()
    assert [x["name"] for x in client.get(f"{p}/datasets").json()] == ["netops-v1"]

    dl = client.get(f"{p}/datasets/{d['id']}/download?split=train")
    assert dl.status_code == 200 and "netops-v1-train.jsonl" in dl.headers["content-disposition"]
    first = json.loads(dl.text.splitlines()[0])
    assert first["messages"][0]["role"] == "system" and first["tools"]
    assert client.get(f"{p}/datasets/{d['id']}/download?split=../../etc").status_code == 422

    assert client.delete(f"{p}/datasets/{d['id']}").status_code == 204
    assert not folder.exists() and client.get(f"{p}/datasets").json() == []


# ---- training + adapters --------------------------------------------------------------


def _dataset(client, ws_id: str, name: str = "netops-v1") -> dict:  # noqa: F811
    runs = _scripted_runs(client, ws_id, (10_000, 6))
    r = client.post(f"/api/workspaces/{ws_id}/datasets", json={"name": name, "run_ids": runs})
    assert r.status_code == 201, r.text
    return r.json()


def test_training_job_lifecycle_and_adapter(client, fake_boxes, fast_polls):  # noqa: F811
    _signup(client)
    ws = _ws(client)
    p = f"/api/workspaces/{ws['id']}"
    box = _add_box(client, ws["id"])
    d = _dataset(client, ws["id"])
    body = {"dataset_id": d["id"], "box_id": box["id"], "base_model": "Qwen/Qwen2.5-0.5B-Instruct",
            "lr": 1e-4, "max_steps": 4}  # fmt: skip
    assert client.post(f"{p}/training", json={**body, "lr": 5}).status_code == 400
    assert client.post(f"{p}/training", json={**body, "base_model": "-x"}).status_code == 400
    r = client.post(f"{p}/training", json=body)
    assert r.status_code == 202, r.text
    tj = r.json()
    assert (
        tj["status"] == "queued"
        and tj["adapter_name"] == "netops-v1"
        and tj["hyperparams"]["lr"] == 1e-4
    )
    assert tj["hyperparams"]["epochs"] == 2 and tj["hyperparams"]["lora_r"] == 16  # CLI defaults
    assert client.post(f"{p}/training", json=body).status_code == 409  # same name / box busy

    _work()  # train_start: upload + detached start
    fake = fake_boxes[box["id"]]
    uploaded = {remote for _, remote in fake.puts}
    assert (
        "/root/envforge/jobs/netops-v1/train.jsonl" in uploaded
        and "/root/envforge/jobs/netops-v1/sft.py" in uploaded
    )
    started = client.get(f"{p}/training/{tj['id']}").json()
    assert started["status"] == "running" and started["started_at"]
    assert "--lr 0.0001" in fake.trainings["netops-v1"]["cmd"]

    _work()  # poll 1: running, first loss point
    mid = client.get(f"{p}/training/{tj['id']}").json()
    assert (
        mid["status"] == "running"
        and mid["total_steps"] == 4
        and mid["losses"][0] == {"step": 1, "loss": 2.0}
    )
    assert "ENVFORGE" in mid["log_tail"]
    _work(2)  # poll 2: still running; poll 3: exit 0 -> fetch adapter
    done = client.get(f"{p}/training/{tj['id']}").json()
    assert done["status"] == "completed", done
    assert (
        done["step"] == 4
        and any("eval_loss" in x for x in done["losses"])
        and done["est_cost_usd"] is not None
    )
    assert "netops-v1" not in fake.trainings  # the container was cleaned up

    adapters = client.get(f"{p}/adapters").json()
    assert len(adapters) == 1
    a = adapters[0]
    assert a["id"] == done["adapter_id"] and a["model_spec"] == "vllm:netops-v1"
    assert a["base_model"] == "Qwen/Qwen2.5-0.5B-Instruct" and a["dataset_name"] == "netops-v1"
    assert a["pack"] == "netops" and a["metrics"]["final_eval_loss"] == 0.61

    tgz = client.get(f"{p}/adapters/{a['id']}/download")
    assert tgz.status_code == 200
    names = tarfile.open(fileobj=io.BytesIO(tgz.content), mode="r:gz").getnames()
    assert "netops-v1/adapter_config.json" in names

    # serve the adapter, then test it: an eval run on seeds 0+ against the box endpoint
    r = client.post(f"{p}/adapters/{a['id']}/serve", json={"box_id": box["id"]})
    assert r.status_code == 202
    _work()
    b = client.get(f"{p}/gpu-boxes").json()["boxes"][0]
    assert b["serve_state"] == "serving" and b["served_adapters"] == ["netops-v1"]
    assert fake.serving[8000]["adapters"] == ["netops-v1"]
    assert client.get(f"{p}/adapters/{a['id']}").json()["served_on"][0]["box_name"] == "gpu-1"
    r = client.post(f"{p}/adapters/{a['id']}/test", json={"box_id": box["id"], "episodes": 3})
    assert r.status_code == 202
    _work()
    test_run_id = client.get(f"{p}/adapters/{a['id']}").json()["test_run_id"]
    run = client.get(f"{p}/runs/{test_run_id}").json()
    assert run["model"] == "vllm:netops-v1" and run["seed_start"] == 0 and run["episodes"] == 3


def test_adapter_is_uploaded_to_a_box_that_lacks_it(client, fake_boxes, fast_polls):  # noqa: F811
    _signup(client)
    ws = _ws(client)
    p = f"/api/workspaces/{ws['id']}"
    trainer = _add_box(client, ws["id"])
    other = _add_box(client, ws["id"], name="gpu-2", host="100.64.0.8")
    d = _dataset(client, ws["id"])
    client.post(f"{p}/training", json={"dataset_id": d["id"], "box_id": trainer["id"],
                                       "base_model": "Qwen/Qwen2.5-0.5B-Instruct"})  # fmt: skip
    _work(4)
    a = client.get(f"{p}/adapters").json()[0]
    client.post(f"{p}/adapters/{a['id']}/serve", json={"box_id": other["id"]})
    _work()
    fake2 = fake_boxes[other["id"]]
    assert any(
        remote == "/root/envforge/adapters/" and Path(local).name == "netops-v1"
        for local, remote in fake2.puts
    )
    assert fake2.serving[8000]["adapters"] == ["netops-v1"]


def test_training_failure_and_cancel(client, fake_boxes, fast_polls):  # noqa: F811
    _signup(client)
    ws = _ws(client)
    p = f"/api/workspaces/{ws['id']}"
    box = _add_box(client, ws["id"])
    d = _dataset(client, ws["id"])
    fake_boxes[box["id"]] = FakeGPUBox("100.64.0.7", train_exit=1)
    body = {"dataset_id": d["id"], "box_id": box["id"], "base_model": "Qwen/Qwen2.5-0.5B-Instruct"}
    tj = client.post(f"{p}/training", json=body).json()
    _work(4)
    failed = client.get(f"{p}/training/{tj['id']}").json()
    assert failed["status"] == "failed" and "CUDA out of memory" in failed["error"]
    assert client.get(f"{p}/adapters").json() == []

    # retry under the same name is allowed after a failure; cancel it while it runs
    fake_boxes[box["id"]] = FakeGPUBox("100.64.0.7", train_polls=50)
    tj2 = client.post(f"{p}/training", json=body).json()
    _work(2)  # start + one poll
    assert client.get(f"{p}/training/{tj2['id']}").json()["status"] == "running"
    r = client.post(f"{p}/training/{tj2['id']}/cancel")
    assert r.json()["cancel_requested"] is True
    _work()
    assert client.get(f"{p}/training/{tj2['id']}").json()["status"] == "cancelled"
    assert "netops-v1" not in fake_boxes[box["id"]].trainings

    # a queued job is cancelled at once and its start job never runs
    tj3 = client.post(f"{p}/training", json=body).json()
    assert client.post(f"{p}/training/{tj3['id']}/cancel").json()["status"] == "cancelled"
    _work()
    assert client.get(f"{p}/training/{tj3['id']}").json()["status"] == "cancelled"


def test_training_needs_admin_and_respects_workspace(client, fake_boxes):  # noqa: F811
    _signup(client)
    ws_b = _second_user(client)
    d = _dataset(client, ws_b["id"], name="bee-data")  # datasets are not gated
    r = client.post(f"/api/workspaces/{ws_b['id']}/training",
                    json={"dataset_id": d["id"], "box_id": "nope", "base_model": "Qwen/Qwen2.5-0.5B-Instruct"})  # fmt: skip
    assert r.status_code == 403
    assert client.get(f"/api/workspaces/{ws_b['id']}/training").json()["can_manage"] is False


def test_stale_compute_job_does_not_leave_box_starting(client, fake_boxes):  # noqa: F811
    _signup(client)
    ws = _ws(client)
    box = _add_box(client, ws["id"])
    client.post(
        f"/api/workspaces/{ws['id']}/gpu-boxes/{box['id']}/serve",
        json={"model": "Qwen/Qwen2.5-0.5B-Instruct"},
    )
    from datetime import UTC, datetime, timedelta

    from envforge.web.models import Job
    from envforge.web.worker import reap_stale_jobs

    with dbmod.session_factory()() as db:
        job = db.query(Job).filter(Job.kind == "box_serve").one()
        job.status, job.heartbeat_at = "running", datetime.now(UTC) - timedelta(hours=1)
        db.commit()
        assert reap_stale_jobs(db) == 1
    b = client.get(f"/api/workspaces/{ws['id']}/gpu-boxes").json()["boxes"][0]
    assert b["serve_state"] == "failed" and "no heartbeat" in b["serve_error"]


def test_delete_workspace_removes_boxes_and_training_rows(client, fake_boxes, fast_polls):  # noqa: F811
    _signup(client)
    ws = _ws(client)
    box = _add_box(client, ws["id"])
    d = _dataset(client, ws["id"])
    client.post(f"/api/workspaces/{ws['id']}/training", json={"dataset_id": d["id"], "box_id": box["id"],
                "base_model": "Qwen/Qwen2.5-0.5B-Instruct"})  # fmt: skip
    client.post("/api/workspaces", json={"name": "other"})
    assert client.delete(f"/api/workspaces/{ws['id']}").status_code == 409  # training queued
    _work(4)
    r = client.delete(f"/api/workspaces/{ws['id']}")
    assert r.status_code == 409 and "GPU boxes" in r.json()["detail"]  # vLLM must be stopped
    assert client.delete(f"/api/workspaces/{ws['id']}/gpu-boxes/{box['id']}").status_code == 202
    _work()
    assert client.delete(f"/api/workspaces/{ws['id']}").status_code == 204
    from envforge.web.models import Adapter, Dataset, GpuBox, TrainingJob

    with dbmod.session_factory()() as db:
        for model in (GpuBox, Dataset, TrainingJob, Adapter):
            assert db.query(model).count() == 0


def test_delete_box_stops_vllm_first(client, fake_boxes):  # noqa: F811
    _signup(client)
    ws = _ws(client)
    p = f"/api/workspaces/{ws['id']}/gpu-boxes"
    box = _add_box(client, ws["id"])
    client.post(f"{p}/{box['id']}/check")
    assert client.delete(f"{p}/{box['id']}").status_code == 409  # busy with the check
    _work()
    client.post(f"{p}/{box['id']}/serve", json={"model": "Qwen/Qwen2.5-0.5B-Instruct"})
    _work()
    assert fake_boxes[box["id"]].serving  # vLLM runs on the machine

    assert client.delete(f"{p}/{box['id']}").status_code == 202
    listed = client.get(p).json()["boxes"]
    assert listed[0]["job"]["kind"] == "box_delete"  # still listed until the job ran
    _work()
    assert not fake_boxes[box["id"]].serving  # the vLLM container was removed
    assert client.get(p).json()["boxes"] == []
    assert not C.known_hosts_path(box["id"]).exists()


def test_unreachable_box_is_kept_until_forced(client, fake_boxes):  # noqa: F811
    _signup(client)
    ws = _ws(client)
    p = f"/api/workspaces/{ws['id']}/gpu-boxes"
    box = _add_box(client, ws["id"], host="down.example.com")
    assert client.delete(f"{p}/{box['id']}").status_code == 202
    _work()
    (b,) = client.get(p).json()["boxes"]  # couldn't stop vLLM: the box is not forgotten
    assert b["delete_failed"] is True and "Couldn't stop vLLM" in b["serve_error"]
    assert client.delete(f"{p}/{box['id']}?force=true").status_code == 204
    assert client.get(p).json()["boxes"] == []


# ---- the real SSH path, with subprocess faked -------------------------------------------


def test_ssh_box_uses_only_the_stored_key_and_removes_it(client, monkeypatch):  # noqa: F811
    """open_box writes the key to a private 600 file for the connection only, pins the
    host key per box, and ignores the worker's agent and ssh config."""
    from envforge.compute.base import CommandResult

    _signup(client)
    ws = _ws(client)
    box = _add_box(client, ws["id"])
    seen: dict = {}

    def fake_runner(argv, *, input=None, timeout=None):
        key_path = Path(argv[argv.index("-i") + 1])
        seen.update(argv=argv, timeout=timeout, mode=key_path.stat().st_mode & 0o777,
                    key=key_path.read_text(), key_path=key_path)  # fmt: skip
        return CommandResult(0, "envforge-ok\n", "")

    monkeypatch.setattr(C, "subprocess_runner", fake_runner)
    from envforge.web.models import GpuBox

    with dbmod.session_factory()() as db:
        row = db.get(GpuBox, box["id"])
        with C.open_box(db, row) as cb:
            cb.run(["echo", "envforge-ok"])
    argv = " ".join(seen["argv"])
    assert seen["mode"] == 0o600 and seen["key"].strip() == KEY.strip()
    assert not seen["key_path"].exists() and not seen["key_path"].parent.exists()
    assert (
        "IdentitiesOnly=yes" in argv
        and "IdentityAgent=none" in argv
        and "ProxyCommand=none" in argv
    )
    assert f"UserKnownHostsFile={C.known_hosts_path(box['id'])}" in argv
    assert seen["argv"][-2] == "root@100.64.0.7"
    assert seen["timeout"] == get_settings().COMPUTE_SSH_TIMEOUT_S  # never unbounded

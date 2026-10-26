"""DPO and GRPO in the web app, end to end on a FakeGPUBox: preference datasets, a DPO job
that continues from an SFT adapter, and a two-round GRPO loop whose rollout runs go through
the runs worker (with a scripted model standing in for vLLM)."""

from __future__ import annotations

import json

import pytest
from test_web_api import _signup, client  # noqa: F401 - shared fixture
from test_web_compute import _add_box, _dataset, _scripted_runs, _work, _ws, fake_boxes, fast_polls  # noqa: F401

from envforge.web import db as dbmod
from envforge.web.settings import get_settings


def _until(client, path: str, done, n: int = 12) -> dict:  # noqa: F811
    """Run compute-worker iterations until `done(GET path)` holds."""
    for _ in range(n):
        body = client.get(path).json()
        if done(body):
            return body
        _work()
    body = client.get(path).json()
    assert done(body), body
    return body


def _sft_adapter(client, ws_id: str, box_id: str, name: str = "netops-v1") -> dict:  # noqa: F811
    p = f"/api/workspaces/{ws_id}"
    d = _dataset(client, ws_id, name)
    r = client.post(f"{p}/training", json={"dataset_id": d["id"], "box_id": box_id,
                                           "base_model": "Qwen/Qwen2.5-0.5B-Instruct", "max_steps": 4})  # fmt: skip
    assert r.status_code == 202, r.text
    _until(client, f"{p}/training/{r.json()['id']}", lambda b: b["status"] == "completed")
    return next(a for a in client.get(f"{p}/adapters").json() if a["name"] == name)


def _student_run(client, ws_id: str) -> str:  # noqa: F811
    """A run that fails every other episode on the same training seeds as the teacher."""
    from envforge.web.worker import run_forever

    r = client.post(f"/api/workspaces/{ws_id}/runs",
                    json={"model": "scripted:error_every=2", "pack": "netops", "episodes": 6,
                          "seed_start": 10_000, "label": "student"})  # fmt: skip
    assert r.status_code == 202, r.text
    assert run_forever(once=True, kind="runs") == 1
    return r.json()["id"]


def test_preference_dataset_and_dpo_from_an_sft_adapter(client, fake_boxes, fast_polls):  # noqa: F811
    _signup(client)
    ws = _ws(client)
    p = f"/api/workspaces/{ws['id']}"
    box = _add_box(client, ws["id"])
    sft = _sft_adapter(client, ws["id"], box["id"])
    teacher = _scripted_runs(client, ws["id"], (10_000, 6))[0]
    student = _student_run(client, ws["id"])

    body = {"run_ids": [teacher, student], "kind": "preference", "val_frac": 0.0}
    preview = client.post(f"{p}/datasets/preview", json=body).json()
    assert preview["kind"] == "preference" and preview["kept"] > 0 and preview["overlap"] == []
    assert preview["rejected_episodes"] == 3 and preview["tasks"] == preview["kept"]
    r = client.post(f"{p}/datasets", json={**body, "name": "netops-pairs"})
    assert r.status_code == 201, r.text
    pairs = r.json()
    assert pairs["kind"] == "preference" and pairs["filters"]["max_pairs_per_task"] == 4
    line = json.loads(
        client.get(f"{p}/datasets/{pairs['id']}/download?split=train").text.splitlines()[0]
    )
    assert {"prompt", "chosen", "rejected"} <= set(line)

    sft_dataset = next(d for d in client.get(f"{p}/datasets").json() if d["kind"] == "sft")
    start = {"method": "dpo", "box_id": box["id"], "base_model": "Qwen/Qwen2.5-0.5B-Instruct",
             "adapter_name": "netops-dpo1", "init_adapter": sft["name"], "beta": 0.2}  # fmt: skip
    r = client.post(f"{p}/training", json={**start, "dataset_id": sft_dataset["id"]})
    assert r.status_code == 400 and "preference" in r.json()["detail"]
    r = client.post(
        f"{p}/training", json={**start, "dataset_id": pairs["id"], "init_adapter": None}
    )
    assert r.status_code == 400 and "adapter to start from" in r.json()["detail"]
    r = client.post(
        f"{p}/training", json={**start, "dataset_id": pairs["id"], "base_model": "Qwen/Other"}
    )
    assert r.status_code == 400 and "only works on that base model" in r.json()["detail"]
    r = client.post(f"{p}/training", json={**start, "dataset_id": pairs["id"], "beta": 50})
    assert r.status_code == 400

    r = client.post(f"{p}/training", json={**start, "dataset_id": pairs["id"]})
    assert r.status_code == 202, r.text
    tj = r.json()
    assert (
        tj["method"] == "dpo"
        and tj["init_adapter"] == "netops-v1"
        and tj["hyperparams"]["beta"] == 0.2
    )
    _work()  # train_start
    cmd = fake_boxes[box["id"]].trainings["netops-dpo1"]["cmd"]
    assert (
        "/job/dpo.py" in cmd and "--init-adapter /adapters/netops-v1" in cmd and "--beta 0.2" in cmd
    )
    done = _until(client, f"{p}/training/{tj['id']}", lambda b: b["status"] == "completed")
    a = next(x for x in client.get(f"{p}/adapters").json() if x["name"] == "netops-dpo1")
    assert (
        a["metrics"]["method"] == "dpo" and a["pack"] == "netops" and done["adapter_id"] == a["id"]
    )


def test_grpo_loop_runs_rounds_of_rollouts_and_updates(client, fake_boxes, fast_polls, monkeypatch):  # noqa: F811
    from envforge.models.scripted import ScriptedModel
    from envforge.web import worker
    from envforge.web.models import Run
    from envforge.web.worker import run_forever

    # the rollout runs ask for vllm:<adapter>; a scripted model that botches every other
    # episode stands in, so each group of attempts has passes and failures to compare
    models: list[str] = []
    monkeypatch.setattr(
        worker, "load_model", lambda spec, **kw: models.append(spec) or ScriptedModel(error_every=2)
    )
    monkeypatch.setenv("VLLM_RUN_CONCURRENCY", "1")  # the scripted stand-in isn't thread-safe
    get_settings.cache_clear()

    _signup(client)
    ws = _ws(client)
    p = f"/api/workspaces/{ws['id']}"
    box = _add_box(client, ws["id"])
    sft = _sft_adapter(client, ws["id"], box["id"])
    body = {"method": "grpo", "box_id": box["id"], "base_model": "Qwen/Qwen2.5-0.5B-Instruct",
            "adapter_name": "netops-grpo", "init_adapter": sft["name"], "pack": "netops",
            "rounds": 2, "tasks_per_round": 2, "group_size": 4, "kl_beta": 0.02}  # fmt: skip
    assert client.post(f"{p}/training", json={**body, "group_size": 1}).status_code == 400
    r = client.post(f"{p}/training", json=body)
    assert r.status_code == 202, r.text
    tj = r.json()
    path = f"{p}/training/{tj['id']}"
    assert tj["grpo"]["phase"] == "rollout" and tj["total_steps"] == 2

    fake = fake_boxes[box["id"]]
    for round_no, adapter in ((1, "netops-v1"), (2, "netops-grpo-r1")):
        _until(client, path, lambda b: b["grpo"]["phase"] == "rollouts")
        assert fake.serving[8000]["adapters"] == [adapter]  # the current policy is served
        # serving and testing on the box are refused while it trains
        assert (
            client.post(f"{p}/gpu-boxes/{box['id']}/serve", json={"model": "x/y"}).status_code
            == 409
        )
        run_id = client.get(path).json()["grpo"]["run_id"]
        with dbmod.session_factory()() as db:
            run = db.get(Run, run_id)
            assert (run.repeats, run.temperature, run.episodes) == (4, 1.0, 8)
            assert run.seed_start == 10_000 + (round_no - 1) * 2 and run.model == f"vllm:{adapter}"
        assert run_forever(once=True, kind="runs") == 1
        _until(
            client,
            path,
            lambda b: b["grpo"]["phase"] in ("training", "fetching") or b["status"] != "running",
        )
        out = "netops-grpo-r1" if round_no == 1 else "netops-grpo"
        cmd = fake.trainings[out]["cmd"]
        assert "/job/grpo_step.py" in cmd and f"--policy-adapter /adapters/{adapter}" in cmd
        assert "--ref-adapter /adapters/netops-v1" in cmd and "--kl-beta 0.02" in cmd
        rounds = client.get(path).json()["grpo"]["rounds"]
        assert rounds[-1]["round"] == round_no and rounds[-1]["informative_groups"] == 2
        assert rounds[-1]["trained_on"] == 8 and 0 < rounds[-1]["mean_reward"] < 1

    done = _until(client, path, lambda b: b["status"] == "completed")
    assert done["step"] == 2 and len(done["grpo"]["rounds"]) == 2
    from envforge import config

    assert not list((config.workspace_dir(ws["slug"]) / "grpo").rglob("*.jsonl"))  # uploaded, then removed
    a = next(x for x in client.get(f"{p}/adapters").json() if x["name"] == "netops-grpo")
    assert (
        a["metrics"]["method"] == "grpo"
        and len(a["metrics"]["rounds"]) == 2
        and a["pack"] == "netops"
    )
    assert [m for m in models if m.startswith("vllm:")] == ["vllm:netops-v1", "vllm:netops-grpo-r1"]
    get_settings.cache_clear()


def test_grpo_needs_a_starting_adapter_and_an_environment(client, fake_boxes):  # noqa: F811
    _signup(client)
    ws = _ws(client)
    box = _add_box(client, ws["id"])
    r = client.post(f"/api/workspaces/{ws['id']}/training",
                    json={"method": "grpo", "box_id": box["id"], "base_model": "Qwen/x", "adapter_name": "g"})  # fmt: skip
    assert r.status_code == 400 and "adapter to start from" in r.json()["detail"]


@pytest.mark.parametrize("bad", [{"kind": "other"}, {"max_pairs_per_task": 0}])
def test_preference_dataset_filters_are_checked(client, bad):  # noqa: F811
    _signup(client)
    ws = _ws(client)
    r = client.post(f"/api/workspaces/{ws['id']}/datasets/preview", json={"run_ids": ["x"], **bad})
    assert r.status_code == 422


def test_runs_take_attempts_per_task_and_temperature(client):  # noqa: F811
    from envforge.core.trajectory import load_run
    from envforge.web.worker import run_forever

    _signup(client)
    ws = _ws(client)
    url = f"/api/workspaces/{ws['id']}/runs"
    body = {"model": "scripted", "pack": "netops", "episodes": 6, "seed_start": 10_000}
    assert client.post(url, json={**body, "repeats": 33}).status_code == 422
    r = client.post(url, json={**body, "repeats": 3, "temperature": 0.8})
    assert r.status_code == 202, r.text
    assert r.json()["repeats"] == 3 and r.json()["temperature"] == 0.8
    assert run_forever(once=True, kind="runs") == 1
    with dbmod.session_factory()() as db:
        from envforge.web.models import Run

        row = db.get(Run, r.json()["id"])
        assert [t.task.seed for t in load_run(row.run_dir)] == [10_000] * 3 + [10_001] * 3


def test_grpo_checks_round_size_and_waits_for_a_run_slot(client, fake_boxes, fast_polls):  # noqa: F811
    _signup(client)
    ws = _ws(client)
    p = f"/api/workspaces/{ws['id']}"
    box = _add_box(client, ws["id"])
    sft = _sft_adapter(client, ws["id"], box["id"])
    body = {"method": "grpo", "box_id": box["id"], "base_model": "Qwen/Qwen2.5-0.5B-Instruct",
            "adapter_name": "g1", "init_adapter": sft["name"], "pack": "netops",
            "rounds": 1, "tasks_per_round": 2, "group_size": 4}  # fmt: skip
    r = client.post(f"{p}/training", json={**body, "tasks_per_round": 200, "group_size": 8})
    assert r.status_code == 400 and "at most 1000" in r.json()["detail"]

    # the workspace's run limit is full: the round waits instead of failing the job
    fillers = [client.post(f"{p}/runs", json={"model": "scripted", "pack": "netops", "episodes": 1}).json()["id"]
               for _ in range(3)]  # fmt: skip
    tj = client.post(f"{p}/training", json=body).json()
    path = f"{p}/training/{tj['id']}"
    _work()  # grpo_rollout: serves, then finds no free run slot
    b = client.get(path).json()
    assert b["status"] == "running" and b["grpo"]["phase"] == "queue_rollout", b
    serve_cmds = len(
        [c for c in fake_boxes[box["id"]].commands if "docker run" in c and "vllm" in c]
    )
    for rid in fillers:
        client.post(f"{p}/runs/{rid}/cancel")
    _until(client, path, lambda x: x["grpo"]["phase"] == "rollouts")
    # the retry found the adapter already served: vLLM was not started again
    assert (
        len([c for c in fake_boxes[box["id"]].commands if "docker run" in c and "vllm" in c])
        == serve_cmds
    )

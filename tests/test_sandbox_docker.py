"""Docker backend tests: the isolation promises in envforge/sandbox/docker.py, checked for real.

Skipped automatically when no Docker daemon is reachable. Run just these with
`uv run pytest -m docker`.
"""

import time

import pytest

from envforge.core.runner import run_episode
from envforge.models import load_model
from envforge.packs.repofix import PACK


def _docker_available() -> bool:
    try:
        import docker

        client = docker.from_env(timeout=5)
        client.ping()
        client.close()
        return True
    except Exception:  # noqa: BLE001 - any failure means "no usable daemon"
        return False


pytestmark = [
    pytest.mark.docker,
    pytest.mark.skipif(not _docker_available(), reason="Docker daemon not reachable"),
]


@pytest.fixture
def sandbox(monkeypatch):
    monkeypatch.setenv("ENVFORGE_SANDBOX", "docker")
    from envforge.sandbox import open_sandbox

    with open_sandbox(files={"pkg/hello.py": "print('hello from the box')\n"}) as sb:
        yield sb


def test_files_round_trip(sandbox):
    sandbox.write_files({"a.txt": "alpha\n", "deep/nested/b.py": "x = 1\n"})
    assert sandbox.read_file("deep/nested/b.py") == "x = 1\n"
    assert sandbox.list_files() == ["a.txt", "deep/nested/b.py", "pkg/hello.py"]
    with pytest.raises(FileNotFoundError):
        sandbox.read_file("missing.txt")
    with pytest.raises(FileNotFoundError):
        sandbox.read_file("deep")  # a directory is not a file
    with pytest.raises(ValueError):
        sandbox.read_file("../etc/passwd")


def test_runs_python(sandbox):
    res = sandbox.exec(["python", "pkg/hello.py"])
    assert res.ok and res.output.strip() == "hello from the box"


def test_runs_as_non_root(sandbox):
    assert sandbox.exec(["id", "-u"]).output.strip() == "65534"


def test_has_no_network(sandbox):
    assert sandbox.exec(["ls", "/sys/class/net"]).output.split() == ["lo"]
    probe = "import socket; socket.create_connection(('1.1.1.1', 53), timeout=3)"
    res = sandbox.exec(["python", "-c", probe], timeout=10)
    assert not res.ok and ("unreachable" in res.output or "OSError" in res.output)


def test_filesystem_is_read_only_except_tmp(sandbox):
    assert not sandbox.exec(["touch", "/etc/owned"]).ok  # read-only root fs
    assert not sandbox.exec(["touch", "/work/pkg/owned.py"]).ok  # code can't rewrite the repo
    assert sandbox.exec(["touch", "/tmp/scratch"]).ok  # tmpfs scratch space works


def test_timeout_kills_the_process(sandbox):
    started = time.monotonic()
    res = sandbox.exec(["python", "-c", "while True: pass"], timeout=1)
    assert res.timed_out and not res.ok
    assert time.monotonic() - started < 10
    # and the box is still usable afterwards
    assert sandbox.exec(["true"]).ok


def test_container_is_hardened(sandbox):
    import docker

    client = docker.from_env()
    attrs = client.containers.get(sandbox.container_id).attrs
    client.close()
    host = attrs["HostConfig"]
    assert host["NetworkMode"] == "none"
    assert host["CapDrop"] == ["ALL"]
    assert "no-new-privileges" in host["SecurityOpt"]
    assert host["ReadonlyRootfs"] is True
    assert host["Memory"] == 512 * 1024 * 1024
    assert host["NanoCpus"] == 1_000_000_000
    assert host["PidsLimit"] == 128
    assert attrs["Config"]["User"] == "65534:65534"
    assert attrs["Config"]["Labels"]["envforge"] == "sandbox"
    assert "PYTHONDONTWRITEBYTECODE=1" in attrs["Config"]["Env"]
    assert not any(m["Type"] == "bind" for m in attrs["Mounts"])  # no host paths shared


def test_container_and_volume_removed_after_close(monkeypatch):
    import docker

    monkeypatch.setenv("ENVFORGE_SANDBOX", "docker")
    from envforge.sandbox import open_sandbox

    sb = open_sandbox(files={"x.py": "pass\n"})
    client = docker.from_env()
    ctr = client.containers.get(sb.container_id)
    volumes = [m["Name"] for m in ctr.attrs["Mounts"] if m["Type"] == "volume"]
    assert volumes
    sb.close()
    sb.close()  # idempotent
    with pytest.raises(docker.errors.NotFound):
        client.containers.get(ctr.id)
    for name in volumes:
        with pytest.raises(docker.errors.NotFound):
            client.volumes.get(name)
    client.close()


def test_repofix_episode_end_to_end_on_docker(monkeypatch):
    import docker

    monkeypatch.setenv("ENVFORGE_SANDBOX", "docker")
    traj = run_episode(PACK, PACK.generate_task(seed=2, tier=3), load_model("scripted"))
    assert traj.success and traj.reward == 1.0, traj.verdict
    client = docker.from_env()
    leftovers = client.containers.list(all=True, filters={"label": "envforge=sandbox"})
    client.close()
    assert leftovers == []

"""Docker backend: one locked-down, throwaway container per sandbox.

Every container gets the same hardening (see `DockerSandbox.start`):

    network_mode="none"              no network at all, not even DNS
    user 65534:65534 ("nobody")      never root inside the box
    cap_drop=["ALL"]                 no Linux capabilities
    no-new-privileges                setuid binaries can't escalate
    read_only=True                   the image's filesystem can't be modified
    /work  64 MB tmpfs volume        the project; root-owned 0755, so code under
                                     test can read it but NOT rewrite it
    /tmp   64 MB tmpfs (noexec)      scratch space for the code under test
    mem 512 MB, 1 CPU, 128 pids      no memory/CPU hogs, no fork bombs
    init=True                        a real PID 1 that reaps zombies
    PYTHONDONTWRITEBYTECODE=1        no __pycache__ (the work dir is read-only anyway)
    labels {"envforge": "sandbox"}   so leftovers can be found and reaped
    auto_remove + `sleep 7200`       even a leaked container deletes itself in 2 h

How files get in and out: the Docker archive API (`put_archive` / `get_archive`),
which streams tar files over the Docker socket. We deliberately do NOT bind-mount
host directories: the web worker itself runs in a container that talks to the
*host's* Docker daemon through a mounted socket, so a path on the worker's
filesystem means nothing to the daemon. Everything goes through the socket.

Why /work is a "tmpfs volume" and not a plain `tmpfs=`: the archive API refuses to
write into a container whose root filesystem is read-only, and it can't see plain
tmpfs mounts at all. It *can* write into volumes. A local-driver volume of type
tmpfs gives us the best of both: RAM-backed, size-capped, writable by the archive
API, and deleted together with the container (`remove(v=True)`).

Timeouts are enforced inside the container with coreutils `timeout`, which kills
the whole process group, so the image must ship it (every Debian/Ubuntu image does).
"""

from __future__ import annotations

import io
import tarfile
import time

import docker
from docker.errors import APIError, DockerException, ImageNotFound, NotFound
from docker.types import DriverConfig, Mount

from envforge.sandbox.base import (
    DEFAULT_IMAGE,
    MAX_OUTPUT_BYTES,
    ExecResult,
    Sandbox,
    SandboxError,
    clip_output,
    safe_relpath,
)

WORKDIR = "/work"
SANDBOX_USER = "65534:65534"  # nobody:nogroup on Debian
LABEL_KEY = "envforge"
LABEL_VALUE = "sandbox"
MAX_LIFETIME_S = 2 * 3600  # the container's main process is `sleep MAX_LIFETIME_S`

_images_checked: set[str] = set()  # images known to be present, so we only look once


def docker_client() -> docker.DockerClient:
    """A client for the daemon in $DOCKER_HOST (default: the local socket)."""
    # Generous HTTP timeout: a streamed exec can legitimately run for minutes.
    return docker.from_env(timeout=600)


def ensure_image(client: docker.DockerClient, image: str) -> None:
    """Pull `image` unless the daemon already has it."""
    if image in _images_checked:
        return
    try:
        client.images.get(image)
    except ImageNotFound:
        try:
            client.images.pull(image)
        except DockerException as e:
            raise SandboxError(f"cannot pull sandbox image {image!r}: {e}") from e
    _images_checked.add(image)


def _tar_of(files: dict[str, str]) -> bytes:
    """Pack {relative path: text} into an in-memory tar archive."""
    buf = io.BytesIO()
    now = int(time.time())
    with tarfile.open(fileobj=buf, mode="w") as tar:
        for path, content in files.items():
            data = content.encode("utf-8")
            info = tarfile.TarInfo(safe_relpath(path))
            info.size = len(data)
            info.mode = 0o644
            info.mtime = now
            tar.addfile(info, io.BytesIO(data))
    return buf.getvalue()


class DockerSandbox(Sandbox):
    """A hardened container with the project in /work. See the module docstring."""

    def __init__(
        self,
        image: str = DEFAULT_IMAGE,
        *,
        mem_limit: str = "512m",
        cpus: float = 1.0,
        pids_limit: int = 128,
        work_size: str = "64m",
    ) -> None:
        self.image = image
        self.mem_limit = mem_limit
        self.cpus = cpus
        self.pids_limit = pids_limit
        self.work_size = work_size
        self._client: docker.DockerClient | None = None
        self._container = None  # docker.models.containers.Container once started

    @property
    def container_id(self) -> str:
        if self._container is None:
            raise SandboxError("sandbox not started")
        return self._container.id

    def start(self) -> None:
        try:
            self._client = docker_client()
            ensure_image(self._client, self.image)
        except DockerException as e:
            raise SandboxError(f"Docker is not reachable: {e}") from e

        labels = {LABEL_KEY: LABEL_VALUE, "envforge.started": str(int(time.time()))}
        work = Mount(
            target=WORKDIR,
            source=None,  # anonymous volume: removed along with the container
            type="volume",
            labels=labels,
            driver_config=DriverConfig(
                "local",
                {
                    "type": "tmpfs",
                    "device": "tmpfs",
                    "o": f"size={self.work_size},mode=0755,nosuid,nodev",
                },
            ),
        )
        try:
            self._container = self._client.containers.run(
                self.image,
                command=["sleep", str(MAX_LIFETIME_S)],
                detach=True,
                auto_remove=True,
                init=True,
                hostname="sandbox",
                labels=labels,
                # ---- isolation
                user=SANDBOX_USER,
                network_mode="none",
                cap_drop=["ALL"],
                security_opt=["no-new-privileges"],
                read_only=True,
                mounts=[work],
                tmpfs={"/tmp": "rw,nosuid,nodev,noexec,size=64m"},
                # ---- resource limits
                mem_limit=self.mem_limit,
                memswap_limit=self.mem_limit,  # same as mem_limit = no swap
                nano_cpus=int(self.cpus * 1e9),
                pids_limit=self.pids_limit,
                # ---- process environment
                working_dir=WORKDIR,
                environment={
                    "HOME": "/tmp",
                    "LANG": "C.UTF-8",
                    "PYTHONDONTWRITEBYTECODE": "1",
                    "PYTHONUNBUFFERED": "1",
                },
            )
        except DockerException as e:
            raise SandboxError(f"cannot start sandbox container: {e}") from e

    def exec(self, cmd: list[str], timeout: float = 30.0) -> ExecResult:
        api = self._api()
        # `timeout -k 2 T cmd`: SIGTERM after T seconds, SIGKILL 2 s later if needed.
        argv = ["timeout", "-k", "2", f"{timeout:g}", *cmd]
        started = time.monotonic()
        try:
            exec_id = api.exec_create(self.container_id, argv, workdir=WORKDIR, user=SANDBOX_USER)
            exec_id = exec_id["Id"]
            # Stream the output so a chatty process can't make us buffer gigabytes.
            kept: list[bytes] = []
            size = 0
            for chunk in api.exec_start(exec_id, stream=True):
                if size <= MAX_OUTPUT_BYTES:
                    kept.append(chunk)
                    size += len(chunk)
            info = api.exec_inspect(exec_id)
            for _ in range(50):  # the exit code can lag the end of the stream slightly
                if not info.get("Running"):
                    break
                time.sleep(0.02)
                info = api.exec_inspect(exec_id)
        except DockerException as e:
            raise SandboxError(f"exec failed: {e}") from e

        code = info.get("ExitCode")
        code = -1 if code is None else int(code)
        elapsed = time.monotonic() - started
        # 124 = timeout sent SIGTERM; 137 = it had to SIGKILL (128 + 9). A 137 that
        # came early is something else (e.g. the OOM killer), not our timeout.
        timed_out = code == 124 or (code == 137 and elapsed >= timeout)
        return ExecResult(exit_code=code, output=clip_output(b"".join(kept)), timed_out=timed_out)

    def write_file(self, path: str, content: str) -> None:
        self.write_files({path: content})

    def write_files(self, files: dict[str, str]) -> None:
        # One tar for the whole batch: one round trip instead of one per file.
        # Parent directories are created by the daemon as it extracts.
        try:
            ok = self._container_or_raise().put_archive(WORKDIR, _tar_of(files))
        except DockerException as e:
            raise SandboxError(f"cannot write files: {e}") from e
        if not ok:
            raise SandboxError("cannot write files: put_archive refused")

    def read_file(self, path: str) -> str:
        rel = safe_relpath(path)
        try:
            stream, _stat = self._container_or_raise().get_archive(f"{WORKDIR}/{rel}")
            data = b"".join(stream)
        except NotFound as e:
            raise FileNotFoundError(path) from e
        except DockerException as e:
            raise SandboxError(f"cannot read {path}: {e}") from e
        with tarfile.open(fileobj=io.BytesIO(data)) as tar:
            member = tar.next()
            if member is None or not member.isfile():
                raise FileNotFoundError(path)  # e.g. it's a directory
            fh = tar.extractfile(member)
            assert fh is not None
            return fh.read().decode("utf-8", errors="replace")

    def list_files(self) -> list[str]:
        res = self.exec(["find", ".", "-type", "f"], timeout=20)
        if not res.ok:
            raise SandboxError(f"cannot list files: {res.output.strip()}")
        paths = (line.strip() for line in res.output.splitlines())
        return sorted(p.removeprefix("./") for p in paths if p.startswith("./"))

    def close(self) -> None:
        if self._container is not None:
            try:
                self._container.remove(force=True, v=True)  # v=True also drops /work
            except NotFound:
                pass  # auto_remove beat us to it
            except APIError as e:
                if e.status_code != 409:  # 409 = removal already in progress
                    raise SandboxError(f"cannot remove container: {e}") from e
            self._container = None
        if self._client is not None:
            self._client.close()
            self._client = None

    # ---- helpers ---------------------------------------------------------
    def _container_or_raise(self):
        if self._container is None:
            raise SandboxError("sandbox not started")
        return self._container

    def _api(self) -> docker.APIClient:
        if self._client is None:
            raise SandboxError("sandbox not started")
        return self._client.api


def reap_sandboxes(older_than_s: float = 0.0) -> int:
    """Remove leftover envforge sandbox containers (and their volumes).

    Normally `close()` does this; reaping is for crashes (a killed worker, a
    power cut). With `older_than_s` > 0 only containers started that long ago
    are touched, so it is safe to run while other workers are busy.
    Returns how many containers were removed.
    """
    client = docker_client()
    cutoff = time.time() - older_than_s
    removed = 0
    try:
        label = f"{LABEL_KEY}={LABEL_VALUE}"
        for ctr in client.containers.list(all=True, filters={"label": label}):
            started = float(ctr.labels.get("envforge.started", "0"))
            if older_than_s and started > cutoff:
                continue
            try:
                ctr.remove(force=True, v=True)
                removed += 1
            except (NotFound, APIError):
                pass  # already going away
        for vol in client.volumes.list(filters={"label": label, "dangling": "true"}):
            try:
                vol.remove()
            except (NotFound, APIError):
                pass  # still in use by a live sandbox
    finally:
        client.close()
    return removed

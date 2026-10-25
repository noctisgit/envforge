"""GPU boxes in the browser: storage, access rules, and the worker jobs that drive them.

A *GPU box* row holds what `envforge compute ...` takes as flags (host, SSH port, user,
runtime) plus how to log in: either its own SSH private key, or (the default) the
*workspace SSH key*, a key pair envforge generates per workspace whose public half you
paste once into RunPod/Vast so every new pod accepts it. Private keys are encrypted with
the same Fernet helper as provider keys; the API only returns public keys and (masked)
fingerprints.

How the control plane reaches vLLM on a box (`vllm_access`): `direct` talks to
`http://<endpoint host>:<port>/v1` (Tailscale, or a port the provider opens); `tunnel`
forwards a local port over the box's SSH connection (compute/tunnel.py), so a pod needs
nothing but SSH reachable and vLLM listens on 127.0.0.1 only. Everything that
touches the box runs in the worker (`envforge-worker --kind compute`) as a row in the
`jobs` table, reusing `envforge.compute` unchanged:

    box_check  ->  compute.check + gpu_info + box_facts   (GPU, VRAM, driver, disk, vLLM)
    box_serve  ->  compute.serve_vllm + wait_healthy      (optionally with LoRA adapters)
    box_stop   ->  compute.stop_vllm
    box_delete ->  compute.stop_vllm, then forget the box (its row, key and known_hosts)

Security rules, all enforced here (routers stay thin):

- A box belongs to one workspace; every lookup is by (workspace, box id), so another
  user's box id answers 404 like any other unknown id.
- Creating, checking, serving on and training on boxes needs an admin while
  `GPU_BOXES_ADMIN_ONLY` is true (the default).
- Job payloads carry *ids only*. The worker connects to the host stored on the box row,
  with that row's key, never to a host named in a payload.
- Model ids and adapter names are validated (regex) and every remote command is built
  as an argv list (`shlex`-quoted by `ComputeBox.run`).
- The decrypted SSH key exists only inside `open_box`: a mode-600 file in a private
  temporary directory, removed when the job's connection closes. It is never logged.
- The workspace SSH key belongs to one workspace: a box only ever logs in with its own
  key or its own workspace's key, so one workspace's key never opens another's pods.
- A box created by a non-admin must resolve to a public address (private ranges and
  the tailnet are admin-only, like vLLM base URLs); checked when saved and again
  before every connection and on its endpoint URL.
"""

from __future__ import annotations

import base64
import contextlib
import hashlib
import ipaddress
import re
import secrets
import shutil
import socket
import tempfile
import threading
import traceback
from collections.abc import Callable, Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from fastapi import HTTPException
from sqlalchemy import func, select, update
from sqlalchemy.orm import Session

from envforge import config
from envforge.compute.base import CommandResult, ComputeBox, ComputeError
from envforge.compute.host import box_facts, check, gpu_info
from envforge.compute.runtime import RUNTIMES, detect_runtime, probe_runtimes
from envforge.compute.ssh import SSHBox, subprocess_runner
from envforge.compute.vllm import (
    ADAPTERS_DIR,
    serve_vllm,
    stop_vllm,
    vllm_logs,
    vllm_status,
    wait_healthy,
)
from envforge.web import crypto
from envforge.web.models import GpuBox, Job, Run, TrainingJob, User, Workspace
from envforge.web.settings import get_settings

#: Job kinds the compute worker claims.
COMPUTE_JOB_KINDS = (
    "box_check", "box_serve", "box_stop", "box_delete", "train_start", "adapter_test"
)  # fmt: skip
ACTIVE = ("queued", "running")

_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{0,62}$")
# Hugging Face ids: "org/name", dots, dashes, underscores. Never a leading "-" (a flag),
# never whitespace, never "..".
_MODEL_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/-]{0,199}$")
_HOST_RE = re.compile(r"^[A-Za-z0-9_.:%\[\]-]+$")
_BIND_RE = re.compile(r"^[A-Za-z0-9_.:%\[\]][A-Za-z0-9_.:%\[\]-]{0,63}$")


# ---- validation and access -------------------------------------------------------


def require_gpu_admin(user: User) -> None:
    """403 unless the user may manage GPU boxes (admins, or anyone if the setting is off)."""
    if get_settings().GPU_BOXES_ADMIN_ONLY and not user.is_admin:
        raise HTTPException(
            403, "GPU boxes are admin-only on this server (GPU_BOXES_ADMIN_ONLY=true)"
        )


def can_manage(user: User) -> bool:
    """Whether the UI should offer the add / check / serve / train buttons."""
    return user.is_admin or not get_settings().GPU_BOXES_ADMIN_ONLY


def check_model_id(model: str) -> str:
    """A Hugging Face model id safe to pass to vLLM / sft.py as one argument."""
    model = (model or "").strip()
    if not _MODEL_RE.match(model) or ".." in model:
        raise ValueError(
            f"invalid model id {model!r}: expected something like Qwen/Qwen2.5-7B-Instruct"
        )
    return model


def check_box_name(name: str) -> str:
    """Box names: short lowercase slugs (shown in the model picker)."""
    name = (name or "").strip()
    if not _NAME_RE.match(name):
        raise ValueError("box name: lowercase letters, digits, '.', '_' or '-' (max 63)")
    return name


def ssh_key_fingerprint(private_key: str) -> str:
    """`SHA256:...` of the public half, as `ssh-keygen -l` prints it.

    Also validates the key: OpenSSH or PEM private keys without a passphrase only (the
    worker connects non-interactively, with BatchMode)."""
    from cryptography.hazmat.primitives import serialization

    data = private_key.strip().encode() + b"\n"
    key: Any = None
    try:
        key = serialization.load_ssh_private_key(data, password=None)
    except TypeError as e:
        raise ValueError(
            "passphrase-protected keys aren't supported; use a dedicated key without one"
        ) from e
    except ValueError:
        try:
            key = serialization.load_pem_private_key(data, password=None)
        except TypeError as e:
            raise ValueError(
                "passphrase-protected keys aren't supported; use a dedicated key without one"
            ) from e
        except ValueError as e:
            raise ValueError(
                "that doesn't look like an SSH private key (OpenSSH or PEM format)"
            ) from e
    blob = key.public_key().public_bytes(
        serialization.Encoding.OpenSSH, serialization.PublicFormat.OpenSSH
    )
    return public_key_fingerprint(blob.decode())


def public_key_fingerprint(public_key: str) -> str:
    """`SHA256:...` of an OpenSSH public key line (`ssh-ed25519 AAAA... comment`)."""
    raw = base64.b64decode(public_key.split()[1])
    digest = base64.b64encode(hashlib.sha256(raw).digest()).decode().rstrip("=")
    return f"SHA256:{digest}"


def mask_fingerprint(fp: str) -> str:
    """`SHA256:AbCdEfGh…wXyZ`: enough to recognise a key, not the whole thing."""
    if not fp.startswith("SHA256:") or len(fp) < 20:
        return "…"
    body = fp[len("SHA256:") :]
    return f"SHA256:{body[:8]}…{body[-4:]}"


def _host_allowed(host: str) -> None:
    """Raise ValueError unless every address `host` resolves to is a public one.

    Applies to boxes added by non-admins (only possible with GPU_BOXES_ADMIN_ONLY=false).
    Private ranges *and the tailnet* are admin-only, the same policy as vLLM base URLs
    (security.check_base_url): otherwise any user could make the worker ssh into, or
    send requests to, the owner's private network.
    """
    bare = host.strip("[]")
    try:
        infos = socket.getaddrinfo(bare, 22, type=socket.SOCK_STREAM)
    except socket.gaierror as e:
        raise ValueError(f"cannot resolve {host}") from e
    for info in infos:
        ip = ipaddress.ip_address(info[4][0])
        if not ip.is_global:
            raise ValueError(
                f"{host} resolves to {ip}, which is not allowed for boxes added by non-admins: "
                "private and Tailscale addresses are admin-only. Use a public host, or ask an admin."
            )


def _url_allowed(url: str) -> str:
    """The endpoint URL of a non-admin's box must point at a public host too."""
    from urllib.parse import urlparse

    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        raise ValueError(f"invalid endpoint URL {url!r}")
    _host_allowed(parsed.hostname)
    return url


def _creator_is_admin(db: Session, box: GpuBox) -> bool:
    u = db.get(User, box.created_by)
    return bool(u and u.is_admin)


# ---- the workspace SSH key ---------------------------------------------------------------


def _new_key_pair(comment: str) -> tuple[str, str]:
    """(OpenSSH private key, `ssh-ed25519 AAAA... comment` public line), no passphrase."""
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric import ed25519

    key = ed25519.Ed25519PrivateKey.generate()
    private = key.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.OpenSSH,
        serialization.NoEncryption(),
    ).decode()
    public = (
        key.public_key()
        .public_bytes(serialization.Encoding.OpenSSH, serialization.PublicFormat.OpenSSH)
        .decode()
    )
    return private, f"{public} envforge-{comment}"


def _store_workspace_key(db: Session, ws: Workspace) -> None:
    private, public = _new_key_pair(ws.slug)
    ws.ssh_key_ciphertext = crypto.encrypt(private)
    ws.ssh_public_key = public
    ws.ssh_key_created_at = datetime.now(UTC)
    db.commit()


def workspace_key_out(ws: Workspace) -> dict[str, Any] | None:
    if not ws.ssh_public_key:
        return None
    return {
        "public_key": ws.ssh_public_key,
        "fingerprint": public_key_fingerprint(ws.ssh_public_key),
        "created_at": ws.ssh_key_created_at,
    }


def workspace_ssh_key(db: Session, ws: Workspace, user: User) -> dict[str, Any] | None:
    """The workspace's public SSH key, generated the first time someone who may manage
    GPU boxes asks for it. Members who can't manage boxes see it once it exists."""
    if not ws.ssh_public_key and can_manage(user):
        _store_workspace_key(db, ws)
    return workspace_key_out(ws)


def rotate_workspace_key(db: Session, ws: Workspace, user: User) -> dict[str, Any] | None:
    """Replace the key pair. Pods that only know the old public key stop accepting the
    boxes that use the workspace key until the new one is added to the provider."""
    require_gpu_admin(user)
    _store_workspace_key(db, ws)
    return workspace_key_out(ws)


def _box_private_key(db: Session, box: GpuBox) -> str:
    """The key this box logs in with: its own, or its workspace's. Worker only."""
    if box.ssh_key_ciphertext:
        return crypto.decrypt(box.ssh_key_ciphertext)
    ws = db.get(Workspace, box.workspace_id)
    if ws is None or not ws.ssh_key_ciphertext:
        raise ComputeError(
            "this box logs in with the workspace SSH key, but the workspace has none yet; "
            "open Settings → GPU boxes to create it"
        )
    return crypto.decrypt(ws.ssh_key_ciphertext)


def default_vllm_access(host: str) -> str:
    """`direct` for tailnet / private addresses and names (Tailscale MagicDNS, ssh aliases),
    `tunnel` for public ones, where only the provider's SSH port is reliably reachable."""
    bare = host.strip("[]")
    try:
        ip = ipaddress.ip_address(bare)
    except ValueError:
        return "direct" if bare.endswith(".ts.net") or "." not in bare else "tunnel"
    return "tunnel" if ip.is_global else "direct"


def get_box(db: Session, ws: Workspace, box_id: str) -> GpuBox:
    """The box with this id *in this workspace*, or 404."""
    box = db.get(GpuBox, box_id)
    if box is None or box.workspace_id != ws.id:
        raise HTTPException(404, "GPU box not found")
    return box


def list_boxes(db: Session, ws: Workspace) -> list[GpuBox]:
    return list(
        db.scalars(select(GpuBox).where(GpuBox.workspace_id == ws.id).order_by(GpuBox.created_at))
    )


def active_box_job(db: Session, box: GpuBox) -> Job | None:
    """The queued/running job on this box, if any (box jobs and training starts)."""
    for job in db.scalars(
        select(Job)
        .where(Job.workspace_id == box.workspace_id, Job.status.in_(ACTIVE))
        .order_by(Job.created_at)
    ):
        if (job.payload or {}).get("box_id") == box.id:
            return job
    return None


def _delete_failed(db: Session, box: GpuBox) -> bool:
    """Whether the box's latest job was a removal that couldn't reach it (the UI then
    offers to forget the box without stopping vLLM)."""
    for job in db.scalars(
        select(Job)
        .where(Job.workspace_id == box.workspace_id, Job.kind.in_(COMPUTE_JOB_KINDS))
        .order_by(Job.created_at.desc())
    ):
        if (job.payload or {}).get("box_id") == box.id:
            return job.kind == "box_delete" and job.status == "failed"
    return False


def box_out(db: Session, box: GpuBox) -> dict[str, Any]:
    """What the API returns for a box: no secret, only masked forms."""
    job = active_box_job(db, box)
    if box.ssh_key_ciphertext:
        fingerprint = box.ssh_key_fingerprint or ""
    else:
        ws = db.get(Workspace, box.workspace_id)
        fingerprint = public_key_fingerprint(ws.ssh_public_key) if ws and ws.ssh_public_key else ""
    return {
        "id": box.id,
        "name": box.name,
        "host": box.host,
        "port": box.port,
        "user": box.user,
        "runtime": box.runtime,
        "detected_runtime": box.detected_runtime,
        "ssh_key_source": "custom" if box.ssh_key_ciphertext else "workspace",
        "ssh_key_fingerprint": mask_fingerprint(fingerprint),
        "vllm_key": box.vllm_key_masked,
        "vllm_key_generated": box.vllm_key_generated,
        "hf_token": box.hf_token_masked,
        "vllm_port": box.vllm_port,
        "endpoint_host": box.endpoint_host,
        "bind": box.bind,
        "vllm_access": box.vllm_access,
        "hourly_usd": box.hourly_usd,
        "status": box.status,
        "check_result": box.check_result,
        "checked_at": box.checked_at,
        "serve_state": box.serve_state,
        "served_model": box.served_model,
        "served_adapters": box.served_adapters or [],
        "endpoint_url": box.endpoint_url,
        "serve_error": box.serve_error,
        "delete_failed": job is None and _delete_failed(db, box),
        "created_at": box.created_at,
        "job": {"id": job.id, "kind": job.kind, "status": job.status, "progress": job.progress}
        if job
        else None,
    }


def create_box(db: Session, ws: Workspace, user: User, data: dict[str, Any]) -> GpuBox:
    """Validate and store a new box. `data` is the request body as a dict."""
    require_gpu_admin(user)
    try:
        name = check_box_name(data["name"])
        host = (data["host"] or "").strip()
        user_name = (data.get("user") or "root").strip()
        port = int(data.get("port") or 22)
        SSHBox(host, user=user_name, port=port)  # same host/user/port rules as the CLI
        runtime = data.get("runtime") or "auto"
        if runtime not in (*RUNTIMES, "auto"):
            raise ValueError("runtime must be docker, pod or auto")
        vllm_port = int(data.get("vllm_port") or 8000)
        if not 1 <= vllm_port <= 65535:
            raise ValueError("vLLM port must be 1-65535")
        endpoint_host = (data.get("endpoint_host") or "").strip() or None
        if endpoint_host and (endpoint_host.startswith("-") or not _HOST_RE.match(endpoint_host)):
            raise ValueError("invalid endpoint host")
        bind = (data.get("bind") or "").strip() or None
        if bind and not _BIND_RE.match(bind):
            raise ValueError("invalid bind address")
        hourly = data.get("hourly_usd")
        if hourly is not None and not 0 <= float(hourly) <= 1000:
            raise ValueError("$/hour must be between 0 and 1000")
        private_key = (data.get("ssh_private_key") or "").strip()
        fingerprint = ssh_key_fingerprint(private_key) if private_key else None
        vllm_access = data.get("vllm_access") or default_vllm_access(host)
        if vllm_access not in ("direct", "tunnel"):
            raise ValueError("vLLM access must be direct or tunnel")
        vllm_key = (data.get("vllm_api_key") or "").strip()
        hf_token = (data.get("hf_token") or "").strip()
        for label, value in (("vLLM API key", vllm_key), ("Hugging Face token", hf_token)):
            if value and (len(value) < 8 or any(c.isspace() for c in value)):
                raise ValueError(f"{label}: at least 8 characters, no spaces")
        if not user.is_admin:
            _host_allowed(host)
            if endpoint_host:
                _host_allowed(endpoint_host)
    except (KeyError, TypeError) as e:
        raise HTTPException(400, f"missing or invalid field: {e}") from e
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    if db.scalar(select(GpuBox).where(GpuBox.workspace_id == ws.id, GpuBox.name == name)):
        raise HTTPException(409, f"a box named {name} already exists in this workspace")

    if not private_key and not ws.ssh_key_ciphertext:
        _store_workspace_key(db, ws)  # no key pasted: log in with the workspace's
    generated = not vllm_key
    vllm_key = vllm_key or secrets.token_urlsafe(32)
    box = GpuBox(
        workspace_id=ws.id,
        name=name,
        host=host,
        port=port,
        user=user_name,
        runtime=runtime,
        ssh_key_ciphertext=crypto.encrypt(private_key + "\n") if private_key else None,
        ssh_key_fingerprint=fingerprint,
        vllm_key_ciphertext=crypto.encrypt(vllm_key),
        vllm_key_masked=crypto.mask(vllm_key),
        vllm_key_generated=generated,
        hf_token_ciphertext=crypto.encrypt(hf_token) if hf_token else None,
        hf_token_masked=crypto.mask(hf_token) if hf_token else None,
        vllm_port=vllm_port,
        endpoint_host=endpoint_host,
        bind=bind,
        vllm_access=vllm_access,
        hourly_usd=float(hourly) if hourly is not None else None,
        created_by=user.id,
    )
    db.add(box)
    db.commit()
    return box


def delete_box(
    db: Session, ws: Workspace, user: User, box: GpuBox, *, force: bool = False
) -> Job | None:
    """Remove a box and what it runs: queues `box_delete`, which stops vLLM on the machine
    and then forgets the box. `force` forgets it at once without touching the machine, for
    a box that can't be reached any more (say, the rental is already gone). Refused while
    a job or a training run is using the box."""
    require_gpu_admin(user)
    if active_box_job(db, box):
        raise HTTPException(409, "a job is running on this box; wait for it to finish")
    busy = db.scalar(
        select(func.count()).select_from(TrainingJob).where(
            TrainingJob.box_id == box.id, TrainingJob.status.in_(("queued", "starting", "running", "fetching"))
        )
    )  # fmt: skip
    if busy:
        raise HTTPException(409, "a training job is running on this box; cancel it first")
    if not force:
        return enqueue_box_job(db, box, user, "box_delete")
    _forget_box(db, box)
    return None


def _forget_box(db: Session, box: GpuBox) -> None:
    """Delete the row (and with it the encrypted key) and the pinned host key."""
    db.execute(update(TrainingJob).where(TrainingJob.box_id == box.id).values(box_id=None))
    db.delete(box)
    db.commit()
    known_hosts_path(box.id).unlink(missing_ok=True)


# ---- jobs (API side) ------------------------------------------------------------------


def enqueue_box_job(
    db: Session, box: GpuBox, user: User, kind: str, payload: dict | None = None
) -> Job:
    """Queue `kind` for this box. One job per box at a time."""
    if active_box_job(db, box):
        raise HTTPException(409, "this box is busy with another job; wait for it to finish")
    job = Job(
        kind=kind,
        workspace_id=box.workspace_id,
        created_by=user.id,
        payload={**(payload or {}), "box_id": box.id},
    )
    db.add(job)
    db.commit()
    return job


def request_check(db: Session, ws: Workspace, user: User, box: GpuBox) -> Job:
    require_gpu_admin(user)
    return enqueue_box_job(db, box, user, "box_check")


def _lora_rank_bucket(ranks: list[int]) -> int:
    """vLLM preallocates for --max-lora-rank; it must be >= every adapter's rank."""
    need = max([16, *ranks])
    return next((r for r in (8, 16, 32, 64, 128, 256, 512) if r >= need), need)


def request_serve(
    db: Session,
    ws: Workspace,
    user: User,
    box: GpuBox,
    *,
    model: str,
    adapters: list[str] | None = None,
    max_model_len: int | None = None,
    gpu_memory_utilization: float = 0.90,
) -> Job:
    """Queue `box_serve`: vLLM with `model` (+ adapters from this workspace)."""
    from envforge.web.models import Adapter

    require_gpu_admin(user)
    try:
        model = check_model_id(model)
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    names: list[str] = []
    ranks: list[int] = []
    for name in adapters or []:
        a = db.scalar(select(Adapter).where(Adapter.workspace_id == ws.id, Adapter.name == name))
        if a is None:
            raise HTTPException(400, f"no adapter named {name!r} in this workspace")
        if a.base_model != model:
            raise HTTPException(400, f"adapter {name} was trained on {a.base_model}, not {model}")
        names.append(a.name)
        if a.lora_rank:
            ranks.append(a.lora_rank)
    if max_model_len is not None and not 256 <= max_model_len <= 1_048_576:
        raise HTTPException(400, "max model length must be 256..1048576")
    if not 0.1 <= gpu_memory_utilization <= 0.98:
        raise HTTPException(400, "GPU memory fraction must be 0.1..0.98")
    job = enqueue_box_job(
        db,
        box,
        user,
        "box_serve",
        {
            "model": model,
            "adapters": names,
            "max_model_len": max_model_len,
            "gpu_memory_utilization": gpu_memory_utilization,
            "max_lora_rank": _lora_rank_bucket(ranks),
        },
    )
    box.serve_state, box.serve_error = "starting", None
    box.served_model, box.served_adapters = model, names
    db.commit()
    return job


def request_stop(db: Session, ws: Workspace, user: User, box: GpuBox) -> Job:
    require_gpu_admin(user)
    job = enqueue_box_job(db, box, user, "box_stop")
    box.serve_state = "stopping"
    db.commit()
    return job


# ---- models served by boxes (picker + runs) ---------------------------------------------


def box_models(db: Session, ws: Workspace) -> list[dict[str, Any]]:
    """`vllm:` models the workspace's serving boxes offer, shaped like catalog entries."""
    out: list[dict[str, Any]] = []
    for box in list_boxes(db, ws):
        if box.serve_state != "serving" or not box.served_model:
            continue
        base = box.served_model
        common = {"provider": "vllm", "owned_by": f"GPU box {box.name}", "context_window": 0,
                  "input_per_mtok": None, "output_per_mtok": None, "released": None}  # fmt: skip
        out.append({**common, "id": f"vllm:{base}", "name": f"{base.split('/')[-1]} · {box.name}",
                    "tags": ["gpu-box", "open-weights"]})  # fmt: skip
        for a in box.served_adapters or []:
            out.append({**common, "id": f"vllm:{a}", "name": f"{a} (LoRA on {base.split('/')[-1]}) · {box.name}",
                        "tags": ["gpu-box", "lora"]})  # fmt: skip
    return out


def box_for_model(db: Session, ws: Workspace, model_spec: str) -> GpuBox | None:
    """The workspace box currently serving `vllm:<id>`, if any."""
    if not model_spec.startswith("vllm:"):
        return None
    model_id = model_spec[len("vllm:") :]
    for box in list_boxes(db, ws):
        if box.serve_state == "serving" and (
            box.served_model == model_id or model_id in (box.served_adapters or [])
        ):
            return box
    return None


def _tunnel_target(box: GpuBox) -> str:
    """Where vLLM listens as seen from the box itself: its bind address, or loopback."""
    if not box.bind or box.bind in ("0.0.0.0", "::", "[::]"):
        return "127.0.0.1"
    return box.bind


@contextlib.contextmanager
def box_base_url(box: GpuBox, cb: ComputeBox) -> Iterator[str]:
    """The base URL the control plane uses for the box's vLLM while the block runs: the
    stored endpoint, or (vllm_access=tunnel) a local port forwarded over `cb`'s SSH."""
    if box.vllm_access == "tunnel":
        with cb.tunnel(box.vllm_port, _tunnel_target(box)) as t:
            yield t.url("/v1")
    else:
        yield box.endpoint_url or ""


def endpoint_for_run(
    db: Session, run: Run, stack: contextlib.ExitStack | None = None
) -> tuple[str, str]:
    """(api_key, base_url) for a run on a box endpoint. Worker only: decrypts the key.

    A tunnel box needs `stack`: the SSH connection and the tunnel are entered on it, so
    they stay open until the caller closes the stack (when the run ends)."""
    box = db.get(GpuBox, run.gpu_box_id) if run.gpu_box_id else None
    if box is None or box.workspace_id != run.workspace_id:
        raise PermissionError("the GPU box this run was queued on no longer exists")
    model_id = run.model.split(":", 1)[1]
    if (
        box.serve_state != "serving"
        or not box.endpoint_url
        or (model_id != box.served_model and model_id not in (box.served_adapters or []))
    ):
        raise RuntimeError(f"GPU box {box.name} is not serving {model_id} any more; serve it again")
    if box.vllm_access == "tunnel":
        if stack is None:
            raise RuntimeError(f"GPU box {box.name} is reached through an SSH tunnel (worker only)")
        cb = stack.enter_context(open_box(db, box))  # re-checks a non-admin's host
        base_url = stack.enter_context(box_base_url(box, cb))
    else:
        base_url = box.endpoint_url
        if not _creator_is_admin(db, box):
            base_url = _url_allowed(base_url)  # re-check: DNS can change
    return crypto.decrypt(box.vllm_key_ciphertext or ""), base_url


# ---- the worker side -------------------------------------------------------------------

BoxFactory = Callable[[Session, GpuBox], contextlib.AbstractContextManager[ComputeBox]]
_box_factory: BoxFactory | None = None


def set_box_factory(factory: BoxFactory | None) -> None:
    """Tests (and offline demos) swap SSH for a fake box: `set_box_factory(lambda db, b: nullcontext(FakeGPUBox()))`."""
    global _box_factory
    _box_factory = factory


def known_hosts_path(box_id: str) -> Path:
    """Per-box known_hosts: the box's host key is pinned on first contact (accept-new)."""
    return config.DATA_DIR / "ssh" / f"{box_id}.known_hosts"


def _capped_runner(
    argv: list[str], *, input: str | None = None, timeout: float | None = None
) -> CommandResult:
    """subprocess_runner with a default timeout, so no ssh/scp call can hang a job forever."""
    return subprocess_runner(
        argv, input=input, timeout=timeout or get_settings().COMPUTE_SSH_TIMEOUT_S
    )


@contextlib.contextmanager
def _ssh_box(db: Session, box: GpuBox) -> Iterator[ComputeBox]:
    """An `SSHBox` for the row, with the decrypted key in a private temp file."""
    if not _creator_is_admin(db, box):
        _host_allowed(box.host)  # DNS can change between save and use
    kh = known_hosts_path(box.id)
    kh.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix="envforge-ssh-"))  # mode 700
    key_file = tmp / "id"
    try:
        key_file.touch(mode=0o600)
        key_file.write_text(_box_private_key(db, box))
        options = {
            "UserKnownHostsFile": str(kh),
            "IdentitiesOnly": "yes",  # only this key, never the worker's agent or files
            "IdentityAgent": "none",
            "ProxyCommand": "none",  # ignore anything an ssh config on the worker says
            "ControlMaster": "no",
            "PermitLocalCommand": "no",
        }
        yield SSHBox(
            box.host,
            user=box.user,
            port=box.port,
            key_path=key_file,
            options=options,
            runner=_capped_runner,
        )
    finally:
        with contextlib.suppress(OSError):
            key_file.write_bytes(b"\0" * 64)  # best effort; the file is removed next
        shutil.rmtree(tmp, ignore_errors=True)


def open_box(db: Session, box: GpuBox) -> contextlib.AbstractContextManager[ComputeBox]:
    """Connect to the box stored on this row (the only way the worker reaches a box)."""
    return (_box_factory or _ssh_box)(db, box)


class JobHeartbeat:
    """Touches jobs.heartbeat_at from a side thread while a long job runs, so
    `reap_stale_jobs` can tell a slow job (pulling an image) from a dead worker."""

    def __init__(self, job_id: str, every_s: float = 30.0) -> None:
        self.job_id, self.every_s = job_id, every_s
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def _loop(self) -> None:
        from envforge.web.db import session_factory

        Session_ = session_factory()
        while not self._stop.wait(self.every_s):
            try:
                with Session_() as db:
                    db.execute(
                        update(Job)
                        .where(Job.id == self.job_id)
                        .values(heartbeat_at=datetime.now(UTC))
                    )
                    db.commit()
            except Exception as e:  # noqa: BLE001 - a missed beat is not fatal
                print(f"[compute] heartbeat: {type(e).__name__}: {e}", flush=True)

    def __enter__(self) -> JobHeartbeat:
        # SQLite (dev/tests) shares one connection across threads: no second writer there
        if not get_settings().DATABASE_URL.startswith("sqlite"):
            self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=5)


def _progress(db: Session, job: Job, text: str) -> None:
    job.progress = text[:500]
    job.heartbeat_at = datetime.now(UTC)
    db.commit()


def resolve_box_runtime(db: Session, box: GpuBox, cb: ComputeBox) -> str:
    """docker / pod: the box's setting, else what we detected before, else ask the box."""
    if box.runtime in RUNTIMES:
        return box.runtime
    if box.detected_runtime in RUNTIMES:
        return box.detected_runtime
    box.detected_runtime = detect_runtime(cb)
    db.commit()
    return box.detected_runtime


def _do_check(db: Session, job: Job, box: GpuBox) -> None:
    _progress(db, job, f"connecting to {box.host}")
    with open_box(db, box) as cb:
        support = None
        try:
            support = probe_runtimes(cb)
        except ComputeError:
            pass  # most likely ssh itself failed; check() reports that clearly
        if box.runtime in RUNTIMES:
            runtime = box.runtime
        else:
            runtime = (support.preferred if support else None) or box.detected_runtime or "docker"
        _progress(db, job, f"running {runtime} readiness checks")
        results = check(cb, runtime=runtime)
        ssh_ok = results[0].ok
        gpus, facts, problems = [], None, []
        if ssh_ok:
            try:
                gpus = [
                    {"name": g.name, "memory_total_mib": g.memory_total_mib, "memory_used_mib": g.memory_used_mib,
                     "utilization_pct": g.utilization_pct} for g in gpu_info(cb)
                ]  # fmt: skip
            except ComputeError as e:
                problems.append(str(e)[:300])
            try:
                facts = box_facts(cb)
            except ComputeError as e:
                problems.append(str(e)[:300])
    box.check_result = {
        "runtime": runtime,
        "support": {"docker": support.docker, "pod": support.pod} if support else None,
        "checks": [{"name": r.name, "ok": r.ok, "detail": r.detail[:300]} for r in results],
        "gpus": gpus,
        "driver": facts.driver if facts else None,
        "cuda": facts.cuda if facts else None,
        "disk_free_gib": facts.disk_free_gib if facts else None,
        "vllm_image": facts.vllm_image if facts else None,
        "vllm_python": facts.vllm_python if facts else None,
        "problems": problems,
    }
    if support and support.preferred and box.runtime == "auto":
        box.detected_runtime = support.preferred
    box.status = (
        "unreachable" if not ssh_ok else ("ok" if all(r.ok for r in results) else "degraded")
    )
    box.checked_at = datetime.now(UTC)
    db.commit()


def serve_on_box(
    db: Session,
    job: Job,
    box: GpuBox,
    *,
    model: str,
    adapters: list[str],
    max_lora_rank: int = 64,
    max_model_len: int | None = None,
    gpu_memory_utilization: float = 0.90,
) -> list[str]:
    """Start vLLM on the box and wait until it answers. Uploads any adapter the box
    doesn't have yet from the workspace's local copy. Returns the served model ids."""
    from envforge.web.models import Adapter

    model = check_model_id(model)
    box.serve_state, box.serve_error = "starting", None
    box.served_model, box.served_adapters = model, list(adapters)
    db.commit()
    api_key = crypto.decrypt(box.vllm_key_ciphertext) if box.vllm_key_ciphertext else None
    hf_token = crypto.decrypt(box.hf_token_ciphertext) if box.hf_token_ciphertext else None
    with open_box(db, box) as cb:
        runtime = resolve_box_runtime(db, box, cb)
        for name in adapters:
            a = db.scalar(
                select(Adapter).where(
                    Adapter.workspace_id == box.workspace_id, Adapter.name == name
                )
            )
            if a is None:
                raise ValueError(f"adapter {name} no longer exists")
            remote = f"{ADAPTERS_DIR}/{a.name}"
            if not cb.run(["test", "-f", f"{remote}/adapter_config.json"], timeout=60).ok:
                local = Path(a.path)
                if not (local / "adapter_config.json").is_file():
                    raise ValueError(
                        f"adapter {name} is not on the box and has no local copy to upload"
                    )
                _progress(db, job, f"uploading adapter {name}")
                cb.run(["mkdir", "-p", ADAPTERS_DIR], timeout=60, check=True)
                cb.put(local, f"{ADAPTERS_DIR}/", recursive=True)
        _progress(db, job, f"starting vLLM with {model}")
        tunnel = box.vllm_access == "tunnel"
        server = serve_vllm(
            cb,
            model,
            port=box.vllm_port,
            api_key=api_key,
            hf_token=hf_token,
            # through a tunnel nothing else needs to reach vLLM: listen on loopback only
            bind=box.bind or ("127.0.0.1" if tunnel else None),
            endpoint_host=box.endpoint_host,
            lora_adapters=adapters,
            max_lora_rank=max_lora_rank,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            runtime=runtime,  # type: ignore[arg-type]
        )
        # a tunnel box's endpoint is the box-local address the tunnel forwards to
        box.endpoint_url = (
            f"http://{_tunnel_target(box)}:{box.vllm_port}/v1" if tunnel else server.base_url
        )
        db.commit()
        if not tunnel and not _creator_is_admin(db, box):
            _url_allowed(server.base_url)
        _progress(db, job, "loading weights (the first start downloads them)")

        def alive() -> bool:
            job.heartbeat_at = datetime.now(UTC)
            db.commit()
            return vllm_status(cb, box.vllm_port, runtime=runtime, probe=False).running  # type: ignore[arg-type]

        try:
            with box_base_url(box, cb) as base_url:
                ids = wait_healthy(
                    base_url,
                    timeout=get_settings().SERVE_HEALTH_TIMEOUT_S,
                    api_key=api_key,
                    is_alive=alive,
                    urlopen=getattr(cb, "urlopen", None),
                )
        except (ComputeError, TimeoutError) as e:
            tail = ""
            with contextlib.suppress(Exception):
                tail = vllm_logs(cb, box.vllm_port, tail=15, runtime=runtime)  # type: ignore[arg-type]
            raise ComputeError(f"{e}\n--- last vLLM log lines ---\n{tail}".strip()[:4000]) from e
    box.serve_state = "serving"
    db.commit()
    return ids


def _do_stop(db: Session, job: Job, box: GpuBox) -> None:
    _progress(db, job, "stopping vLLM")
    with open_box(db, box) as cb:
        runtime = resolve_box_runtime(db, box, cb)
        stop_vllm(cb, box.vllm_port, runtime=runtime)  # type: ignore[arg-type]
    box.serve_state, box.serve_error = "stopped", None
    box.served_model = box.served_adapters = box.endpoint_url = None
    db.commit()


def _do_delete(db: Session, job: Job, box: GpuBox) -> None:
    """Stop vLLM (a no-op if nothing runs), then forget the box. If the box can't be
    reached, this fails and the box stays, so a running vLLM is never silently orphaned."""
    _progress(db, job, "stopping vLLM before removing the box")
    with open_box(db, box) as cb:
        runtime = resolve_box_runtime(db, box, cb)
        stop_vllm(cb, box.vllm_port, runtime=runtime)  # type: ignore[arg-type]
    _forget_box(db, box)


def _user_error(e: BaseException) -> str:
    return f"{type(e).__name__}: {e}"[:4000]


def fail_related(db: Session, job: Job, error: str) -> None:
    """After a compute job failed (or its worker died): leave the box / training job in
    a state the UI can explain, never stuck in 'starting'."""
    payload = job.payload or {}
    box = db.get(GpuBox, payload.get("box_id")) if payload.get("box_id") else None
    if box is not None:
        if job.kind in ("box_serve", "adapter_test") and box.serve_state == "starting":
            box.serve_state, box.serve_error = "failed", error
        elif job.kind == "box_stop" and box.serve_state == "stopping":
            box.serve_state, box.serve_error = "failed", error
        elif job.kind == "box_check" and box.status == "unknown":
            box.status = "unreachable"
        elif job.kind == "box_delete":
            box.serve_error = f"Couldn't stop vLLM, so the box was kept: {error}"
    if job.kind == "train_start" and (tj := db.get(TrainingJob, payload.get("training_job_id"))):
        if tj.status in ("queued", "starting"):
            tj.status, tj.error, tj.finished_at = "failed", error, datetime.now(UTC)
    db.commit()


def process_compute_job(db: Session, job: Job) -> None:
    """Run one compute job and record how it ended (done / failed with a readable error)."""
    from envforge.web import training_service as T

    try:
        with JobHeartbeat(job.id):
            payload = job.payload or {}
            box = db.get(GpuBox, payload.get("box_id"))
            if box is None or box.workspace_id != job.workspace_id:
                raise ValueError("the GPU box for this job no longer exists")
            if job.kind == "box_check":
                _do_check(db, job, box)
            elif job.kind == "box_serve":
                serve_on_box(
                    db,
                    job,
                    box,
                    model=payload["model"],
                    adapters=list(payload.get("adapters") or []),
                    max_lora_rank=int(payload.get("max_lora_rank") or 64),
                    max_model_len=payload.get("max_model_len"),
                    gpu_memory_utilization=float(payload.get("gpu_memory_utilization") or 0.90),
                )
            elif job.kind == "box_stop":
                _do_stop(db, job, box)
            elif job.kind == "box_delete":
                _do_delete(db, job, box)
            elif job.kind == "train_start":
                T.run_train_start(db, job, box)
            elif job.kind == "adapter_test":
                T.run_adapter_test(db, job, box)
            else:
                raise ValueError(f"unknown compute job kind {job.kind}")
        job.status = "done"
    except Exception as e:  # noqa: BLE001
        traceback.print_exc()  # detail for the worker log; the key never reaches it
        db.rollback()
        job.status, job.error = "failed", _user_error(e)
        fail_related(db, job, job.error)
    finally:
        job.progress = ""
        job.finished_at = datetime.now(UTC)
        db.commit()


def work_once(db: Session) -> bool:
    """Claim and run one compute job; else advance running training jobs. True if a job ran."""
    from envforge.web import training_service as T
    from envforge.web.builder import claim_job

    job = claim_job(db, COMPUTE_JOB_KINDS)
    if job is not None:
        process_compute_job(db, job)
        return True
    T.poll_trainings(db)
    return False

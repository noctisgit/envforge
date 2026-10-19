"""Model catalog: which models exist, whether they can call tools, and what they cost.

The source of truth is the Vercel AI Gateway's public model list
(https://ai-gateway.vercel.sh/v1/models — no key needed). It lists 400+ models with
per-token prices and capability tags. We derive native ids from it as well:

    vercel:anthropic/claude-sonnet-4.5   ->  anthropic:claude-sonnet-4-5   (dots become dashes)
    vercel:openai/gpt-5                  ->  openai:gpt-5

so one fetch prices runs for every API provider. `vllm` models are self-hosted
(no per-token price) and `scripted` is free.

The list is cached in memory and on disk (data/cache/), so evaluation keeps working
offline once it has been fetched, and a failed fetch is not retried for a while.
Set ENVFORGE_OFFLINE=1 to never touch the network (tests do).
"""

from __future__ import annotations

import json
import os
import re
import threading
import time
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from envforge import config

GATEWAY_MODELS_URL = "https://ai-gateway.vercel.sh/v1/models"
CACHE_TTL_S = 6 * 3600  # refresh the list at most every 6 hours
RETRY_AFTER_FAILURE_S = 600  # after a failed fetch, wait before trying again

_lock = threading.Lock()
_cache: list[dict] | None = None
_cache_at = 0.0
_failed_at = 0.0


@dataclass
class ModelInfo:
    """One selectable model, as the UI and the cost calculator see it."""

    id: str  # full spec, e.g. "vercel:anthropic/claude-sonnet-4.5"
    provider: str  # anthropic | openai | vercel | vllm | scripted
    name: str  # human name, e.g. "Claude Sonnet 4.5"
    owned_by: str = ""
    context_window: int = 0
    input_per_mtok: float | None = None  # USD per 1M input tokens
    output_per_mtok: float | None = None  # USD per 1M output tokens
    tags: list[str] = field(default_factory=list)
    released: int | None = None  # unix seconds

    def to_dict(self) -> dict:
        return asdict(self)


SCRIPTED = ModelInfo(
    id="scripted",
    provider="scripted",
    name="Scripted (reference solver, offline)",
    owned_by="envforge",
    input_per_mtok=0.0,
    output_per_mtok=0.0,
    tags=["tool-use", "free"],
)


# ---- fetching -----------------------------------------------------------------


def _cache_file() -> Path:
    return config.DATA_DIR / "cache" / "gateway_models.json"


def _offline() -> bool:
    return os.getenv("ENVFORGE_OFFLINE", "").lower() in ("1", "true", "yes")


def _fetch() -> list[dict]:
    req = urllib.request.Request(GATEWAY_MODELS_URL, headers={"User-Agent": "envforge"})
    with urllib.request.urlopen(req, timeout=8) as resp:  # noqa: S310 - fixed https URL
        return json.loads(resp.read())["data"]


def gateway_models(*, refresh: bool = False) -> list[dict]:
    """Raw gateway model entries. Memory cache → network → disk cache → []."""
    global _cache, _cache_at, _failed_at
    with _lock:
        now = time.time()
        fresh = _cache is not None and now - _cache_at < CACHE_TTL_S
        if fresh and not refresh:
            return _cache  # type: ignore[return-value]
        may_fetch = not _offline() and (refresh or now - _failed_at > RETRY_AFTER_FAILURE_S)
        if may_fetch:
            try:
                data = _fetch()
                _cache, _cache_at = data, now
                try:
                    _cache_file().parent.mkdir(parents=True, exist_ok=True)
                    _cache_file().write_text(json.dumps(data))
                except OSError:
                    pass  # read-only data dir: memory cache is enough
                return data
            except Exception:  # noqa: BLE001 - network is optional
                _failed_at = now
        if _cache is None and _cache_file().exists():
            try:
                _cache, _cache_at = json.loads(_cache_file().read_text()), now
            except (OSError, json.JSONDecodeError):
                pass
        return _cache or []


def set_gateway_models(data: list[dict]) -> None:
    """Install a catalog directly (tests, or an air-gapped box with a saved list)."""
    global _cache, _cache_at
    with _lock:
        _cache, _cache_at = data, time.time()


# ---- shaping ------------------------------------------------------------------


def _per_mtok(price: str | float | None) -> float | None:
    """Gateway prices are USD per token as strings; we show USD per 1M tokens."""
    if price in (None, ""):
        return None
    return round(float(price) * 1_000_000, 6)


def _usable(entry: dict) -> bool:
    """Only chat models that can call tools work with the agent loop. Batch variants
    are a different API, so we hide them."""
    return (
        entry.get("type") == "language"
        and "tool-use" in entry.get("tags", [])
        and not entry["id"].endswith("-batch")
    )


def _native_id(provider: str, entry: dict) -> str | None:
    """Map a gateway entry to the provider's own API id, or None if it has no native twin.
    Open-weights models (e.g. openai/gpt-oss) are served by third parties, not the
    provider's own API, so they have no native id."""
    if "open-weights" in entry.get("tags", []):
        return None
    owner, _, model = entry["id"].partition("/")
    if provider == "anthropic" and owner == "anthropic" and not model.endswith("-fast"):
        return model.replace(".", "-")
    if provider == "openai" and owner == "openai":
        return model
    return None


def _info(provider: str, model_id: str, e: dict) -> ModelInfo:
    pricing = e.get("pricing") or {}
    return ModelInfo(
        id=f"{provider}:{model_id}",
        provider=provider,
        name=e.get("name") or model_id,
        owned_by=e.get("owned_by", ""),
        context_window=int(e.get("context_window") or 0),
        input_per_mtok=_per_mtok(pricing.get("input")),
        output_per_mtok=_per_mtok(pricing.get("output")),
        tags=list(e.get("tags", [])),
        released=e.get("released"),
    )


def _undated(model_id: str) -> str:
    """claude-haiku-4-5-20251001 -> claude-haiku-4-5 (providers list dated snapshots;
    the undated alias is what people type and what the gateway mapping produces)."""
    return re.sub(r"-\d{8}$", "", model_id)


def list_models(
    providers: set[str] | list[str], native_ids: dict[str, list[str]] | None = None
) -> list[ModelInfo]:
    """Every tool-capable model reachable through the given providers.

    `native_ids` maps a provider to the model ids its key reported when it was saved
    (see web/services.py). When given, native models are limited to those, because
    the gateway also serves models a provider has retired from its own API. `vllm`
    contributes nothing here (its models are whatever you serve); the UI lets you
    type a custom `vllm:<model>` id. `scripted` is always included.
    """
    wanted = set(providers)
    allowed = {p: {_undated(i) for i in ids} for p, ids in (native_ids or {}).items()}
    out: list[ModelInfo] = [SCRIPTED]
    covered: dict[str, set[str]] = {"anthropic": set(), "openai": set()}
    entries = [e for e in gateway_models() if _usable(e)]
    for e in entries:
        if "vercel" in wanted:
            out.append(_info("vercel", e["id"], e))
        for native in ("anthropic", "openai"):
            if native in wanted and (nid := _native_id(native, e)):
                if native in allowed and nid not in allowed[native]:
                    continue  # the key can't use it (retired, or not on this account)
                covered[native].add(nid)
                out.append(_info(native, nid, e))
    # Anthropic models the key reports that the gateway doesn't list yet: still
    # selectable, just without a price. (OpenAI's list also has embeddings, audio and
    # image models, so there we only trust ids the catalog knows are chat + tools.)
    if "anthropic" in wanted:
        for mid in sorted(allowed.get("anthropic", set()) - covered["anthropic"]):
            out.append(ModelInfo(id=f"anthropic:{mid}", provider="anthropic", name=mid))
    # whatever the user's vLLM box served when its URL was saved (self-hosted: $0/token)
    if "vllm" in wanted:
        for mid in sorted((native_ids or {}).get("vllm", [])):
            out.append(ModelInfo(id=f"vllm:{mid}", provider="vllm", name=mid, owned_by="self-hosted",
                                 input_per_mtok=0.0, output_per_mtok=0.0, tags=["tool-use", "open-weights"]))  # fmt: skip
    return out


# ---- pricing ------------------------------------------------------------------


def price_per_token(spec: str) -> tuple[float, float] | None:
    """(input, output) USD per token for a model spec, or None if unknown.

    scripted and vllm are free in token terms (self-hosted GPU time is not billed
    per token), so they return (0, 0) without touching the catalog.
    """
    provider, _, model = spec.partition(":")
    if provider in ("scripted", "vllm") or spec == "scripted":
        return (0.0, 0.0)
    model = _undated(model) if provider == "anthropic" else model
    for e in gateway_models():
        if provider == "vercel":
            match = e["id"] == model
        else:
            match = _native_id(provider, e) == model
        if match:
            p = e.get("pricing") or {}
            if p.get("input") is None or p.get("output") is None:
                return None
            return (float(p["input"]), float(p["output"]))
    return None


def cache_prices(spec: str) -> tuple[float, float] | None:
    """(cache read, cache write) USD per token, or None if the model isn't priced.

    From the gateway catalog when it lists them; otherwise Anthropic's published
    multipliers for native Claude (read 0.1x, 5-minute write 1.25x of input) and, for
    everything else, no discount (a conservative estimate).
    """
    provider, _, model = spec.partition(":")
    if provider in ("scripted", "vllm") or spec == "scripted":
        return (0.0, 0.0)
    base = price_per_token(spec)
    if base is None:
        return None
    for e in gateway_models():
        match = (
            e["id"] == model if provider == "vercel" else _native_id(provider, e) == _undated(model)
        )
        if match:
            p = e.get("pricing") or {}
            read = p.get("input_cache_read")
            write = p.get("input_cache_write")
            if read is not None:
                return (float(read), float(write) if write is not None else base[0])
            break
    if provider == "anthropic" or model.startswith("anthropic/"):
        return (base[0] * 0.1, base[0] * 1.25)
    return (base[0], base[0])


def usage_cost(spec: str, usage: Any) -> float | None:
    """Estimated USD for a Usage (see core/trajectory.py), pricing cached tokens properly.

    `usage.input_tokens` is the whole prompt; the cached read/write parts are subsets.
    """
    inp = getattr(usage, "input_tokens", 0) or 0
    out = getattr(usage, "output_tokens", 0) or 0
    read = getattr(usage, "cache_read_tokens", 0) or 0
    write = getattr(usage, "cache_write_tokens", 0) or 0
    if not (inp or out):
        return 0.0
    price = price_per_token(spec)
    if price is None:
        return None
    if not (read or write):
        return inp * price[0] + out * price[1]
    cp = cache_prices(spec) or (price[0], price[0])
    uncached = max(0, inp - read - write)
    return uncached * price[0] + read * cp[0] + write * cp[1] + out * price[1]


def cost_usd(spec: str, input_tokens: int, output_tokens: int) -> float | None:
    """Estimated cost of a number of tokens on a model, or None if the price is unknown.

    This uses list prices and ignores prompt-caching discounts and long-context tiers,
    so treat it as an estimate.
    """
    if input_tokens == 0 and output_tokens == 0:
        return 0.0
    price = price_per_token(spec)
    if price is None:
        return None
    return input_tokens * price[0] + output_tokens * price[1]

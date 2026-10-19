"""Open-weight models on the Hugging Face Hub: search, and what it takes to train them.

Used by the web UI's "Training base model" picker. The Hub's public search API needs
no key (https://huggingface.co/api/models), so the list is always current. For each
result we add what matters for envforge:

- `params`         parameter count (from the repo's safetensors index)
- `lora_vram_gb`   a rough GPU-memory estimate for LoRA SFT (see `estimate_lora_vram_gb`)
- `tool_parser`    the vLLM `--tool-call-parser` for the family, when known — envforge's
                   runner needs structured tool calls, so serving needs the right parser
- `trainable`      False for formats our LoRA/vLLM pipeline can't use (GGUF, MLX, ONNX,
                   GPTQ/AWQ/bitsandbytes quantisations)

Results are cached in memory for a while; set ENVFORGE_OFFLINE=1 to never call out
(tests do), in which case a short built-in list of known-good models is returned.
"""

from __future__ import annotations

import json
import os
import re
import threading
import time
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass, field
from typing import Any

HUB_API = "https://huggingface.co/api/models"
CACHE_TTL_S = 30 * 60
_lock = threading.Lock()
_cache: dict[tuple, tuple[float, list[HubModel]]] = {}

# Formats the LoRA SFT script and vLLM LoRA serving can't start from.
_UNTRAINABLE = re.compile(
    r"(?i)(gguf|[-_.]mlx\b|mlx-community|onnx|gptq|awq|exl2|bnb|-4bit|-8bit|int4|int8|fp8|nvfp4|mxfp4|fp4)"
)

# vLLM tool-call parser per model family (https://docs.vllm.ai — tool calling).
_PARSERS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"(?i)qwen(2\.5|3)"), "hermes"),
    (re.compile(r"(?i)llama-?3"), "llama3_json"),
    (re.compile(r"(?i)mistral|ministral|mixtral"), "mistral"),
    (re.compile(r"(?i)granite"), "granite"),
    (re.compile(r"(?i)hermes"), "hermes"),
    (re.compile(r"(?i)internlm"), "internlm"),
    (re.compile(r"(?i)phi-?4"), "phi4_mini_json"),
]

# Shown when the Hub can't be reached (or offline mode): small, tool-capable, open licences.
RECOMMENDED = [
    ("Qwen/Qwen2.5-3B-Instruct", 3_090_000_000),
    ("Qwen/Qwen2.5-7B-Instruct", 7_620_000_000),
    ("Qwen/Qwen3-4B", 4_022_000_000),
    ("Qwen/Qwen3-8B", 8_191_000_000),
    ("meta-llama/Llama-3.2-3B-Instruct", 3_213_000_000),
    ("mistralai/Mistral-7B-Instruct-v0.3", 7_248_000_000),
]


@dataclass
class HubModel:
    id: str
    params: int | None = None
    downloads: int = 0
    likes: int = 0
    gated: bool = False  # needs accepting a licence + an HF token
    license: str | None = None
    last_modified: str | None = None
    tool_parser: str | None = None
    lora_vram_gb: float | None = None
    trainable: bool = True
    tags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def tool_parser_for(model_id: str) -> str | None:
    for rx, parser in _PARSERS:
        if rx.search(model_id):
            return parser
    return None


def estimate_lora_vram_gb(params: int | None, max_len: int = 8192) -> float | None:
    """Rough peak GPU memory for LoRA SFT with gradient checkpointing, in GB.

    bf16 weights take 2 bytes per parameter; LoRA's own weights, gradients and optimizer
    state are small; activations at `max_len` tokens with checkpointing and the output
    layer add a few GB. It's an estimate to pick a GPU class, not a promise.
    """
    if not params:
        return None
    weights = params * 2 / 1e9
    activations = 2.0 + 3.0 * (max_len / 8192) * (params / 7e9) ** 0.5
    return round(weights * 1.1 + activations, 1)


def _offline() -> bool:
    return os.getenv("ENVFORGE_OFFLINE", "").lower() in ("1", "true", "yes")


def _shape(m: dict[str, Any]) -> HubModel:
    st = m.get("safetensors") or {}
    params = st.get("total")
    tags = m.get("tags") or []
    lic = next((t.split(":", 1)[1] for t in tags if t.startswith("license:")), None)
    mid = m["id"]
    return HubModel(
        id=mid,
        params=params,
        downloads=m.get("downloads") or 0,
        likes=m.get("likes") or 0,
        gated=bool(m.get("gated")),
        license=lic or (m.get("cardData") or {}).get("license"),
        last_modified=m.get("lastModified"),
        tool_parser=tool_parser_for(mid),
        lora_vram_gb=estimate_lora_vram_gb(params),
        trainable=bool(params) and not _UNTRAINABLE.search(mid),
        tags=[t for t in tags if t in ("conversational", "text-generation")],
    )


def _recommended(query: str) -> list[HubModel]:
    q = query.lower()
    return [
        HubModel(
            id=i,
            params=p,
            tool_parser=tool_parser_for(i),
            lora_vram_gb=estimate_lora_vram_gb(p),
            gated=i.startswith("meta-llama/"),
        )  # fmt: skip
        for i, p in RECOMMENDED
        if not q or q in i.lower()
    ]


def search(
    query: str = "", *, sort: str = "downloads", limit: int = 30
) -> tuple[list[HubModel], bool]:
    """(models, live). Text-generation repos matching `query`, most downloaded first.

    `live` is False when the Hub couldn't be reached and the built-in list was used.
    """
    sort = sort if sort in ("downloads", "likes", "lastModified", "trendingScore") else "downloads"
    key = (query.strip().lower(), sort, limit)
    with _lock:
        hit = _cache.get(key)
        if hit and time.time() - hit[0] < CACHE_TTL_S:
            return hit[1], True
    if _offline():
        return _recommended(query), False
    params = [
        ("pipeline_tag", "text-generation"),
        ("sort", sort),
        ("direction", "-1"),
        ("limit", str(min(max(limit * 2, 20), 100))),  # over-fetch: untrainable ones are dropped
    ]
    if query.strip():
        params.append(("search", query.strip()))
    for e in ("safetensors", "downloads", "likes", "gated", "tags", "lastModified", "cardData"):
        params.append(("expand[]", e))
    url = f"{HUB_API}?{urllib.parse.urlencode(params)}"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "envforge"})
        with urllib.request.urlopen(req, timeout=10) as r:  # noqa: S310 - fixed https host
            raw = json.loads(r.read())
    except Exception:  # noqa: BLE001 - the Hub is optional; fall back to known models
        return _recommended(query), False
    models = [m for m in (_shape(x) for x in raw) if m.trainable][:limit]
    with _lock:
        _cache[key] = (time.time(), models)
    return models, True

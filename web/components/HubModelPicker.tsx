"use client";

// Searchable picker for open-weight models on the Hugging Face Hub (the training base
// model). Results come live from the Hub through /api/hub/models, with what matters for
// envforge: size, a rough LoRA GPU-memory estimate, licence, gated, and the vLLM
// tool-call parser for the family. Any Hugging Face id can still be typed.

import { useEffect, useMemo, useRef, useState } from "react";
import { api } from "@/lib/api";

interface HubModel {
  id: string;
  params: number | null;
  downloads: number;
  likes: number;
  gated: boolean;
  license: string | null;
  last_modified: string | null;
  tool_parser: string | null;
  lora_vram_gb: number | null;
  trainable: boolean;
}

type Sort = "downloads" | "likes" | "trendingScore" | "lastModified";

const size = (p: number | null) => (p ? (p >= 1e9 ? `${(p / 1e9).toFixed(p >= 1e10 ? 0 : 1)}B` : `${Math.round(p / 1e6)}M`) : "?");
const count = (n: number) => (n >= 1e6 ? `${(n / 1e6).toFixed(1)}M` : n >= 1e3 ? `${Math.round(n / 1e3)}k` : String(n));
const gpuClass = (gb: number | null) => (gb === null ? "" : gb <= 22 ? "24 GB card" : gb <= 38 ? "40 GB card" : gb <= 75 ? "80 GB card" : "multi-GPU");

export function HubModelPicker({ value, onChange, disabled }: { value: string; onChange: (id: string) => void; disabled?: boolean }) {
  const [open, setOpen] = useState(false);
  const [query, setQuery] = useState("");
  const [sort, setSort] = useState<Sort>("downloads");
  const [models, setModels] = useState<HubModel[] | null>(null);
  const [live, setLive] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [active, setActive] = useState(0);
  const [selected, setSelected] = useState<HubModel | null>(null);
  const root = useRef<HTMLDivElement>(null);
  const search = useRef<HTMLInputElement>(null);

  // debounced live search while open
  useEffect(() => {
    if (!open) return;
    setModels(null);
    setError(null);
    const t = setTimeout(() => {
      api<{ live: boolean; models: HubModel[] }>(`/api/hub/models?q=${encodeURIComponent(query)}&sort=${sort}`)
        .then((r) => {
          setModels(r.models);
          setLive(r.live);
          setActive(0);
        })
        .catch((e) => {
          setModels([]);
          setError(e instanceof Error ? e.message : String(e));
        });
    }, 300);
    return () => clearTimeout(t);
  }, [open, query, sort]);

  // details for the current value (one cheap lookup when the page loads)
  useEffect(() => {
    if (!value) return;
    api<{ models: HubModel[] }>(`/api/hub/models?q=${encodeURIComponent(value)}`)
      .then((r) => setSelected(r.models.find((m) => m.id === value) ?? null))
      .catch(() => {});
  }, [value]);

  useEffect(() => {
    if (!open) return;
    setTimeout(() => search.current?.focus(), 0);
    const onDown = (e: MouseEvent) => {
      if (root.current && !root.current.contains(e.target as Node)) setOpen(false);
    };
    document.addEventListener("mousedown", onDown);
    return () => document.removeEventListener("mousedown", onDown);
  }, [open]);

  const custom = useMemo(() => (/^[\w.-]+\/[\w.-]+$/.test(query.trim()) && !models?.some((m) => m.id === query.trim()) ? query.trim() : null), [query, models]);
  const options = [...(custom ? [custom] : []), ...(models ?? []).map((m) => m.id)];

  function choose(id: string) {
    onChange(id);
    setSelected(models?.find((m) => m.id === id) ?? null);
    setOpen(false);
    setQuery("");
  }

  return (
    <div ref={root} className="relative">
      <button
        type="button"
        className="input flex items-center justify-between gap-2 text-left"
        onClick={() => !disabled && setOpen((o) => !o)}
        disabled={disabled}
        aria-haspopup="listbox"
        aria-expanded={open}
      >
        <span className="mono min-w-0 truncate">{value || "choose a base model"}</span>
        <span style={{ color: "var(--muted)" }}>▾</span>
      </button>
      {selected && (
        <div className="mt-1 text-xs" style={{ color: "var(--muted)" }}>
          {size(selected.params)} params · ≈{selected.lora_vram_gb ?? "?"} GB for LoRA ({gpuClass(selected.lora_vram_gb)}) · {selected.license ?? "licence ?"}
          {selected.tool_parser ? ` · vLLM parser ${selected.tool_parser}` : " · tool parser unknown"}
          {selected.gated && <span style={{ color: "var(--warn)" }}> · gated: needs an HF token</span>}
        </div>
      )}

      {open && (
        <div
          className="panel absolute z-20 mt-1 flex max-h-[28rem] w-[min(44rem,calc(100vw-2rem))] flex-col shadow-2xl"
          onKeyDown={(e) => {
            if (e.key === "ArrowDown") {
              e.preventDefault();
              setActive((i) => Math.min(i + 1, options.length - 1));
            } else if (e.key === "ArrowUp") {
              e.preventDefault();
              setActive((i) => Math.max(i - 1, 0));
            } else if (e.key === "Enter" && options[active]) {
              e.preventDefault();
              choose(options[active]);
            } else if (e.key === "Escape") setOpen(false);
          }}
        >
          <div className="space-y-2 border-b p-3" style={{ borderColor: "var(--line)" }}>
            <input
              ref={search}
              className="input"
              placeholder="Search Hugging Face — e.g. qwen2.5 instruct, llama 3.2, mistral… or type org/model"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              role="combobox"
              aria-expanded
              aria-controls="hub-options"
            />
            <div className="flex flex-wrap items-center gap-2 text-xs" style={{ color: "var(--muted)" }}>
              <span>sort</span>
              {(
                [
                  ["downloads", "most downloaded"],
                  ["likes", "most liked"],
                  ["trendingScore", "trending"],
                  ["lastModified", "recently updated"],
                ] as [Sort, string][]
              ).map(([k, label]) => (
                <button
                  key={k}
                  type="button"
                  onClick={() => setSort(k)}
                  className="rounded-full border px-2 py-0.5"
                  style={{ borderColor: sort === k ? "var(--accent)" : "var(--line)", color: sort === k ? "var(--accent)" : "var(--muted)" }}
                >
                  {label}
                </button>
              ))}
              <span className="ml-auto">
                {error ? "model search unavailable" : live ? "live from huggingface.co" : "Hub unreachable — showing known-good models"}
              </span>
            </div>
          </div>
          <div id="hub-options" role="listbox" className="overflow-y-auto py-1">
            {!models && <div className="px-3 py-2 text-sm" style={{ color: "var(--muted)" }}>Searching…</div>}
            {custom && (
              <Row active={active === 0} onPick={() => choose(custom)} onHover={() => setActive(0)}>
                <span>
                  Use <span className="mono">{custom}</span>
                </span>
              </Row>
            )}
            {models?.map((m, i) => {
              const idx = i + (custom ? 1 : 0);
              return (
                <Row key={m.id} active={active === idx} selected={m.id === value} onPick={() => choose(m.id)} onHover={() => setActive(idx)}>
                  <div className="min-w-0 flex-1">
                    <div className="mono truncate">{m.id}</div>
                    <div className="text-xs" style={{ color: "var(--muted)" }}>
                      {m.license ?? "licence ?"} · ↓{count(m.downloads)} · ♥{count(m.likes)}
                      {m.tool_parser ? ` · parser ${m.tool_parser}` : ""}
                      {m.gated && <span style={{ color: "var(--warn)" }}> · gated</span>}
                    </div>
                  </div>
                  <div className="shrink-0 text-right text-xs">
                    <div className="font-medium">{size(m.params)}</div>
                    <div style={{ color: "var(--muted)" }}>≈{m.lora_vram_gb ?? "?"} GB · {gpuClass(m.lora_vram_gb)}</div>
                  </div>
                </Row>
              );
            })}
            {error && (
              <div className="px-3 py-2 text-sm" style={{ color: "var(--bad)" }}>
                Couldn&apos;t search models: {error}. You can still type a full id like <span className="mono">org/model</span>.
              </div>
            )}
            {models && models.length === 0 && !custom && !error && (
              <div className="px-3 py-2 text-sm" style={{ color: "var(--muted)" }}>
                No trainable text models match. Type a full id like <span className="mono">org/model</span> to use it anyway.
              </div>
            )}
          </div>
          <div className="border-t px-3 py-2 text-[11px]" style={{ borderColor: "var(--line)", color: "var(--muted)" }}>
            GPU estimate: LoRA SFT at 8k tokens with gradient checkpointing — a guide to pick a GPU class, not a guarantee. Quantised
            (GGUF/AWQ/GPTQ/FP4…) repos are hidden: train from the original weights.
          </div>
        </div>
      )}
    </div>
  );
}

function Row({ active, selected, onPick, onHover, children }: { active: boolean; selected?: boolean; onPick: () => void; onHover: () => void; children: React.ReactNode }) {
  return (
    <div
      role="option"
      aria-selected={selected}
      onMouseDown={(e) => e.preventDefault()}
      onClick={onPick}
      onMouseEnter={onHover}
      className="flex cursor-pointer items-center gap-3 px-3 py-2 text-sm"
      style={{ background: active ? "var(--line)" : undefined, borderLeft: `2px solid ${selected ? "var(--accent)" : "transparent"}` }}
    >
      {children}
    </div>
  );
}

"use client";

// Searchable model selector. Lists only models the workspace can actually run (it has
// a key for the provider), from the server's catalog (Vercel AI Gateway's public list,
// mapped to native Anthropic/OpenAI ids). Any `provider:model` can still be typed in,
// which is how self-hosted `vllm:` models are chosen.

import { useEffect, useMemo, useRef, useState } from "react";
import { Api, type ModelInfo, type ModelsResponse } from "@/lib/api";

const PROVIDER_LABEL: Record<string, string> = {
  scripted: "Offline",
  anthropic: "Anthropic",
  openai: "OpenAI",
  vercel: "Vercel AI Gateway",
  vllm: "vLLM (self-hosted)",
};
const SHOWN_TAGS = ["reasoning", "vision", "open-weights", "fast"];
const MAX_ROWS = 150;

type Sort = "name" | "price" | "newest" | "context";

// one fetch per workspace per page load, shared by every picker on the page
const cache = new Map<string, Promise<ModelsResponse>>();
function loadModels(wsId: string, refresh = false): Promise<ModelsResponse> {
  if (refresh || !cache.has(wsId)) cache.set(wsId, Api.models(wsId, refresh));
  return cache.get(wsId)!;
}

/** Call after the user's provider keys change: the available models change with them. */
export function invalidateModelCache() {
  cache.clear();
}

const price = (m: ModelInfo) => (m.input_per_mtok ?? Infinity) + (m.output_per_mtok ?? Infinity);
const usd = (x: number | null) => (x === null ? "?" : x < 0.1 ? `$${x.toFixed(3)}` : `$${x.toFixed(2)}`);
const ctx = (n: number) => (n >= 1_000_000 ? `${n / 1_000_000}M` : n >= 1000 ? `${Math.round(n / 1000)}k` : n ? String(n) : "");

export function ModelPicker({
  wsId,
  value,
  onChange,
  disabled,
}: {
  wsId: string;
  value: string;
  onChange: (id: string) => void;
  disabled?: boolean;
}) {
  const [data, setData] = useState<ModelsResponse | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [open, setOpen] = useState(false);
  const [query, setQuery] = useState("");
  const [provider, setProvider] = useState<string>("all");
  const [sort, setSort] = useState<Sort>("name");
  const [active, setActive] = useState(0);
  const root = useRef<HTMLDivElement>(null);
  const search = useRef<HTMLInputElement>(null);
  const list = useRef<HTMLDivElement>(null);

  useEffect(() => {
    setData(null);
    loadModels(wsId)
      .then(setData)
      .catch((e) => setErr(String(e)));
  }, [wsId]);

  // close on outside click
  useEffect(() => {
    if (!open) return;
    const onDown = (e: MouseEvent) => {
      if (root.current && !root.current.contains(e.target as Node)) setOpen(false);
    };
    document.addEventListener("mousedown", onDown);
    return () => document.removeEventListener("mousedown", onDown);
  }, [open]);

  useEffect(() => {
    if (open) {
      setQuery("");
      setActive(0);
      setTimeout(() => search.current?.focus(), 0);
    }
  }, [open]);

  const models = data?.models ?? [];
  const providers = useMemo(() => Array.from(new Set(models.map((m) => m.provider))), [models]);
  const selected = models.find((m) => m.id === value);

  const filtered = useMemo(() => {
    const words = query.toLowerCase().split(/\s+/).filter(Boolean);
    const rows = models.filter((m) => {
      if (provider !== "all" && m.provider !== provider) return false;
      const hay = `${m.id} ${m.name} ${m.owned_by} ${m.tags.join(" ")}`.toLowerCase();
      return words.every((w) => hay.includes(w));
    });
    const by: Record<Sort, (a: ModelInfo, b: ModelInfo) => number> = {
      name: (a, b) => a.provider.localeCompare(b.provider) || a.name.localeCompare(b.name),
      price: (a, b) => price(a) - price(b),
      newest: (a, b) => (b.released ?? 0) - (a.released ?? 0),
      context: (a, b) => b.context_window - a.context_window,
    };
    return rows.sort((a, b) => (a.provider === "scripted" ? -1 : b.provider === "scripted" ? 1 : by[sort](a, b)));
  }, [models, query, provider, sort]);

  const shown = filtered.slice(0, MAX_ROWS);
  // a typed `provider:model` that isn't in the list can still be used (vllm, new models)
  const custom = query.includes(":") && !models.some((m) => m.id === query.trim()) ? query.trim() : null;
  const options: string[] = [...(custom ? [custom] : []), ...shown.map((m) => m.id)];

  function choose(id: string) {
    onChange(id);
    setOpen(false);
  }

  function onKey(e: React.KeyboardEvent) {
    if (e.key === "ArrowDown") {
      e.preventDefault();
      setActive((i) => Math.min(i + 1, options.length - 1));
    } else if (e.key === "ArrowUp") {
      e.preventDefault();
      setActive((i) => Math.max(i - 1, 0));
    } else if (e.key === "Enter") {
      e.preventDefault();
      if (options[active]) choose(options[active]);
    } else if (e.key === "Escape") {
      setOpen(false);
    }
  }

  useEffect(() => {
    list.current?.querySelector(`[data-idx="${active}"]`)?.scrollIntoView({ block: "nearest" });
  }, [active]);

  const hasVllm = data?.providers.includes("vllm");

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
        <span className="min-w-0 truncate">
          {selected ? (
            <>
              {selected.name} <span className="mono text-xs" style={{ color: "var(--muted)" }}>{selected.id}</span>
            </>
          ) : (
            <span className="mono">{value || "choose a model"}</span>
          )}
        </span>
        <span style={{ color: "var(--muted)" }}>▾</span>
      </button>
      {value && data && !selected && !value.startsWith("vllm:") && (
        <div className="mt-1 text-xs" style={{ color: "var(--warn)" }}>
          Not available with your keys — add a key for its provider under Provider keys, or check the id.
        </div>
      )}

      {open && (
        <div
          className="panel absolute z-20 mt-1 flex max-h-[28rem] w-[min(44rem,calc(100vw-2rem))] flex-col shadow-2xl"
          style={{ background: "var(--panel)" }}
          onKeyDown={onKey}
        >
          <div className="space-y-2 border-b p-3" style={{ borderColor: "var(--line)" }}>
            <input
              ref={search}
              className="input"
              placeholder="Search models — e.g. claude sonnet, qwen, cheap reasoning… or type provider:model"
              value={query}
              onChange={(e) => {
                setQuery(e.target.value);
                setActive(0);
              }}
              role="combobox"
              aria-controls="model-options"
              aria-expanded
            />
            <div className="flex flex-wrap items-center gap-2 text-xs">
              {["all", ...providers].map((p) => (
                <button
                  key={p}
                  type="button"
                  onClick={() => {
                    setProvider(p);
                    setActive(0);
                  }}
                  className="rounded-full border px-2 py-0.5"
                  style={{
                    borderColor: provider === p ? "var(--accent)" : "var(--line)",
                    color: provider === p ? "var(--accent)" : "var(--muted)",
                  }}
                >
                  {p === "all" ? "All" : PROVIDER_LABEL[p] ?? p}
                </button>
              ))}
              <span className="ml-auto" style={{ color: "var(--muted)" }}>
                sort
              </span>
              <select className="input !w-auto py-0.5 text-xs" value={sort} onChange={(e) => setSort(e.target.value as Sort)}>
                <option value="name">name</option>
                <option value="price">cheapest</option>
                <option value="newest">newest</option>
                <option value="context">context</option>
              </select>
              <button
                type="button"
                className="underline"
                style={{ color: "var(--muted)" }}
                title="Re-fetch the model list"
                onClick={() => loadModels(wsId, true).then(setData)}
              >
                refresh
              </button>
            </div>
          </div>

          <div ref={list} id="model-options" role="listbox" className="overflow-y-auto py-1">
            {err && <div className="px-3 py-2 text-sm" style={{ color: "var(--bad)" }}>{err}</div>}
            {!data && !err && <div className="px-3 py-2 text-sm" style={{ color: "var(--muted)" }}>Loading models…</div>}
            {data && !data.catalog_available && (
              <div className="px-3 py-2 text-xs" style={{ color: "var(--warn)" }}>
                The model catalog couldn&apos;t be fetched; type a provider:model id.
              </div>
            )}
            {data && data.providers.length === 0 && (
              <div className="px-3 py-2 text-xs" style={{ color: "var(--muted)" }}>
                Only the offline model is available. Add a provider key under Provider keys (sidebar) to see real models.
              </div>
            )}
            {custom && (
              <Row idx={0} active={active === 0} onPick={() => choose(custom)} onHover={() => setActive(0)}>
                <span>
                  Use custom id <span className="mono">{custom}</span>
                </span>
              </Row>
            )}
            {shown.map((m, i) => {
              const idx = i + (custom ? 1 : 0);
              return (
                <Row key={m.id} idx={idx} active={active === idx} selected={m.id === value} onPick={() => choose(m.id)} onHover={() => setActive(idx)}>
                  <div className="min-w-0 flex-1">
                    <div className="flex items-center gap-2">
                      <span className="truncate">{m.name}</span>
                      {m.tags
                        .filter((t) => SHOWN_TAGS.includes(t))
                        .map((t) => (
                          <span key={t} className="rounded px-1 text-[10px] uppercase tracking-wide" style={{ background: "var(--line)", color: "var(--muted)" }}>
                            {t}
                          </span>
                        ))}
                    </div>
                    <div className="mono truncate text-xs" style={{ color: "var(--muted)" }}>
                      {m.id}
                    </div>
                  </div>
                  <div className="shrink-0 text-right text-xs" style={{ color: "var(--muted)" }}>
                    <div>{PROVIDER_LABEL[m.provider] ?? m.provider}</div>
                    <div className="mono">
                      {m.provider === "scripted" ? "free" : `${usd(m.input_per_mtok)} / ${usd(m.output_per_mtok)} per M`}
                      {ctx(m.context_window) && ` · ${ctx(m.context_window)}`}
                    </div>
                  </div>
                </Row>
              );
            })}
            {filtered.length > MAX_ROWS && (
              <div className="px-3 py-2 text-xs" style={{ color: "var(--muted)" }}>
                Showing {MAX_ROWS} of {filtered.length}. Refine the search to see more.
              </div>
            )}
            {data && filtered.length === 0 && !custom && (
              <div className="px-3 py-2 text-sm" style={{ color: "var(--muted)" }}>
                No match.{hasVllm && " For your vLLM box type vllm:<model id it serves>."}
              </div>
            )}
          </div>
          {hasVllm && (
            <div className="border-t px-3 py-2 text-xs" style={{ borderColor: "var(--line)", color: "var(--muted)" }}>
              Self-hosted: type <span className="mono">vllm:Qwen/Qwen2.5-7B-Instruct</span> (whatever your vLLM box serves).
            </div>
          )}
        </div>
      )}
    </div>
  );
}

function Row({
  idx,
  active,
  selected,
  onPick,
  onHover,
  children,
}: {
  idx: number;
  active: boolean;
  selected?: boolean;
  onPick: () => void;
  onHover: () => void;
  children: React.ReactNode;
}) {
  return (
    <div
      data-idx={idx}
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

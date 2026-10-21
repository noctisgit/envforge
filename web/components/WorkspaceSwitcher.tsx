"use client";

// Sidebar workspace switcher: a searchable dropdown in the same style as the model
// picker, with "new workspace" and "settings" at the bottom. Keyboard: ↑ ↓ Enter Esc.

import Link from "next/link";
import { useEffect, useMemo, useRef, useState } from "react";
import { WorkspaceMark as Mark } from "@/components/Avatar";
import { Api, ApiError } from "@/lib/api";
import { useWorkspace } from "@/lib/workspace";

export function WorkspaceSwitcher() {
  const { workspaces, current, select, reload } = useWorkspace();
  const [open, setOpen] = useState(false);
  const [query, setQuery] = useState("");
  const [active, setActive] = useState(0);
  const [creating, setCreating] = useState(false);
  const [newName, setNewName] = useState("");
  const [err, setErr] = useState<string | null>(null);
  const root = useRef<HTMLDivElement>(null);
  const trigger = useRef<HTMLButtonElement>(null);
  const search = useRef<HTMLInputElement>(null);
  // The menu is wider than the sidebar, and the sidebar scrolls (which would clip it),
  // so it is positioned against the viewport, just under the trigger.
  const [pos, setPos] = useState<{ top: number; left: number; width: number } | null>(null);

  useEffect(() => {
    if (!open) return;
    setQuery("");
    setActive(0);
    setCreating(false);
    setErr(null);
    const place = () => {
      const r = trigger.current?.getBoundingClientRect();
      if (!r) return;
      const width = Math.min(Math.max(r.width, 288), window.innerWidth - 16);
      setPos({ top: r.bottom + 8, left: Math.max(8, Math.min(r.left, window.innerWidth - width - 8)), width });
    };
    place();
    setTimeout(() => search.current?.focus({ preventScroll: true }), 0);
    const onDown = (e: MouseEvent) => {
      if (root.current && !root.current.contains(e.target as Node)) setOpen(false);
    };
    document.addEventListener("mousedown", onDown);
    window.addEventListener("resize", place);
    window.addEventListener("scroll", place, true);
    return () => {
      document.removeEventListener("mousedown", onDown);
      window.removeEventListener("resize", place);
      window.removeEventListener("scroll", place, true);
    };
  }, [open]);

  const filtered = useMemo(() => {
    const q = query.toLowerCase().trim();
    return workspaces.filter((w) => !q || w.name.toLowerCase().includes(q) || w.slug.includes(q));
  }, [workspaces, query]);

  if (!current) return null;

  function choose(id: string) {
    select(id);
    setOpen(false);
  }

  async function create() {
    const name = newName.trim();
    if (!name) return;
    try {
      const ws = await Api.createWorkspace(name);
      await reload();
      select(ws.id);
      setNewName("");
      setOpen(false);
    } catch (e) {
      setErr(e instanceof ApiError ? e.message : String(e));
    }
  }

  function onKey(e: React.KeyboardEvent) {
    if (creating) return;
    if (e.key === "ArrowDown") {
      e.preventDefault();
      setActive((i) => Math.min(i + 1, filtered.length - 1));
    } else if (e.key === "ArrowUp") {
      e.preventDefault();
      setActive((i) => Math.max(i - 1, 0));
    } else if (e.key === "Enter" && filtered[active]) {
      e.preventDefault();
      choose(filtered[active].id);
    } else if (e.key === "Escape") {
      setOpen(false);
    }
  }

  return (
    <div ref={root} className="relative">
      <button
        ref={trigger}
        type="button"
        onClick={() => setOpen((o) => !o)}
        aria-haspopup="listbox"
        aria-expanded={open}
        aria-label="Switch workspace"
        className="flex w-full items-center gap-2 rounded-md border px-2 py-1.5 text-sm hover:border-[var(--accent)]"
        style={{ borderColor: "var(--line)", background: "var(--input-bg)" }}
      >
        <Mark name={current.name} />
        <span className="min-w-0 flex-1 truncate text-left">{current.name}</span>
        <span className="text-xs" style={{ color: "var(--muted)" }}>
          ▾
        </span>
      </button>

      {open && pos && (
        <div
          className="panel fixed z-50 overflow-hidden"
          style={{ top: pos.top, left: pos.left, width: pos.width, boxShadow: "var(--shadow-dialog)" }}
          onKeyDown={onKey}
        >
          <div className="border-b p-2" style={{ borderColor: "var(--line)" }}>
            <input
              ref={search}
              className="input"
              placeholder={`Search ${workspaces.length} workspace${workspaces.length === 1 ? "" : "s"}…`}
              value={query}
              onChange={(e) => {
                setQuery(e.target.value);
                setActive(0);
              }}
              role="combobox"
              aria-expanded
              aria-controls="ws-options"
            />
          </div>
          <div id="ws-options" role="listbox" className="max-h-72 overflow-y-auto py-1">
            {filtered.length === 0 && (
              <div className="px-3 py-2 text-sm" style={{ color: "var(--muted)" }}>
                No workspace matches “{query}”.
              </div>
            )}
            {filtered.map((w, i) => (
              <div
                key={w.id}
                role="option"
                aria-selected={w.id === current.id}
                onMouseEnter={() => setActive(i)}
                onClick={() => choose(w.id)}
                className="flex cursor-pointer items-center gap-3 px-3 py-2 text-sm"
                style={{ background: i === active ? "var(--line)" : undefined }}
              >
                <Mark name={w.name} />
                <div className="min-w-0 flex-1">
                  <div className="truncate">{w.name}</div>
                  <div className="mono truncate text-[11px]" style={{ color: "var(--muted)" }}>
                    {w.default_model}
                  </div>
                </div>
                {w.id === current.id && <span style={{ color: "var(--accent)" }}>✓</span>}
              </div>
            ))}
          </div>
          <div className="border-t p-2" style={{ borderColor: "var(--line)" }}>
            {creating ? (
              <div className="space-y-2">
                <div className="flex gap-2">
                  <input
                    className="input"
                    autoFocus
                    placeholder="workspace name"
                    maxLength={120}
                    value={newName}
                    onChange={(e) => setNewName(e.target.value)}
                    onKeyDown={(e) => {
                      if (e.key === "Enter") create();
                      if (e.key === "Escape") setCreating(false);
                    }}
                  />
                  <button className="btn" onClick={create} disabled={!newName.trim()}>
                    Create
                  </button>
                </div>
                {err && (
                  <div className="text-xs" style={{ color: "var(--bad)" }}>
                    {err}
                  </div>
                )}
              </div>
            ) : (
              <div className="flex items-center justify-between text-sm">
                <button className="rounded px-2 py-1 hover:bg-[var(--line)]" onClick={() => setCreating(true)}>
                  + New workspace
                </button>
                <Link href="/settings" className="rounded px-2 py-1 hover:bg-[var(--line)]" style={{ color: "var(--muted)" }} onClick={() => setOpen(false)}>
                  Settings
                </Link>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

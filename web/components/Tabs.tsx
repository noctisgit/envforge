"use client";

// A small accessible tab bar (WAI-ARIA tabs pattern) plus a hook that keeps the
// active tab in the URL hash, so a tab is linkable and survives a reload.
//
//   const [tab, setTab] = useHashTab(["overview", "tools"], "overview");
//   <Tabs idPrefix="env" tabs={[{ id: "overview", label: "Overview" }, ...]} active={tab} onChange={setTab} />
//   <div {...tabPanelProps("env", "overview")} hidden={tab !== "overview"}>…</div>
//
// Arrow keys (and Home/End) move between tabs; only the active tab is in the Tab order.
// variant "underline" is the page-level bar (accent underline); "pills" is a compact
// secondary selector for use inside a tab.

import { useCallback, useEffect, useRef, useState } from "react";

export interface TabItem {
  id: string;
  label: string;
  count?: number | string;
}

export function Tabs({
  tabs,
  active,
  onChange,
  idPrefix,
  label,
  variant = "underline",
}: {
  tabs: TabItem[];
  active: string;
  onChange: (id: string) => void;
  idPrefix: string;
  label?: string;
  variant?: "underline" | "pills";
}) {
  const refs = useRef<(HTMLButtonElement | null)[]>([]);

  function onKeyDown(e: React.KeyboardEvent, i: number) {
    let next = -1;
    if (e.key === "ArrowRight") next = (i + 1) % tabs.length;
    else if (e.key === "ArrowLeft") next = (i - 1 + tabs.length) % tabs.length;
    else if (e.key === "Home") next = 0;
    else if (e.key === "End") next = tabs.length - 1;
    if (next < 0) return;
    e.preventDefault();
    onChange(tabs[next].id);
    refs.current[next]?.focus();
  }

  // keep the active tab visible when the bar scrolls horizontally (narrow screens);
  // only the bar scrolls, never the page
  const bar = useRef<HTMLDivElement>(null);
  const activeIndex = tabs.findIndex((t) => t.id === active);
  useEffect(() => {
    const el = refs.current[activeIndex];
    const box = bar.current;
    if (!el || !box || box.scrollWidth <= box.clientWidth) return;
    const left = el.offsetLeft; // the bar is position:relative, so this is relative to it
    if (left < box.scrollLeft) box.scrollLeft = left - 8;
    else if (left + el.offsetWidth > box.scrollLeft + box.clientWidth) box.scrollLeft = left + el.offsetWidth - box.clientWidth + 8;
  }, [activeIndex]);

  const underline = variant === "underline";
  return (
    <div
      ref={bar}
      role="tablist"
      aria-label={label}
      className={underline ? "relative flex gap-1 overflow-x-auto border-b" : "relative flex flex-wrap gap-1.5"}
      style={underline ? { borderColor: "var(--line)", scrollbarWidth: "thin" } : undefined}
    >
      {tabs.map((t, i) => {
        const on = t.id === active;
        return (
          <button
            key={t.id}
            ref={(el) => {
              refs.current[i] = el;
            }}
            id={`${idPrefix}-tab-${t.id}`}
            role="tab"
            type="button"
            aria-selected={on}
            aria-controls={`${idPrefix}-panel-${t.id}`}
            tabIndex={on ? 0 : -1}
            onClick={() => onChange(t.id)}
            onKeyDown={(e) => onKeyDown(e, i)}
            className={
              underline
                ? "-mb-px shrink-0 whitespace-nowrap rounded-t-md px-3 py-2 text-sm outline-none focus-visible:bg-white/5"
                : "whitespace-nowrap rounded-full border px-2.5 py-1 text-xs outline-none focus-visible:ring-1 focus-visible:ring-[var(--accent)]"
            }
            style={
              underline
                ? {
                    color: on ? "var(--fg)" : "var(--muted)",
                    fontWeight: on ? 500 : 400,
                    borderBottom: `2px solid ${on ? "var(--accent)" : "transparent"}`,
                  }
                : {
                    color: on ? "var(--accent)" : "var(--muted)",
                    borderColor: on ? "var(--accent)" : "var(--line)",
                    background: on ? "var(--accent-tint)" : "transparent",
                  }
            }
          >
            {t.label}
            {t.count !== undefined && (
              <span className="ml-1.5 tabular-nums" style={{ color: on && underline ? "var(--muted)" : undefined, opacity: on ? 1 : 0.8 }}>
                ({t.count})
              </span>
            )}
          </button>
        );
      })}
    </div>
  );
}

/** Props for the element that holds one tab's content. */
export function tabPanelProps(idPrefix: string, id: string) {
  return {
    id: `${idPrefix}-panel-${id}`,
    role: "tabpanel",
    "aria-labelledby": `${idPrefix}-tab-${id}`,
    tabIndex: 0,
  } as const;
}

/**
 * Active tab mirrored in the URL hash (`#tools`). The default tab keeps a clean URL.
 * Unknown hashes fall back to the default. Uses replaceState, so switching tabs
 * doesn't pile up history entries; a pasted link or a reload restores the tab.
 */
export function useHashTab(ids: readonly string[], fallback: string): [string, (id: string) => void] {
  const [tab, setTab] = useState(fallback);
  const key = ids.join("|");

  useEffect(() => {
    const read = () => {
      const h = decodeURIComponent(window.location.hash.slice(1));
      setTab(ids.includes(h) ? h : fallback);
    };
    read();
    window.addEventListener("hashchange", read);
    return () => window.removeEventListener("hashchange", read);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [key, fallback]);

  const select = useCallback(
    (id: string) => {
      setTab(id);
      const { pathname, search } = window.location;
      window.history.replaceState(window.history.state, "", id === fallback ? pathname + search : `${pathname}${search}#${id}`);
    },
    [fallback],
  );
  return [tab, select];
}

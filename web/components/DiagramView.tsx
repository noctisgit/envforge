"use client";

// Renders any pack's Diagram (envforge/core/env.py). Nothing here knows about a
// domain: nodes and edges carry a few standard states (ok, broken, target, path,
// inactive), free-form kinds and badges, and optional fixed positions.
//
//   layout "graph": SVG; fixed x/y when the pack gives them, else a deterministic
//                   force layout; nodes sharing a `group` get a labelled zone
//   layout "tree":  a collapsible outline using each node's `parent`
//
// Click a node for its `detail` text.

import { useEffect, useMemo, useRef, useState } from "react";
import type { Diagram, DiagramEdge, DiagramNode } from "@/lib/api";

const NODE_STYLE: Record<string, { stroke: string; fill: string; dash?: string; opacity?: number }> = {
  ok: { stroke: "var(--line)", fill: "var(--panel)" },
  broken: { stroke: "var(--bad)", fill: "var(--bad-tint)" },
  target: { stroke: "var(--accent)", fill: "var(--accent-tint)" },
  inactive: { stroke: "var(--muted)", fill: "var(--panel)", dash: "4 3", opacity: 0.55 },
};
const EDGE_STYLE: Record<string, { stroke: string; width: number; dash?: string; opacity?: number }> = {
  ok: { stroke: "var(--diagram-edge)", width: 1.5 },
  path: { stroke: "var(--accent)", width: 2.5 },
  broken: { stroke: "var(--bad)", width: 2.5, dash: "6 4" },
  inactive: { stroke: "var(--diagram-edge)", width: 1.5, dash: "3 4", opacity: 0.5 },
};
const STATE_LABEL: Record<string, string> = {
  broken: "faulty / where it fails",
  target: "named in the goal",
  path: "the goal's traffic gets through here",
  inactive: "disabled / unused",
};
const nodeStyle = (s: string) => NODE_STYLE[s] ?? NODE_STYLE.ok;
const edgeStyle = (s: string) => EDGE_STYLE[s] ?? EDGE_STYLE.ok;

export function DiagramView({ diagram, height = 460 }: { diagram: Diagram; height?: number }) {
  const [selected, setSelected] = useState<string | null>(null);
  const node = diagram.nodes.find((n) => n.id === selected) ?? null;
  useEffect(() => {
    if (selected && !diagram.nodes.some((n) => n.id === selected)) setSelected(null);
  }, [diagram, selected]);

  return (
    <div className="space-y-3">
      {diagram.title && <div className="text-sm font-semibold">{diagram.title}</div>}
      {diagram.layout === "tree" ? (
        <TreeView diagram={diagram} selected={selected} onSelect={setSelected} />
      ) : (
        <GraphView diagram={diagram} height={height} selected={selected} onSelect={setSelected} />
      )}
      <Legend diagram={diagram} />
      {diagram.notes.length > 0 && (
        <ul className="space-y-0.5 text-sm">
          {diagram.notes.map((n, i) => (
            <li key={i} className="mono text-xs" style={{ color: /\bOK\b/.test(n) ? "var(--good)" : "var(--fg)" }}>
              {n}
            </li>
          ))}
        </ul>
      )}
      {node ? (
        <div className="rounded-md border p-3 text-sm" style={{ borderColor: "var(--line)", background: "var(--input-bg)" }}>
          <div className="mb-1 flex flex-wrap items-center gap-2">
            <span className="font-medium">{node.label}</span>
            <span className="text-xs" style={{ color: "var(--muted)" }}>
              {node.kind}
              {node.group ? ` · ${node.group}` : ""}
            </span>
            {node.state !== "ok" && <StateChip state={node.state} />}
            {node.badges.map((b) => (
              <Badge key={b} text={b} />
            ))}
          </div>
          {node.detail && <pre className="mono overflow-x-auto whitespace-pre-wrap text-xs" style={{ color: "var(--muted)" }}>{node.detail}</pre>}
        </div>
      ) : (
        <div className="text-xs" style={{ color: "var(--muted)" }}>
          Click a node for details.
        </div>
      )}
    </div>
  );
}

function StateChip({ state }: { state: string }) {
  const s = nodeStyle(state);
  return (
    <span className="rounded px-1.5 text-[10px] uppercase tracking-wide" style={{ border: `1px solid ${s.stroke}`, color: s.stroke }}>
      {state}
    </span>
  );
}

function Badge({ text }: { text: string }) {
  return (
    <span className="rounded px-1 text-[10px] uppercase tracking-wide" style={{ background: "var(--line)", color: "var(--muted)" }}>
      {text}
    </span>
  );
}

function Legend({ diagram }: { diagram: Diagram }) {
  const states = new Set([...diagram.nodes.map((n) => n.state), ...diagram.edges.map((e) => e.state)]);
  const kinds = Array.from(new Set(diagram.nodes.map((n) => n.kind)));
  const badges = Array.from(new Set(diagram.nodes.flatMap((n) => n.badges)));
  const shown = Object.keys(STATE_LABEL).filter((s) => states.has(s));
  return (
    <div className="flex flex-wrap items-center gap-x-4 gap-y-1 text-xs" style={{ color: "var(--muted)" }}>
      {shown.map((s) => (
        <span key={s} className="flex items-center gap-1.5">
          {s === "path" ? (
            <svg width="18" height="6" aria-hidden>
              <line x1="0" y1="3" x2="18" y2="3" stroke={edgeStyle(s).stroke} strokeWidth="2.5" />
            </svg>
          ) : (
            <span className="inline-block h-3 w-3 rounded-sm" style={{ border: `1.5px ${nodeStyle(s).dash ? "dashed" : "solid"} ${nodeStyle(s).stroke}`, background: nodeStyle(s).fill }} />
          )}
          {STATE_LABEL[s]}
        </span>
      ))}
      {kinds.length > 1 && <span>kinds: {kinds.join(", ")}</span>}
      {badges.length > 0 && (
        <span className="flex flex-wrap items-center gap-1">
          badges: {badges.map((b) => <Badge key={b} text={b} />)}
        </span>
      )}
    </div>
  );
}

// ---- graph -------------------------------------------------------------------

/** Deterministic force layout for nodes without fixed positions (Fruchterman–Reingold). */
function layout(nodes: DiagramNode[], edges: DiagramEdge[]): Map<string, { x: number; y: number }> {
  const pos = new Map<string, { x: number; y: number }>();
  const free = nodes.filter((n) => n.x === null || n.y === null);
  nodes.forEach((n) => {
    if (n.x !== null && n.y !== null) pos.set(n.id, { x: n.x, y: n.y });
  });
  if (!free.length) return pos;
  const sorted = [...free].sort((a, b) => a.id.localeCompare(b.id));
  sorted.forEach((n, i) => {
    const a = (2 * Math.PI * i) / sorted.length;
    pos.set(n.id, { x: 0.5 + 0.35 * Math.cos(a), y: 0.5 + 0.35 * Math.sin(a) });
  });
  const k = Math.sqrt(1 / nodes.length) * 0.9;
  const fixed = new Set(nodes.filter((n) => n.x !== null && n.y !== null).map((n) => n.id));
  for (let it = 0; it < 300; it++) {
    const t = 0.1 * (1 - it / 300);
    const disp = new Map(nodes.map((n) => [n.id, { x: 0, y: 0 }]));
    for (const a of nodes)
      for (const b of nodes) {
        if (a.id === b.id) continue;
        const pa = pos.get(a.id)!;
        const pb = pos.get(b.id)!;
        const dx = pa.x - pb.x || 1e-4;
        const dy = pa.y - pb.y || 1e-4;
        const d = Math.hypot(dx, dy);
        const f = (k * k) / d;
        disp.get(a.id)!.x += (dx / d) * f;
        disp.get(a.id)!.y += (dy / d) * f;
      }
    for (const e of edges) {
      const pa = pos.get(e.source);
      const pb = pos.get(e.target);
      if (!pa || !pb) continue;
      const dx = pa.x - pb.x;
      const dy = pa.y - pb.y;
      const d = Math.hypot(dx, dy) || 1e-4;
      const f = (d * d) / k;
      disp.get(e.source)!.x -= (dx / d) * f;
      disp.get(e.source)!.y -= (dy / d) * f;
      disp.get(e.target)!.x += (dx / d) * f;
      disp.get(e.target)!.y += (dy / d) * f;
    }
    for (const n of free) {
      const p = pos.get(n.id)!;
      const d = disp.get(n.id)!;
      const len = Math.hypot(d.x, d.y) || 1;
      p.x = Math.min(0.95, Math.max(0.05, p.x + (d.x / len) * Math.min(len, t)));
      p.y = Math.min(0.95, Math.max(0.05, p.y + (d.y / len) * Math.min(len, t)));
    }
  }
  if (!fixed.size) {
    // stretch the free layout to fill the canvas
    const xs = free.map((n) => pos.get(n.id)!.x);
    const ys = free.map((n) => pos.get(n.id)!.y);
    const [x0, x1, y0, y1] = [Math.min(...xs), Math.max(...xs), Math.min(...ys), Math.max(...ys)];
    for (const n of free) {
      const p = pos.get(n.id)!;
      p.x = x1 > x0 ? 0.06 + ((p.x - x0) / (x1 - x0)) * 0.88 : 0.5;
      p.y = y1 > y0 ? 0.08 + ((p.y - y0) / (y1 - y0)) * 0.84 : 0.5;
    }
  }
  return pos;
}

function GraphView({ diagram, height, selected, onSelect }: { diagram: Diagram; height: number; selected: string | null; onSelect: (id: string | null) => void }) {
  const wrap = useRef<HTMLDivElement>(null);
  const [w, setW] = useState(800);
  useEffect(() => {
    if (!wrap.current) return;
    const ro = new ResizeObserver(([e]) => setW(Math.max(320, e.contentRect.width)));
    ro.observe(wrap.current);
    return () => ro.disconnect();
  }, []);

  // positions only depend on structure, so replay frames don't make nodes jump
  const structure = diagram.nodes.map((n) => `${n.id}:${n.x}:${n.y}`).join("|") + "#" + diagram.edges.map((e) => `${e.source}>${e.target}`).join("|");
  // eslint-disable-next-line react-hooks/exhaustive-deps
  const pos = useMemo(() => layout(diagram.nodes, diagram.edges), [structure]);
  const H = height;
  const P = 40;
  const px = (id: string) => {
    const p = pos.get(id) ?? { x: 0.5, y: 0.5 };
    return { x: P + p.x * (w - 2 * P), y: P + p.y * (H - 2 * P) };
  };
  const boxW = (n: DiagramNode) => Math.max(44, n.label.length * 7.2 + 22);
  const BOX_H = 28;

  const groups = new Map<string, DiagramNode[]>();
  diagram.nodes.forEach((n) => n.group && groups.set(n.group, [...(groups.get(n.group) ?? []), n]));
  const nodeById = new Map(diagram.nodes.map((n) => [n.id, n]));

  return (
    <div ref={wrap} className="w-full overflow-x-auto rounded-md border" style={{ borderColor: "var(--line)", background: "var(--code-bg)" }}>
      <svg width={w} height={H} role="img" aria-label={diagram.title || "environment diagram"} onClick={() => onSelect(null)}>
        <defs>
          <marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
            <path d="M0,0 L10,5 L0,10 z" fill="var(--diagram-edge)" />
          </marker>
        </defs>
        {/* zones */}
        {Array.from(groups.entries())
          .filter(([, ns]) => ns.length > 1)
          .map(([g, ns]) => {
            const pts = ns.map((n) => ({ ...px(n.id), hw: boxW(n) / 2 }));
            const x0 = Math.min(...pts.map((p) => p.x - p.hw)) - 12;
            const x1 = Math.max(...pts.map((p) => p.x + p.hw)) + 12;
            const y0 = Math.min(...pts.map((p) => p.y)) - BOX_H / 2 - 22;
            const y1 = Math.max(...pts.map((p) => p.y)) + BOX_H / 2 + 18;
            return (
              <g key={g}>
                <rect x={x0} y={y0} width={x1 - x0} height={y1 - y0} rx={10} fill="rgba(255,255,255,0.02)" stroke="var(--diagram-zone-line)" />
                <text x={x0 + 8} y={y0 + 14} fontSize={10} fill="var(--muted)" letterSpacing="0.06em">
                  {g.toUpperCase()}
                </text>
              </g>
            );
          })}
        {/* edges */}
        {diagram.edges.map((e, i) => {
          if (!nodeById.has(e.source) || !nodeById.has(e.target)) return null;
          const a = px(e.source);
          const b = px(e.target);
          const s = edgeStyle(e.state);
          const mx = (a.x + b.x) / 2;
          const my = (a.y + b.y) / 2;
          return (
            <g key={i} opacity={s.opacity ?? 1}>
              <line x1={a.x} y1={a.y} x2={b.x} y2={b.y} stroke={s.stroke} strokeWidth={s.width} strokeDasharray={s.dash} markerEnd={e.directed ? "url(#arrow)" : undefined} />
              {e.label && (
                <>
                  <rect x={mx - e.label.length * 3 - 4} y={my - 8} width={e.label.length * 6 + 8} height={14} rx={3} fill="var(--code-bg)" />
                  <text x={mx} y={my + 3} fontSize={9.5} textAnchor="middle" fill="var(--muted)" className="mono">
                    {e.label}
                  </text>
                </>
              )}
            </g>
          );
        })}
        {/* nodes */}
        {diagram.nodes.map((n) => {
          const { x, y } = px(n.id);
          const bw = boxW(n);
          const s = nodeStyle(n.state);
          const isSel = n.id === selected;
          return (
            <g
              key={n.id}
              transform={`translate(${x - bw / 2},${y - BOX_H / 2})`}
              opacity={s.opacity ?? 1}
              className="cursor-pointer"
              onClick={(ev) => {
                ev.stopPropagation();
                onSelect(isSel ? null : n.id);
              }}
            >
              <title>{`${n.label} (${n.kind})${n.detail ? "\n" + n.detail : ""}`}</title>
              <rect width={bw} height={BOX_H} rx={7} fill={s.fill} stroke={isSel ? "var(--fg)" : s.stroke} strokeWidth={isSel ? 2 : 1.5} strokeDasharray={s.dash} />
              <text x={bw / 2} y={BOX_H / 2 + 4} fontSize={12} textAnchor="middle" fill="var(--fg)">
                {n.label}
              </text>
              <text x={bw / 2} y={-4} fontSize={9} textAnchor="middle" fill="var(--muted)">
                {n.kind}
              </text>
              {n.badges.length > 0 && (
                <text x={bw / 2} y={BOX_H + 11} fontSize={9} textAnchor="middle" fill="var(--warn)">
                  {n.badges.join(" · ")}
                </text>
              )}
            </g>
          );
        })}
      </svg>
    </div>
  );
}

// ---- tree --------------------------------------------------------------------

function TreeView({ diagram, selected, onSelect }: { diagram: Diagram; selected: string | null; onSelect: (id: string | null) => void }) {
  const children = useMemo(() => {
    const m = new Map<string | null, DiagramNode[]>();
    diagram.nodes.forEach((n) => m.set(n.parent, [...(m.get(n.parent) ?? []), n]));
    return m;
  }, [diagram]);
  const [closed, setClosed] = useState<Set<string>>(new Set());

  const render = (parent: string | null, depth: number): React.ReactNode =>
    (children.get(parent) ?? []).map((n) => {
      const kids = children.get(n.id);
      const isOpen = !closed.has(n.id);
      const s = nodeStyle(n.state);
      return (
        <div key={n.id}>
          <div
            className="flex cursor-pointer items-center gap-2 rounded px-2 py-1 text-sm"
            style={{ paddingLeft: 8 + depth * 18, background: n.id === selected ? "var(--line)" : undefined, opacity: s.opacity ?? 1 }}
            onClick={() => onSelect(n.id === selected ? null : n.id)}
          >
            <span
              className="w-3 text-xs"
              style={{ color: "var(--muted)" }}
              onClick={(e) => {
                if (!kids) return;
                e.stopPropagation();
                const next = new Set(closed);
                if (isOpen) next.add(n.id);
                else next.delete(n.id);
                setClosed(next);
              }}
            >
              {kids ? (isOpen ? "▾" : "▸") : "·"}
            </span>
            <span className="mono" style={{ color: n.state === "ok" ? "var(--fg)" : s.stroke }}>
              {n.label}
            </span>
            <span className="text-[10px]" style={{ color: "var(--muted)" }}>
              {n.kind}
            </span>
            {n.state !== "ok" && <StateChip state={n.state} />}
            {n.badges.map((b) => (
              <Badge key={b} text={b} />
            ))}
          </div>
          {kids && isOpen && render(n.id, depth + 1)}
        </div>
      );
    });

  return (
    <div className="max-h-[480px] overflow-auto rounded-md border py-2" style={{ borderColor: "var(--line)", background: "var(--code-bg)" }}>
      {render(null, 0)}
    </div>
  );
}

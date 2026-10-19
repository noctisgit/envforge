"use client";

// Minimal single-series bar chart (daily usage). One hue, no legend (the title names
// the series), recessive grid, hover tooltip per bar. Plain SVG, no chart library.

import { useEffect, useRef, useState } from "react";

export interface Bar {
  label: string; // x label, e.g. "2026-10-19"
  value: number;
  detail?: string; // extra tooltip line
}

const H = 160;
const PAD = { top: 10, right: 8, bottom: 22, left: 44 };

function niceMax(v: number): number {
  if (v <= 0) return 1;
  const p = 10 ** Math.floor(Math.log10(v));
  return [1, 2, 2.5, 5, 10].map((m) => m * p).find((x) => x >= v) ?? v;
}

export function BarChart({ data, format, color = "var(--accent)" }: { data: Bar[]; format: (v: number) => string; color?: string }) {
  const wrap = useRef<HTMLDivElement>(null);
  const [w, setW] = useState(600);
  const [hover, setHover] = useState<number | null>(null);

  useEffect(() => {
    if (!wrap.current) return;
    const ro = new ResizeObserver(([e]) => setW(Math.max(240, e.contentRect.width)));
    ro.observe(wrap.current);
    return () => ro.disconnect();
  }, []);

  const max = niceMax(Math.max(0, ...data.map((d) => d.value)));
  const innerW = w - PAD.left - PAD.right;
  const innerH = H - PAD.top - PAD.bottom;
  const slot = innerW / Math.max(1, data.length);
  const barW = Math.max(2, Math.min(28, slot - 2)); // 2px surface gap between bars
  const y = (v: number) => PAD.top + innerH - (v / max) * innerH;
  const ticks = [0, max / 2, max];
  const every = Math.ceil(data.length / Math.max(2, Math.floor(innerW / 64))); // x-label density

  return (
    <div ref={wrap} className="relative w-full" onMouseLeave={() => setHover(null)}>
      <svg width={w} height={H} role="img" aria-label="daily usage bar chart">
        {ticks.map((t) => (
          <g key={t}>
            <line x1={PAD.left} x2={w - PAD.right} y1={y(t)} y2={y(t)} stroke="var(--line)" strokeWidth={1} />
            <text x={PAD.left - 6} y={y(t) + 4} textAnchor="end" fontSize={10} fill="var(--muted)">
              {format(t)}
            </text>
          </g>
        ))}
        {data.map((d, i) => {
          const x = PAD.left + i * slot + (slot - barW) / 2;
          const top = y(d.value);
          const h = PAD.top + innerH - top;
          const r = Math.min(4, barW / 2, h);
          return (
            <g key={d.label}>
              {/* hit target: the whole column, bigger than the bar */}
              <rect x={PAD.left + i * slot} y={PAD.top} width={slot} height={innerH} fill="transparent" onMouseEnter={() => setHover(i)} />
              {d.value > 0 && (
                <path
                  d={`M${x},${top + h} V${top + r} Q${x},${top} ${x + r},${top} H${x + barW - r} Q${x + barW},${top} ${x + barW},${top + r} V${top + h} Z`}
                  fill={color}
                  opacity={hover === null || hover === i ? 1 : 0.45}
                  pointerEvents="none"
                />
              )}
              {i % every === 0 && (
                <text x={PAD.left + i * slot + slot / 2} y={H - 6} textAnchor="middle" fontSize={10} fill="var(--muted)">
                  {d.label.slice(5)}
                </text>
              )}
            </g>
          );
        })}
      </svg>
      {hover !== null && data[hover] && (
        <div
          className="panel pointer-events-none absolute z-10 px-2 py-1 text-xs"
          style={{
            left: Math.min(w - 150, Math.max(0, PAD.left + hover * slot + slot / 2 - 60)),
            top: 0,
            minWidth: 120,
          }}
        >
          <div style={{ color: "var(--muted)" }}>{data[hover].label}</div>
          <div className="font-semibold">{format(data[hover].value)}</div>
          {data[hover].detail && <div style={{ color: "var(--muted)" }}>{data[hover].detail}</div>}
        </div>
      )}
    </div>
  );
}

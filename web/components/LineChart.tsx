"use client";

// A small inline-SVG line chart for a training loss curve: train loss as a line, eval
// loss as dots, a recessive grid, hover readout. No chart library.

import { useEffect, useRef, useState } from "react";

export interface Series {
  name: string;
  color: string;
  points: { x: number; y: number }[];
  dots?: boolean; // draw points only (sparse series such as eval loss)
}

const H = 180;
const PAD = { top: 10, right: 12, bottom: 24, left: 44 };

export function LineChart({ series, xLabel = "step", format = (v: number) => v.toFixed(3) }: { series: Series[]; xLabel?: string; format?: (v: number) => string }) {
  const wrap = useRef<HTMLDivElement>(null);
  const [w, setW] = useState(600);
  const [hover, setHover] = useState<number | null>(null);

  useEffect(() => {
    if (!wrap.current) return;
    const ro = new ResizeObserver(([e]) => setW(Math.max(240, e.contentRect.width)));
    ro.observe(wrap.current);
    return () => ro.disconnect();
  }, []);

  const all = series.flatMap((s) => s.points);
  if (all.length === 0) return <div className="hint">No loss reported yet.</div>;
  const xMin = Math.min(...all.map((p) => p.x));
  const xMax = Math.max(...all.map((p) => p.x), xMin + 1);
  const yMax = Math.max(...all.map((p) => p.y)) * 1.05 || 1;
  const yMin = Math.min(0, ...all.map((p) => p.y));
  const iw = w - PAD.left - PAD.right;
  const ih = H - PAD.top - PAD.bottom;
  const x = (v: number) => PAD.left + ((v - xMin) / (xMax - xMin)) * iw;
  const y = (v: number) => PAD.top + ih - ((v - yMin) / (yMax - yMin)) * ih;
  const ticks = [0, 0.25, 0.5, 0.75, 1].map((t) => yMin + t * (yMax - yMin));

  function onMove(e: React.MouseEvent<SVGSVGElement>) {
    const rect = e.currentTarget.getBoundingClientRect();
    const px = ((e.clientX - rect.left) / rect.width) * w;
    setHover(Math.round(xMin + ((px - PAD.left) / iw) * (xMax - xMin)));
  }
  const nearest = (s: Series) =>
    hover === null || s.points.length === 0 ? null : s.points.reduce((a, b) => (Math.abs(b.x - hover) < Math.abs(a.x - hover) ? b : a));

  return (
    <div ref={wrap} className="relative min-w-0">
      <svg width={w} height={H} role="img" aria-label={`Loss by ${xLabel}`} onMouseMove={onMove} onMouseLeave={() => setHover(null)}>
        {ticks.map((t) => (
          <g key={t}>
            <line x1={PAD.left} x2={w - PAD.right} y1={y(t)} y2={y(t)} stroke="var(--line)" strokeWidth={1} />
            <text x={PAD.left - 6} y={y(t) + 3} textAnchor="end" fontSize={10} fill="var(--muted)">
              {format(t)}
            </text>
          </g>
        ))}
        <text x={w - PAD.right} y={H - 6} textAnchor="end" fontSize={10} fill="var(--muted)">
          {xLabel} {xMax}
        </text>
        <text x={PAD.left} y={H - 6} fontSize={10} fill="var(--muted)">
          {xMin}
        </text>
        {series.map((s) =>
          s.dots ? (
            s.points.map((p) => <circle key={`${s.name}-${p.x}`} cx={x(p.x)} cy={y(p.y)} r={3.5} fill={s.color} />)
          ) : (
            <polyline
              key={s.name}
              fill="none"
              stroke={s.color}
              strokeWidth={2}
              strokeLinejoin="round"
              points={s.points.map((p) => `${x(p.x)},${y(p.y)}`).join(" ")}
            />
          ),
        )}
        {hover !== null && hover >= xMin && hover <= xMax && <line x1={x(hover)} x2={x(hover)} y1={PAD.top} y2={PAD.top + ih} stroke="var(--muted)" strokeDasharray="3 3" />}
      </svg>
      <div className="hint flex flex-wrap gap-x-4 gap-y-1">
        {series.map((s) => {
          const n = nearest(s);
          return (
            <span key={s.name} className="inline-flex items-center gap-1.5">
              <span aria-hidden className="inline-block h-2 w-2 rounded-full" style={{ background: s.color }} />
              {s.name}
              {n && (
                <span className="num" style={{ color: "var(--fg)" }}>
                  {format(n.y)} @ {n.x}
                </span>
              )}
            </span>
          );
        })}
      </div>
    </div>
  );
}

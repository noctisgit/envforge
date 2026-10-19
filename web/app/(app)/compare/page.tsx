"use client";

// Side-by-side metrics for several runs — different models on the same environment,
// or the same model before/after training. Mirrors `envforge compare`.

import Link from "next/link";
import { useSearchParams } from "next/navigation";
import { Suspense, useEffect, useState } from "react";
import { ExportButton } from "@/components/ExportButton";
import { Api, fmtCost, fmtTokens, type Metrics, type Run } from "@/lib/api";
import { useWorkspace } from "@/lib/workspace";

const pct = (x: number) => `${Math.round(x * 100)}%`;

type Col = { key: string; label: string; get: (m: Metrics) => string; best?: (a: Metrics) => number };

const COLS: Col[] = [
  { key: "n", label: "n", get: (m) => String(m.n) },
  { key: "solve", label: "solve", get: (m) => pct(m.solve_rate), best: (m) => m.solve_rate },
  { key: "reward", label: "reward", get: (m) => m.mean_reward.toFixed(2), best: (m) => m.mean_reward },
  { key: "steps", label: "steps", get: (m) => m.mean_steps.toFixed(1), best: (m) => -m.mean_steps },
  { key: "toolerr", label: "tool err", get: (m) => pct(m.tool_error_rate), best: (m) => -m.tool_error_rate },
  { key: "coll", label: "collateral", get: (m) => pct(m.collateral_rate), best: (m) => -m.collateral_rate },
  { key: "budget", label: "budget out", get: (m) => pct(m.max_steps_rate), best: (m) => -m.max_steps_rate },
  { key: "tokens", label: "tokens/ep", get: (m) => (m.n ? fmtTokens(Math.round((m.input_tokens + m.output_tokens) / m.n)) : "—") },
  { key: "cost", label: "cost/ep", get: (m) => (m.n && m.cost_usd !== null ? fmtCost(m.cost_usd / m.n) : "?") },
];

function CompareInner() {
  const params = useSearchParams();
  const ids = (params.get("runs") ?? "").split(",").filter(Boolean);
  const { current: ws } = useWorkspace();
  const [runs, setRuns] = useState<Run[]>([]);
  const [err, setErr] = useState<string | null>(null);
  const key = ids.join(",");

  useEffect(() => {
    if (!ws || !key) return;
    Promise.all(key.split(",").map((id) => Api.run(ws.id, id)))
      .then(setRuns)
      .catch((e) => setErr(String(e)));
  }, [ws, key]);

  if (!ws) return null;
  if (err) return <div style={{ color: "var(--bad)" }}>{err}</div>;
  const withMetrics = runs.filter((r) => r.metrics);
  const tiers = Array.from(new Set(withMetrics.flatMap((r) => Object.keys(r.metrics!.by_tier)))).sort();

  const table = (pick: (r: Run) => Metrics | undefined) => {
    const ms = withMetrics.map(pick);
    return (
      <table className="data">
        <thead>
          <tr>
            <th>run</th>
            <th>model</th>
            {COLS.map((c) => (
              <th key={c.key}>{c.label}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {withMetrics.map((r, i) => (
            <tr key={r.id}>
              <td>
                <Link href={`/runs/${encodeURIComponent(r.id)}`} className="mono" style={{ color: "var(--accent)" }}>
                  {r.label || r.id.slice(0, 28)}
                </Link>
              </td>
              <td className="mono">{r.model}</td>
              {COLS.map((c) => {
                const m = ms[i];
                if (!m) return <td key={c.key}>—</td>;
                // bold the best value in each column (only where "best" is meaningful)
                const scores = c.best ? ms.map((x) => (x ? c.best!(x) : -Infinity)) : [];
                const isBest = c.best && ms.length > 1 && c.best(m) === Math.max(...scores);
                return (
                  <td key={c.key} className="tabular-nums" style={isBest ? { fontWeight: 600, color: "var(--fg)" } : { color: "var(--muted)" }}>
                    {c.get(m)}
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
    );
  };

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-center gap-3">
        <Link href="/runs" style={{ color: "var(--muted)" }}>
          ← Runs
        </Link>
        <h1 className="text-lg font-semibold">Compare {runs.length} runs</h1>
        <div className="ml-auto">
          <ExportButton wsId={ws.id} runIds={ids} />
        </div>
      </div>
      {runs.length > withMetrics.length && (
        <div className="text-sm" style={{ color: "var(--warn)" }}>
          {runs.length - withMetrics.length} run(s) have no metrics yet and are hidden.
        </div>
      )}
      {new Set(withMetrics.map((r) => r.pack)).size > 1 && (
        <div className="text-sm" style={{ color: "var(--warn)" }}>
          These runs use different environments; the numbers aren&apos;t directly comparable.
        </div>
      )}
      <div className="panel overflow-x-auto p-2">
        <div className="px-2 pt-2 text-sm font-semibold">Overall</div>
        {table((r) => r.metrics?.overall)}
      </div>
      {tiers.map((t) => (
        <div key={t} className="panel overflow-x-auto p-2">
          <div className="px-2 pt-2 text-sm font-semibold">Tier {t}</div>
          {table((r) => r.metrics?.by_tier[t])}
        </div>
      ))}
    </div>
  );
}

export default function ComparePage() {
  // useSearchParams needs a Suspense boundary in the app router
  return (
    <Suspense fallback={null}>
      <CompareInner />
    </Suspense>
  );
}

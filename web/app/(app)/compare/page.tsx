"use client";

// Side-by-side metrics for several runs — different models on the same environment,
// or the same model before/after training. Mirrors `envforge compare`.

import Link from "next/link";
import { useSearchParams } from "next/navigation";
import { Suspense, useEffect, useState } from "react";
import { ExportButton } from "@/components/ExportButton";
import { NextStep } from "@/components/NextStep";
import { Page, PageHeader, Panel, rateTone, toneColor } from "@/components/Page";
import { Tabs } from "@/components/Tabs";
import { Api, ApiError, fmtCost, fmtTokens, type Metrics, type Run } from "@/lib/api";
import { modelRole } from "@/lib/journey";
import { useWorkspace } from "@/lib/workspace";

const pct = (x: number) => `${(x * 100).toFixed(1)}%`;

// One colour per run, in this fixed order (never cycled): the first run is the
// baseline in grey. Bars also carry their value as text, so colour is never the only cue.
const SERIES = ["var(--muted)", "var(--accent)", "var(--fg)", "var(--diagram-edge)"];
const seriesColor = (i: number) => SERIES[i] ?? "var(--diagram-edge)";

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

function Swatch({ i }: { i: number }) {
  return <span aria-hidden className="inline-block h-2.5 w-2.5 shrink-0 rounded-sm" style={{ background: seriesColor(i) }} />;
}

function Big({ label, value, color }: { label: string; value: string; color?: string }) {
  return (
    <div>
      <div className="label mb-0">{label}</div>
      <div className="num text-2xl font-semibold" style={{ color }}>
        {value}
      </div>
    </div>
  );
}

function CompareInner() {
  const params = useSearchParams();
  const ids = (params.get("runs") ?? "").split(",").filter(Boolean);
  const { current: ws } = useWorkspace();
  const [runs, setRuns] = useState<Run[]>([]);
  const [err, setErr] = useState<string | null>(null);
  const [scope, setScope] = useState("overall");
  const key = ids.join(",");

  useEffect(() => {
    if (!ws || !key) return;
    Promise.all(key.split(",").map((id) => Api.run(ws.id, id)))
      .then(setRuns)
      .catch((e) => setErr(e instanceof ApiError ? e.message : String(e)));
  }, [ws, key]);

  if (!ws) return null;
  if (err)
    return (
      <Page>
        <PageHeader back={{ href: "/runs", label: "Runs" }} title="Compare" description={<span style={{ color: "var(--bad)" }}>{err}</span>} />
      </Page>
    );
  const withMetrics = runs.filter((r) => r.metrics);
  const tiers = Array.from(new Set(withMetrics.flatMap((r) => Object.keys(r.metrics!.by_tier)))).sort();
  const scopes = [{ id: "overall", label: "overall" }, ...tiers.map((t) => ({ id: t, label: `tier ${t}` }))];
  const pick = (r: Run, s: string) => (s === "overall" ? r.metrics?.overall : r.metrics?.by_tier[s]);

  const packs = new Set(withMetrics.map((r) => r.pack));
  const sameSeeds = withMetrics.length > 1 && withMetrics.every((r) => r.seed_start === withMetrics[0].seed_start && r.tiers.join() === withMetrics[0].tiers.join());
  const summary = [
    packs.size === 1 ? `same environment (${[...packs][0]})` : null,
    sameSeeds ? `same seed start (${withMetrics[0].seed_start}) and tiers` : null,
    withMetrics.length > 1 ? "the best value in each column is highlighted" : null,
  ]
    .filter(Boolean)
    .join(" · ");

  const base = withMetrics[0]?.metrics?.overall;
  const bestSolve = Math.max(...withMetrics.map((r) => r.metrics!.overall.solve_rate));

  // the distillation hint: a strong API model clearly ahead of an open one on the same tasks
  const solve = (r: Run) => r.metrics!.overall.solve_rate;
  const topTeacher = withMetrics.filter((r) => modelRole(r.model) === "teacher").sort((a, b) => solve(b) - solve(a))[0];
  const topOpen = withMetrics.filter((r) => modelRole(r.model).startsWith("student")).sort((a, b) => solve(b) - solve(a))[0];
  const gap = topTeacher && topOpen && packs.size === 1 ? solve(topTeacher) - solve(topOpen) : 0;

  const s = scopes.some((x) => x.id === scope) ? scope : "overall";
  const ms = withMetrics.map((r) => pick(r, s));

  return (
    <Page>
      <PageHeader
        back={{ href: "/runs", label: "Runs" }}
        title={`Compare ${runs.length} runs`}
        description={summary || undefined}
        actions={<ExportButton wsId={ws.id} runIds={ids} />}
      />
      {runs.length > withMetrics.length && (
        <div className="text-sm" style={{ color: "var(--warn)" }}>
          {runs.length - withMetrics.length} run(s) have no metrics yet and are hidden.
        </div>
      )}
      {packs.size > 1 && (
        <div className="text-sm" style={{ color: "var(--warn)" }}>
          These runs use different environments; the numbers aren&apos;t directly comparable.
        </div>
      )}

      {withMetrics.length > 0 && (
        <div className="grid grid-cols-1 gap-4 md:grid-cols-2 2xl:grid-cols-3">
          {withMetrics.map((r, i) => {
            const m = r.metrics!.overall;
            const delta = base && i > 0 ? (m.solve_rate - base.solve_rate) * 100 : null;
            const best = withMetrics.length > 1 && m.solve_rate === bestSolve;
            const tone = rateTone(m.solve_rate);
            return (
              <div key={r.id} className="panel flex min-w-0 flex-col gap-2 p-5" style={best ? { borderColor: "var(--accent)" } : undefined}>
                <div className="flex flex-wrap items-center gap-2">
                  <Swatch i={i} />
                  <Link href={`/runs/${encodeURIComponent(r.id)}`} className="min-w-0 font-medium [overflow-wrap:anywhere] hover:underline">
                    {r.label || <span className="mono">{r.id}</span>}
                  </Link>
                  {r.label && (
                    <span className="mono text-xs" style={{ color: "var(--muted)" }}>
                      {r.id.slice(-6)}
                    </span>
                  )}
                  {i === 0 && withMetrics.length > 1 && <span className="pill pill-tag">baseline</span>}
                  {delta !== null && (
                    <span className={`pill ${delta > 0 ? "pill-completed" : delta < 0 ? "pill-failed" : "pill-queued"}`}>
                      {delta > 0 ? "+" : ""}
                      {delta.toFixed(1)} pts
                    </span>
                  )}
                </div>
                <div className="mono text-xs [overflow-wrap:anywhere]" style={{ color: "var(--muted)" }}>
                  {r.model}
                </div>
                <div className="mt-2 flex flex-wrap gap-6">
                  <Big label="solve" value={pct(m.solve_rate)} color={toneColor(tone)} />
                  <Big label="reward" value={m.mean_reward.toFixed(2)} />
                  <Big label="steps" value={m.mean_steps.toFixed(1)} />
                  <Big label="episodes" value={String(m.n)} />
                </div>
              </div>
            );
          })}
        </div>
      )}

      {withMetrics.length > 0 && (
        <Panel title="Solve rate by tier" hint={withMetrics.length > 1 ? "one bar per run, in the order of the cards above · scale 0–100%" : "scale 0–100%"} flush>
          <div className="grid grid-cols-1 gap-6 px-4 py-5 sm:grid-cols-2 xl:grid-cols-4">
            {scopes.map((sc) => (
              <figure key={sc.id} className="flex min-w-0 flex-col gap-2">
                <figcaption className="label mb-0">{sc.label}</figcaption>
                <div className="flex h-[120px] items-end gap-1.5 border-b" style={{ borderColor: "var(--line)" }}>
                  {withMetrics.map((r, i) => {
                    const m = pick(r, sc.id);
                    const v = m ? m.solve_rate : 0;
                    const name = r.label || r.id;
                    return (
                      <div
                        key={r.id}
                        role="img"
                        aria-label={`${name}: ${m ? pct(v) : "no episodes"}`}
                        title={`${name} · ${m ? `${pct(v)} of ${m.n}` : "no episodes"}`}
                        className="flex-1 rounded-t"
                        style={{ height: m ? `${Math.max(v * 100, 1)}%` : 0, background: seriesColor(i) }}
                      />
                    );
                  })}
                </div>
                <div className="num flex justify-between gap-1 text-xs">
                  {withMetrics.map((r, i) => {
                    const m = pick(r, sc.id);
                    return (
                      <span key={r.id} className="flex-1 truncate text-center" style={{ color: i === 0 ? "var(--muted)" : "var(--fg)" }}>
                        {m ? pct(m.solve_rate) : "—"}
                      </span>
                    );
                  })}
                </div>
              </figure>
            ))}
          </div>
        </Panel>
      )}

      <Panel
        title="All metrics"
        aside={scopes.length > 1 ? <Tabs idPrefix="cmp-scope" variant="pills" label="Scope" tabs={scopes} active={s} onChange={setScope} /> : undefined}
        flush
      >
        <table className="data">
          <thead>
            <tr>
              <th>run</th>
              <th>model</th>
              {COLS.map((c) => (
                <th key={c.key} className="text-right">
                  {c.label}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {withMetrics.map((r, i) => (
              <tr key={r.id}>
                <td>
                  <span className="inline-flex items-center gap-2">
                    <Swatch i={i} />
                    <Link href={`/runs/${encodeURIComponent(r.id)}`} className="hover:underline" style={{ color: "var(--accent)" }}>
                      {r.label || <span className="mono">{r.id.slice(0, 28)}</span>}
                    </Link>
                  </span>
                </td>
                <td className="mono text-xs">{r.model}</td>
                {COLS.map((c) => {
                  const m = ms[i];
                  if (!m)
                    return (
                      <td key={c.key} className="text-right">
                        —
                      </td>
                    );
                  // highlight the best value in each column (only where "best" is meaningful)
                  const scores = c.best ? ms.map((x) => (x ? c.best!(x) : -Infinity)) : [];
                  const isBest = c.best && ms.length > 1 && c.best(m) === Math.max(...scores);
                  return (
                    <td
                      key={c.key}
                      className={`num text-right ${c.key === "cost" ? "mono" : ""}`}
                      style={isBest ? { fontWeight: 600, color: "var(--fg)" } : { color: "var(--muted)" }}
                    >
                      {c.get(m)}
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </Panel>
      {gap >= 0.1 && (
        <NextStep title={`${topTeacher!.model} solves ${Math.round(gap * 100)} points more than ${topOpen!.model.slice(5)}`} actions={<Link href="/train" className="btn">Train the open model on its solutions</Link>}>
          That gap is what distillation closes: collect the stronger model&apos;s clean solves on training seeds and fine-tune the open model on them.
        </NextStep>
      )}
    </Page>
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

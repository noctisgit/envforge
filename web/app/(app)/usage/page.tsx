"use client";

import { useEffect, useState } from "react";
import { BarChart } from "@/components/BarChart";
import { Api, fmtCost, fmtTokens, type Usage, type UsageBucket } from "@/lib/api";
import { useWorkspace } from "@/lib/workspace";

const RANGES = [7, 30, 90];

export default function UsagePage() {
  const { current: ws } = useWorkspace();
  const [days, setDays] = useState(30);
  const [scope, setScope] = useState<"workspace" | "me">("workspace");
  const [usage, setUsage] = useState<Usage | null>(null);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    if (!ws) return;
    setUsage(null);
    (scope === "workspace" ? Api.usage(ws.id, days) : Api.myUsage(days)).then(setUsage).catch((e) => setErr(String(e)));
  }, [ws, days, scope]);

  if (!ws) return null;

  return (
    <div className="space-y-6">
      {/* filters: one row above the charts */}
      <div className="flex flex-wrap items-center gap-3">
        <h1 className="text-lg font-semibold">Usage</h1>
        <div className="flex gap-1 text-sm">
          <Toggle on={scope === "workspace"} onClick={() => setScope("workspace")}>
            {ws.name}
          </Toggle>
          <Toggle on={scope === "me"} onClick={() => setScope("me")}>
            Me, all workspaces
          </Toggle>
        </div>
        <div className="ml-auto flex gap-1 text-sm">
          {RANGES.map((d) => (
            <Toggle key={d} on={days === d} onClick={() => setDays(d)}>
              {d}d
            </Toggle>
          ))}
        </div>
      </div>

      {err && <div style={{ color: "var(--bad)" }}>{err}</div>}
      {!usage ? (
        <div style={{ color: "var(--muted)" }}>Loading…</div>
      ) : (
        <>
          <div className="grid grid-cols-2 gap-3 md:grid-cols-5">
            <Tile label="runs" value={usage.totals.runs.toLocaleString()} />
            <Tile label="episodes" value={usage.totals.episodes.toLocaleString()} />
            <Tile
              label="input tokens"
              value={fmtTokens(usage.totals.input_tokens)}
              note={
                usage.totals.cache_read_tokens
                  ? `${Math.round((100 * usage.totals.cache_read_tokens) / Math.max(1, usage.totals.input_tokens))}% served from prompt cache`
                  : undefined
              }
            />
            <Tile label="output tokens" value={fmtTokens(usage.totals.output_tokens)} />
            <Tile
              label="est. cost"
              value={fmtCost(usage.totals.cost_usd)}
              note={usage.totals.unpriced_runs ? `+ ${usage.totals.unpriced_runs} unpriced run(s)` : "at list price"}
            />
          </div>

          {usage.builder && usage.builder.jobs > 0 && (
            <div className="panel flex flex-wrap items-center gap-x-6 gap-y-1 p-4 text-sm">
              <span className="font-semibold">Environment builder</span>
              <span>{usage.builder.jobs} chat turn(s)</span>
              <span>
                {fmtTokens(usage.builder.input_tokens)} in · {fmtTokens(usage.builder.output_tokens)} out
              </span>
              <span>est. {fmtCost(usage.builder.cost_usd)}</span>
              <span className="text-xs" style={{ color: "var(--muted)" }}>
                (includes task extraction; not part of the run totals above)
              </span>
            </div>
          )}

          <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
            <div className="panel p-4">
              <div className="mb-2 text-sm font-semibold">Tokens per day</div>
              <BarChart
                data={usage.by_day.map((d) => ({
                  label: d.date,
                  value: d.input_tokens + d.output_tokens,
                  detail: `${fmtTokens(d.input_tokens)} in · ${fmtTokens(d.output_tokens)} out · ${d.runs} run(s)`,
                }))}
                format={fmtTokens}
              />
            </div>
            <div className="panel p-4">
              <div className="mb-2 text-sm font-semibold">Estimated cost per day</div>
              <BarChart
                data={usage.by_day.map((d) => ({ label: d.date, value: d.cost_usd, detail: `${d.episodes} episode(s)` }))}
                format={(v) => fmtCost(v)}
                color="var(--warn)"
              />
            </div>
          </div>

          <div className="panel overflow-x-auto p-2">
            <div className="px-2 pt-2 text-sm font-semibold">By model</div>
            <BucketTable rows={usage.by_model.map((r) => ({ key: r.model, label: <span className="mono">{r.model}</span>, ...r }))} />
          </div>

          <div className="panel overflow-x-auto p-2">
            <div className="px-2 pt-2 text-sm font-semibold">{scope === "workspace" ? "By member" : "By workspace"}</div>
            <BucketTable
              rows={usage.groups.map((r) => ({
                key: r.user_id ?? r.workspace_id ?? "?",
                label: scope === "workspace" ? r.email : r.name,
                ...r,
              }))}
            />
          </div>

          <p className="text-xs" style={{ color: "var(--muted)" }}>
            Cost is estimated from token counts at the Vercel AI Gateway&apos;s list prices (also used for native Anthropic/OpenAI
            ids); prompt-caching discounts are ignored. Self-hosted vLLM and the scripted model count as $0.
          </p>
        </>
      )}
    </div>
  );
}

function Toggle({ on, onClick, children }: { on: boolean; onClick: () => void; children: React.ReactNode }) {
  return (
    <button
      type="button"
      onClick={onClick}
      className="rounded-md border px-2.5 py-1"
      style={{ borderColor: on ? "var(--accent)" : "var(--line)", color: on ? "var(--accent)" : "var(--muted)" }}
    >
      {children}
    </button>
  );
}

function Tile({ label, value, note }: { label: string; value: string; note?: string }) {
  return (
    <div className="panel p-4">
      <div className="label">{label}</div>
      <div className="mt-1 text-2xl font-semibold tabular-nums">{value}</div>
      {note && (
        <div className="mt-1 text-xs" style={{ color: "var(--muted)" }}>
          {note}
        </div>
      )}
    </div>
  );
}

function BucketTable({ rows }: { rows: (UsageBucket & { key: string; label: React.ReactNode })[] }) {
  if (rows.length === 0)
    return (
      <div className="p-3 text-sm" style={{ color: "var(--muted)" }}>
        No runs in this period.
      </div>
    );
  return (
    <table className="data">
      <thead>
        <tr>
          <th></th>
          <th>runs</th>
          <th>episodes</th>
          <th>input</th>
          <th>output</th>
          <th>est. cost</th>
        </tr>
      </thead>
      <tbody>
        {rows.map((r) => (
          <tr key={r.key}>
            <td>{r.label}</td>
            <td className="tabular-nums">{r.runs}</td>
            <td className="tabular-nums">{r.episodes}</td>
            <td className="tabular-nums">{fmtTokens(r.input_tokens)}</td>
            <td className="tabular-nums">{fmtTokens(r.output_tokens)}</td>
            <td className="tabular-nums">
              {fmtCost(r.cost_usd)}
              {r.unpriced_runs > 0 && <span style={{ color: "var(--muted)" }}> +?</span>}
            </td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

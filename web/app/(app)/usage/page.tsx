"use client";

import { useEffect, useState } from "react";
import { BarChart } from "@/components/BarChart";
import { Empty, Page, PageHeader, Panel, StatCard, StatGrid } from "@/components/Page";
import { Tabs } from "@/components/Tabs";
import { useToast } from "@/components/Toast";
import { Api, fmtCost, fmtTokens, type Usage, type UsageBucket } from "@/lib/api";
import { useWorkspace } from "@/lib/workspace";

const RANGES = [7, 30, 90];

export default function UsagePage() {
  const { current: ws } = useWorkspace();
  const [days, setDays] = useState(30);
  const [scope, setScope] = useState<"workspace" | "me">("workspace");
  const [usage, setUsage] = useState<Usage | null>(null);
  const toast = useToast();

  useEffect(() => {
    if (!ws) return;
    setUsage(null);
    (scope === "workspace" ? Api.usage(ws.id, days) : Api.myUsage(days)).then(setUsage).catch((e) => toast.error("Couldn't load usage", e));
  }, [ws, days, scope, toast]);

  if (!ws) return null;

  return (
    <Page>
      {/* filters: one row above the charts */}
      <PageHeader
        title="Usage"
        meta={
          <Tabs
            idPrefix="usage-scope"
            variant="pills"
            label="Scope"
            tabs={[
              { id: "workspace", label: ws.name },
              { id: "me", label: "Me, all workspaces" },
            ]}
            active={scope}
            onChange={(id) => setScope(id as "workspace" | "me")}
          />
        }
        actions={
          <Tabs
            idPrefix="usage-range"
            variant="pills"
            label="Range"
            tabs={RANGES.map((d) => ({ id: String(d), label: `${d}d` }))}
            active={String(days)}
            onChange={(id) => setDays(Number(id))}
          />
        }
      />

      {!usage ? (
        <div style={{ color: "var(--muted)" }}>Loading…</div>
      ) : (
        <>
          <StatGrid cols={5}>
            <StatCard label="runs" value={usage.totals.runs.toLocaleString()} note={`last ${days} days`} />
            <StatCard label="episodes" value={usage.totals.episodes.toLocaleString()} />
            <StatCard
              label="input tokens"
              value={fmtTokens(usage.totals.input_tokens)}
              note={
                usage.totals.cache_read_tokens
                  ? `${Math.round((100 * usage.totals.cache_read_tokens) / Math.max(1, usage.totals.input_tokens))}% served from prompt cache`
                  : undefined
              }
            />
            <StatCard label="output tokens" value={fmtTokens(usage.totals.output_tokens)} />
            <StatCard
              label="est. cost"
              mono
              value={fmtCost(usage.totals.cost_usd)}
              note={usage.totals.unpriced_runs ? `at list price · + ${usage.totals.unpriced_runs} unpriced run(s)` : "at list price"}
            />
          </StatGrid>

          {usage.builder && usage.builder.jobs > 0 && (
            <div className="panel flex flex-wrap items-center gap-x-6 gap-y-1 p-4">
              <span className="font-medium">Environment builder</span>
              <span>
                <span className="num">{usage.builder.jobs}</span> <span style={{ color: "var(--muted)" }}>chat turn(s)</span>
              </span>
              <span className="num">
                {fmtTokens(usage.builder.input_tokens)} in · {fmtTokens(usage.builder.output_tokens)} out
              </span>
              <span className="mono">est. {fmtCost(usage.builder.cost_usd)}</span>
              <span className="hint">includes task extraction · not part of the run totals above</span>
            </div>
          )}

          <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
            <Panel title="Tokens per day" hint="input + output, hover a bar for the split">
              <BarChart
                data={usage.by_day.map((d) => ({
                  label: d.date,
                  value: d.input_tokens + d.output_tokens,
                  detail: `${fmtTokens(d.input_tokens)} in · ${fmtTokens(d.output_tokens)} out · ${d.runs} run(s)`,
                }))}
                format={fmtTokens}
              />
            </Panel>
            <Panel title="Estimated cost per day" hint="USD · vLLM and scripted count as $0">
              <BarChart
                data={usage.by_day.map((d) => ({ label: d.date, value: d.cost_usd, detail: `${d.episodes} episode(s)` }))}
                format={(v) => fmtCost(v)}
                color="var(--warn)"
              />
            </Panel>
          </div>

          <div className="grid grid-cols-1 items-start gap-6 xl:grid-cols-2">
            <Panel title="By model" flush>
              <BucketTable rows={usage.by_model.map((r) => ({ key: r.model, label: <span className="mono">{r.model}</span>, ...r }))} />
            </Panel>

            <Panel title={scope === "workspace" ? "By member" : "By workspace"} flush>
              <BucketTable
                rows={usage.groups.map((r) => ({
                  key: r.user_id ?? r.workspace_id ?? "?",
                  label: scope === "workspace" ? r.email : r.name,
                  ...r,
                }))}
              />
            </Panel>
          </div>

          <p className="hint">
            Cost is estimated from token counts at the Vercel AI Gateway&apos;s list prices (also used for native Anthropic/OpenAI ids); prompt-caching
            discounts are ignored. Self-hosted vLLM and the scripted model count as $0.
          </p>
        </>
      )}
    </Page>
  );
}

function BucketTable({ rows }: { rows: (UsageBucket & { key: string; label: React.ReactNode })[] }) {
  if (rows.length === 0) return <Empty>No runs in this period.</Empty>;
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
            <td className="num">{r.runs}</td>
            <td className="num">{r.episodes}</td>
            <td className="num">{fmtTokens(r.input_tokens)}</td>
            <td className="num">{fmtTokens(r.output_tokens)}</td>
            <td className="num mono">
              {fmtCost(r.cost_usd)}
              {r.unpriced_runs > 0 && <span style={{ color: "var(--muted)" }}> +?</span>}
            </td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

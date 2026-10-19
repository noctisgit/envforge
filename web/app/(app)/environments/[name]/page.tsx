"use client";

// One environment, entirely from its pack's describe() and diagram() hooks (via
// /api/packs/{name}): what it is, its "dimensions", its world, tools, tiers, reward
// rules, extra tables, and a live task preview with the reference solution.
//
// Laid out as tabs (Overview · World · Tools · Rules · Preview); the active tab lives
// in the URL hash (#tools) so it can be linked to and survives a reload. Nothing here
// knows about a particular pack.

import Link from "next/link";
import { useParams } from "next/navigation";
import { useEffect, useState } from "react";
import { DiagramView } from "@/components/DiagramView";
import { Page, PageHeader, Panel } from "@/components/Page";
import { Tabs, tabPanelProps, useHashTab, type TabItem } from "@/components/Tabs";
import { Api, ApiError, type Pack, type PackDetail, type TaskPreview, type ToolSpec } from "@/lib/api";
import { useWorkspace } from "@/lib/workspace";

const TAB_IDS = ["overview", "world", "tools", "rules", "preview"] as const;
const P = "env"; // id prefix for tab / panel elements

export default function EnvironmentPage() {
  const { name } = useParams<{ name: string }>();
  const { current: ws } = useWorkspace();
  const [pack, setPack] = useState<PackDetail | null>(null);
  const [listing, setListing] = useState<Pack | null>(null); // custom flag + builder project
  const [err, setErr] = useState<string | null>(null);
  const [tab, setTab] = useHashTab(TAB_IDS, "overview");
  // tabs mount the first time they're opened and then stay mounted (hidden), so the
  // preview keeps its seed/tier and the diagrams keep their selection
  const [visited, setVisited] = useState<Set<string>>(() => new Set());
  useEffect(() => setVisited((v) => (v.has(tab) ? v : new Set(v).add(tab))), [tab]);

  useEffect(() => {
    if (!ws) return;
    const packName = decodeURIComponent(name);
    Api.pack(ws.id, packName)
      .then(setPack)
      .catch((e) => setErr(e instanceof ApiError ? e.message : String(e)));
    Api.wsPacks(ws.id)
      .then((ps) => setListing(ps.find((p) => p.name === packName) ?? null))
      .catch(() => setListing(null));
  }, [name, ws]);

  if (err) return <div style={{ color: "var(--bad)" }}>{err}</div>;
  if (!pack) return <div style={{ color: "var(--muted)" }}>Loading…</div>;

  const tabs: TabItem[] = [
    { id: "overview", label: "Overview" },
    { id: "world", label: "World" },
    { id: "tools", label: "Tools", count: pack.tools.length + 1 },
    { id: "rules", label: "Rules", count: pack.tables.length ? `${pack.tables.length} table${pack.tables.length === 1 ? "" : "s"}` : undefined },
    { id: "preview", label: "Task preview" },
  ];
  const panel = (id: string) => ({ ...tabPanelProps(P, id), hidden: tab !== id, className: "outline-none" });

  return (
    <Page>
      <PageHeader
        back={{ href: "/environments", label: "Environments" }}
        title={pack.name}
        meta={
          <>
            <span className="pill pill-queued">{pack.kind === "sim" ? "simulation" : "container sandbox"}</span>
            {listing?.custom && <span className="pill pill-running">built in this workspace</span>}
          </>
        }
        actions={
          <>
            {listing?.project_id && (
              <Link href={`/builder/${listing.project_id}`} className="btn-ghost">
                Open in Builder
              </Link>
            )}
            <Link href={`/runs?pack=${encodeURIComponent(pack.name)}`} className="btn">
              Run it
            </Link>
          </>
        }
        description={<span style={{ color: "var(--fg)" }}>{pack.description}</span>}
      >
        <div className="text-xs" style={{ color: "var(--muted)" }}>
          tiers {pack.tiers.join(", ")} · up to {pack.max_steps} turns · {pack.tools.length} tools + finish
          {pack.replayable && " · replayable"}
        </div>
      </PageHeader>

      <div className="space-y-6">
        <Tabs idPrefix={P} label={`${pack.name} sections`} tabs={tabs} active={tab} onChange={setTab} />

        <div {...panel("overview")}>
          <OverviewTab pack={pack} />
        </div>
        <div {...panel("world")}>{visited.has("world") && <WorldTab pack={pack} onPreview={() => setTab("preview")} />}</div>
        <div {...panel("tools")}>{visited.has("tools") && <ToolsTab pack={pack} />}</div>
        <div {...panel("rules")}>{visited.has("rules") && <RulesTab pack={pack} />}</div>
        <div {...panel("preview")}>{visited.has("preview") && ws && <Preview pack={pack} wsId={ws.id} />}</div>
      </div>
    </Page>
  );
}

const Muted = ({ children }: { children: React.ReactNode }) => (
  <p className="text-sm" style={{ color: "var(--muted)" }}>
    {children}
  </p>
);

// ---- Overview ---------------------------------------------------------------------------

function OverviewTab({ pack }: { pack: PackDetail }) {
  const overview = pack.overview && pack.overview !== pack.description ? pack.overview : "";
  return (
    <div className="space-y-6">
      <Panel title="How an episode works">
        {overview && <p className="max-w-4xl text-sm leading-relaxed">{overview}</p>}
        <EpisodeFlow pack={pack} />
      </Panel>

      <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
        <Panel title="Dimensions" hint="LLM-agent environments have no tensors: observations and actions are text. This is what can be counted.">
          <table className="data">
            <tbody>
              {pack.dimensions.map(([k, v]) => (
                <tr key={k}>
                  <td className="whitespace-nowrap" style={{ color: "var(--muted)" }}>
                    {k}
                  </td>
                  <td>{v}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </Panel>
        <Panel title="The world">
          {pack.world.length ? (
            <ul className="list-disc space-y-1 pl-5 text-sm">
              {pack.world.map((w, i) => (
                <li key={i}>{w}</li>
              ))}
            </ul>
          ) : (
            <Muted>{pack.description}</Muted>
          )}
          {(pack.observation || pack.action) && (
            <dl className="grid grid-cols-1 gap-3 border-t pt-3 text-sm" style={{ borderColor: "var(--line)" }}>
              {pack.observation && (
                <div>
                  <dt className="label">the agent sees</dt>
                  <dd>{pack.observation}</dd>
                </div>
              )}
              {pack.action && (
                <div>
                  <dt className="label">the agent can</dt>
                  <dd>{pack.action}</dd>
                </div>
              )}
            </dl>
          )}
        </Panel>
      </div>
    </div>
  );
}

function EpisodeFlow({ pack }: { pack: PackDetail }) {
  const steps = [
    ["Task", "generated from (seed, tier), deterministically; the agent gets a goal, the verifier keeps hidden facts"],
    ["Agent", "reads the system prompt, the goal and the first observation"],
    ["Loop", `up to ${pack.max_steps} turns: the model calls tools, the environment returns text (errors included)`],
    ["Verifier", "checks the end state in code — no model, no human"],
    ["Reward", pack.reward_rules.length ? pack.reward_rules.map((r) => r.reward.toFixed(1)).join(" / ") : "a scalar"],
  ];
  return (
    <ol className="flex flex-col gap-2 md:flex-row md:items-stretch md:gap-0">
      {steps.map(([t, d], i) => (
        <li key={t} className="flex min-w-0 flex-1 items-stretch">
          <div className="min-w-0 flex-1 rounded-md border px-3 py-2.5 text-sm" style={{ borderColor: "var(--line)", background: "#0f1317" }}>
            <div className="font-medium">
              <span className="mono text-xs" style={{ color: "var(--accent)" }}>
                {i + 1}
              </span>{" "}
              {t}
            </div>
            <div className="text-xs leading-snug" style={{ color: "var(--muted)" }}>
              {d}
            </div>
          </div>
          {i < steps.length - 1 && (
            <span aria-hidden className="hidden w-5 shrink-0 items-center justify-center text-xs md:flex" style={{ color: "var(--muted)" }}>
              →
            </span>
          )}
        </li>
      ))}
    </ol>
  );
}

// ---- World ------------------------------------------------------------------------------

function WorldTab({ pack, onPreview }: { pack: PackDetail; onPreview: () => void }) {
  if (!pack.world_diagram)
    return (
      <Panel title="World">
        <Muted>
          This pack doesn&apos;t draw a diagram yet. Packs can add one by implementing <span className="mono">diagram()</span> (see docs/PACKS.md).
        </Muted>
      </Panel>
    );
  return (
    <Panel
      title="World"
      hint="A sample task: seed 0, lowest tier. Click a node for its details."
      aside={
        <button className="btn-ghost" onClick={onPreview}>
          Generate other tasks →
        </button>
      }
    >
      <DiagramView diagram={pack.world_diagram} height={520} />
    </Panel>
  );
}

// ---- Tools ------------------------------------------------------------------------------

function paramText(t: ToolSpec): React.ReactNode {
  const props = t.parameters.properties ?? {};
  const req = new Set(t.parameters.required ?? []);
  const names = Object.keys(props);
  if (!names.length) return <span style={{ color: "var(--muted)" }}>—</span>;
  // one parameter per line; enums get spaces around "|" so long value lists wrap
  return (
    <ul className="mono space-y-0.5 text-xs [overflow-wrap:anywhere]">
      {names.map((n) => {
        const p = props[n];
        const type = p.enum ? p.enum.map(String).join(" | ") : (p.type ?? "any");
        return (
          <li key={n}>
            {n}
            {req.has(n) ? "" : "?"}: <span style={{ color: "var(--muted)" }}>{type}</span>
          </li>
        );
      })}
    </ul>
  );
}

function matches(t: ToolSpec, q: string): boolean {
  if (!q) return true;
  const hay = [t.name, t.description, ...Object.keys(t.parameters.properties ?? {})].join(" ").toLowerCase();
  return q
    .toLowerCase()
    .split(/\s+/)
    .every((w) => hay.includes(w));
}

function ToolsTab({ pack }: { pack: PackDetail }) {
  const [cat, setCat] = useState("all");
  const [q, setQ] = useState("");
  const groups = [
    { id: "observe", title: "Observe (read-only)", tools: pack.tools.filter((t) => t.category === "observe") },
    { id: "act", title: "Act (change state)", tools: pack.tools.filter((t) => t.category === "act") },
    { id: "other", title: "Tools", tools: pack.tools.filter((t) => !t.category) },
    { id: "builtin", title: "Added by envforge to every pack", tools: [pack.finish_tool] },
  ].filter((g) => g.tools.length > 0);
  const total = pack.tools.length + 1;
  const cats: TabItem[] = [
    { id: "all", label: "All", count: total },
    ...groups.map((g) => ({
      id: g.id,
      label: { observe: "Observe", act: "Act", other: "Uncategorised", builtin: "Built-in" }[g.id] ?? g.id,
      count: g.tools.length,
    })),
  ];
  const shown = groups
    .filter((g) => cat === "all" || g.id === cat)
    .map((g) => ({ ...g, tools: g.tools.filter((t) => matches(t, q.trim())) }))
    .filter((g) => g.tools.length > 0);
  const nShown = shown.reduce((n, g) => n + g.tools.length, 0);

  return (
    <Panel title={`Tools (${pack.tools.length} + finish)`} hint="What the model may call, with the JSON-schema parameters it sees (? = optional).">
      <div className="flex flex-wrap items-center gap-3">
        <Tabs idPrefix="tools" label="Tool category" variant="pills" tabs={cats} active={cat} onChange={setCat} />
        <div className="ml-auto flex w-full items-center gap-2 sm:w-auto">
          <input
            className="input sm:!w-64"
            type="search"
            placeholder="Filter by name, parameter, description…"
            aria-label="Filter tools"
            value={q}
            onChange={(e) => setQ(e.target.value)}
          />
        </div>
      </div>
      <div {...tabPanelProps("tools", cat)} className="space-y-5 outline-none">
        {q.trim() && (
          <div className="text-xs" style={{ color: "var(--muted)" }}>
            {nShown} of {cat === "all" ? total : groups.find((g) => g.id === cat)?.tools.length} tools match
          </div>
        )}
        {shown.map((g) => (
          <ToolTable key={g.id} title={g.title} tools={g.tools} />
        ))}
        {nShown === 0 && <Muted>No tools match “{q}”.</Muted>}
      </div>
    </Panel>
  );
}

function ToolTable({ title, tools }: { title: string; tools: ToolSpec[] }) {
  return (
    <div>
      <div className="label mb-1">
        {title} · {tools.length}
      </div>
      {/* fixed layout: column widths hold no matter how long an enum or description is */}
      <table className="data sm:table-fixed">
        <thead>
          <tr>
            <th className="sm:w-[22%]">tool</th>
            <th className="hidden sm:table-cell sm:w-[33%]">parameters</th>
            <th className="hidden sm:table-cell">description</th>
          </tr>
        </thead>
        <tbody>
          {tools.map((t) => (
            <tr key={t.name}>
              <td className="[overflow-wrap:anywhere]">
                <div className="mono text-xs font-medium">{t.name}</div>
                {/* narrow screens: one stacked cell instead of three cramped columns */}
                <div className="mt-1 space-y-1 sm:hidden">
                  {Object.keys(t.parameters.properties ?? {}).length > 0 && <div>{paramText(t)}</div>}
                  <div style={{ color: "var(--muted)" }}>{t.description}</div>
                </div>
              </td>
              <td className="hidden sm:table-cell">{paramText(t)}</td>
              <td className="hidden leading-snug sm:table-cell" style={{ color: "var(--muted)" }}>
                {t.description}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

// ---- Rules ------------------------------------------------------------------------------

function RulesTab({ pack }: { pack: PackDetail }) {
  const [sel, setSel] = useState(0);
  const tables = pack.tables;
  const t = tables[Math.min(sel, tables.length - 1)];
  return (
    <div className="space-y-6">
      <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
        <Panel title="Difficulty tiers">
          <table className="data">
            <tbody>
              {pack.tiers.map((t) => (
                <tr key={t}>
                  <td className="whitespace-nowrap font-medium">tier {t}</td>
                  <td>{pack.tier_notes[String(t)] ?? "—"}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </Panel>
        <Panel title="Reward" hint="Computed by code at the end of the episode.">
          {pack.reward_rules.length ? (
            <table className="data">
              <tbody>
                {pack.reward_rules.map((r, i) => (
                  <tr key={i}>
                    <td
                      className="mono w-12 whitespace-nowrap font-semibold"
                      style={{ color: r.reward >= 1 ? "var(--good)" : r.reward > 0 ? "var(--warn)" : "var(--bad)" }}
                    >
                      {r.reward.toFixed(1)}
                    </td>
                    <td>{r.outcome}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          ) : (
            <Muted>Defined by the pack&apos;s verify().</Muted>
          )}
        </Panel>
      </div>

      {t && (
        <Panel title="Reference tables" hint="What the task generator draws from and the verifier checks against.">
          {tables.length > 1 && (
            <Tabs
              idPrefix="tables"
              label="Reference tables"
              variant="pills"
              tabs={tables.map((x, i) => ({ id: String(i), label: x.title, count: x.rows.length }))}
              active={String(Math.min(sel, tables.length - 1))}
              onChange={(id) => setSel(Number(id))}
            />
          )}
          <div {...(tables.length > 1 ? tabPanelProps("tables", String(Math.min(sel, tables.length - 1))) : {})} className="space-y-2 outline-none">
            <div>
              <h3 className="text-sm font-medium">{t.title}</h3>
              {t.note && (
                <p className="max-w-4xl text-xs" style={{ color: "var(--muted)" }}>
                  {t.note}
                </p>
              )}
            </div>
            <div className="overflow-x-auto">
              <table className="data">
                <thead>
                  <tr>
                    {t.columns.map((c) => (
                      <th key={c}>{c}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {t.rows.map((r, i) => (
                    <tr key={i}>
                      {r.map((c, j) => (
                        <td key={j} className={j === 0 ? "font-medium" : ""}>
                          {c}
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </Panel>
      )}
    </div>
  );
}

// ---- Task preview -----------------------------------------------------------------------

function Preview({ pack, wsId }: { pack: PackDetail; wsId: string }) {
  const [seed, setSeed] = useState(0);
  const [tier, setTier] = useState(pack.tiers[pack.tiers.length - 1]);
  const [p, setP] = useState<TaskPreview | null>(null);
  const [view, setView] = useState<"before" | "after">("before");
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  async function load(s = seed, t = tier) {
    setBusy(true);
    setErr(null);
    try {
      setP(await Api.preview(wsId, pack.name, s, t));
      setView("before");
    } catch (e) {
      setErr(e instanceof ApiError ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  }
  useEffect(() => {
    load();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [pack.name]);

  const diagram = view === "after" && p?.diagram_after ? p.diagram_after : p?.diagram_before;

  return (
    <div className="space-y-6">
      <Panel
        title="Task preview"
        hint="Generate any task the way a run would. No model is called and nothing is billed. Hidden facts are what the verifier knows and the agent doesn't."
      >
        <div className="flex flex-wrap items-end gap-3">
          <div>
            <label className="label" htmlFor="pv-seed">
              Seed
            </label>
            <input
              id="pv-seed"
              className="input !w-28"
              type="number"
              min={0}
              value={seed}
              onChange={(e) => setSeed(Math.max(0, Number(e.target.value)))}
              onKeyDown={(e) => e.key === "Enter" && load()}
            />
          </div>
          <div>
            <label className="label" htmlFor="pv-tier">
              Tier
            </label>
            <select id="pv-tier" className="input !w-24" value={tier} onChange={(e) => setTier(Number(e.target.value))}>
              {pack.tiers.map((t) => (
                <option key={t} value={t}>
                  {t}
                </option>
              ))}
            </select>
          </div>
          <button className="btn h-[38px]" onClick={() => load()} disabled={busy}>
            {busy ? "Generating…" : "Generate"}
          </button>
          <button
            className="btn-ghost h-[38px]"
            onClick={() => {
              const s = Math.floor(Math.random() * 100000);
              setSeed(s);
              load(s);
            }}
            disabled={busy}
          >
            Random seed
          </button>
        </div>
        {err && <div style={{ color: "var(--bad)" }}>{err}</div>}
        {p && (
          <div className="grid grid-cols-1 gap-4 border-t pt-4 lg:grid-cols-2" style={{ borderColor: "var(--line)" }}>
            <div className="space-y-1">
              <div className="label">goal (what the agent is told) · {p.task.id}</div>
              <div className="text-sm">{p.task.goal}</div>
            </div>
            <div className="space-y-1">
              <div className="label">hidden (what only the verifier knows)</div>
              <HiddenView hidden={p.task.hidden} />
            </div>
          </div>
        )}
      </Panel>

      {p && diagram && (
        <Panel
          title={p.diagram_after ? "Before and after" : "Starting state"}
          aside={
            p.diagram_after && (
              <div className="flex gap-1 text-sm" role="group" aria-label="Diagram state">
                {(["before", "after"] as const).map((v) => (
                  <button
                    key={v}
                    onClick={() => setView(v)}
                    aria-pressed={view === v}
                    className="rounded-md border px-2.5 py-1"
                    style={{ borderColor: view === v ? "var(--accent)" : "var(--line)", color: view === v ? "var(--accent)" : "var(--muted)" }}
                  >
                    {v === "before" ? "Broken (start)" : "After the reference fix"}
                  </button>
                ))}
              </div>
            )
          }
        >
          <DiagramView diagram={diagram} />
        </Panel>
      )}

      {p && (
        <div className="grid grid-cols-1 gap-6 xl:grid-cols-[minmax(0,3fr)_minmax(0,2fr)]">
          {p.solution.length > 0 ? (
            <Panel
              title={`Reference solution (${p.solution.length} call${p.solution.length === 1 ? "" : "s"})`}
              hint={p.verdict ? `Scored by the verifier: reward ${p.verdict.reward.toFixed(1)}${p.verdict.success ? " (solved)" : ""}.` : undefined}
            >
              <ol className="space-y-2">
                {p.solution.map((c, i) => (
                  <li key={i} className="flex gap-2 text-sm">
                    <span className="mono w-5 shrink-0 text-right text-xs leading-5" style={{ color: "var(--muted)" }}>
                      {i + 1}
                    </span>
                    <div className="min-w-0">
                      <span className="mono break-all" style={{ color: "var(--accent)" }}>
                        {c.name}(
                        {Object.entries(c.args)
                          .map(([k, v]) => `${k}=${JSON.stringify(v)}`)
                          .join(", ")}
                        )
                      </span>
                      {c.observation && (
                        <pre className="mono whitespace-pre-wrap text-xs" style={{ color: "var(--muted)" }}>
                          {c.observation}
                        </pre>
                      )}
                    </div>
                  </li>
                ))}
              </ol>
            </Panel>
          ) : (
            <Panel title="Reference solution">
              <Muted>This pack&apos;s reference solver made no calls for this task.</Muted>
            </Panel>
          )}
          <Panel title="What the model receives" hint="The system prompt and first observation, exactly as sent.">
            <details>
              <summary className="label cursor-pointer">system prompt</summary>
              <pre className="mono mt-2 max-h-[28rem] overflow-y-auto whitespace-pre-wrap text-xs" style={{ color: "var(--muted)" }}>
                {p.system_prompt}
              </pre>
            </details>
            {p.first_observation && (
              <details open>
                <summary className="label cursor-pointer">first observation</summary>
                <pre className="mono mt-2 max-h-[28rem] overflow-y-auto whitespace-pre-wrap text-xs" style={{ color: "var(--muted)" }}>
                  {`TASK\n${p.task.goal}\n\nINITIAL STATE\n${p.first_observation}`}
                </pre>
              </details>
            )}
          </Panel>
        </div>
      )}
    </div>
  );
}

function HiddenView({ hidden }: { hidden: Record<string, unknown> }) {
  const entries = Object.entries(hidden);
  if (!entries.length)
    return (
      <div className="text-sm" style={{ color: "var(--muted)" }}>
        —
      </div>
    );
  return (
    <dl className="space-y-1 text-sm">
      {entries.map(([k, v]) => (
        <div key={k} className="flex gap-2">
          <dt className="mono shrink-0 text-xs" style={{ color: "var(--muted)" }}>
            {k}
          </dt>
          <dd className="mono min-w-0 text-xs">
            {Array.isArray(v) && v.length === 0 ? (
              <span style={{ color: "var(--muted)" }}>none</span>
            ) : Array.isArray(v) && v.every((x) => typeof x === "string") ? (
              <ul className="list-disc pl-4">
                {v.map((x, i) => (
                  <li key={i}>{x}</li>
                ))}
              </ul>
            ) : (
              <span className="break-all">{JSON.stringify(v)}</span>
            )}
          </dd>
        </div>
      ))}
    </dl>
  );
}

"use client";

import Link from "next/link";
import { useRouter } from "next/navigation";
import { useCallback, useEffect, useState } from "react";
import { useConfirm } from "@/components/ConfirmDialog";
import { ExportButton } from "@/components/ExportButton";
import { ModelPicker } from "@/components/ModelPicker";
import { Empty, Page, PageHeader, Panel, rateTone, StatCard, StatGrid, toneColor } from "@/components/Page";
import { Tabs } from "@/components/Tabs";
import { errorText, useToast } from "@/components/Toast";
import { Api, fmtCost, fmtTokens, type Pack, type Run } from "@/lib/api";
import { EVAL_SEED, TRAIN_SEED } from "@/lib/journey";
import { useWorkspace } from "@/lib/workspace";

const pct = (x: number) => `${Math.round(x * 100)}%`;
const WEEK_MS = 7 * 24 * 3600 * 1000;
const isActive = (r: Run) => r.status === "queued" || r.status === "running";

type Filter = "all" | "active" | "completed" | "failed" | "cancelled";
const matches = (r: Run, f: Filter) => f === "all" || (f === "active" ? isActive(r) : r.status === f);

/** Headline numbers for the last 7 days, from the runs list alone. */
function weekStats(runs: Run[]) {
  const since = Date.now() - WEEK_MS;
  const recent = runs.filter((r) => new Date(r.created_at).getTime() >= since);
  const done = runs.filter((r) => r.status === "completed" && new Date(r.finished_at ?? r.created_at).getTime() >= since);
  const scored = done.filter((r) => r.metrics && r.metrics.overall.n > 0);
  const n = scored.reduce((a, r) => a + r.metrics!.overall.n, 0);
  const solved = scored.reduce((a, r) => a + r.metrics!.overall.solve_rate * r.metrics!.overall.n, 0);
  return {
    completed: done.length,
    episodes: done.reduce((a, r) => a + r.progress, 0),
    solveRate: n ? solved / n : null,
    solveEpisodes: n,
    cost: recent.reduce((a, r) => a + (r.cost_usd ?? 0), 0),
    unpriced: recent.filter((r) => r.cost_usd === null).length,
  };
}

export default function RunsPage() {
  const { current: ws } = useWorkspace();
  const router = useRouter();
  const [runs, setRuns] = useState<Run[]>([]);
  const [packs, setPacks] = useState<Pack[]>([]);
  const [model, setModel] = useState("");
  const [pack, setPack] = useState("");
  const [episodes, setEpisodes] = useState(10);
  const [tiers, setTiers] = useState<number[]>([]);
  const [label, setLabel] = useState("");
  const [seedStart, setSeedStart] = useState(0);
  // sampling: several varied attempts per task (on-policy preference pairs, GRPO-style data)
  const [showSampling, setShowSampling] = useState(false);
  const [repeats, setRepeats] = useState(1);
  const [temperature, setTemperature] = useState(0);
  const [busy, setBusy] = useState(false);
  const [picked, setPicked] = useState<string[]>([]);
  const [filter, setFilter] = useState<Filter>("all");
  const [showForm, setShowForm] = useState<boolean | null>(null); // null until the first load decides
  const [slots, setSlots] = useState<number | null>(null);
  const confirm = useConfirm();
  const toast = useToast();

  useEffect(() => {
    Api.info()
      .then((i) => setSlots(i.max_active_runs))
      .catch(() => {});
  }, []);

  useEffect(() => {
    if (ws)
      Api.wsPacks(ws.id)
        .then(setPacks)
        .catch(() => {});
  }, [ws]);
  useEffect(() => {
    if (!ws) return;
    // "Run it" on an environment page and the steps on Home link here with the form
    // pre-filled: ?pack= &model= &seed= &episodes= &label=
    const q = new URLSearchParams(window.location.search);
    const num = (k: string) => (q.get(k) !== null && Number.isFinite(Number(q.get(k))) ? Math.max(0, Math.floor(Number(q.get(k)))) : null);
    setModel(q.get("model") || ws.default_model);
    setPack(q.get("pack") || ws.default_pack);
    setEpisodes(num("episodes") || ws.default_episodes);
    setSeedStart(num("seed") ?? 0);
    setLabel(q.get("label") || "");
    if ([...q.keys()].length) {
      setShowForm(true);
      setTimeout(() => document.getElementById("new-run")?.scrollIntoView({ block: "start" }), 50);
    }
    setPicked([]);
  }, [ws]);

  const load = useCallback(async () => {
    if (!ws) return;
    const rs = await Api.runs(ws.id);
    setRuns(rs);
    setShowForm((s) => (s === null ? rs.length === 0 : s)); // open the form for a first run
  }, [ws]);

  useEffect(() => {
    load().catch(() => {});
  }, [load]);

  // poll while anything is still in flight
  const active = runs.some(isActive);
  useEffect(() => {
    if (!active) return;
    const t = setInterval(() => load().catch(() => {}), 3000);
    return () => clearInterval(t);
  }, [active, load]);

  if (!ws) return null;
  const currentPack = packs.find((p) => p.name === pack);
  const packTiers = currentPack?.tiers ?? [];

  async function submit(e: React.FormEvent) {
    e.preventDefault();
    setBusy(true);
    try {
      const run = await Api.createRun(ws!.id, { model, pack, episodes, tiers, label, seed_start: seedStart, repeats, temperature });
      setLabel("");
      toast({
        tone: "success",
        title: "Run queued",
        message: `${episodes} episode${episodes === 1 ? "" : "s"} of ${pack} with ${model}.`,
        action: { label: "Open", onClick: () => router.push(`/runs/${encodeURIComponent(run.id)}`) },
      });
      await load();
    } catch (ex) {
      toast.error("Couldn't queue the run", ex);
    } finally {
      setBusy(false);
    }
  }

  const toggle = (id: string) => setPicked((p) => (p.includes(id) ? p.filter((x) => x !== id) : [...p, id]));

  async function remove(ids: string[]) {
    const chosen = runs.filter((r) => ids.includes(r.id));
    const active = chosen.filter(isActive);
    const deletable = chosen.filter((r) => !active.includes(r));
    if (!deletable.length) {
      toast.warning("Nothing deleted", "Queued or running runs can't be deleted — cancel them first.");
      return;
    }
    const one = deletable.length === 1;
    const ok = await confirm({
      title: one ? "Delete this run?" : `Delete ${deletable.length} runs?`,
      body: (
        <div className="space-y-2">
          <p>
            {one ? <span className="mono">{deletable[0].label || deletable[0].id}</span> : `${deletable.length} runs`} and {one ? "its" : "their"}{" "}
            {deletable.reduce((n, r) => n + r.progress, 0)} recorded episodes will be permanently deleted. Transcripts and training-data exports from{" "}
            {one ? "it" : "them"} will no longer be available.
          </p>
          <p>Your usage history keeps their token and cost totals.</p>
          {active.length > 0 && <p style={{ color: "var(--warn)" }}>{active.length} queued/running run(s) will be skipped.</p>}
        </div>
      ),
      confirmLabel: one ? "Delete run" : `Delete ${deletable.length} runs`,
      tone: "danger",
    });
    if (!ok) return;
    const failed: string[] = [];
    for (const r of deletable) {
      try {
        await Api.deleteRun(ws!.id, r.id);
      } catch (e) {
        failed.push(`${r.id}: ${errorText(e)}`);
      }
    }
    setPicked((p) => p.filter((id) => !deletable.some((r) => r.id === id)));
    const deleted = deletable.length - failed.length;
    if (deleted) toast.success(deleted === 1 ? "Run deleted" : `${deleted} runs deleted`);
    if (failed.length) toast.error("Some runs were not deleted", failed.join("; "));
    await load();
  }

  const stats = weekStats(runs);
  const activeRuns = runs.filter(isActive);
  const running = runs.find((r) => r.status === "running");
  const count = (f: Filter) => runs.filter((r) => matches(r, f)).length;
  const filters: { id: Filter; label: string; count: number }[] = (["all", "active", "completed", "failed", "cancelled"] as Filter[])
    .map((id) => ({ id, label: id, count: count(id) }))
    .filter((f) => f.id !== "cancelled" || f.count > 0 || filter === "cancelled");
  const shown = runs.filter((r) => matches(r, filter));

  return (
    <Page>
      <PageHeader
        title="Runs"
        description={
          <>
            Evaluate a model on an environment, then open, compare, export or delete the results.
            {slots !== null && ` Up to ${slots} can be queued or running at once.`}
          </>
        }
        actions={
          <button className={showForm ? "btn-ghost" : "btn"} onClick={() => setShowForm(!showForm)} aria-expanded={!!showForm} aria-controls="new-run">
            {showForm ? "Hide form" : "New run"}
          </button>
        }
      />

      <StatGrid>
        <StatCard
          label="Active"
          value={activeRuns.length}
          unit={slots !== null ? `of ${slots} slots` : undefined}
          note={running ? `${running.label || running.id} · episode ${running.progress} / ${running.episodes}` : activeRuns.length ? "waiting for a worker" : "nothing running"}
        />
        <StatCard label="Completed · 7d" value={stats.completed} note={`${stats.episodes.toLocaleString()} episodes`} />
        <StatCard
          label="Solve rate · 7d"
          value={stats.solveRate === null ? "—" : `${(stats.solveRate * 100).toFixed(1)}%`}
          tone={rateTone(stats.solveRate)}
          note={stats.solveEpisodes ? `across ${stats.solveEpisodes.toLocaleString()} scored episodes` : "no finished runs this week"}
        />
        <StatCard label="Est. cost · 7d" value={fmtCost(stats.cost)} mono note={stats.unpriced ? `at list price · ${stats.unpriced} unpriced` : "at list price"} />
      </StatGrid>

      {showForm && (
      <div id="new-run">
      <Panel title="New run" hint="Defaults come from workspace settings. Nothing is billed until the worker starts.">
        <form onSubmit={submit} className="grid grid-cols-1 gap-4 md:grid-cols-6 2xl:grid-cols-12">
          <div className="md:col-span-3 2xl:col-span-4">
            <label className="label">Model</label>
            <ModelPicker wsId={ws.id} value={model} onChange={setModel} />
          </div>
          <div className="md:col-span-2 2xl:col-span-2">
            <label className="label">Environment</label>
            <select
              className="input"
              value={pack}
              onChange={(e) => {
                setPack(e.target.value);
                setTiers([]);
              }}
            >
              {packs.map((p) => (
                <option key={p.name} value={p.name}>
                  {p.name} {p.kind === "sandbox" ? "(sandbox)" : ""}
                </option>
              ))}
            </select>
            {currentPack && (
              <div className="hint mt-1 line-clamp-2" title={currentPack.description}>
                {currentPack.description}
              </div>
            )}
          </div>
          <div>
            <label className="label">Episodes</label>
            <input className="input" type="number" min={1} max={1000} value={episodes} onChange={(e) => setEpisodes(Number(e.target.value))} />
          </div>
          <div className="md:col-span-2 2xl:col-span-2">
            <label className="label">Tiers (none = all)</label>
            <div className="flex gap-3 pt-2 text-sm">
              {packTiers.map((t) => (
                <label key={t} className="flex items-center gap-1">
                  <input
                    type="checkbox"
                    checked={tiers.includes(t)}
                    onChange={(e) => setTiers(e.target.checked ? [...tiers, t].sort() : tiers.filter((x) => x !== t))}
                  />
                  {t}
                </label>
              ))}
            </div>
          </div>
          <div className="md:col-span-2 2xl:col-span-2">
            <label className="label" htmlFor="run-purpose">
              Purpose
            </label>
            <select
              id="run-purpose"
              className="input"
              value={seedStart === EVAL_SEED ? "eval" : seedStart === TRAIN_SEED ? "train" : "custom"}
              onChange={(e) => setSeedStart(e.target.value === "eval" ? EVAL_SEED : e.target.value === "train" ? TRAIN_SEED : seedStart >= 100 && seedStart < TRAIN_SEED ? seedStart : 100)}
            >
              <option value="eval">Evaluation · seeds 0+</option>
              <option value="train">Training data · seeds 10000+</option>
              <option value="custom">Custom seed range</option>
            </select>
            {seedStart !== EVAL_SEED && seedStart !== TRAIN_SEED ? (
              <input
                aria-label="Seed start"
                className="input mt-2"
                type="number"
                min={0}
                value={seedStart}
                onChange={(e) => setSeedStart(Math.max(0, Number(e.target.value)))}
              />
            ) : (
              <div className="hint mt-1">
                {seedStart === TRAIN_SEED
                  ? "tasks a model can learn from; never used for scoring"
                  : "the fixed tasks every model is scored on"}
              </div>
            )}
          </div>
          <div className="md:col-span-2 2xl:col-span-1">
            <label className="label">Label</label>
            <input className="input" value={label} onChange={(e) => setLabel(e.target.value)} placeholder="optional" maxLength={200} />
          </div>
          <div>
            {/* an empty label line keeps the button level with the inputs */}
            <span className="label invisible hidden md:block" aria-hidden>
              queue
            </span>
            <button className="btn btn-field w-full" disabled={busy || !model.trim()}>
              Queue run
            </button>
          </div>
          <div className="md:col-span-6 2xl:col-span-12">
            <button type="button" className="hint underline" onClick={() => setShowSampling((x) => !x)} aria-expanded={showSampling}>
              {showSampling ? "Hide" : "Show"} sampling{repeats > 1 || temperature > 0 ? ` (${repeats} attempts per task, temperature ${temperature})` : ""}
            </button>
            {showSampling && (
              <div className="mt-3 grid grid-cols-2 items-start gap-4 md:grid-cols-6 2xl:grid-cols-12">
                <div className="md:col-span-2">
                  <label className="label" htmlFor="run-repeats">
                    Attempts per task
                  </label>
                  <input id="run-repeats" className="input" type="number" min={1} max={32} value={repeats} onChange={(e) => setRepeats(Math.max(1, Math.min(32, Number(e.target.value))))} />
                  <p className="hint mt-1">
                    {repeats > 1 ? `${episodes} episodes = ${Math.ceil(episodes / repeats)} tasks × ${repeats} attempts` : "each episode is a different task"}
                  </p>
                </div>
                <div className="md:col-span-2">
                  <label className="label" htmlFor="run-temp">
                    Temperature
                  </label>
                  <input id="run-temp" className="input" type="number" min={0} max={2} step={0.1} value={temperature} onChange={(e) => setTemperature(Math.max(0, Math.min(2, Number(e.target.value))))} />
                  <p className="hint mt-1">0 = the model&apos;s single best guess; 0.7–1 makes repeated attempts differ</p>
                </div>
                <p className="hint md:col-span-2 2xl:col-span-8 self-center">
                  Several varied attempts at the same training task give pairs of your model&apos;s own success and failure: the best data for DPO. Keep evaluation runs at 1
                  attempt and temperature 0.
                </p>
              </div>
            )}
          </div>
        </form>
      </Panel>
      </div>
      )}

      <Panel
        title="History"
        hint={picked.length ? undefined : "Select runs to compare, export or delete them."}
        aside={<Tabs idPrefix="runs-filter" variant="pills" label="Filter runs" tabs={filters} active={filter} onChange={(id) => setFilter(id as Filter)} />}
        footer={
          <>
            <span>
              Showing {shown.length} of {runs.length} run{runs.length === 1 ? "" : "s"}
              {active && " · refreshes every 3 s while a run is active"}
            </span>
            <span>Deleting keeps token and cost totals in Usage.</span>
          </>
        }
        flush
      >
        {picked.length > 0 && (
          <div className="flex flex-wrap items-center gap-2 border-b px-4 py-2.5" style={{ borderColor: "var(--line)", background: "var(--accent-tint)" }}>
            <span className="mr-1 font-medium">{picked.length} selected</span>
            <button className="btn" disabled={picked.length < 2} title={picked.length < 2 ? "Pick at least two runs" : undefined} onClick={() => router.push(`/compare?runs=${picked.map(encodeURIComponent).join(",")}`)}>
              Compare
            </button>
            <ExportButton wsId={ws.id} runIds={picked} />
            <button className="btn-danger" onClick={() => remove(picked)}>
              Delete
            </button>
            <button className="btn-ghost ml-auto" onClick={() => setPicked([])}>
              Clear
            </button>
          </div>
        )}
        {runs.length === 0 ? (
          <Empty>
            No runs yet. A run plays a model through an environment&apos;s tasks and scores every attempt. Start with model{" "}
            <span className="mono">scripted</span>: it needs no key and finishes in seconds. Not sure where to begin? See{" "}
            <Link href="/home" className="underline">
              Home
            </Link>
            .
          </Empty>
        ) : shown.length === 0 ? (
          <Empty>No {filter} runs.</Empty>
        ) : (
          <table className="data">
            <thead>
              <tr>
                <th className="w-8">
                  <input
                    type="checkbox"
                    aria-label="select all shown runs"
                    checked={shown.every((r) => picked.includes(r.id))}
                    onChange={(e) =>
                      setPicked((p) => (e.target.checked ? Array.from(new Set([...p, ...shown.map((r) => r.id)])) : p.filter((id) => !shown.some((r) => r.id === id))))
                    }
                  />
                </th>
                <th>run</th>
                <th>model</th>
                <th>env</th>
                <th>status</th>
                <th className="text-right">progress</th>
                <th className="text-right">solve</th>
                <th className="text-right">reward</th>
                <th className="text-right">tokens</th>
                <th className="text-right">cost</th>
                <th>created</th>
                <th></th>
              </tr>
            </thead>
            <tbody>
              {shown.map((r) => {
                const tone = rateTone(r.metrics?.overall.solve_rate);
                return (
                  <tr key={r.id}>
                    <td>
                      <input type="checkbox" checked={picked.includes(r.id)} onChange={() => toggle(r.id)} aria-label={`select ${r.label || r.id}`} />
                    </td>
                    <td className="min-w-40 max-w-[22rem]">
                      <Link href={`/runs/${encodeURIComponent(r.id)}`} className="[overflow-wrap:anywhere] hover:underline" style={{ color: "var(--accent)" }}>
                        {r.label || <span className="mono">{r.id}</span>}
                      </Link>
                      {r.label && (
                        <span className="mono ml-2 text-xs" style={{ color: "var(--muted)" }}>
                          {r.id.slice(-6)}
                        </span>
                      )}
                    </td>
                    <td className="mono text-xs [overflow-wrap:anywhere]">{r.model}</td>
                    <td>{r.pack}</td>
                    <td>
                      <span className={`pill pill-${r.status}`}>{r.status}</span>
                    </td>
                    <td className="num text-right">
                      {r.progress}/{r.episodes}
                    </td>
                    <td className="num text-right" style={{ color: toneColor(tone) }}>
                      {r.metrics ? pct(r.metrics.overall.solve_rate) : "—"}
                    </td>
                    <td className="num text-right">{r.metrics ? r.metrics.overall.mean_reward.toFixed(2) : "—"}</td>
                    <td className="num text-right">{fmtTokens(r.input_tokens + r.output_tokens)}</td>
                    <td className="num mono text-right">{fmtCost(r.cost_usd)}</td>
                    <td className="whitespace-nowrap" style={{ color: "var(--muted)" }}>
                      {new Date(r.created_at).toLocaleString(undefined, { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" })}
                    </td>
                    <td>
                      {!isActive(r) && (
                        <button className="text-xs hover:underline" style={{ color: "var(--muted)" }} onClick={() => remove([r.id])} aria-label={`delete ${r.label || r.id}`}>
                          Delete
                        </button>
                      )}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        )}
      </Panel>
    </Page>
  );
}

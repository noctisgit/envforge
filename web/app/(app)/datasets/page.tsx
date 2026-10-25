"use client";

// Datasets: rejection sampling in the browser. Pick finished runs on training seeds,
// set the filters, watch the live preview (counts, split, overlap with evaluation runs),
// save a named dataset. Two kinds: SFT examples (solved episodes to imitate) and
// preference pairs for DPO (a solved and a failed episode of the same task). Saved
// datasets feed the Fine-tune page and can be downloaded.

import Link from "next/link";
import { useSearchParams } from "next/navigation";
import { Suspense, useCallback, useEffect, useMemo, useState } from "react";
import { useConfirm } from "@/components/ConfirmDialog";
import { newRunHref } from "@/components/Journey";
import { Empty, Page, PageHeader, Panel, StatCard, StatGrid } from "@/components/Page";
import { useToast } from "@/components/Toast";
import { Compute, downloadFile, type DatasetKind, type DatasetPreview, type DatasetRow, type DatasetSource } from "@/lib/api";
import { TRAIN_SEED } from "@/lib/journey";
import { finalName, typingName } from "@/lib/names";
import { useWorkspace } from "@/lib/workspace";

const pct = (x: number | null) => (x === null ? "–" : `${Math.round(x * 100)}%`);

function Preview({ p, loading }: { p: DatasetPreview | null; loading: boolean }) {
  if (!p) return <p className="hint">{loading ? "Building a preview…" : "Pick at least one run to see what the dataset would contain."}</p>;
  const pairs = p.kind === "preference";
  return (
    <div className="space-y-3" aria-live="polite" style={{ opacity: loading ? 0.6 : 1 }}>
      <StatGrid cols={4}>
        <StatCard
          label={pairs ? "Pairs" : "Examples"}
          value={p.kept}
          note={pairs ? `${p.chosen_episodes ?? 0} solved · ${p.rejected_episodes ?? 0} failed episodes` : p.episodes_seen !== undefined ? `of ${p.episodes_seen} episodes` : undefined}
          tone={p.kept ? undefined : "bad"}
        />
        <StatCard label="Train / val" value={`${p.n_train} / ${p.n_val}`} note="split by task" />
        <StatCard
          label={pairs ? "Tasks with a pair" : "Tasks"}
          value={p.tasks}
          note={
            pairs
              ? `${p.tasks_without_rejected ?? 0} never failed · ${p.tasks_without_chosen ?? 0} never solved`
              : p.examples_per_task
                ? `${p.examples_per_task.min}–${p.examples_per_task.max} examples each`
                : undefined
          }
        />
        <StatCard label="Eval overlap" value={p.overlap.length ? `${p.overlap.reduce((n, o) => n + o.tasks, 0)} tasks` : "none"} tone={p.overlap.length ? "bad" : "good"} note={p.overlap.length ? `in ${p.overlap.length} eval run(s)` : "evaluation seeds stay clean"} />
      </StatGrid>
      {p.by_pack_tier && Object.keys(p.by_pack_tier).length > 0 && (
        <div className="hint">
          {Object.entries(p.by_pack_tier)
            .map(([k, n]) => `${k}: ${n}`)
            .join(" · ")}
          {p.models && ` · from ${Object.entries(p.models).map(([m, n]) => `${m} ×${n}`).join(", ")}`}
        </div>
      )}
      {p.overlap.map((o) => (
        <p key={o.run_id} className="text-sm" style={{ color: "var(--bad)" }}>
          <Link href={`/runs/${encodeURIComponent(o.run_id)}`} className="underline">
            {o.label || o.model}
          </Link>{" "}
          shares {o.tasks} task{o.tasks === 1 ? "" : "s"} with this dataset ({o.sample.join(", ")}): its score would measure memory, not skill.
        </p>
      ))}
      {p.warnings
        .filter((w) => !w.includes("share tasks"))
        .map((w) => (
          <p key={w} className="hint" style={{ color: p.kept ? "var(--warn)" : "var(--bad)" }}>
            {w}
          </p>
        ))}
    </div>
  );
}

function DatasetsInner() {
  const { current: ws } = useWorkspace();
  const params = useSearchParams();
  const toast = useToast();
  const confirm = useConfirm();
  const [sources, setSources] = useState<DatasetSource[] | null>(null);
  const [datasets, setDatasets] = useState<DatasetRow[] | null>(null);
  const [picked, setPicked] = useState<string[]>([]);
  const [minReward, setMinReward] = useState(1);
  const [valPct, setValPct] = useState(5);
  const [bestPerTask, setBestPerTask] = useState(true);
  const [kind, setKind] = useState<DatasetKind>("sft");
  const [maxPairs, setMaxPairs] = useState(4);
  const [preview, setPreview] = useState<DatasetPreview | null>(null);
  const [previewing, setPreviewing] = useState(false);
  const [name, setName] = useState("");
  const [nameEdited, setNameEdited] = useState(false); // once you type, no more suggestions
  const [saving, setSaving] = useState(false);
  const [showIneligible, setShowIneligible] = useState(false);

  const load = useCallback(async () => {
    if (!ws) return;
    const [s, d] = await Promise.all([Compute.datasetSources(ws.id), Compute.datasets(ws.id)]);
    setSources(s);
    setDatasets(d);
    return s;
  }, [ws]);

  useEffect(() => {
    setSources(null);
    setDatasets(null);
    load()
      .then((s) => {
        const wanted = (params.get("runs") ?? "").split(",").filter(Boolean);
        if (s && wanted.length) setPicked(s.filter((x) => x.eligible && wanted.includes(x.id)).map((x) => x.id));
      })
      .catch((e) => toast.error("Couldn't load datasets", e));
  }, [load, params, toast]);

  const eligible = useMemo(() => (sources ?? []).filter((s) => s.eligible), [sources]);
  const ineligible = useMemo(() => (sources ?? []).filter((s) => !s.eligible), [sources]);
  const wantedIneligible = useMemo(() => {
    const wanted = (params.get("runs") ?? "").split(",").filter(Boolean);
    return ineligible.filter((s) => wanted.includes(s.id));
  }, [params, ineligible]);

  // live preview, debounced
  useEffect(() => {
    if (!ws || picked.length === 0) {
      setPreview(null);
      return;
    }
    setPreviewing(true);
    const t = setTimeout(() => {
      Compute.previewDataset(ws.id, { run_ids: picked, min_reward: minReward, val_frac: valPct / 100, best_per_task: bestPerTask, kind, max_pairs_per_task: maxPairs })
        .then(setPreview)
        .catch((e) => {
          setPreview(null);
          toast.error("Preview failed", e);
        })
        .finally(() => setPreviewing(false));
    }, 400);
    return () => clearTimeout(t);
  }, [ws, picked, minReward, valPct, bestPerTask, kind, maxPairs, toast]);

  // suggest a name from the environment
  useEffect(() => {
    if (name || nameEdited || picked.length === 0 || !sources) return;
    const first = sources.find((s) => s.id === picked[0]);
    if (first) setName(finalName(`${first.pack}-${kind === "preference" ? "pairs" : "v"}${(datasets?.length ?? 0) + 1}`));
  }, [picked, sources, datasets, name, nameEdited, kind]);

  if (!ws) return null;

  async function save(e: React.FormEvent) {
    e.preventDefault();
    setSaving(true);
    try {
      const d = await Compute.saveDataset(ws!.id, { name: finalName(name), run_ids: picked, min_reward: minReward, val_frac: valPct / 100, best_per_task: bestPerTask, kind, max_pairs_per_task: maxPairs });
      toast.success("Dataset saved", `${d.name}: ${d.n_train} training and ${d.n_val} validation ${kind === "preference" ? "pairs" : "examples"}.`);
      setName("");
      setNameEdited(false);
      setPicked([]);
      await load();
    } catch (ex) {
      toast.error("Couldn't save the dataset", ex);
    } finally {
      setSaving(false);
    }
  }

  async function remove(d: DatasetRow) {
    const ok = await confirm({
      title: `Delete dataset “${d.name}”?`,
      body: "Its files are removed from the server. Adapters already trained on it keep working and remember its name.",
      confirmLabel: "Delete dataset",
      tone: "danger",
    });
    if (!ok) return;
    try {
      await Compute.deleteDataset(ws!.id, d.id);
      toast.success("Dataset deleted");
      await load();
    } catch (ex) {
      toast.error("Couldn't delete the dataset", ex);
    }
  }

  const toggle = (id: string) => setPicked((xs) => (xs.includes(id) ? xs.filter((x) => x !== id) : [...xs, id]));

  return (
    <Page>
      <PageHeader
        title="Datasets"
        description={
          <>
            Training data from a strong model&apos;s solutions (<em>rejection sampling</em>): keep the episodes that scored well, one per task, and hold some tasks
            out for validation. Or <em>preference pairs</em> for DPO: a solved and a failed episode of the same task. Only runs on training seeds ({TRAIN_SEED}+)
            can be used, so evaluation runs on seeds 0–99 stay a fair test.
          </>
        }
      />

      <Panel title="New dataset" hint="Pick runs, set the filters, check the preview, save.">
        {!sources ? (
          <p className="hint">Loading runs…</p>
        ) : eligible.length === 0 ? (
          <div className="space-y-3">
            <p style={{ color: "var(--muted)" }}>
              No finished runs on training seeds yet. Run a strong model (the teacher) on seeds {TRAIN_SEED}+ first; a few hundred episodes is a good start.
            </p>
            <Link href={newRunHref({ seed: TRAIN_SEED, episodes: 200, label: "teacher · training data" })} className="btn">
              Start a run on training seeds
            </Link>
          </div>
        ) : (
          <form onSubmit={save} className="space-y-5">
            <fieldset>
              <legend className="label">What to build</legend>
              <div className="flex flex-wrap gap-2" role="radiogroup">
                {(
                  [
                    ["sft", "SFT examples", "solved episodes to imitate"],
                    ["preference", "Preference pairs (DPO)", "a solved and a failed episode of the same task"],
                  ] as const
                ).map(([k, label, note]) => (
                  <label key={k} className={`inset-card flex cursor-pointer items-start gap-2 px-3 py-2 text-sm ${kind === k ? "" : "opacity-70"}`} style={kind === k ? { borderColor: "var(--accent)" } : undefined}>
                    <input type="radio" name="ds-kind" className="mt-1" checked={kind === k} onChange={() => setKind(k)} />
                    <span>
                      <span className="block">{label}</span>
                      <span className="hint">{note}</span>
                    </span>
                  </label>
                ))}
              </div>
              {kind === "preference" && (
                <p className="hint mt-2">
                  The failed side should come from the model you want to improve: run your adapter on training seeds ({TRAIN_SEED}+) and pick that run together
                  with a teacher run on the same seeds.
                </p>
              )}
            </fieldset>
            {wantedIneligible.length > 0 && (
              <p className="text-sm" style={{ color: "var(--warn)" }}>
                {wantedIneligible.length} of the runs you came from can&apos;t be used: {wantedIneligible[0].reason}.
              </p>
            )}
            <div className="overflow-x-auto">
              <table className="data inset">
                <thead>
                  <tr>
                    <th className="w-8" aria-label="Use" />
                    <th>Run</th>
                    <th>Environment</th>
                    <th>Seeds</th>
                    <th className="text-right">Solved</th>
                  </tr>
                </thead>
                <tbody>
                  {[...eligible, ...(showIneligible ? ineligible : [])].map((s) => (
                    <tr key={s.id} style={{ opacity: s.eligible ? 1 : 0.55 }}>
                      <td>
                        <input type="checkbox" aria-label={`Use ${s.label || s.id}`} disabled={!s.eligible} checked={picked.includes(s.id)} onChange={() => toggle(s.id)} />
                      </td>
                      <td className="min-w-0">
                        <div className="truncate">{s.label || s.model}</div>
                        <div className="mono hint truncate">{s.eligible ? s.model : s.reason}</div>
                      </td>
                      <td>{s.pack}</td>
                      <td className="num">
                        {s.seed_start}–{s.seed_start + s.episodes - 1}
                      </td>
                      <td className="num text-right">{pct(s.solve_rate)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            {ineligible.length > 0 && (
              <button type="button" className="hint underline" onClick={() => setShowIneligible((x) => !x)}>
                {showIneligible ? "Hide" : "Show"} {ineligible.length} run{ineligible.length === 1 ? "" : "s"} that can&apos;t be used (evaluation seeds)
              </button>
            )}

            <div className="grid grid-cols-1 items-end gap-4 sm:grid-cols-3">
              <div>
                <label className="label" htmlFor="ds-reward">
                  {kind === "preference" ? "Solved side: minimum reward" : "Minimum reward"}
                </label>
                <select id="ds-reward" className="input" value={minReward} onChange={(e) => setMinReward(Number(e.target.value))}>
                  <option value={1}>1.0 (clean solves only)</option>
                  <option value={0.8}>0.8</option>
                  <option value={0.5}>0.5</option>
                  <option value={0.3}>0.3 (incl. collateral)</option>
                  <option value={0}>0 (everything)</option>
                </select>
              </div>
              <div>
                <label className="label" htmlFor="ds-val">
                  Validation split (% of tasks)
                </label>
                <input id="ds-val" className="input" type="number" min={0} max={50} step={1} value={valPct} onChange={(e) => setValPct(Math.max(0, Math.min(50, Number(e.target.value))))} />
              </div>
              {kind === "preference" ? (
                <div>
                  <label className="label" htmlFor="ds-pairs">
                    Pairs per task (max)
                  </label>
                  <input id="ds-pairs" className="input" type="number" min={1} max={32} step={1} value={maxPairs} onChange={(e) => setMaxPairs(Math.max(1, Math.min(32, Number(e.target.value))))} />
                </div>
              ) : (
                <label className="flex items-center gap-2 pb-2 text-sm">
                  <input type="checkbox" checked={bestPerTask} onChange={(e) => setBestPerTask(e.target.checked)} />
                  Best episode per task only
                </label>
              )}
            </div>

            <Preview p={preview} loading={previewing} />

            <div className="flex flex-wrap items-end gap-2 border-t pt-4" style={{ borderColor: "var(--line)" }}>
              <div className="min-w-0 flex-1 basis-60">
                <label className="label" htmlFor="ds-name">
                  Name
                </label>
                <input
                  id="ds-name"
                  className="input mono"
                  maxLength={63}
                  placeholder="netops-v1"
                  value={name}
                  onChange={(e) => {
                    setNameEdited(true);
                    setName(typingName(e.target.value));
                  }}
                />
              </div>
              <button className="btn btn-field" disabled={saving || !finalName(name) || picked.length === 0 || !preview?.kept}>
                {saving ? "Saving…" : "Save dataset"}
              </button>
            </div>
          </form>
        )}
      </Panel>

      <Panel title="Saved datasets" flush hint={datasets ? `${datasets.length}` : undefined}>
        {!datasets ? (
          <Empty>Loading…</Empty>
        ) : datasets.length === 0 ? (
          <Empty>No datasets yet. Save one above; it then shows up on the Training page.</Empty>
        ) : (
          <table className="data">
            <thead>
              <tr>
                <th>Name</th>
                <th>Environment</th>
                <th className="text-right">Train / val</th>
                <th className="hidden md:table-cell">Filters</th>
                <th className="hidden md:table-cell">Created</th>
                <th />
              </tr>
            </thead>
            <tbody>
              {datasets.map((d) => (
                <tr key={d.id}>
                  <td>
                    <span className="mono">{d.name}</span>
                    {d.kind === "preference" && <span className="pill pill-queued ml-2">DPO pairs</span>}
                  </td>
                  <td>{d.packs.join(", ")}</td>
                  <td className="num text-right">
                    {d.n_train} / {d.n_val}
                  </td>
                  <td className="hint hidden md:table-cell">
                    reward ≥ {d.filters.min_reward} ·{" "}
                    {d.kind === "preference" ? `≤ ${d.filters.max_pairs_per_task ?? 4} pairs per task` : d.filters.best_per_task ? "best per task" : "all passes"} ·{" "}
                    {Math.round(d.filters.val_frac * 100)}% val ·{" "}
                    {d.source_runs.length} run{d.source_runs.length === 1 ? "" : "s"}
                  </td>
                  <td className="hint hidden md:table-cell">{new Date(d.created_at).toLocaleDateString()}</td>
                  <td>
                    <div className="flex flex-wrap justify-end gap-1">
                      <Link href={`/training?dataset=${d.id}`} className="btn btn-sm">
                        Train
                      </Link>
                      <button
                        className="btn-ghost btn-sm"
                        onClick={() => downloadFile(`/api/workspaces/${ws.id}/datasets/${d.id}/download?split=train`, `${d.name}-train.jsonl`).catch((e) => toast.error("Download failed", e))}
                      >
                        train.jsonl
                      </button>
                      {d.n_val > 0 && (
                        <button
                          className="btn-ghost btn-sm"
                          onClick={() => downloadFile(`/api/workspaces/${ws.id}/datasets/${d.id}/download?split=val`, `${d.name}-val.jsonl`).catch((e) => toast.error("Download failed", e))}
                        >
                          val.jsonl
                        </button>
                      )}
                      <button className="btn-danger btn-sm" onClick={() => remove(d)}>
                        Delete
                      </button>
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </Panel>
    </Page>
  );
}

export default function DatasetsPage() {
  // useSearchParams needs a Suspense boundary in the app router
  return (
    <Suspense fallback={null}>
      <DatasetsInner />
    </Suspense>
  );
}

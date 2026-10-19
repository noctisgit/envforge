"use client";

import Link from "next/link";
import { useRouter } from "next/navigation";
import { useCallback, useEffect, useState } from "react";
import { useConfirm } from "@/components/ConfirmDialog";
import { ExportButton } from "@/components/ExportButton";
import { ModelPicker } from "@/components/ModelPicker";
import { Api, ApiError, fmtCost, fmtTokens, type Pack, type Run } from "@/lib/api";
import { useWorkspace } from "@/lib/workspace";

const pct = (x: number) => `${Math.round(x * 100)}%`;

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
  const [msg, setMsg] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [picked, setPicked] = useState<string[]>([]);
  const confirm = useConfirm();

  useEffect(() => {
    if (ws) Api.wsPacks(ws.id).then(setPacks).catch(() => {});
  }, [ws]);
  useEffect(() => {
    if (!ws) return;
    setModel(ws.default_model);
    // "Run it" on an environment page links here with ?pack=<name>
    const wanted = new URLSearchParams(window.location.search).get("pack");
    setPack(wanted || ws.default_pack);
    setEpisodes(ws.default_episodes);
    setPicked([]);
  }, [ws]);

  const load = useCallback(async () => {
    if (!ws) return;
    setRuns(await Api.runs(ws.id));
  }, [ws]);

  useEffect(() => {
    load().catch(() => {});
  }, [load]);

  // poll while anything is still in flight
  const active = runs.some((r) => r.status === "queued" || r.status === "running");
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
    setMsg(null);
    try {
      await Api.createRun(ws!.id, { model, pack, episodes, tiers, label, seed_start: seedStart });
      setLabel("");
      await load();
    } catch (ex) {
      setMsg(ex instanceof ApiError ? ex.message : String(ex));
    } finally {
      setBusy(false);
    }
  }

  const toggle = (id: string) => setPicked((p) => (p.includes(id) ? p.filter((x) => x !== id) : [...p, id]));

  async function remove(ids: string[]) {
    const chosen = runs.filter((r) => ids.includes(r.id));
    const active = chosen.filter((r) => r.status === "queued" || r.status === "running");
    const deletable = chosen.filter((r) => !active.includes(r));
    if (!deletable.length) {
      setMsg("Queued or running runs can't be deleted — cancel them first.");
      return;
    }
    const one = deletable.length === 1;
    const ok = await confirm({
      title: one ? "Delete this run?" : `Delete ${deletable.length} runs?`,
      body: (
        <div className="space-y-2">
          <p>
            {one ? (
              <span className="mono">{deletable[0].label || deletable[0].id}</span>
            ) : (
              `${deletable.length} runs`
            )}{" "}
            and {one ? "its" : "their"} {deletable.reduce((n, r) => n + r.progress, 0)} recorded episodes will be permanently deleted.
            Transcripts and training-data exports from {one ? "it" : "them"} will no longer be available.
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
        failed.push(`${r.id}: ${e instanceof ApiError ? e.message : String(e)}`);
      }
    }
    setPicked((p) => p.filter((id) => !deletable.some((r) => r.id === id)));
    setMsg(failed.length ? `Some runs were not deleted: ${failed.join("; ")}` : null);
    await load();
  }

  return (
    <div className="space-y-6">
      <form onSubmit={submit} className="panel grid grid-cols-1 gap-4 p-5 md:grid-cols-6">
        <div className="md:col-span-6 text-sm font-semibold">New run</div>
        <div className="md:col-span-3">
          <label className="label">Model</label>
          <ModelPicker wsId={ws.id} value={model} onChange={setModel} />
        </div>
        <div className="md:col-span-2">
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
        </div>
        <div>
          <label className="label">Episodes</label>
          <input className="input" type="number" min={1} max={1000} value={episodes} onChange={(e) => setEpisodes(Number(e.target.value))} />
        </div>
        {currentPack && (
          <div className="md:col-span-6 -mt-2 text-xs" style={{ color: "var(--muted)" }}>
            {currentPack.description}
          </div>
        )}
        <div className="md:col-span-2">
          <label className="label">Tiers (none = all)</label>
          <div className="flex gap-3 pt-2 text-sm">
            {packTiers.map((t) => (
              <label key={t} className="flex items-center gap-1">
                <input type="checkbox" checked={tiers.includes(t)} onChange={(e) => setTiers(e.target.checked ? [...tiers, t].sort() : tiers.filter((x) => x !== t))} />
                {t}
              </label>
            ))}
          </div>
        </div>
        <div>
          <label className="label" title="Each seed is a different generated task. Collect training data on seeds 10000+, evaluate on 0–99, so evaluation never reuses a training task.">
            Seed start ⓘ
          </label>
          <input className="input" type="number" min={0} value={seedStart} onChange={(e) => setSeedStart(Math.max(0, Number(e.target.value)))} />
          <div className="mt-1 text-[11px]" style={{ color: "var(--muted)" }}>
            {seedStart >= 10000 ? "training range" : seedStart < 100 ? "evaluation range" : "custom range"}
          </div>
        </div>
        <div className="md:col-span-2">
          <label className="label">Label</label>
          <input className="input" value={label} onChange={(e) => setLabel(e.target.value)} placeholder="optional" maxLength={200} />
        </div>
        <div className="flex items-end">
          <button className="btn" disabled={busy || !model.trim()}>
            Queue run
          </button>
        </div>
        {msg && (
          <div className="md:col-span-6 text-sm" style={{ color: "var(--bad)" }}>
            {msg}
          </div>
        )}
      </form>

      {picked.length > 0 && (
        <div className="panel flex flex-wrap items-center gap-3 p-3 text-sm">
          <span>{picked.length} selected</span>
          <button className="btn" disabled={picked.length < 2} onClick={() => router.push(`/compare?runs=${picked.map(encodeURIComponent).join(",")}`)}>
            Compare
          </button>
          <ExportButton wsId={ws.id} runIds={picked} />
          <button className="btn-ghost" style={{ color: "var(--bad)" }} onClick={() => remove(picked)}>
            Delete
          </button>
          <button className="btn-ghost ml-auto" onClick={() => setPicked([])}>
            Clear
          </button>
        </div>
      )}

      <div className="panel overflow-x-auto p-2">
        {runs.length === 0 ? (
          <div className="p-4 text-sm" style={{ color: "var(--muted)" }}>
            No runs yet. Start with model <span className="mono">scripted</span>; it needs no key.
          </div>
        ) : (
          <table className="data">
            <thead>
              <tr>
                <th></th>
                <th>run</th>
                <th>model</th>
                <th>env</th>
                <th>status</th>
                <th>progress</th>
                <th>solve</th>
                <th>reward</th>
                <th>tokens</th>
                <th>cost</th>
                <th>created</th>
                <th></th>
              </tr>
            </thead>
            <tbody>
              {runs.map((r) => (
                <tr key={r.id}>
                  <td>
                    <input type="checkbox" checked={picked.includes(r.id)} onChange={() => toggle(r.id)} aria-label={`select ${r.id}`} />
                  </td>
                  <td>
                    <Link href={`/runs/${encodeURIComponent(r.id)}`} className="mono" style={{ color: "var(--accent)" }}>
                      {r.label || r.id}
                    </Link>
                  </td>
                  <td className="mono">{r.model}</td>
                  <td>{r.pack}</td>
                  <td>
                    <span className={`pill pill-${r.status}`}>{r.status}</span>
                  </td>
                  <td>
                    {r.progress}/{r.episodes}
                  </td>
                  <td>{r.metrics ? pct(r.metrics.overall.solve_rate) : "—"}</td>
                  <td>{r.metrics ? r.metrics.overall.mean_reward.toFixed(2) : "—"}</td>
                  <td className="tabular-nums">{fmtTokens(r.input_tokens + r.output_tokens)}</td>
                  <td className="tabular-nums">{fmtCost(r.cost_usd)}</td>
                  <td style={{ color: "var(--muted)" }}>{new Date(r.created_at).toLocaleString()}</td>
                  <td>
                    {r.status !== "queued" && r.status !== "running" && (
                      <button className="text-xs underline" style={{ color: "var(--muted)" }} onClick={() => remove([r.id])} aria-label={`delete ${r.id}`}>
                        Delete
                      </button>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </div>
  );
}

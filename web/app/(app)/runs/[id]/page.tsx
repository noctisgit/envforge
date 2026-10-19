"use client";

import Link from "next/link";
import { useParams, useRouter } from "next/navigation";
import { useCallback, useEffect, useState } from "react";
import { useConfirm } from "@/components/ConfirmDialog";
import { DiagramView } from "@/components/DiagramView";
import { ExportButton } from "@/components/ExportButton";
import { MetricsTable } from "@/components/MetricsTable";
import { Api, ApiError, fmtCost, fmtTokens, type EpisodeSummary, type ReplayFrame, type Run, type Trajectory } from "@/lib/api";
import { useWorkspace } from "@/lib/workspace";

export default function RunDetailPage() {
  const { id } = useParams<{ id: string }>();
  const runId = decodeURIComponent(id);
  const { current: ws } = useWorkspace();
  const [run, setRun] = useState<Run | null>(null);
  const [episodes, setEpisodes] = useState<EpisodeSummary[]>([]);
  const [open, setOpen] = useState<number | null>(null);
  const confirm = useConfirm();
  const router = useRouter();
  const [traj, setTraj] = useState<Trajectory | null>(null);
  const [err, setErr] = useState<string | null>(null);

  const load = useCallback(async () => {
    if (!ws) return;
    try {
      const [r, eps] = await Promise.all([Api.run(ws.id, runId), Api.episodes(ws.id, runId)]);
      setRun(r);
      setEpisodes(eps);
    } catch (ex) {
      setErr(ex instanceof ApiError ? ex.message : String(ex));
    }
  }, [ws, runId]);

  useEffect(() => {
    load();
  }, [load]);

  const active = run?.status === "queued" || run?.status === "running";
  useEffect(() => {
    if (!active) return;
    const t = setInterval(load, 3000);
    return () => clearInterval(t);
  }, [active, load]);

  useEffect(() => {
    if (!ws || open === null) return;
    setTraj(null);
    Api.episode(ws.id, runId, open).then(setTraj).catch(() => {});
  }, [ws, runId, open]);

  if (err) return <div style={{ color: "var(--bad)" }}>{err}</div>;
  if (!ws || !run) return null;

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-center gap-3">
        <Link href="/runs" style={{ color: "var(--muted)" }}>
          ← Runs
        </Link>
        <h1 className="mono text-lg font-semibold">{run.label || run.id}</h1>
        <span className={`pill pill-${run.status}`}>{run.status}</span>
        <div className="ml-auto flex items-center gap-2">
          {episodes.length > 0 && <ExportButton wsId={ws.id} runIds={[run.id]} />}
          {!active && (
            <button
              className="btn-ghost"
              style={{ color: "var(--bad)" }}
              onClick={async () => {
                const ok = await confirm({
                  title: "Delete this run?",
                  body: `${run.label || run.id} and its ${run.progress} recorded episodes will be permanently deleted. Your usage history keeps its token and cost totals.`,
                  confirmLabel: "Delete run",
                  tone: "danger",
                });
                if (!ok) return;
                await Api.deleteRun(ws.id, run.id);
                router.replace("/runs");
              }}
            >
              Delete
            </button>
          )}
          {active && (
            <button
              className="btn-ghost"
              onClick={async () => {
                const ok = await confirm({
                  title: "Cancel this run?",
                  body: "The worker stops after the episode it is on. Finished episodes and their usage are kept.",
                  confirmLabel: "Cancel run",
                  cancelLabel: "Keep running",
                  tone: "danger",
                });
                if (ok) setRun(await Api.cancelRun(ws.id, run.id));
              }}
            >
              Cancel
            </button>
          )}
        </div>
      </div>

      <div className="panel grid grid-cols-2 gap-4 p-5 text-sm md:grid-cols-4">
        <Field k="model" v={<span className="mono">{run.model}</span>} />
        <Field k="pack" v={run.pack} />
        <Field k="progress" v={`${run.progress}/${run.episodes}`} />
        <Field k="tiers" v={run.tiers.length ? run.tiers.join(", ") : "all"} />
        <Field k="seed start" v={run.seed_start} />
        <Field k="temperature" v={run.temperature} />
        <Field k="started" v={run.started_at ? new Date(run.started_at).toLocaleString() : "—"} />
        <Field k="finished" v={run.finished_at ? new Date(run.finished_at).toLocaleString() : "—"} />
        <Field
          k="input tokens"
          v={`${fmtTokens(run.input_tokens)}${run.cache_read_tokens ? ` · ${Math.round((100 * run.cache_read_tokens) / Math.max(1, run.input_tokens))}% cached` : ""}`}
        />
        <Field k="output tokens" v={fmtTokens(run.output_tokens)} />
        <Field k="est. cost" v={fmtCost(run.cost_usd)} />
        <Field k="cost / episode" v={run.progress && run.cost_usd !== null ? fmtCost(run.cost_usd / run.progress) : "—"} />
      </div>

      {run.error && (
        <pre className="panel mono overflow-x-auto whitespace-pre-wrap p-4 text-xs" style={{ color: "var(--bad)" }}>
          {run.error}
        </pre>
      )}

      {run.metrics && (
        <div className="panel overflow-x-auto p-2">
          <MetricsTable overall={run.metrics.overall} byTier={run.metrics.by_tier} />
        </div>
      )}

      {episodes.length > 0 && (
        <div className="panel overflow-x-auto p-2">
          <table className="data">
            <thead>
              <tr>
                <th>#</th>
                <th>task</th>
                <th>tier</th>
                <th>result</th>
                <th>reward</th>
                <th>steps</th>
                <th>calls</th>
                <th>tool err</th>
                <th>finished</th>
                <th>collateral</th>
              </tr>
            </thead>
            <tbody>
              {episodes.map((e) => (
                <tr key={e.episode} onClick={() => setOpen(open === e.episode ? null : e.episode)} className="cursor-pointer" style={open === e.episode ? { background: "var(--line)" } : undefined}>
                  <td>{e.episode}</td>
                  <td className="mono">{e.task_id}</td>
                  <td>{e.tier}</td>
                  <td style={{ color: e.success ? "var(--good)" : "var(--bad)" }}>{e.success ? "solved" : "failed"}</td>
                  <td>{e.reward.toFixed(2)}</td>
                  <td>{e.steps}</td>
                  <td>{e.tool_calls}</td>
                  <td>{e.tool_errors}</td>
                  <td>{e.finished_reason}</td>
                  <td style={{ color: e.collateral.length ? "var(--warn)" : undefined }}>{e.collateral.join(", ") || "—"}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {open !== null && (
        <>
          <Replay wsId={ws.id} runId={run.id} episode={open} />
          <Transcript traj={traj} episode={open} />
        </>
      )}
    </div>
  );
}

function Field({ k, v }: { k: string; v: React.ReactNode }) {
  return (
    <div>
      <div className="label">{k}</div>
      <div>{v}</div>
    </div>
  );
}

function Transcript({ traj, episode }: { traj: Trajectory | null; episode: number }) {
  if (!traj) return <div className="panel p-5 text-sm">Loading episode {episode}…</div>;
  return (
    <div className="panel space-y-4 p-5 text-sm">
      <div className="font-semibold">Episode {episode}</div>
      <div>
        <div className="label">goal</div>
        <div>{traj.task.goal}</div>
      </div>
      <details>
        <summary className="label cursor-pointer">system prompt</summary>
        <pre className="mono whitespace-pre-wrap text-xs" style={{ color: "var(--muted)" }}>
          {traj.system_prompt}
        </pre>
      </details>
      {traj.steps.map((s) => (
        <div key={s.index} className="space-y-2 border-t pt-3" style={{ borderColor: "var(--line)" }}>
          <div className="label">step {s.index}</div>
          {s.assistant_text && <div className="whitespace-pre-wrap">{s.assistant_text}</div>}
          {s.tool_calls.map((c, i) => (
            <div key={c.id} className="space-y-1">
              <div className="mono text-xs" style={{ color: "var(--accent)" }}>
                {c.name}({JSON.stringify(c.args)})
              </div>
              {s.observations[i] !== undefined && (
                <pre className="mono overflow-x-auto whitespace-pre-wrap text-xs" style={{ color: s.errors[i] ? "var(--bad)" : "var(--muted)" }}>
                  {s.observations[i]}
                </pre>
              )}
            </div>
          ))}
        </div>
      ))}
      <div className="border-t pt-3" style={{ borderColor: "var(--line)" }}>
        <div className="label">verdict</div>
        <div style={{ color: traj.verdict?.success ? "var(--good)" : "var(--bad)" }}>
          {traj.verdict ? `${traj.verdict.success ? "solved" : "failed"} · reward ${traj.verdict.reward.toFixed(2)}` : "none"} · {traj.finished_reason}
          {traj.error && ` · ${traj.error}`}
        </div>
        <pre className="mono mt-2 overflow-x-auto whitespace-pre-wrap text-xs" style={{ color: "var(--muted)" }}>
          {JSON.stringify(traj.verdict?.details ?? {}, null, 2)}
        </pre>
      </div>
    </div>
  );
}

function Replay({ wsId, runId, episode }: { wsId: string; runId: string; episode: number }) {
  const [frames, setFrames] = useState<ReplayFrame[] | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [i, setI] = useState(0);
  const [onlyChanges, setOnlyChanges] = useState(false);

  useEffect(() => {
    setFrames(null);
    setErr(null);
    setI(0);
    Api.replay(wsId, runId, episode)
      .then((r) => setFrames(r.frames))
      .catch((e) => setErr(e instanceof ApiError ? e.message : String(e)));
  }, [wsId, runId, episode]);

  if (err)
    return (
      <div className="panel p-4 text-xs" style={{ color: "var(--muted)" }}>
        Replay not available: {err}
      </div>
    );
  if (!frames) return <div className="panel p-4 text-sm">Replaying episode {episode}…</div>;

  // indexes of the frames to step through
  const idx = frames.map((_, k) => k).filter((k) => k === 0 || !onlyChanges || frames[k].changed);
  const pos = Math.min(i, idx.length - 1);
  const f = frames[idx[pos]];
  return (
    <div className="panel space-y-3 p-5">
      <div className="flex flex-wrap items-center gap-3 text-sm">
        <span className="font-semibold">Replay · episode {episode}</span>
        <span style={{ color: "var(--muted)" }}>
          {pos === 0 ? "start (before any action)" : `call ${pos} of ${idx.length - 1}`}
        </span>
        <div className="ml-auto flex items-center gap-2">
          <label className="flex items-center gap-1 text-xs" style={{ color: "var(--muted)" }}>
            <input
              type="checkbox"
              checked={onlyChanges}
              onChange={(e) => {
                setOnlyChanges(e.target.checked);
                setI(0);
              }}
            />
            only calls that changed something
          </label>
          <button className="btn-ghost py-0.5" disabled={pos === 0} onClick={() => setI(pos - 1)}>
            ← Prev
          </button>
          <button className="btn-ghost py-0.5" disabled={pos >= idx.length - 1} onClick={() => setI(pos + 1)}>
            Next →
          </button>
        </div>
      </div>
      <input type="range" min={0} max={idx.length - 1} value={pos} onChange={(e) => setI(Number(e.target.value))} className="w-full" aria-label="replay position" />
      {f.call && (
        <div className="text-sm">
          <span className="mono" style={{ color: f.error ? "var(--bad)" : "var(--accent)" }}>
            {f.call.name}({Object.entries(f.call.args).map(([k, v]) => `${k}=${JSON.stringify(v)}`).join(", ")})
          </span>
          {f.changed && <span className="ml-2 text-xs" style={{ color: "var(--warn)" }}>changed the world</span>}
          {f.observation && (
            <pre className="mono mt-1 max-h-40 overflow-auto whitespace-pre-wrap text-xs" style={{ color: "var(--muted)" }}>
              {f.observation}
            </pre>
          )}
        </div>
      )}
      <DiagramView diagram={f.diagram} height={420} />
    </div>
  );
}

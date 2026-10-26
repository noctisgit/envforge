"use client";

import Link from "next/link";
import { useParams, useRouter } from "next/navigation";
import { useCallback, useEffect, useState } from "react";
import { useConfirm } from "@/components/ConfirmDialog";
import { DiagramView } from "@/components/DiagramView";
import { ExportButton } from "@/components/ExportButton";
import { MetricsTable } from "@/components/MetricsTable";
import { newRunHref } from "@/components/Journey";
import { NextStep } from "@/components/NextStep";
import { Empty, Facts, Page, PageHeader, Panel } from "@/components/Page";
import { useToast } from "@/components/Toast";
import { ALL_TASKS_TIER, Api, ApiError, fmtCost, fmtTokens, type EpisodeSummary, type ReplayFrame, type Run, type Trajectory } from "@/lib/api";
import { EVAL_SEED, TRAIN_SEED, isTrainingRun, modelRole } from "@/lib/journey";
import { useWorkspace } from "@/lib/workspace";

const time = (iso: string | null) => (iso ? new Date(iso).toLocaleString(undefined, { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" }) : "—");

function duration(a: string | null, b: string | null): string {
  if (!a || !b) return "";
  const s = Math.max(0, Math.round((new Date(b).getTime() - new Date(a).getTime()) / 1000));
  return s < 90 ? `${s} s` : s < 5400 ? `${Math.round(s / 60)} min` : `${(s / 3600).toFixed(1)} h`;
}

export default function RunDetailPage() {
  const { id } = useParams<{ id: string }>();
  const runId = decodeURIComponent(id);
  const { current: ws } = useWorkspace();
  const [run, setRun] = useState<Run | null>(null);
  const [episodes, setEpisodes] = useState<EpisodeSummary[]>([]);
  const [open, setOpen] = useState<number | null>(null);
  const confirm = useConfirm();
  const toast = useToast();
  const router = useRouter();
  const [traj, setTraj] = useState<Trajectory | null>(null);
  const [err, setErr] = useState<string | null>(null);

  const load = useCallback(async () => {
    if (!ws) return;
    try {
      const [r, eps] = await Promise.all([Api.run(ws.id, runId), Api.episodes(ws.id, runId)]);
      setRun(r);
      setEpisodes(eps);
      setOpen((o) => (o === null && eps.length ? eps[0].episode : o)); // show the first replay straight away
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
    Api.episode(ws.id, runId, open)
      .then(setTraj)
      .catch(() => {});
  }, [ws, runId, open]);

  if (err)
    return (
      <Page>
        <PageHeader back={{ href: "/runs", label: "Runs" }} title="Run not available" description={<span style={{ color: "var(--bad)" }}>{err}</span>} />
      </Page>
    );
  if (!ws || !run) return null;

  async function remove() {
    const ok = await confirm({
      title: "Delete this run?",
      body: `${run!.label || run!.id} and its ${run!.progress} recorded episodes will be permanently deleted. Training-data exports already downloaded are not affected. Your usage history keeps its token and cost totals.`,
      confirmLabel: "Delete run",
      tone: "danger",
      requireText: "delete",
    });
    if (!ok) return;
    try {
      await Api.deleteRun(ws!.id, run!.id);
      toast.success("Run deleted");
      router.replace("/runs");
    } catch (e) {
      toast.error("Couldn't delete the run", e);
    }
  }

  async function cancel() {
    const ok = await confirm({
      title: "Stop this run?",
      body: "The worker stops after the episode it is on. Finished episodes and their usage are kept and scored.",
      confirmLabel: "Stop run",
      cancelLabel: "Keep running",
      tone: "danger",
    });
    if (!ok) return;
    try {
      setRun(await Api.cancelRun(ws!.id, run!.id));
      toast.info("Stopping", "The worker stops after the current episode.");
    } catch (e) {
      toast.error("Couldn't stop the run", e);
    }
  }

  const cachedPct = run.cache_read_tokens ? Math.round((100 * run.cache_read_tokens) / Math.max(1, run.input_tokens)) : 0;
  const took = duration(run.started_at, run.finished_at);

  return (
    <Page>
      <PageHeader
        back={{ href: "/runs", label: "Runs" }}
        title={
          <>
            {run.label || <span className="mono">{run.id}</span>}
            {run.label && (
              <span className="mono ml-2 text-sm font-normal" style={{ color: "var(--muted)" }}>
                {run.id}
              </span>
            )}
          </>
        }
        meta={<span className={`pill pill-${run.status}`}>{run.status}</span>}
        actions={
          <>
            {episodes.length > 0 && <ExportButton wsId={ws.id} runIds={[run.id]} />}
            {active ? (
              <button className="btn-danger" onClick={cancel}>
                Stop run
              </button>
            ) : (
              <button className="btn-danger" onClick={remove}>
                Delete
              </button>
            )}
          </>
        }
      >
        <Facts
          items={[
            { label: "model", value: <span className="mono text-xs">{run.model}</span> },
            { label: "pack", value: <Link href={`/environments/${encodeURIComponent(run.pack)}`}>{run.pack}</Link> },
            { label: "progress", value: <span className="num">{run.progress} / {run.episodes}</span> },
            {
              label: "tiers · seed start",
              value: (
                <>
                  {run.tiers.length === 1 && run.tiers[0] === ALL_TASKS_TIER ? "every task once" : run.tiers.length ? run.tiers.join(", ") : "all"} · <span className="mono">{run.seed_start}</span>
                </>
              ),
            },
            { label: "temperature", value: <span className="num">{run.temperature}</span> },
            { label: "max steps", value: <span className="num">{run.max_steps ?? "pack default"}</span> },
            { label: "created", value: time(run.created_at) },
            { label: "started → finished", value: `${time(run.started_at)} → ${time(run.finished_at)}${took ? ` · ${took}` : ""}` },
            {
              label: "input tokens",
              value: (
                <span className="num">
                  {run.input_tokens.toLocaleString()}
                  {cachedPct > 0 && <span style={{ color: "var(--muted)" }}> · {cachedPct}% cached</span>}
                </span>
              ),
            },
            { label: "output tokens", value: <span className="num">{run.output_tokens.toLocaleString()}</span> },
            { label: "est. cost", value: <span className="mono">{fmtCost(run.cost_usd)}</span> },
            { label: "cost / episode", value: <span className="mono">{run.progress && run.cost_usd !== null ? fmtCost(run.cost_usd / run.progress) : "—"}</span> },
          ]}
        />
      </PageHeader>

      {run.status === "completed" && <RunNextStep run={run} wsId={ws.id} />}

      {run.error && (
        <Panel title="Error" tone="danger">
          <pre className="code-block" style={{ color: "var(--bad)" }}>
            {run.error}
          </pre>
        </Panel>
      )}

      {run.metrics && (
        <Panel title="Metrics" hint="overall, then per tier · solve ≥ 80% good · 40–80% warn · below bad" flush>
          <MetricsTable overall={run.metrics.overall} byTier={run.metrics.by_tier} />
        </Panel>
      )}

      <div className="grid grid-cols-1 items-start gap-6 xl:grid-cols-2">
        <Panel title="Episodes" hint={episodes.length ? "click a row to replay it" : undefined} flush>
          {episodes.length === 0 ? (
            <Empty>{active ? "No episodes finished yet — this page refreshes every 3 s." : "This run recorded no episodes."}</Empty>
          ) : (
            <table className="data">
              <thead>
                <tr>
                  <th>#</th>
                  <th>task</th>
                  <th className="text-right">tier</th>
                  <th>result</th>
                  <th className="text-right">reward</th>
                  <th className="text-right">steps</th>
                  <th className="text-right">calls</th>
                  <th className="text-right">tool err</th>
                  <th>finished</th>
                  <th>collateral</th>
                </tr>
              </thead>
              <tbody>
                {episodes.map((e) => {
                  const on = open === e.episode;
                  return (
                    <tr
                      key={e.episode}
                      onClick={() => setOpen(e.episode)}
                      className={`cursor-pointer ${on ? "font-semibold" : ""}`}
                      style={on ? { background: "var(--accent-tint)" } : undefined}
                    >
                      <td>
                        <button
                          type="button"
                          className="btn-ghost btn-sm"
                          aria-pressed={on}
                          aria-label={`replay episode ${e.episode}`}
                          style={on ? { borderColor: "var(--accent)", color: "var(--accent)" } : undefined}
                          onClick={(ev) => {
                            ev.stopPropagation();
                            setOpen(e.episode);
                          }}
                        >
                          #{e.episode}
                        </button>
                      </td>
                      <td className="mono text-xs">{e.task_id}</td>
                      <td className="num text-right">{e.tier}</td>
                      <td style={{ color: e.success ? "var(--good)" : "var(--bad)" }}>{e.success ? "solved" : "failed"}</td>
                      <td className="num text-right">{e.reward.toFixed(2)}</td>
                      <td className="num text-right">{e.steps}</td>
                      <td className="num text-right">{e.tool_calls}</td>
                      <td className="num text-right">{e.tool_errors}</td>
                      <td className="mono text-xs">{e.finished_reason}</td>
                      <td style={{ color: e.collateral.length ? "var(--warn)" : undefined }}>{e.collateral.join(", ") || "—"}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          )}
        </Panel>

        {open !== null && <Replay wsId={ws.id} runId={run.id} episode={open} traj={traj} />}
      </div>

      {open !== null && <Transcript traj={traj} episode={open} />}
    </Page>
  );
}

function Verdict({ traj }: { traj: Trajectory }) {
  const v = traj.verdict;
  return (
    <div className="space-y-1.5 border-t pt-3" style={{ borderColor: "var(--line)" }}>
      <div className="label mb-0">verdict</div>
      <div>
        <span className="font-medium" style={{ color: v?.success ? "var(--good)" : "var(--bad)" }}>
          {v ? (v.success ? "solved" : "failed") : "no verdict"}
        </span>
        {v && (
          <>
            {" "}
            · reward <span className="mono">{v.reward.toFixed(2)}</span>
          </>
        )}{" "}
        · finished_reason <span className="mono">{traj.finished_reason}</span>
        {traj.error && <span style={{ color: "var(--bad)" }}> · {traj.error}</span>}
      </div>
    </div>
  );
}

function Transcript({ traj, episode }: { traj: Trajectory | null; episode: number }) {
  if (!traj)
    return (
      <Panel title={`Transcript · episode ${episode}`}>
        <div style={{ color: "var(--muted)" }}>Loading…</div>
      </Panel>
    );
  return (
    <Panel title={`Transcript · episode ${episode}`} hint={`${traj.steps.length} step${traj.steps.length === 1 ? "" : "s"} · task ${traj.task.id}`}>
      <div>
        <div className="label">goal</div>
        <div className="whitespace-pre-wrap">{traj.task.goal}</div>
      </div>
      <details>
        <summary className="label mb-0 cursor-pointer">system prompt</summary>
        <pre className="code-block mt-2" style={{ color: "var(--muted)" }}>
          {traj.system_prompt}
        </pre>
      </details>
      {traj.steps.map((s) => (
        <div key={s.index} className="space-y-2 border-t pt-3" style={{ borderColor: "var(--line)" }}>
          <div className="label mb-0">step {s.index}</div>
          {s.assistant_text && <div className="whitespace-pre-wrap">{s.assistant_text}</div>}
          {s.tool_calls.map((c, i) => (
            <div key={c.id} className="space-y-1">
              <div className="mono text-xs [overflow-wrap:anywhere]" style={{ color: "var(--accent)" }}>
                {c.name}({JSON.stringify(c.args)})
              </div>
              {s.observations[i] !== undefined && (
                <pre className="code-block" style={{ color: s.errors[i] ? "var(--bad)" : "var(--muted)" }}>
                  {s.observations[i]}
                </pre>
              )}
            </div>
          ))}
        </div>
      ))}
      <Verdict traj={traj} />
      <details>
        <summary className="label mb-0 cursor-pointer">verifier details</summary>
        <pre className="code-block mt-2" style={{ color: "var(--muted)" }}>
          {JSON.stringify(traj.verdict?.details ?? {}, null, 2)}
        </pre>
      </details>
    </Panel>
  );
}

function Replay({ wsId, runId, episode, traj }: { wsId: string; runId: string; episode: number; traj: Trajectory | null }) {
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

  const title = `Replay · episode ${episode}`;
  if (err)
    return (
      <Panel title={title}>
        <div style={{ color: "var(--muted)" }}>Replay not available: {err}</div>
      </Panel>
    );
  if (!frames)
    return (
      <Panel title={title}>
        <div style={{ color: "var(--muted)" }}>Replaying…</div>
      </Panel>
    );

  // indexes of the frames to step through
  const idx = frames.map((_, k) => k).filter((k) => k === 0 || !onlyChanges || frames[k].changed);
  const pos = Math.min(i, idx.length - 1);
  const f = frames[idx[pos]];
  return (
    <Panel
      title={title}
      hint={pos === 0 ? "start (before any action)" : `call ${pos} of ${idx.length - 1}`}
      aside={
        <>
          <button className="btn-ghost" disabled={pos === 0} onClick={() => setI(pos - 1)}>
            ← Prev
          </button>
          <button className="btn-ghost" disabled={pos >= idx.length - 1} onClick={() => setI(pos + 1)}>
            Next →
          </button>
        </>
      }
      flush
    >
      <div className="space-y-3 p-4">
        <div className="flex flex-wrap items-center gap-3">
          <input
            type="range"
            min={0}
            max={idx.length - 1}
            value={pos}
            onChange={(e) => setI(Number(e.target.value))}
            className="min-w-40 flex-1 accent-[var(--accent)]"
            aria-label="replay position"
          />
          <label className="hint flex items-center gap-1.5">
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
        </div>
        <div className="code-block" style={{ color: f.call ? (f.error ? "var(--bad)" : "var(--accent)") : "var(--muted)" }}>
          {f.call
            ? `${f.call.name}(${Object.entries(f.call.args)
                .map(([k, v]) => `${k}=${JSON.stringify(v)}`)
                .join(", ")})`
            : "start (before any action)"}
          {f.changed && <span style={{ color: "var(--warn)" }}> · changed the world</span>}
        </div>
        <DiagramView diagram={f.diagram} height={420} />
        {f.observation && (
          <details open className="border-t pt-3" style={{ borderColor: "var(--line)" }}>
            <summary className="label mb-0 cursor-pointer">observation</summary>
            <pre className="code-block mt-2 max-h-48 overflow-auto">{f.observation}</pre>
          </details>
        )}
        {traj && <Verdict traj={traj} />}
      </div>
    </Panel>
  );
}

/** What usually follows a finished run, depending on who ran (teacher / student / adapter) and on which seeds. */
function RunNextStep({ run, wsId }: { run: Run; wsId: string }) {
  const role = modelRole(run.model);
  const again = (o: { model?: string; seed?: number; label?: string }) => newRunHref({ pack: run.pack, episodes: run.episodes, ...o });
  if (role === "demo")
    return (
      <NextStep title="Try a real model on the same tasks" actions={<><Link href="/account#keys" className="btn-ghost">Add a provider key</Link><Link href={again({ seed: run.seed_start })} className="btn">Run another model</Link></>}>
        The scripted model always wins: it replays the reference solution. A real model shows what an agent actually manages.
      </NextStep>
    );
  if (role === "teacher" && isTrainingRun(run))
    return (
      <NextStep
        title="Turn these solves into training data"
        actions={<><ExportButton wsId={wsId} runIds={[run.id]} /><Link href="/train" className="btn">Continue in Train a model</Link></>}
      >
        This run used training seeds, so its clean solves can teach an open model without leaking into the evaluation.
      </NextStep>
    );
  if (role === "teacher")
    return (
      <NextStep
        title="Compare it, or use it as a teacher"
        actions={<><Link href={again({ seed: run.seed_start })} className="btn-ghost">Run another model on the same tasks</Link><Link href={again({ model: run.model, seed: TRAIN_SEED, label: "teacher · training data" })} className="btn">Collect training data with this model</Link></>}
      >
        Same environment and seeds is the only fair comparison. To train an open model to do this, run it again on training seeds (10000+).
      </NextStep>
    );
  if (role === "student")
    return (
      <NextStep title="This is your baseline. Next: give it something to learn from" actions={<Link href="/train" className="btn">Continue in Train a model</Link>}>
        Collect a strong model&apos;s solutions on training seeds, build a dataset and fine-tune a LoRA adapter for {run.model.slice(5)}.
      </NextStep>
    );
  return (
    <NextStep title="Compare the trained model with its baseline" actions={<Link href="/train" className="btn">See before / after in Train a model</Link>}>
      {run.seed_start === EVAL_SEED ? "Same evaluation seeds as the baseline: the difference is what training taught it." : "Run it on the evaluation seeds (0+) to compare it fairly with the baseline."}
    </NextStep>
  );
}

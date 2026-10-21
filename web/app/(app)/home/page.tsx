"use client";

// Home: "what do you want to do?" the first time, then the step-by-step guide for that
// goal, with each step's status worked out from this workspace's real data.

import Link from "next/link";
import { useEffect, useState } from "react";
import { Journey, useJourneyData } from "@/components/Journey";
import { Empty, Page, PageHeader, Panel } from "@/components/Page";
import { useAuthState } from "@/lib/auth";
import { fmtCost } from "@/lib/api";
import { GOALS, loadGoal, modelRole, saveGoal, type GoalId } from "@/lib/journey";
import { useWorkspace } from "@/lib/workspace";

/** The whole pipeline in one line, so the steps have a picture to hang on. */
function PipelineStrip({ compact = false, upTo = 6 }: { compact?: boolean; upTo?: number }) {
  const stages = [
    { n: "1", t: "Environment", d: "a world with tools, tasks and a scorer" },
    { n: "2", t: "Evaluate", d: "run models, score every episode" },
    { n: "3", t: "Dataset", d: "keep the strong model's clean solves" },
    { n: "4", t: "Train", d: "LoRA fine-tune an open model on a GPU" },
    { n: "5", t: "Test", d: "same tasks again: did it improve?" },
    { n: "6", t: "Use it", d: "an OpenAI-compatible endpoint" },
  ];
  return (
    <ol className={compact ? "grid gap-2" : "grid grid-cols-2 gap-2 md:grid-cols-3 xl:grid-cols-6"} aria-label="How envforge works">
      {stages.map((s, i) => (
        <li key={s.n} className="inset-card relative p-3">
          <div className="label mb-0.5" style={{ color: i < upTo ? "var(--accent)" : "var(--muted)" }}>
            {s.n} · {s.t}
          </div>
          <div className="hint">{s.d}</div>
        </li>
      ))}
    </ol>
  );
}

function GoalChooser({ onPick, current }: { onPick: (g: GoalId) => void; current?: GoalId | null }) {
  return (
    <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
      {GOALS.map((g) => (
        <button
          key={g.id}
          type="button"
          onClick={() => onPick(g.id)}
          aria-pressed={current === g.id}
          className="panel flex flex-col gap-2 p-5 text-left hover:border-[var(--accent)]"
          style={current === g.id ? { borderColor: "var(--accent)" } : undefined}
        >
          <span className="text-lg font-semibold">{g.title}</span>
          <span style={{ color: "var(--muted)" }}>{g.blurb}</span>
          <span className="hint mt-auto pt-2">
            You end up with: <span style={{ color: "var(--fg)" }}>{g.outcome}</span>
          </span>
        </button>
      ))}
    </div>
  );
}

export default function HomePage() {
  const { current: ws } = useWorkspace();
  const { user } = useAuthState();
  const { data, reload } = useJourneyData(ws);
  const [goal, setGoal] = useState<GoalId | null>(null);
  const [choosing, setChoosing] = useState(false);
  const [ready, setReady] = useState(false);

  useEffect(() => {
    if (!ws) return;
    setGoal(loadGoal(ws.id));
    setChoosing(false);
    setReady(true);
  }, [ws]);

  if (!ws || !ready) return null;
  const first = user?.display_name?.split(" ")[0] || "";

  function pick(g: GoalId) {
    saveGoal(ws!.id, g);
    setGoal(g);
    setChoosing(false);
  }

  if (!goal || choosing)
    return (
      <Page>
        <PageHeader
          title={goal ? "Change your goal" : `Welcome${first ? `, ${first}` : ""}. What do you want to do?`}
          description="Pick one and envforge shows you the steps, in order, and what each one produces. You can change it any time; nothing is lost."
          actions={
            goal ? (
              <button className="btn-ghost" onClick={() => setChoosing(false)}>
                Cancel
              </button>
            ) : undefined
          }
        />
        <GoalChooser onPick={pick} current={goal} />
        <Panel title="How it works" hint="The full pipeline. Comparing models needs steps 1–2; training an open model uses all six.">
          <PipelineStrip />
        </Panel>
      </Page>
    );

  const g = GOALS.find((x) => x.id === goal)!;
  const recent = data ? [...data.runs].sort((a, b) => b.created_at.localeCompare(a.created_at)).slice(0, 6) : [];

  return (
    <Page>
      <PageHeader
        title={g.title}
        description={
          <>
            {g.blurb} You end up with {g.outcome}.
          </>
        }
        actions={
          <button className="btn-ghost" onClick={() => setChoosing(true)}>
            Change goal
          </button>
        }
      />
      <div className="grid grid-cols-1 items-start gap-6 xl:grid-cols-[minmax(0,1fr)_22rem]">
        <div className="min-w-0">{data ? <Journey goal={goal} ws={ws} data={data} onChange={() => reload().catch(() => {})} /> : <div className="hint">Loading…</div>}</div>
        <div className="space-y-6">
          <Panel title="Recent runs" flush aside={<Link href="/runs" className="hint hover:text-[var(--fg)]">all runs →</Link>}>
            {recent.length === 0 ? (
              <Empty>Nothing yet. Your runs show up here as they go.</Empty>
            ) : (
              <ul>
                {recent.map((r) => (
                  <li key={r.id} className="border-b last:border-b-0" style={{ borderColor: "var(--line)" }}>
                    <Link href={`/runs/${encodeURIComponent(r.id)}`} className="flex items-center gap-3 px-4 py-2.5 hover:bg-[color-mix(in_srgb,var(--fg)_2.5%,transparent)]">
                      <span className="min-w-0 flex-1">
                        <span className="block truncate">{r.label || r.model}</span>
                        <span className="hint block truncate">
                          {modelRole(r.model)} · {r.pack} · seeds {r.seed_start}+
                          {r.metrics ? ` · ${Math.round(r.metrics.overall.solve_rate * 100)}% solved` : ""}
                          {r.cost_usd ? ` · ${fmtCost(r.cost_usd)}` : ""}
                        </span>
                      </span>
                      <span className={`pill pill-${r.status}`}>{r.status}</span>
                    </Link>
                  </li>
                ))}
              </ul>
            )}
          </Panel>
          <Panel title="The pipeline" hint={goal === "distill" ? "Your goal uses all six stages." : "Your goal covers the highlighted stages."}>
            <PipelineStrip compact upTo={goal === "distill" ? 6 : 2} />
          </Panel>
        </div>
      </div>
    </Page>
  );
}

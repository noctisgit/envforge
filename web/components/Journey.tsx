"use client";

// The step-by-step guide for one goal (lib/journey.ts): a numbered list of steps, each
// with its status and the one action that moves it forward. Steps that still run on the
// command line show the exact commands, filled in with this workspace's runs and models.

import Link from "next/link";
import { useRouter } from "next/navigation";
import { useCallback, useEffect, useMemo, useState } from "react";
import { CodeBlock } from "@/components/CodeBlock";
import { ExportButton } from "@/components/ExportButton";
import { useToast } from "@/components/Toast";
import { Api, type Run, type Workspace } from "@/lib/api";
import {
  EVAL_SEED,
  TRAIN_SEED,
  isEvalRun,
  isTrainingRun,
  loadMarks,
  modelRole,
  saveMark,
  stepsFor,
  type GoalId,
  type JourneyData,
  type Step,
} from "@/lib/journey";

/** Keys, runs, packs and builder projects for the journey; refreshes while a run is in flight. */
export function useJourneyData(ws: Workspace | null) {
  const [data, setData] = useState<JourneyData | null>(null);
  const load = useCallback(async () => {
    if (!ws) return;
    const [keys, runs, packs, projects] = await Promise.all([
      Api.keys().catch(() => []),
      Api.runs(ws.id),
      Api.wsPacks(ws.id).catch(() => []),
      Api.projects(ws.id).catch(() => []),
    ]);
    setData({ keys, runs, packs, projects, marks: loadMarks(ws.id) });
  }, [ws]);
  useEffect(() => {
    setData(null);
    load().catch(() => {});
  }, [load]);
  const active = data?.runs.some((r) => r.status === "queued" || r.status === "running");
  useEffect(() => {
    if (!active) return;
    const t = setInterval(() => load().catch(() => {}), 4000);
    return () => clearInterval(t);
  }, [active, load]);
  return { data, reload: load };
}

/** A link to the Runs page with the new-run form pre-filled. */
export function newRunHref(o: { pack?: string; model?: string; seed?: number; episodes?: number; label?: string }) {
  const q = new URLSearchParams();
  if (o.pack) q.set("pack", o.pack);
  if (o.model) q.set("model", o.model);
  if (o.seed !== undefined) q.set("seed", String(o.seed));
  if (o.episodes) q.set("episodes", String(o.episodes));
  if (o.label) q.set("label", o.label);
  return `/runs?${q.toString()}#new-run`;
}

const compareHref = (rs: Run[]) => `/compare?runs=${rs.map((r) => encodeURIComponent(r.id)).join(",")}`;
const latest = (rs: Run[]) => [...rs].sort((a, b) => b.created_at.localeCompare(a.created_at));

/** The newest completed run per model, so a comparison has one column per model. */
function onePerModel(rs: Run[]) {
  const seen = new Set<string>();
  return latest(rs).filter((r) => (seen.has(r.model) ? false : (seen.add(r.model), true)));
}

const DOT: Record<Step["status"], { bg: string; fg: string; border: string }> = {
  done: { bg: "var(--good)", fg: "var(--on-accent)", border: "var(--good)" },
  current: { bg: "var(--accent)", fg: "var(--on-accent)", border: "var(--accent)" },
  todo: { bg: "transparent", fg: "var(--muted)", border: "var(--line)" },
  optional: { bg: "transparent", fg: "var(--muted)", border: "var(--line)" },
};

const STATUS_LABEL: Record<Step["status"], string> = { done: "done", current: "do this next", todo: "to do", optional: "optional" };

export function Journey({ goal, ws, data, onChange }: { goal: GoalId; ws: Workspace; data: JourneyData; onChange: () => void }) {
  const steps = useMemo(() => stepsFor(goal, data), [goal, data]);
  const [openId, setOpenId] = useState<string | null>(null);
  const current = steps.find((s) => s.status === "current")?.id ?? null;
  const shown = openId ?? current;
  const doneCount = steps.filter((s) => s.status === "done").length;

  const mark = (id: string, on: boolean) => {
    saveMark(ws.id, id, on);
    onChange();
  };

  return (
    <div className="space-y-3">
      <div className="flex flex-wrap items-center gap-3">
        <div className="h-1.5 min-w-40 flex-1 overflow-hidden rounded-full" style={{ background: "var(--line)" }} aria-hidden>
          <div className="h-full rounded-full" style={{ width: `${(doneCount / steps.length) * 100}%`, background: "var(--good)" }} />
        </div>
        <span className="hint num">
          {doneCount} of {steps.length} steps done
        </span>
      </div>
      <ol className="space-y-2">
        {steps.map((s, i) => {
          const open = s.id === shown;
          const dot = DOT[s.status];
          return (
            <li
              key={s.id}
              className="panel"
              style={s.status === "current" ? { borderColor: "color-mix(in srgb, var(--accent) 55%, var(--line))" } : undefined}
            >
              <button
                type="button"
                className="flex w-full items-start gap-3 px-4 py-3 text-left"
                aria-expanded={open}
                aria-controls={`step-${s.id}`}
                onClick={() => setOpenId(open ? "" : s.id)}
              >
                <span
                  aria-hidden
                  className="mt-0.5 inline-flex h-6 w-6 shrink-0 items-center justify-center rounded-full border text-xs font-semibold"
                  style={{ background: dot.bg, color: dot.fg, borderColor: dot.border }}
                >
                  {s.status === "done" ? "✓" : i + 1}
                </span>
                <span className="min-w-0 flex-1">
                  <span className="flex flex-wrap items-center gap-x-2 gap-y-1">
                    <span className={`font-medium ${s.status === "done" ? "" : ""}`} style={{ color: s.status === "todo" || s.status === "optional" ? "var(--muted)" : "var(--fg)" }}>
                      {s.title}
                    </span>
                    <span
                      className={`pill ${s.status === "done" ? "pill-completed" : s.status === "current" ? "pill-running" : "pill-queued"}`}
                    >
                      {STATUS_LABEL[s.status]}
                    </span>
                    {s.cli && s.status !== "done" && <span className="pill pill-tag">command line</span>}
                  </span>
                  {s.detail && <span className="hint mt-0.5 block [overflow-wrap:anywhere]">{s.detail}</span>}
                </span>
                <span aria-hidden className="mt-1 text-xs" style={{ color: "var(--muted)" }}>
                  {open ? "▴" : "▾"}
                </span>
              </button>
              {open && (
                <div id={`step-${s.id}`} className="space-y-3 border-t px-4 py-4 sm:pl-13" style={{ borderColor: "var(--line)" }}>
                  <p style={{ color: "var(--muted)" }}>{s.why}</p>
                  <StepBody step={s} goal={goal} ws={ws} data={data} onChange={onChange} />
                  {s.manual && (
                    <label className="flex items-center gap-2 text-sm" style={{ color: "var(--muted)" }}>
                      <input type="checkbox" checked={s.status === "done"} onChange={(e) => mark(s.id, e.target.checked)} />
                      I&apos;ve done this
                    </label>
                  )}
                </div>
              )}
            </li>
          );
        })}
      </ol>
    </div>
  );
}

/** The actions (and, for command-line steps, the commands) for one step. */
function StepBody({ step, ws, data, onChange }: { step: Step; goal: GoalId; ws: Workspace; data: JourneyData; onChange: () => void }) {
  const router = useRouter();
  const toast = useToast();
  const [busy, setBusy] = useState(false);
  const done = data.runs.filter((r) => r.status === "completed");
  const evalRuns = done.filter(isEvalRun);
  const teacherEval = latest(evalRuns.filter((r) => modelRole(r.model) === "teacher"));
  const studentEval = latest(evalRuns.filter((r) => modelRole(r.model) === "student"));
  const adapterEval = latest(evalRuns.filter((r) => modelRole(r.model) === "student + adapter"));
  const teacherTrain = latest(done.filter((r) => isTrainingRun(r) && modelRole(r.model) === "teacher"));
  const pack = teacherEval[0]?.pack ?? teacherTrain[0]?.pack ?? ws.default_pack;
  const teacherModel = teacherEval[0]?.model ?? (modelRole(ws.default_model) === "teacher" ? ws.default_model : "anthropic:claude-sonnet-5-5");
  const vllm = data.keys.find((k) => k.provider === "vllm" && k.base_url);
  const base = ws.train_base_model || "Qwen/Qwen2.5-3B-Instruct";
  const datasetName = `${pack.replace(/[^a-z0-9_-]/gi, "-")}-v1`.toLowerCase();
  const markAnd = (id: string) => () => {
    saveMark(ws.id, id, true);
    onChange();
  };

  async function demoRun() {
    setBusy(true);
    try {
      const run = await Api.createRun(ws.id, { model: "scripted", pack: ws.default_pack || "netops", episodes: 5, seed_start: EVAL_SEED, label: "scripted demo" });
      toast({ tone: "success", title: "Demo run queued", message: "It takes a few seconds.", action: { label: "Open run", onClick: () => router.push(`/runs/${encodeURIComponent(run.id)}`) } });
      onChange();
    } catch (e) {
      toast.error("Couldn't queue the demo run", e);
    } finally {
      setBusy(false);
    }
  }

  const keysLink = (
    <Link href="/account#keys" className="btn">
      Add a provider key
    </Link>
  );

  // the shell preamble every command-line step shares: the CLI reads the web app's runs
  const preamble = `# On the envforge server, in the folder where you ran ./install.sh (once: uv sync).
# The web app keeps its runs in a Docker volume; point the CLI at it (needs root):
export ENVFORGE_DATA_DIR=$(docker volume inspect -f '{{ .Mountpoint }}' envforge_envdata)
RUNS=$ENVFORGE_DATA_DIR/workspaces/${ws.slug}/runs`;
  const gpuEnv = `# your GPU box over Tailscale (see docs/COMPUTE.md); drop RUNTIME on a plain VM
export ENVFORGE_GPU_HOST=gpu-1 ENVFORGE_GPU_KEY=~/.ssh/id_ed25519_gpu ENVFORGE_GPU_RUNTIME=pod`;

  switch (step.id) {
    case "key":
      return <div className="flex flex-wrap gap-2">{keysLink}</div>;

    case "demo-run":
      return (
        <div className="flex flex-wrap gap-2">
          <button className="btn" disabled={busy} onClick={demoRun}>
            Run scripted on {ws.default_pack || "netops"} (5 episodes)
          </button>
          <Link href={newRunHref({ model: "scripted" })} className="btn-ghost">
            Choose the settings myself
          </Link>
        </div>
      );

    case "replay": {
      const r = latest(done)[0];
      return r ? (
        <Link href={`/runs/${encodeURIComponent(r.id)}`} className="btn" onClick={markAnd("replay")}>
          Open {r.label || "the latest run"}
        </Link>
      ) : (
        <p className="hint">Finish the demo run first.</p>
      );
    }

    case "envs":
      return (
        <Link href="/environments" className="btn" onClick={markAnd("envs")}>
          Browse environments
        </Link>
      );

    case "first-model":
      return (
        <Link href={newRunHref({ seed: EVAL_SEED })} className="btn">
          Start a run
        </Link>
      );

    case "second-model": {
      const first = latest(evalRuns)[0];
      return (
        <Link href={newRunHref({ pack: first?.pack, seed: EVAL_SEED, episodes: first?.episodes })} className="btn">
          Run another model{first ? ` on ${first.pack}` : ""}
        </Link>
      );
    }

    case "compare": {
      const byPack = new Map<string, Run[]>();
      for (const r of evalRuns) byPack.set(r.pack, [...(byPack.get(r.pack) ?? []), r]);
      const best = [...byPack.values()].map(onePerModel).sort((a, b) => b.length - a.length)[0] ?? [];
      return best.length >= 2 ? (
        <Link href={compareHref(best.slice(0, 4))} className="btn" onClick={markAnd("compare")}>
          Compare {best.length > 4 ? "the latest 4" : best.length} models on {best[0].pack}
        </Link>
      ) : (
        <p className="hint">Run at least two models on the same environment first.</p>
      );
    }

    case "describe":
      return (
        <Link href="/builder" className="btn">
          Open the Builder
        </Link>
      );

    case "tasks":
    case "publish": {
      const p = data.projects[0];
      return p ? (
        <Link href={`/builder/${p.id}${step.id === "tasks" ? "#data" : ""}`} className="btn">
          Open {p.title || p.name}
        </Link>
      ) : (
        <p className="hint">Start a project in the Builder first.</p>
      );
    }

    case "run-custom": {
      const custom = data.packs.find((p) => p.custom);
      return (
        <Link href={newRunHref({ pack: custom?.name, seed: EVAL_SEED })} className="btn">
          Run a model{custom ? ` on ${custom.name}` : ""}
        </Link>
      );
    }

    case "teacher-eval":
      return (
        <div className="flex flex-wrap gap-2">
          <Link href={newRunHref({ pack, model: teacherModel, seed: EVAL_SEED, episodes: 50, label: "teacher · eval" })} className="btn">
            Run {teacherModel.split(":").pop()} on evaluation seeds
          </Link>
          <span className="hint self-center">50 episodes is enough for a first score.</span>
        </div>
      );

    case "gpu":
      return (
        <div className="space-y-3">
          <p>
            Rent a GPU with ~24 GB or more (an L40 or A40 on RunPod) running{" "}
            <span className="mono">ghcr.io/noctisgit/envforge-gpu</span>, join it to your tailnet as <span className="mono">gpu-1</span>, then start vLLM from the
            server:
          </p>
          <CodeBlock
            label="serve the open model"
            code={`${gpuEnv}
uv run envforge compute check
uv run envforge compute vllm-up ${base} --bind 127.0.0.1`}
          />
          <p>
            Then add the vLLM provider with base URL <span className="mono">http://gpu-1.&lt;your-tailnet&gt;.ts.net:8000/v1</span>.
          </p>
          <div className="flex flex-wrap gap-2">
            <Link href="/account#keys" className="btn">
              Add the vLLM URL
            </Link>
          </div>
        </div>
      );

    case "student-eval": {
      const ref = teacherEval[0];
      return (
        <div className="flex flex-wrap gap-2">
          <Link href={newRunHref({ pack, model: `vllm:${base}`, seed: EVAL_SEED, episodes: ref?.episodes ?? 50, label: "student · baseline" })} className="btn">
            Run {base.split("/").pop()} on the same seeds
          </Link>
          {!vllm && <span className="hint self-center">Needs the vLLM provider from the previous step.</span>}
        </div>
      );
    }

    case "teacher-train":
      return (
        <div className="flex flex-wrap gap-2">
          <Link href={newRunHref({ pack, model: teacherModel, seed: TRAIN_SEED, episodes: 200, label: "teacher · training data" })} className="btn">
            Run {teacherModel.split(":").pop()} on training seeds
          </Link>
          <span className="hint self-center">Seeds 10000+ never overlap the evaluation seeds, so the test stays fair. Check the cost estimate in the form first.</span>
        </div>
      );

    case "dataset": {
      const ids = teacherTrain.map((r) => r.id);
      return (
        <div className="space-y-3">
          {ids.length === 0 ? (
            <p className="hint">Collect the teacher&apos;s solutions on training seeds first.</p>
          ) : (
            <>
              <CodeBlock
                label="build the dataset (clean solves, one per task, with a validation split)"
                code={`${preamble}
uv run envforge train dataset ${ids.map((id) => `$RUNS/${id}`).join(" \\\n    ")} \\
    --name ${datasetName} --min-reward 1.0 --workspace ${ws.slug}${
                  studentEval[0]
                    ? `
# make sure the evaluation run shares no task with the dataset
uv run envforge train check-overlap ${datasetName} $RUNS/${studentEval[0].id} --workspace ${ws.slug}`
                    : ""
                }`}
              />
              <div className="flex flex-wrap items-center gap-2">
                <span className="hint">Or just look at the data:</span>
                <ExportButton wsId={ws.id} runIds={ids} />
              </div>
            </>
          )}
        </div>
      );
    }

    case "train":
      return (
        <CodeBlock
          label="fine-tune on the GPU box, then copy the adapter home"
          code={`${preamble}
${gpuEnv}
uv run envforge train sft ${datasetName} --base ${base} --workspace ${ws.slug}
# add --max-steps 20 for a quick smoke test first; --detach to leave it running
uv run envforge train fetch ${datasetName} --workspace ${ws.slug}`}
        />
      );

    case "test": {
      const baseline = studentEval[0];
      const adapter = adapterEval[0];
      return (
        <div className="space-y-3">
          <CodeBlock
            label="serve the base model with your adapter"
            code={`${gpuEnv}
uv run envforge compute vllm-up ${base} --bind 127.0.0.1 --lora ${datasetName} --max-lora-rank 16`}
          />
          <div className="flex flex-wrap gap-2">
            <Link href={newRunHref({ pack, model: `vllm:${datasetName}`, seed: EVAL_SEED, episodes: baseline?.episodes ?? 50, label: "student + adapter" })} className="btn">
              Run vllm:{datasetName} on evaluation seeds
            </Link>
            {baseline && adapter && (
              <Link href={compareHref([baseline, adapter, ...(teacherEval[0] ? [teacherEval[0]] : [])])} className="btn-ghost">
                Compare before / after{teacherEval[0] ? " / teacher" : ""}
              </Link>
            )}
          </div>
        </div>
      );
    }

    case "use": {
      const url = vllm?.base_url ?? "http://gpu-1.<your-tailnet>.ts.net:8000/v1";
      const model = adapterEval[0]?.model.slice(5) ?? datasetName;
      return (
        <div className="space-y-3">
          <p>vLLM speaks the OpenAI API, so any OpenAI client, agent framework or chat UI (e.g. Open WebUI) can use your model:</p>
          <CodeBlock
            label="python"
            code={`from openai import OpenAI

client = OpenAI(base_url="${url}", api_key="<your vLLM key, or anything>")
reply = client.chat.completions.create(
    model="${model}",
    messages=[{"role": "user", "content": "…"}],
)
print(reply.choices[0].message.content)`}
          />
          <p className="hint">
            The adapter files are on the server under <span className="mono">workspaces/{ws.slug}/adapters/{model}/</span> in the data volume. They only
            work on top of <span className="mono">{base}</span>; merge them into it to run with Ollama or LM Studio.
          </p>
        </div>
      );
    }
  }
  return null;
}

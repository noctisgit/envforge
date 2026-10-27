"use client";

// Models: the LoRA adapters trained in this workspace, with where they came from
// (dataset, base model, training job), and what to do with them: serve on a GPU box,
// test on the evaluation seeds (then compare with the teacher and the base model),
// download, or call from your own code.

import Link from "next/link";
import { useCallback, useEffect, useState } from "react";
import { CodeBlock } from "@/components/CodeBlock";
import { useConfirm } from "@/components/ConfirmDialog";
import { GrpoRounds } from "@/components/GrpoRounds";
import { invalidateModelCache } from "@/components/ModelPicker";
import { Empty, Page, PageHeader, Panel } from "@/components/Page";
import { useToast } from "@/components/Toast";
import { Api, Compute, downloadFile, type AdapterRow, type GpuBox, type Run } from "@/lib/api";
import { isEvalRun, modelRole } from "@/lib/journey";
import { usePoll } from "@/lib/poll";
import { useWorkspace } from "@/lib/workspace";

const num = (v: unknown, d = 3) => (typeof v === "number" ? v.toFixed(d) : "–");


/** Newest completed evaluation run on `pack` matching `pred`. */
function latestEval(runs: Run[], pack: string | null, pred: (r: Run) => boolean) {
  return runs
    .filter((r) => r.status === "completed" && isEvalRun(r) && (!pack || r.pack === pack) && pred(r))
    .sort((a, b) => b.created_at.localeCompare(a.created_at))[0];
}

function AdapterCard({
  a,
  wsId,
  boxes,
  runs,
  canManage,
  reload,
}: {
  a: AdapterRow;
  wsId: string;
  boxes: GpuBox[];
  runs: Run[];
  canManage: boolean;
  reload: () => void;
}) {
  const toast = useToast();
  const confirm = useConfirm();
  const [boxId, setBoxId] = useState(a.served_on[0]?.box_id ?? boxes[0]?.id ?? "");
  const [episodes, setEpisodes] = useState(50);
  const box = boxes.find((b) => b.id === boxId);
  const busy = !!box?.job;
  const served = a.served_on[0];
  const plan = a.test_plan;
  const testRun = a.test_run_id ? runs.find((r) => r.id === a.test_run_id) : undefined;
  const teacher = latestEval(runs, a.pack, (r) => modelRole(r.model) === "teacher");
  const baseRun = latestEval(runs, a.pack, (r) => r.model === `vllm:${a.base_model}`);
  const compare = testRun && testRun.status === "completed" ? [testRun, teacher, baseRun].filter(Boolean) as Run[] : [];

  async function act(what: string, fn: () => Promise<unknown>, done: string) {
    try {
      await fn();
      toast.success(done);
      reload();
    } catch (ex) {
      toast.error(`Couldn't ${what}`, ex);
    }
  }

  async function remove() {
    const ok = await confirm({
      title: `Delete model “${a.name}”?`,
      body: "Removes the adapter files from this server. A copy may remain on the GPU box it was trained on.",
      confirmLabel: "Delete",
      tone: "danger",
    });
    if (ok) await act("delete", () => Compute.deleteAdapter(wsId, a.id), "Model deleted");
  }

  const url = served?.endpoint_url ?? "http://<gpu-box>:8000/v1";
  const python = `from openai import OpenAI

client = OpenAI(base_url="${url}", api_key=os.environ["VLLM_API_KEY"])
reply = client.chat.completions.create(
    model="${a.name}",
    messages=[{"role": "user", "content": "…"}],
)
print(reply.choices[0].message.content)`;
  const curl = `curl ${url}/chat/completions \\
  -H "Authorization: Bearer $VLLM_API_KEY" -H "Content-Type: application/json" \\
  -d '{"model": "${a.name}", "messages": [{"role": "user", "content": "…"}]}'`;

  return (
    <section id={a.id} className="panel min-w-0 scroll-mt-20 space-y-4 p-5" aria-label={`Model ${a.name}`}>
      <div className="flex flex-wrap items-start gap-x-4 gap-y-2">
        <div className="min-w-0 flex-1">
          <div className="flex flex-wrap items-center gap-2">
            <h2 className="mono text-lg font-semibold">{a.model_spec}</h2>
            {served ? <span className="pill pill-completed">serving on {served.box_name}</span> : <span className="pill pill-queued">not served</span>}
          </div>
          <div className="hint mt-1 [overflow-wrap:anywhere]">
            LoRA{a.lora_rank ? ` r${a.lora_rank}` : ""} on <span className="mono">{a.base_model}</span> ·{" "}
            {a.method === "grpo" ? (
              <span>
                GRPO from <span className="mono">{a.init_adapter}</span> in {a.pack}, {a.metrics.rounds?.length ?? 0} round{a.metrics.rounds?.length === 1 ? "" : "s"}
              </span>
            ) : (
              <>
                {a.method === "dpo" && (
                  <span>
                    DPO from <span className="mono">{a.init_adapter}</span> on{" "}
                  </span>
                )}
                {a.method !== "dpo" && "trained on "}
                {a.dataset_deleted ? (
                  <span>{a.dataset_name ?? "?"} (deleted)</span>
                ) : (
                  <Link href="/datasets" className="underline">
                    {a.dataset_name}
                  </Link>
                )}{" "}
                ({a.source_runs.length} source run{a.source_runs.length === 1 ? "" : "s"})
              </>
            )}{" "}
            ·{" "}
            <Link href="/training" className="underline">
              training job
            </Link>{" "}
            · {new Date(a.created_at).toLocaleString()}
          </div>
        </div>
        <div className="flex flex-wrap gap-2">
          <button className="btn-ghost" onClick={() => downloadFile(`/api/workspaces/${wsId}/adapters/${a.id}/download`, `${a.name}.tar.gz`).catch((e) => toast.error("Download failed", e))}>
            Download .tar.gz
          </button>
          {canManage && (
            <button className="btn-danger" onClick={remove}>
              Delete
            </button>
          )}
        </div>
      </div>

      {a.method === "grpo" ? (
        <GrpoRounds rounds={a.metrics.rounds ?? []} />
      ) : (
        <dl className="grid grid-cols-2 gap-x-4 gap-y-2 sm:grid-cols-4">
          <div>
            <dt className="label mb-0">{a.method === "dpo" ? "Train pairs" : "Train examples"}</dt>
            <dd className="num">{a.metrics.n_train ?? "–"}</dd>
          </div>
          <div>
            <dt className="label mb-0">Steps</dt>
            <dd className="num">{a.metrics.global_step ?? "–"}</dd>
          </div>
          <div>
            <dt className="label mb-0">Final train loss</dt>
            <dd className="num">{num(a.metrics.final_train_loss)}</dd>
          </div>
          <div>
            <dt className="label mb-0">{a.method === "dpo" ? "Held-out pairs preferred" : "Final eval loss"}</dt>
            <dd className="num">
              {a.method === "dpo"
                ? typeof a.metrics.final_eval_accuracy === "number"
                  ? `${Math.round(a.metrics.final_eval_accuracy * 100)}%`
                  : "–"
                : num(a.metrics.final_eval_loss)}
            </dd>
          </div>
        </dl>
      )}

      <div className="space-y-3 border-t pt-4" style={{ borderColor: "var(--line)" }}>
        <div className="font-semibold">Test it</div>
        <p className="hint">
          {plan?.every_task_once ? (
            <>
              Serves base + adapter on the box (if it isn&apos;t already), then scores it on every held-out task of <span className="mono">{plan.pack}</span> once (
              {plan.episodes} episodes): the newest version of its environment. Score the teacher and the untrained base model the same way (Runs, &quot;every task
              once&quot;) and Compare them on exactly the same tasks.
            </>
          ) : (
            <>
              Serves base + adapter on the box (if it isn&apos;t already), then runs it on the evaluation seeds (0+) of {plan?.pack ?? a.pack ?? "its environment"}:
              the same tasks the teacher and the untrained base model were scored on.
            </>
          )}
        </p>
        {boxes.length === 0 ? (
          <p className="hint">
            <Link href="/settings/gpu-boxes" className="underline">
              Add a GPU box
            </Link>{" "}
            first.
          </p>
        ) : canManage ? (
          <div className="flex flex-wrap items-end gap-2">
            <div className="min-w-0 basis-48">
              <label className="label" htmlFor={`box-${a.id}`}>
                GPU box
              </label>
              <select id={`box-${a.id}`} className="input" value={boxId} onChange={(e) => setBoxId(e.target.value)}>
                {boxes.map((b) => (
                  <option key={b.id} value={b.id}>
                    {b.name} ({b.serve_state})
                  </option>
                ))}
              </select>
            </div>
            {!plan?.every_task_once && (
              <div className="w-28">
                <label className="label" htmlFor={`ep-${a.id}`}>
                  Episodes
                </label>
                <input id={`ep-${a.id}`} className="input" type="number" min={1} max={1000} value={episodes} onChange={(e) => setEpisodes(Number(e.target.value))} />
              </div>
            )}
            <button className="btn btn-field" disabled={busy || !boxId || !a.pack} onClick={() => act("start the test", () => Compute.testAdapter(wsId, a.id, boxId, episodes), "Test queued: serving, then an evaluation run")}>
              {plan?.every_task_once ? "Test on every held-out task" : "Test on eval seeds"}
            </button>
            <button
              className="btn-ghost btn-field"
              disabled={busy || !boxId}
              onClick={() => act("serve it", () => Compute.serveAdapter(wsId, a.id, boxId).then(invalidateModelCache), "Serving queued")}
            >
              Serve only
            </button>
            {busy && <span className="hint self-center">{box?.job?.progress || "box busy…"}</span>}
          </div>
        ) : (
          <p className="hint">Serving and testing on GPU boxes is admin-only on this server.</p>
        )}
        {testRun && (
          <div className="flex flex-wrap items-center gap-2 text-sm">
            <span>Test run:</span>
            <Link href={`/runs/${encodeURIComponent(testRun.id)}`} className="underline">
              {testRun.label || testRun.id}
            </Link>
            <span className={`pill pill-${testRun.status}`}>{testRun.status}</span>
            {testRun.metrics && <span className="num">{Math.round(testRun.metrics.overall.solve_rate * 100)}% solved</span>}
            {compare.length >= 2 && (
              <Link href={`/compare?runs=${compare.map((r) => encodeURIComponent(r.id)).join(",")}`} className="btn btn-sm">
                Compare with {[teacher && "the teacher", baseRun && "the base model"].filter(Boolean).join(" and ")}
              </Link>
            )}
          </div>
        )}
      </div>

      <div className="space-y-3 border-t pt-4" style={{ borderColor: "var(--line)" }}>
        <div className="font-semibold">Use it</div>
        {served ? (
          <p>
            Endpoint <span className="mono [overflow-wrap:anywhere]">{served.endpoint_url}</span>, model <span className="mono">{a.name}</span>. vLLM speaks the
            OpenAI API, so any OpenAI client or agent framework works. Use the vLLM API key you set on the box (a generated key is never shown).
          </p>
        ) : (
          <p className="hint">Serve it on a box to get an endpoint; the snippets fill in its URL.</p>
        )}
        <CodeBlock label="curl" code={curl} />
        <CodeBlock label="python (openai SDK)" code={`import os\n${python}`} />
        <p className="hint">
          The download is a standard PEFT adapter: load it on top of <span className="mono">{a.base_model}</span> with transformers + peft, or with vLLM
          <span className="mono"> --lora-modules</span>. GGUF / Ollama export: coming soon.
        </p>
      </div>
    </section>
  );
}

export default function ModelsPage() {
  const { current: ws } = useWorkspace();
  const toast = useToast();
  const [adapters, setAdapters] = useState<AdapterRow[] | null>(null);
  const [boxes, setBoxes] = useState<GpuBox[]>([]);
  const [canManage, setCanManage] = useState(false);
  const [runs, setRuns] = useState<Run[]>([]);

  const load = useCallback(() => {
    if (!ws) return;
    Promise.all([Compute.adapters(ws.id), Compute.boxes(ws.id), Api.runs(ws.id)])
      .then(([a, b, r]) => {
        setAdapters(a);
        setBoxes(b.boxes);
        setCanManage(b.can_manage);
        setRuns(r);
      })
      .catch((e) => toast.error("Couldn't load models", e));
  }, [ws, toast]);
  useEffect(() => {
    setAdapters(null);
    load();
  }, [load]);
  const active = boxes.some((b) => b.job) || runs.some((r) => r.status === "queued" || r.status === "running");
  usePoll(load, active, 4000);

  if (!ws) return null;
  return (
    <Page>
      <PageHeader
        title="Models"
        description="Open models you fine-tuned: LoRA adapters with their lineage. Test one on the evaluation seeds to see what it learned, serve it, or take it with you."
      />
      {!adapters ? (
        <div className="hint">Loading…</div>
      ) : adapters.length === 0 ? (
        <Panel title="No trained models yet" flush>
          <Empty>
            A model shows up here when a training job finishes.{" "}
            <Link href="/training" className="underline">
              Fine-tune one
            </Link>{" "}
            from a{" "}
            <Link href="/datasets" className="underline">
              dataset
            </Link>{" "}
            on a{" "}
            <Link href="/settings/gpu-boxes" className="underline">
              GPU box
            </Link>
            .
          </Empty>
        </Panel>
      ) : (
        adapters.map((a) => <AdapterCard key={a.id} a={a} wsId={ws.id} boxes={boxes} runs={runs} canManage={canManage} reload={load} />)
      )}
    </Page>
  );
}

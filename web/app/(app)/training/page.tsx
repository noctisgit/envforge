"use client";

// Training: fine-tune an open model with LoRA on one of the workspace's GPU boxes. Three
// methods: SFT (imitate a dataset's solved episodes), DPO (prefer the solved side of
// preference pairs, continuing from an adapter) and GRPO (reinforcement learning in an
// environment: rounds of attempts scored by its verifier). The compute worker uploads the
// data and scripts, starts the job detached and polls it; this page shows status, steps,
// the loss curve (and GRPO's rounds), the log tail, elapsed time and an estimated cost.

import Link from "next/link";
import { useSearchParams } from "next/navigation";
import { Suspense, useCallback, useEffect, useMemo, useState } from "react";
import { useConfirm } from "@/components/ConfirmDialog";
import { GrpoRounds } from "@/components/GrpoRounds";
import { HubModelPicker } from "@/components/HubModelPicker";
import { LineChart, type Series } from "@/components/LineChart";
import { Empty, Facts, Page, PageHeader, Panel } from "@/components/Page";
import { useToast } from "@/components/Toast";
import {
  Api,
  Compute,
  fmtCost,
  fmtDuration,
  type AdapterRow,
  type DatasetRow,
  type GpuBox,
  type Pack,
  type TrainingJob,
  type TrainingList,
  type TrainMethod,
  type TrainParams,
} from "@/lib/api";
import { usePoll } from "@/lib/poll";
import { useWorkspace } from "@/lib/workspace";

const ACTIVE: TrainingJob["status"][] = ["queued", "starting", "running", "fetching"];
const isActive = (j: TrainingJob) => ACTIVE.includes(j.status);
const PILL: Record<TrainingJob["status"], string> = {
  queued: "pill-queued",
  starting: "pill-running",
  running: "pill-running",
  fetching: "pill-running",
  completed: "pill-completed",
  failed: "pill-failed",
  cancelled: "pill-cancelled",
};

const METHODS: { id: TrainMethod; label: string; note: string }[] = [
  { id: "sft", label: "SFT", note: "imitate solved episodes" },
  { id: "dpo", label: "DPO", note: "prefer solved over failed attempts" },
  { id: "grpo", label: "GRPO", note: "learn from its own scored attempts" },
];

type Param = { key: keyof TrainParams; label: string; step: string; hint: string };
const PARAMS: Record<TrainMethod, Param[]> = {
  sft: [
    { key: "epochs", label: "Epochs", step: "0.5", hint: "passes over the data" },
    { key: "lr", label: "Learning rate", step: "0.00001", hint: "2e-4 is the usual LoRA rate" },
    { key: "lora_r", label: "LoRA rank", step: "1", hint: "adapter size (16 is plenty)" },
    { key: "lora_alpha", label: "LoRA alpha", step: "1", hint: "usually 2× the rank" },
    { key: "max_len", label: "Max tokens", step: "256", hint: "longer episodes are dropped" },
    { key: "batch_size", label: "Batch size", step: "1", hint: "sequences per GPU step" },
    { key: "grad_accum", label: "Grad. accumulation", step: "1", hint: "effective batch = batch × this" },
    { key: "max_steps", label: "Max steps", step: "1", hint: "optional: e.g. 20 for a smoke test" },
  ],
  dpo: [
    { key: "beta", label: "Beta", step: "0.01", hint: "how hard to prefer; higher stays nearer the start" },
    { key: "epochs", label: "Epochs", step: "0.5", hint: "passes over the pairs" },
    { key: "lr", label: "Learning rate", step: "0.000001", hint: "lower than SFT's (2e-5)" },
    { key: "max_len", label: "Max tokens", step: "256", hint: "pairs with a longer side are dropped" },
    { key: "grad_accum", label: "Grad. accumulation", step: "1", hint: "pairs per optimizer step" },
    { key: "max_steps", label: "Max steps", step: "1", hint: "optional: e.g. 10 for a smoke test" },
  ],
  grpo: [
    { key: "rounds", label: "Rounds", step: "1", hint: "attempt → score → update, this many times" },
    { key: "tasks_per_round", label: "Tasks per round", step: "1", hint: "different training tasks each round" },
    { key: "group_size", label: "Attempts per task", step: "1", hint: "the group each attempt is compared with" },
    { key: "temperature", label: "Temperature", step: "0.1", hint: "variety between attempts" },
    { key: "lr", label: "Learning rate", step: "0.000001", hint: "small: 1e-5" },
    { key: "kl_beta", label: "KL weight", step: "0.01", hint: "pull back toward the starting adapter" },
    { key: "clip_eps", label: "Clip range", step: "0.05", hint: "max change per update (PPO clip)" },
    { key: "max_len", label: "Max tokens", step: "256", hint: "longer attempts are dropped" },
  ],
};

const PHASE: Record<string, string> = {
  rollout: "serving the current adapter",
  queue_rollout: "starting the next round",
  rollouts: "the model is attempting this round's tasks",
  train: "scoring the attempts and starting the update",
  training: "updating the adapter",
  fetching: "copying the final adapter to the server",
};

function JobDetail({ wsId, id, canManage, onChange }: { wsId: string; id: string; canManage: boolean; onChange: () => void }) {
  const toast = useToast();
  const confirm = useConfirm();
  const [job, setJob] = useState<TrainingJob | null>(null);
  const load = useCallback(() => {
    Compute.training(wsId, id)
      .then(setJob)
      .catch(() => {});
  }, [wsId, id]);
  useEffect(() => {
    setJob(null);
    load();
  }, [load]);
  usePoll(load, !!job && isActive(job), 4000);

  if (!job) return <div className="hint">Loading…</div>;
  const losses = job.losses ?? [];
  const progress = job.total_steps ? Math.min(1, job.step / job.total_steps) : 0;
  const grpo = job.method === "grpo" ? job.grpo : null;

  async function cancel() {
    const ok = await confirm({
      title: `Cancel training ${job!.adapter_name}?`,
      body:
        job!.method === "grpo"
          ? "The current round is stopped at the next poll (its rollout run is cancelled, or the update on the box is stopped). Finished rounds are discarded."
          : "The job is stopped on the GPU box at the next poll. Steps done so far are discarded.",
      confirmLabel: "Cancel training",
      cancelLabel: "Keep training",
      tone: "danger",
    });
    if (!ok) return;
    try {
      setJob(await Compute.cancelTraining(wsId, job!.id));
      onChange();
    } catch (ex) {
      toast.error("Couldn't cancel", ex);
    }
  }

  const series: Series[] =
    job.method === "grpo"
      ? [
          { name: "loss", color: "var(--accent)", points: losses.filter((p) => p.loss !== undefined).map((p) => ({ x: p.step, y: p.loss! })) },
          { name: "KL to start", color: "var(--warn)", points: losses.filter((p) => p.kl !== undefined).map((p) => ({ x: p.step, y: p.kl! })) },
        ]
      : [
          { name: "train loss", color: "var(--accent)", points: losses.filter((p) => p.loss !== undefined).map((p) => ({ x: p.step, y: p.loss! })) },
          { name: "eval loss", color: "var(--warn)", dots: true, points: losses.filter((p) => p.eval_loss !== undefined).map((p) => ({ x: p.step, y: p.eval_loss! })) },
          ...(job.method === "dpo"
            ? [{ name: "chosen preferred", color: "var(--good)", points: losses.filter((p) => p.accuracy !== undefined).map((p) => ({ x: p.step, y: p.accuracy! })) }]
            : []),
        ];

  return (
    <Panel
      title={
        <span className="flex flex-wrap items-center gap-2">
          <span className="mono">{job.adapter_name}</span>
          <span className="pill pill-queued">{job.method.toUpperCase()}</span>
          <span className={`pill ${PILL[job.status]}`}>{job.cancel_requested && isActive(job) ? "cancelling" : job.status}</span>
        </span>
      }
      aside={
        <div className="flex flex-wrap gap-2">
          {job.adapter_id && (
            <Link href={`/models#${job.adapter_id}`} className="btn">
              Open the model
            </Link>
          )}
          {canManage && isActive(job) && !job.cancel_requested && (
            <button className="btn-danger" onClick={cancel}>
              Cancel
            </button>
          )}
        </div>
      }
    >
      <Facts
        items={[
          { label: "Base model", value: <span className="mono">{job.base_model}</span> },
          job.method === "sft"
            ? { label: "Dataset", value: job.dataset_name ?? "deleted" }
            : { label: "Starts from", value: <span className="mono">{job.init_adapter ?? "?"}</span> },
          job.method === "grpo" ? { label: "Environment", value: job.pack ?? "?" } : { label: job.method === "dpo" ? "Pairs" : "GPU box", value: job.method === "dpo" ? (job.dataset_name ?? "deleted") : (job.box_name ?? "removed") },
          grpo
            ? { label: "Round", value: <span className="num">{`${Math.min(grpo.round, grpo.rounds_total)} / ${grpo.rounds_total}`}</span> }
            : { label: "Step", value: <span className="num">{job.total_steps ? `${job.step} / ${job.total_steps}` : job.step || "–"}</span> },
          grpo ? { label: "GPU box", value: job.box_name ?? "removed" } : { label: "Epoch", value: <span className="num">{job.epoch?.toFixed(2) ?? "–"}</span> },
          {
            label: "Elapsed · cost",
            value: (
              <span className="num">
                {job.started_at ? fmtDuration(job.elapsed_s) : "not started"}
                {job.est_cost_usd !== null && ` · ~${fmtCost(job.est_cost_usd)}`}
              </span>
            ),
          },
        ]}
      />
      {job.total_steps ? (
        <div className="h-1.5 overflow-hidden rounded-full" style={{ background: "var(--line)" }} aria-label={`${Math.round(progress * 100)}% done`}>
          <div className="h-full rounded-full" style={{ width: `${progress * 100}%`, background: job.status === "failed" ? "var(--bad)" : "var(--accent)" }} />
        </div>
      ) : null}
      {grpo && isActive(job) && (
        <p className="hint">
          Round {Math.min(grpo.round, grpo.rounds_total)}: {PHASE[grpo.phase] ?? grpo.phase}
          {grpo.phase === "rollouts" && grpo.run_id && (
            <>
              {" "}
              (
              <Link href={`/runs/${encodeURIComponent(grpo.run_id)}`} className="underline">
                watch the run
              </Link>
              )
            </>
          )}
          .
        </p>
      )}
      {!grpo && job.status === "queued" && <p className="hint">Waiting for the compute worker to upload the dataset and start training.</p>}
      {!grpo && job.status === "starting" && <p className="hint">Uploading the dataset and starting training; the first run on a box also downloads the base model.</p>}
      {!grpo && job.status === "fetching" && <p className="hint">Training finished; copying the adapter to the server.</p>}
      {job.error && (
        <pre className="code-block max-h-40 overflow-auto" style={{ color: "var(--bad)" }}>
          {job.error}
        </pre>
      )}
      {grpo && (
        <div>
          <div className="label">Rounds</div>
          <GrpoRounds rounds={grpo.rounds} />
        </div>
      )}
      <div>
        <div className="label">{job.method === "dpo" ? "Loss and preference accuracy" : job.method === "grpo" ? "Update loss and KL (all rounds)" : "Loss"}</div>
        <LineChart series={series} />
      </div>
      <div>
        <div className="label">Log (last lines)</div>
        <pre className="code-block max-h-64 overflow-auto" style={{ whiteSpace: "pre", overflowWrap: "normal" }}>
          {job.log_tail || (isActive(job) ? "no output yet" : "no log")}
        </pre>
        {job.polled_at && isActive(job) && <div className="hint mt-1">last read from the box {new Date(job.polled_at).toLocaleTimeString()}</div>}
      </div>
    </Panel>
  );
}

function TrainingInner() {
  const { current: ws } = useWorkspace();
  const params = useSearchParams();
  const toast = useToast();
  const [list, setList] = useState<TrainingList | null>(null);
  const [datasets, setDatasets] = useState<DatasetRow[]>([]);
  const [boxes, setBoxes] = useState<GpuBox[]>([]);
  const [adapters, setAdapters] = useState<AdapterRow[]>([]);
  const [packs, setPacks] = useState<Pack[]>([]);
  const [selected, setSelected] = useState<string | null>(null);
  const [method, setMethod] = useState<TrainMethod>("sft");
  const [datasetId, setDatasetId] = useState("");
  const [boxId, setBoxId] = useState("");
  const [base, setBase] = useState("");
  const [initAdapter, setInitAdapter] = useState("");
  const [pack, setPack] = useState("");
  const [adapterName, setAdapterName] = useState("");
  const [hp, setHp] = useState<Partial<Record<keyof TrainParams, string>>>({});
  const [showHp, setShowHp] = useState(false);
  const [busy, setBusy] = useState(false);

  const load = useCallback(async () => {
    if (!ws) return;
    const [l, d, b, a] = await Promise.all([Compute.trainings(ws.id), Compute.datasets(ws.id), Compute.boxes(ws.id), Compute.adapters(ws.id)]);
    setList(l);
    setDatasets(d);
    setBoxes(b.boxes);
    setAdapters(a);
    return { l, d, b, a };
  }, [ws]);

  useEffect(() => {
    if (ws) Api.wsPacks(ws.id).then(setPacks).catch(() => {});
  }, [ws]);

  useEffect(() => {
    setList(null);
    load()
      .then((r) => {
        if (!r) return;
        const want = params.get("dataset");
        const wanted = want ? r.d.find((x) => x.id === want) : undefined;
        if (wanted) setMethod(wanted.kind === "preference" ? "dpo" : "sft");
        setDatasetId((cur) => cur || wanted?.id || r.d.find((x) => x.kind !== "preference")?.id || "");
        setBoxId((cur) => cur || r.b.boxes[0]?.id || "");
        setBase((cur) => cur || r.l.defaults.base_model);
        setSelected((cur) => cur || r.l.jobs[0]?.id || null);
      })
      .catch((e) => toast.error("Couldn't load training jobs", e));
  }, [load, params, toast]);

  const anyActive = !!list?.jobs.some(isActive);
  usePoll(() => load().catch(() => {}), anyActive, 5000);

  const wantKind = method === "dpo" ? "preference" : "sft";
  const usable = useMemo(() => datasets.filter((d) => d.kind === wantKind), [datasets, wantKind]);
  const dataset = useMemo(() => usable.find((d) => d.id === datasetId), [usable, datasetId]);
  const box = useMemo(() => boxes.find((b) => b.id === boxId), [boxes, boxId]);
  const init = useMemo(() => adapters.find((a) => a.name === initAdapter), [adapters, initAdapter]);

  // switching method: pick a dataset of the right kind, and default the starting adapter
  useEffect(() => {
    if (method !== "grpo" && !usable.some((d) => d.id === datasetId)) setDatasetId(usable[0]?.id ?? "");
    if (method !== "sft" && !initAdapter && adapters[0]) setInitAdapter(adapters[0].name);
  }, [method, usable, datasetId, adapters, initAdapter]);
  // DPO and GRPO continue an adapter: its base model and (GRPO) environment come with it
  useEffect(() => {
    if (method !== "sft" && init) {
      setBase(init.base_model);
      if (method === "grpo") setPack((cur) => cur || init.pack || "");
    }
  }, [method, init]);

  if (!ws) return null;
  const canManage = !!list?.can_manage;
  const defaults: Partial<TrainParams> = list?.method_defaults?.[method] ?? {};

  async function start(e: React.FormEvent) {
    e.preventDefault();
    setBusy(true);
    try {
      const numbers = Object.fromEntries(
        Object.entries(hp)
          .filter(([k, v]) => v !== undefined && v !== "" && PARAMS[method].some((p) => p.key === k))
          .map(([k, v]) => [k, Number(v)]),
      );
      const job = await Compute.startTraining(ws!.id, {
        method,
        dataset_id: method === "grpo" ? null : datasetId,
        box_id: boxId,
        base_model: base,
        adapter_name: adapterName || undefined,
        init_adapter: method === "sft" ? null : initAdapter,
        pack: method === "grpo" ? pack : null,
        ...numbers,
      });
      toast.success("Training queued", `${job.adapter_name} on ${job.box_name}.`);
      setAdapterName("");
      setSelected(job.id);
      await load();
    } catch (ex) {
      toast.error("Couldn't start training", ex);
    } finally {
      setBusy(false);
    }
  }

  const missing =
    boxes.length === 0
      ? "box"
      : method === "sft" && usable.length === 0
        ? "dataset"
        : method === "dpo" && usable.length === 0
          ? "pairs"
          : method !== "sft" && adapters.length === 0
            ? "adapter"
            : null;
  const suggestedName = method === "grpo" ? (init ? `${init.name}-grpo` : "") : method === "dpo" ? (init ? `${init.name}-dpo` : "") : (dataset?.name ?? "");
  const canStart = !busy && !!boxId && !!base && (method === "grpo" ? !!initAdapter && !!pack : !!datasetId) && (method === "sft" || !!initAdapter);

  return (
    <Page>
      <PageHeader
        title="Fine-tune"
        description={
          <>
            Train a LoRA adapter for an open model on a GPU box. The base model&apos;s weights stay as they are; the adapter is a small add-on (tens of MB) that you serve
            on top of it. Start with <b>SFT</b> on a teacher&apos;s solutions; then go past imitation with <b>DPO</b> (pairs of solved and failed attempts) or{" "}
            <b>GRPO</b> (rounds of the model&apos;s own attempts, scored by the environment).
          </>
        }
      />

      {list && !canManage && (
        <div className="inset-card px-4 py-3 text-sm" role="note">
          Training on GPU boxes is admin-only on this server. You can follow jobs here; an admin can start them.
        </div>
      )}

      {list && canManage && (
        <Panel title="New training job">
          <form onSubmit={start} className="space-y-4">
            <fieldset>
              <legend className="label">Method</legend>
              <div className="flex flex-wrap gap-2" role="radiogroup">
                {METHODS.map((m) => (
                  <label
                    key={m.id}
                    className={`inset-card flex cursor-pointer items-start gap-2 px-3 py-2 text-sm ${method === m.id ? "" : "opacity-70"}`}
                    style={method === m.id ? { borderColor: "var(--accent)" } : undefined}
                  >
                    <input type="radio" name="tr-method" className="mt-1" checked={method === m.id} onChange={() => setMethod(m.id)} />
                    <span>
                      <span className="block font-semibold">{m.label}</span>
                      <span className="hint">{m.note}</span>
                    </span>
                  </label>
                ))}
              </div>
            </fieldset>

            {missing ? (
              <p>
                <span style={{ color: "var(--muted)" }}>
                  {missing === "box" && "Add a GPU box first: training runs there, not on this server."}
                  {missing === "dataset" && "Build a dataset first: training data comes from a strong model's solutions."}
                  {missing === "pairs" && "Build a preference dataset first (Datasets → Preference pairs): solved and failed attempts at the same tasks."}
                  {missing === "adapter" && "Train an SFT adapter first: DPO and GRPO continue from one."}
                </span>{" "}
                {missing === "adapter" ? (
                  <button type="button" className="btn btn-sm ml-1" onClick={() => setMethod("sft")}>
                    Use SFT
                  </button>
                ) : (
                  <Link href={missing === "box" ? "/settings/gpu-boxes" : "/datasets"} className="btn btn-sm ml-1">
                    {missing === "box" ? "Add a GPU box" : "Go to Datasets"}
                  </Link>
                )}
              </p>
            ) : (
              <>
                <div className="grid grid-cols-1 items-start gap-4 md:grid-cols-2 xl:grid-cols-4">
                  {method !== "grpo" && (
                    <div>
                      <label className="label" htmlFor="tr-ds">
                        {method === "dpo" ? "Preference pairs" : "Dataset"}
                      </label>
                      <select id="tr-ds" className="input" value={datasetId} onChange={(e) => setDatasetId(e.target.value)}>
                        {usable.map((d) => (
                          <option key={d.id} value={d.id}>
                            {d.name} ({d.n_train} train)
                          </option>
                        ))}
                      </select>
                    </div>
                  )}
                  {method !== "sft" && (
                    <div>
                      <label className="label" htmlFor="tr-init">
                        Start from adapter
                      </label>
                      <select id="tr-init" className="input" value={initAdapter} onChange={(e) => setInitAdapter(e.target.value)}>
                        {adapters.map((a) => (
                          <option key={a.id} value={a.name}>
                            {a.name} ({a.method.toUpperCase()})
                          </option>
                        ))}
                      </select>
                      <p className="hint mt-1">{method === "dpo" ? "usually your SFT adapter; also the reference DPO measures against" : "the policy to improve, and the KL anchor"}</p>
                    </div>
                  )}
                  {method === "grpo" && (
                    <div>
                      <label className="label" htmlFor="tr-pack">
                        Environment
                      </label>
                      <select id="tr-pack" className="input" value={pack} onChange={(e) => setPack(e.target.value)}>
                        <option value="">pick one…</option>
                        {packs.map((p) => (
                          <option key={p.name} value={p.name}>
                            {p.name}
                          </option>
                        ))}
                      </select>
                      <p className="hint mt-1">training seeds 10000+ only; evaluation tasks stay unseen</p>
                    </div>
                  )}
                  <div>
                    <label className="label" htmlFor="tr-box">
                      GPU box
                    </label>
                    <select id="tr-box" className="input" value={boxId} onChange={(e) => setBoxId(e.target.value)}>
                      {boxes.map((b) => (
                        <option key={b.id} value={b.id}>
                          {b.name} {b.check_result?.gpus[0] ? `· ${b.check_result.gpus[0].name}` : ""} {b.status === "ok" ? "" : `(${b.status === "unknown" ? "not checked" : b.status})`}
                        </option>
                      ))}
                    </select>
                    {box && box.status !== "ok" && (
                      <p className="hint mt-1" style={{ color: "var(--warn)" }}>
                        Check this box first on <Link href="/settings/gpu-boxes" className="underline">GPU boxes</Link>.
                      </p>
                    )}
                    {box && box.serve_state !== "stopped" && method !== "grpo" && (
                      <p className="hint mt-1" style={{ color: "var(--warn)" }}>
                        {box.name} is serving {box.served_model ?? "a model"}: starting training stops vLLM first, since serving and training can&apos;t share the GPU&apos;s
                        memory. Serve the trained adapter afterwards from Models.
                      </p>
                    )}
                    {method === "grpo" && <p className="hint mt-1">GRPO serves and stops vLLM on this box itself, every round.</p>}
                  </div>
                  <div className="md:col-span-2">
                    <label className="label">Base model</label>
                    {method === "sft" ? <HubModelPicker value={base} onChange={setBase} /> : <div className="input mono opacity-80">{base || "comes with the adapter"}</div>}
                  </div>
                  <div>
                    <label className="label" htmlFor="tr-name">
                      Adapter name
                    </label>
                    <input
                      id="tr-name"
                      className="input mono"
                      maxLength={method === "grpo" ? 58 : 63}
                      placeholder={suggestedName || "netops-v1"}
                      value={adapterName}
                      onChange={(e) => setAdapterName(e.target.value.toLowerCase().replace(/[^a-z0-9._-]/g, "-"))}
                    />
                    <p className="hint mt-1">Served as vllm:{adapterName || suggestedName || "<name>"}</p>
                  </div>
                </div>
                {method === "grpo" && (
                  <p className="hint">
                    Each round: serve the adapter, let it attempt {hp.tasks_per_round || defaults.tasks_per_round} training tasks {hp.group_size || defaults.group_size} times each
                    (a normal run you can watch), score every attempt, then update toward the attempts that beat their group. Allow roughly 15–30 min per round on a
                    3B model.
                  </p>
                )}
                <button type="button" className="hint underline" onClick={() => setShowHp((x) => !x)} aria-expanded={showHp}>
                  {showHp ? "Hide" : "Show"} {method === "grpo" ? "loop and update settings" : "hyperparameters"} (defaults are the scripts&apos;)
                </button>
                {showHp && (
                  <div className="grid grid-cols-2 items-start gap-4 md:grid-cols-4">
                    {PARAMS[method].map((p) => (
                      <div key={p.key}>
                        <label className="label" htmlFor={`hp-${p.key}`}>
                          {p.label}
                        </label>
                        <input
                          id={`hp-${p.key}`}
                          className="input num"
                          type="number"
                          step={p.step}
                          placeholder={String(defaults[p.key] ?? "")}
                          value={hp[p.key] ?? ""}
                          onChange={(e) => setHp((x) => ({ ...x, [p.key]: e.target.value }))}
                        />
                        <p className="hint mt-1">{p.hint}</p>
                      </div>
                    ))}
                  </div>
                )}
                <div className="flex flex-wrap items-center justify-end gap-3">
                  {box?.hourly_usd !== null && box?.hourly_usd !== undefined && <span className="hint">{box.name} costs ${box.hourly_usd}/h while it runs</span>}
                  <button className="btn" disabled={!canStart}>
                    {busy ? "Queuing…" : method === "grpo" ? "Start GRPO" : "Start training"}
                  </button>
                </div>
              </>
            )}
          </form>
        </Panel>
      )}

      <div className="grid grid-cols-1 items-start gap-6 xl:grid-cols-[22rem_minmax(0,1fr)]">
        <Panel title="Jobs" flush>
          {!list ? (
            <Empty>Loading…</Empty>
          ) : list.jobs.length === 0 ? (
            <Empty>No training jobs yet. Start one above; progress shows up here as it runs.</Empty>
          ) : (
            <ul>
              {list.jobs.map((j) => (
                <li key={j.id} className="border-b last:border-b-0" style={{ borderColor: "var(--line)" }}>
                  <button
                    type="button"
                    onClick={() => setSelected(j.id)}
                    aria-current={selected === j.id}
                    className="flex w-full items-center gap-3 px-4 py-2.5 text-left hover:bg-[color-mix(in_srgb,var(--fg)_2.5%,transparent)]"
                    style={selected === j.id ? { background: "var(--accent-tint)" } : undefined}
                  >
                    <span className="min-w-0 flex-1">
                      <span className="mono block truncate">{j.adapter_name}</span>
                      <span className="hint block truncate">
                        {j.method.toUpperCase()} · {j.base_model.split("/").pop()} · {j.box_name ?? "?"} ·{" "}
                        {j.method === "grpo" && j.total_steps ? `round ${Math.min(j.step + 1, j.total_steps)}/${j.total_steps}` : j.total_steps ? `${j.step}/${j.total_steps}` : new Date(j.created_at).toLocaleDateString()}
                      </span>
                    </span>
                    <span className={`pill ${PILL[j.status]}`}>{j.status}</span>
                  </button>
                </li>
              ))}
            </ul>
          )}
        </Panel>
        <div className="min-w-0">{selected && <JobDetail key={selected} wsId={ws.id} id={selected} canManage={canManage} onChange={() => load().catch(() => {})} />}</div>
      </div>
    </Page>
  );
}

export default function TrainingPage() {
  return (
    <Suspense fallback={null}>
      <TrainingInner />
    </Suspense>
  );
}

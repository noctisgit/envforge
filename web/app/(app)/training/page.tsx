"use client";

// Training: LoRA supervised fine-tuning of an open model on a saved dataset, on one of
// the workspace's GPU boxes. The compute worker uploads the data and sft.py, starts the
// job detached, and polls it; this page shows status, steps, the loss curve, the log
// tail, elapsed time and an estimated cost, and can cancel it.

import Link from "next/link";
import { useSearchParams } from "next/navigation";
import { Suspense, useCallback, useEffect, useMemo, useState } from "react";
import { useConfirm } from "@/components/ConfirmDialog";
import { HubModelPicker } from "@/components/HubModelPicker";
import { LineChart } from "@/components/LineChart";
import { Empty, Facts, Page, PageHeader, Panel } from "@/components/Page";
import { useToast } from "@/components/Toast";
import { Compute, fmtCost, fmtDuration, type DatasetRow, type GpuBox, type TrainingJob, type TrainingList, type TrainParams } from "@/lib/api";
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

const PARAMS: { key: keyof TrainParams; label: string; step: string; hint: string }[] = [
  { key: "epochs", label: "Epochs", step: "0.5", hint: "passes over the data" },
  { key: "lr", label: "Learning rate", step: "0.00001", hint: "2e-4 is the usual LoRA rate" },
  { key: "lora_r", label: "LoRA rank", step: "1", hint: "adapter size (16 is plenty)" },
  { key: "lora_alpha", label: "LoRA alpha", step: "1", hint: "usually 2× the rank" },
  { key: "max_len", label: "Max tokens", step: "256", hint: "longer episodes are dropped" },
  { key: "batch_size", label: "Batch size", step: "1", hint: "sequences per GPU step" },
  { key: "grad_accum", label: "Grad. accumulation", step: "1", hint: "effective batch = batch × this" },
  { key: "max_steps", label: "Max steps", step: "1", hint: "optional: e.g. 20 for a smoke test" },
];

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

  async function cancel() {
    const ok = await confirm({
      title: `Cancel training ${job!.adapter_name}?`,
      body: "The job is stopped on the GPU box at the next poll. Steps done so far are discarded.",
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

  return (
    <Panel
      title={
        <span className="flex flex-wrap items-center gap-2">
          <span className="mono">{job.adapter_name}</span>
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
          { label: "Dataset", value: job.dataset_name ?? "deleted" },
          { label: "GPU box", value: job.box_name ?? "removed" },
          { label: "Step", value: <span className="num">{job.total_steps ? `${job.step} / ${job.total_steps}` : job.step || "–"}</span> },
          { label: "Epoch", value: <span className="num">{job.epoch?.toFixed(2) ?? "–"}</span> },
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
        <div className="h-1.5 overflow-hidden rounded-full" style={{ background: "var(--line)" }} aria-label={`${Math.round(progress * 100)}% of steps`}>
          <div className="h-full rounded-full" style={{ width: `${progress * 100}%`, background: job.status === "failed" ? "var(--bad)" : "var(--accent)" }} />
        </div>
      ) : null}
      {job.status === "queued" && <p className="hint">Waiting for the compute worker to upload the dataset and start training.</p>}
      {job.status === "starting" && <p className="hint">Uploading the dataset and starting training; the first run on a box also downloads the base model.</p>}
      {job.status === "fetching" && <p className="hint">Training finished; copying the adapter to the server.</p>}
      {job.error && (
        <pre className="code-block max-h-40 overflow-auto" style={{ color: "var(--bad)" }}>
          {job.error}
        </pre>
      )}
      <div>
        <div className="label">Loss</div>
        <LineChart
          series={[
            { name: "train loss", color: "var(--accent)", points: losses.filter((p) => p.loss !== undefined).map((p) => ({ x: p.step, y: p.loss! })) },
            { name: "eval loss", color: "var(--warn)", dots: true, points: losses.filter((p) => p.eval_loss !== undefined).map((p) => ({ x: p.step, y: p.eval_loss! })) },
          ]}
        />
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
  const [selected, setSelected] = useState<string | null>(null);
  const [datasetId, setDatasetId] = useState("");
  const [boxId, setBoxId] = useState("");
  const [base, setBase] = useState("");
  const [adapterName, setAdapterName] = useState("");
  const [hp, setHp] = useState<Partial<Record<keyof TrainParams, string>>>({});
  const [showHp, setShowHp] = useState(false);
  const [busy, setBusy] = useState(false);

  const load = useCallback(async () => {
    if (!ws) return;
    const [l, d, b] = await Promise.all([Compute.trainings(ws.id), Compute.datasets(ws.id), Compute.boxes(ws.id)]);
    setList(l);
    setDatasets(d);
    setBoxes(b.boxes);
    return { l, d, b };
  }, [ws]);

  useEffect(() => {
    setList(null);
    load()
      .then((r) => {
        if (!r) return;
        const want = params.get("dataset");
        setDatasetId((cur) => cur || (want && r.d.some((x) => x.id === want) ? want : r.d[0]?.id) || "");
        setBoxId((cur) => cur || r.b.boxes[0]?.id || "");
        setBase((cur) => cur || r.l.defaults.base_model);
        setSelected((cur) => cur || r.l.jobs[0]?.id || null);
      })
      .catch((e) => toast.error("Couldn't load training jobs", e));
  }, [load, params, toast]);

  const anyActive = !!list?.jobs.some(isActive);
  usePoll(() => load().catch(() => {}), anyActive, 5000);

  const dataset = useMemo(() => datasets.find((d) => d.id === datasetId), [datasets, datasetId]);
  const box = useMemo(() => boxes.find((b) => b.id === boxId), [boxes, boxId]);

  if (!ws) return null;
  const canManage = !!list?.can_manage;

  async function start(e: React.FormEvent) {
    e.preventDefault();
    setBusy(true);
    try {
      const numbers = Object.fromEntries(
        Object.entries(hp)
          .filter(([, v]) => v !== undefined && v !== "")
          .map(([k, v]) => [k, Number(v)]),
      );
      const job = await Compute.startTraining(ws!.id, {
        dataset_id: datasetId,
        box_id: boxId,
        base_model: base,
        adapter_name: adapterName || undefined,
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

  const ready = datasets.length > 0 && boxes.length > 0;

  return (
    <Page>
      <PageHeader
        title="Fine-tune"
        description={
          <>
            Train a LoRA adapter for an open model on one of your datasets, on a GPU box. The base model&apos;s weights stay as they are; the adapter is a small add-on
            (tens of MB) that you serve on top of it. A 7B model on a few hundred examples takes roughly 20–60 minutes.
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
          {!ready ? (
            <div className="space-y-3">
              {datasets.length === 0 && (
                <p>
                  <span style={{ color: "var(--muted)" }}>Build a dataset first: training data comes from a strong model&apos;s solutions.</span>{" "}
                  <Link href="/datasets" className="btn btn-sm ml-1">
                    Go to Datasets
                  </Link>
                </p>
              )}
              {boxes.length === 0 && (
                <p>
                  <span style={{ color: "var(--muted)" }}>Add a GPU box first: training runs there, not on this server.</span>{" "}
                  <Link href="/settings/gpu-boxes" className="btn btn-sm ml-1">
                    Add a GPU box
                  </Link>
                </p>
              )}
            </div>
          ) : (
            <form onSubmit={start} className="space-y-4">
              <div className="grid grid-cols-1 items-start gap-4 md:grid-cols-2 xl:grid-cols-4">
                <div>
                  <label className="label" htmlFor="tr-ds">
                    Dataset
                  </label>
                  <select id="tr-ds" className="input" value={datasetId} onChange={(e) => setDatasetId(e.target.value)}>
                    {datasets.map((d) => (
                      <option key={d.id} value={d.id}>
                        {d.name} ({d.n_train} train)
                      </option>
                    ))}
                  </select>
                </div>
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
                  {box && box.serve_state !== "stopped" && (
                    <p className="hint mt-1" style={{ color: "var(--warn)" }}>
                      {box.name} is serving {box.served_model ?? "a model"}: starting training stops vLLM first, since serving and training can&apos;t share the GPU&apos;s
                      memory. Serve the trained adapter afterwards from Models.
                    </p>
                  )}
                </div>
                <div className="md:col-span-2">
                  <label className="label">Base model</label>
                  <HubModelPicker value={base} onChange={setBase} />
                </div>
                <div>
                  <label className="label" htmlFor="tr-name">
                    Adapter name
                  </label>
                  <input
                    id="tr-name"
                    className="input mono"
                    maxLength={63}
                    placeholder={dataset?.name ?? "netops-v1"}
                    value={adapterName}
                    onChange={(e) => setAdapterName(e.target.value.toLowerCase().replace(/[^a-z0-9._-]/g, "-"))}
                  />
                  <p className="hint mt-1">Served as vllm:{adapterName || dataset?.name || "<name>"}</p>
                </div>
              </div>
              <button type="button" className="hint underline" onClick={() => setShowHp((x) => !x)} aria-expanded={showHp}>
                {showHp ? "Hide" : "Show"} hyperparameters (defaults are the CLI&apos;s)
              </button>
              {showHp && list && (
                <div className="grid grid-cols-2 items-start gap-4 md:grid-cols-4">
                  {PARAMS.map((p) => (
                    <div key={p.key}>
                      <label className="label" htmlFor={`hp-${p.key}`}>
                        {p.label}
                      </label>
                      <input
                        id={`hp-${p.key}`}
                        className="input num"
                        type="number"
                        step={p.step}
                        placeholder={String(list.defaults[p.key] ?? "")}
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
                <button className="btn" disabled={busy || !datasetId || !boxId || !base}>
                  {busy ? "Queuing…" : "Start training"}
                </button>
              </div>
            </form>
          )}
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
                        {j.base_model.split("/").pop()} · {j.box_name ?? "?"} · {j.total_steps ? `${j.step}/${j.total_steps}` : new Date(j.created_at).toLocaleDateString()}
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

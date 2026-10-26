"use client";

// Settings → GPU boxes: the machines this workspace serves open models on and trains
// LoRA adapters on. Add one (logging in with the workspace SSH key, or a pasted key stored
// encrypted; only fingerprints come back), check it, start / stop vLLM. Every action is a
// worker job; the page polls while one runs.

import Link from "next/link";
import { useCallback, useEffect, useState } from "react";
import { CodeBlock } from "@/components/CodeBlock";
import { useConfirm } from "@/components/ConfirmDialog";
import { HubModelPicker } from "@/components/HubModelPicker";
import { invalidateModelCache } from "@/components/ModelPicker";
import { Empty, Page, PageHeader, Panel } from "@/components/Page";
import { SettingsTabs } from "@/components/SettingsTabs";
import { useToast } from "@/components/Toast";
import { Compute, type BoxIn, type BoxesResponse, type GpuBox, type WorkspaceSshKey } from "@/lib/api";
import { usePoll } from "@/lib/poll";
import { useWorkspace } from "@/lib/workspace";

const STATUS_PILL: Record<GpuBox["status"], string> = { unknown: "pill-queued", ok: "pill-completed", degraded: "pill-cancelled", unreachable: "pill-failed" };
const SERVE_PILL: Record<GpuBox["serve_state"], string> = {
  stopped: "pill-queued",
  starting: "pill-running",
  serving: "pill-completed",
  stopping: "pill-running",
  failed: "pill-failed",
};
const JOB_LABEL: Record<string, string> = { box_check: "Checking", box_serve: "Starting vLLM", box_stop: "Stopping vLLM", box_delete: "Stopping vLLM and removing the box", train_start: "Starting training", adapter_test: "Testing adapter" };

const gib = (mib: number | null | undefined) => (mib ? `${(mib / 1024).toFixed(0)} GB` : "?");
const ago = (iso: string | null) => (iso ? new Date(iso).toLocaleString() : "never");

const BLANK: BoxIn = { name: "", host: "", port: 22, user: "root", ssh_private_key: "", runtime: "auto", vllm_port: 8000, vllm_access: null };

// The workspace's own SSH key: paste the public half into the provider once and every new
// pod accepts it, so adding a box needs only its address.
function WorkspaceKeyPanel({ wsId, canManage }: { wsId: string; canManage: boolean }) {
  const toast = useToast();
  const confirm = useConfirm();
  const [key, setKey] = useState<WorkspaceSshKey | null | undefined>(undefined);
  useEffect(() => {
    setKey(undefined);
    Compute.sshKey(wsId)
      .then((r) => setKey(r.key))
      .catch((e) => toast.error("Couldn't load the workspace SSH key", e));
  }, [wsId, toast]);

  async function rotate() {
    const ok = await confirm({
      title: "Replace the workspace SSH key?",
      body: "Boxes that log in with the workspace key will be refused by pods that only have the old public key. Add the new one to RunPod / Vast, then start new pods (or add it to running ones).",
      confirmLabel: "Replace key",
      tone: "danger",
    });
    if (!ok) return;
    try {
      setKey((await Compute.rotateSshKey(wsId)).key);
      toast.success("New workspace SSH key", "Add the new public key to your GPU provider.");
    } catch (ex) {
      toast.error("Couldn't replace the key", ex);
    }
  }

  if (key === undefined || (key === null && !canManage)) return null;
  return (
    <Panel title="Workspace SSH key" hint="Add this public key to your GPU provider once; every new pod then lets envforge in, with nothing to paste per box.">
      {key === null ? (
        <p className="hint">No key yet.</p>
      ) : (
        <div className="space-y-3">
          <CodeBlock code={key.public_key} label="Public key" wrap />
          <ul className="hint list-disc space-y-1 pl-5">
            <li>
              <b>RunPod:</b> Settings → SSH Public Keys → paste, save. Pods started from then on accept it (expose TCP port 22 if you don&apos;t use Tailscale).
            </li>
            <li>
              <b>Vast:</b> Account → Keys → add SSH key.
            </li>
            <li>
              <b>Your own VM:</b> append it to <span className="mono">~/.ssh/authorized_keys</span>.
            </li>
          </ul>
          <div className="flex flex-wrap items-center gap-3">
            <span className="mono hint [overflow-wrap:anywhere]">{key.fingerprint}</span>
            {canManage && (
              <button type="button" className="btn-ghost btn-sm" onClick={rotate}>
                Replace key
              </button>
            )}
          </div>
        </div>
      )}
    </Panel>
  );
}

function AddBoxForm({ wsId, onDone, onCancel }: { wsId: string; onDone: () => void; onCancel?: () => void }) {
  const toast = useToast();
  const [f, setF] = useState<BoxIn>(BLANK);
  const [hourly, setHourly] = useState("");
  const [advanced, setAdvanced] = useState(false);
  const [ownKey, setOwnKey] = useState(false);
  const [busy, setBusy] = useState(false);
  const set = <K extends keyof BoxIn>(k: K, v: BoxIn[K]) => setF((x) => ({ ...x, [k]: v }));

  async function submit(e: React.FormEvent) {
    e.preventDefault();
    setBusy(true);
    try {
      await Compute.addBox(wsId, {
        ...f,
        ssh_private_key: ownKey ? f.ssh_private_key : null,
        endpoint_host: f.endpoint_host || null,
        bind: f.bind || null,
        vllm_api_key: f.vllm_api_key || null,
        hf_token: f.hf_token || null,
        hourly_usd: hourly.trim() ? Number(hourly) : null,
      });
      toast.success("GPU box added", "Run a check to see its GPU and whether vLLM is ready.");
      setF(BLANK); // the key leaves the page's memory as soon as it's stored
      setHourly("");
      onDone();
    } catch (ex) {
      toast.error("Couldn't add the box", ex);
    } finally {
      setBusy(false);
    }
  }

  return (
    <form onSubmit={submit} className="space-y-4" aria-label="Add a GPU box" autoComplete="off">
      <div className="grid grid-cols-1 items-start gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <div>
          <label className="label" htmlFor="box-name">Name</label>
          <input id="box-name" className="input" required maxLength={63} placeholder="gpu-1" value={f.name} onChange={(e) => set("name", e.target.value.toLowerCase())} />
        </div>
        <div className="sm:col-span-2 lg:col-span-1">
          <label className="label" htmlFor="box-host">Host</label>
          <input id="box-host" className="input mono" required placeholder="100.x.y.z (Tailscale) or the pod's public IP" value={f.host} onChange={(e) => set("host", e.target.value.trim())} />
        </div>
        <div>
          <label className="label" htmlFor="box-port">SSH port</label>
          <input id="box-port" className="input" type="number" min={1} max={65535} value={f.port} onChange={(e) => set("port", Number(e.target.value))} />
        </div>
        <div>
          <label className="label" htmlFor="box-user">User</label>
          <input id="box-user" className="input mono" value={f.user} onChange={(e) => set("user", e.target.value.trim())} />
        </div>
      </div>
      <div>
        <label className="label" htmlFor="box-login">SSH login</label>
        <select id="box-login" className="input" value={ownKey ? "own" : "workspace"} onChange={(e) => setOwnKey(e.target.value === "own")}>
          <option value="workspace">Workspace SSH key (added to your provider once)</option>
          <option value="own">A private key for this box only</option>
        </select>
        {ownKey ? (
          <>
            <textarea
              id="box-key"
              aria-label="SSH private key"
              className="input mono mt-2 h-28 text-xs"
              required
              spellCheck={false}
              autoComplete="off"
              placeholder={"-----BEGIN OPENSSH PRIVATE KEY-----\n…"}
              value={f.ssh_private_key ?? ""}
              onChange={(e) => set("ssh_private_key", e.target.value)}
            />
            <p className="hint mt-1">
              Use a dedicated key without a passphrase, authorised only on this box. It&apos;s stored encrypted and only the worker decrypts it; this page will
              only ever show its fingerprint.
            </p>
          </>
        ) : (
          <p className="hint mt-1">The box must accept the workspace public key shown above (RunPod / Vast add it to every pod once it&apos;s in your account).</p>
        )}
      </div>
      <div className="grid grid-cols-1 items-start gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <div>
          <label className="label" htmlFor="box-runtime">Runtime</label>
          <select id="box-runtime" className="input" value={f.runtime} onChange={(e) => set("runtime", e.target.value as BoxIn["runtime"])}>
            <option value="auto">auto-detect</option>
            <option value="docker">docker (VM / bare metal)</option>
            <option value="pod">pod (RunPod / Vast, envforge GPU image)</option>
          </select>
        </div>
        <div className="sm:col-span-2">
          <label className="label" htmlFor="box-access">How envforge reaches vLLM</label>
          <select id="box-access" className="input" value={f.vllm_access ?? ""} onChange={(e) => set("vllm_access", (e.target.value || null) as BoxIn["vllm_access"])}>
            <option value="">auto: direct on Tailscale, SSH tunnel for a public IP</option>
            <option value="direct">direct (Tailscale, or an open port)</option>
            <option value="tunnel">SSH tunnel (only the SSH port is reachable)</option>
          </select>
        </div>
        <div>
          <label className="label" htmlFor="box-price">$ / hour (optional)</label>
          <input id="box-price" className="input" inputMode="decimal" placeholder="0.79" value={hourly} onChange={(e) => setHourly(e.target.value)} />
        </div>
      </div>
      <button type="button" className="hint underline" onClick={() => setAdvanced((a) => !a)} aria-expanded={advanced}>
        {advanced ? "Hide" : "Show"} serving options
      </button>
      {advanced && (
        <div className="grid grid-cols-1 items-start gap-4 sm:grid-cols-2 lg:grid-cols-3">
          <div>
            <label className="label" htmlFor="box-vport">vLLM port</label>
            <input id="box-vport" className="input" type="number" min={1} max={65535} value={f.vllm_port} onChange={(e) => set("vllm_port", Number(e.target.value))} />
          </div>
          <div>
            <label className="label" htmlFor="box-ehost">Endpoint host</label>
            <input id="box-ehost" className="input mono" placeholder="defaults to the host" value={f.endpoint_host ?? ""} onChange={(e) => set("endpoint_host", e.target.value.trim())} />
          </div>
          <div>
            <label className="label" htmlFor="box-bind">Bind address</label>
            <input id="box-bind" className="input mono" placeholder="all interfaces (e.g. 127.0.0.1 on a pod)" value={f.bind ?? ""} onChange={(e) => set("bind", e.target.value.trim())} />
          </div>
          <div>
            <label className="label" htmlFor="box-vkey">vLLM API key</label>
            <input id="box-vkey" className="input mono" type="password" autoComplete="new-password" placeholder="generated if empty" value={f.vllm_api_key ?? ""} onChange={(e) => set("vllm_api_key", e.target.value)} />
            <p className="hint mt-1">Set your own to call the endpoint from outside envforge; a generated one is never shown.</p>
          </div>
          <div>
            <label className="label" htmlFor="box-hf">Hugging Face token</label>
            <input id="box-hf" className="input mono" type="password" autoComplete="new-password" placeholder="optional: else your account token (Provider keys)" value={f.hf_token ?? ""} onChange={(e) => set("hf_token", e.target.value)} />
          </div>
        </div>
      )}
      <div className="flex flex-wrap justify-end gap-2">
        {onCancel && (
          <button type="button" className="btn-ghost" onClick={onCancel}>
            Cancel
          </button>
        )}
        <button className="btn" disabled={busy || !f.name || !f.host || (ownKey && !f.ssh_private_key)}>
          {busy ? "Adding…" : "Add box"}
        </button>
      </div>
    </form>
  );
}

// A random key in the browser: 32 bytes as base64url (what `secrets.token_urlsafe(32)` gives).
function randomKey(): string {
  const bytes = crypto.getRandomValues(new Uint8Array(32));
  return btoa(String.fromCharCode(...bytes)).replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/, "");
}

// The key vLLM requires. Set your own to call the endpoint yourself (curl, an app); it is
// shown here only while you type or generate it, then stored encrypted and never returned.
function VllmKey({ box, wsId, canManage, reload }: { box: GpuBox; wsId: string; canManage: boolean; reload: () => void }) {
  const toast = useToast();
  const [editing, setEditing] = useState(false);
  const [key, setKey] = useState("");
  const [copied, setCopied] = useState(false);
  const [busy, setBusy] = useState(false);
  const stopped = box.serve_state === "stopped" || box.serve_state === "failed";

  async function save() {
    setBusy(true);
    try {
      await Compute.setVllmKey(wsId, box.id, key);
      toast.success("vLLM API key changed", "It takes effect the next time vLLM starts on this box.");
      setKey("");
      setEditing(false);
      reload();
    } catch (ex) {
      toast.error("Couldn't change the key", ex);
    } finally {
      setBusy(false);
    }
  }

  async function copy() {
    try {
      await navigator.clipboard.writeText(key);
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    } catch {
      /* clipboard blocked: the key is still selectable */
    }
  }

  return (
    <div className="space-y-2">
      <div className="flex flex-wrap items-center gap-2 text-sm">
        <span className="label mb-0">vLLM API key</span>
        <span className="mono">{box.vllm_key ?? "none"}</span>
        <span className="hint">{box.vllm_key_generated ? "generated by envforge, never shown" : "set by you"}</span>
        {canManage && !editing && (
          <button type="button" className="btn-ghost btn-sm" disabled={!stopped || !!box.job} onClick={() => setEditing(true)}>
            Change
          </button>
        )}
        {canManage && !stopped && !editing && <span className="hint">stop vLLM to change it</span>}
      </div>
      {editing && (
        <div className="space-y-2">
          <div className="flex flex-wrap items-center gap-2">
            <input
              className="input mono min-w-0 flex-1 basis-72"
              aria-label="New vLLM API key"
              autoComplete="off"
              spellCheck={false}
              placeholder="at least 16 characters, no spaces"
              value={key}
              onChange={(e) => setKey(e.target.value.trim())}
            />
            <button type="button" className="btn-ghost btn-sm" onClick={() => setKey(randomKey())}>
              Generate
            </button>
            <button type="button" className="btn-ghost btn-sm" disabled={!key} onClick={copy}>
              {copied ? "Copied" : "Copy"}
            </button>
            <button type="button" className="btn btn-sm" disabled={busy || key.length < 16} onClick={save}>
              {busy ? "Saving…" : "Save"}
            </button>
            <button
              type="button"
              className="btn-ghost btn-sm"
              onClick={() => {
                setKey("");
                setEditing(false);
              }}
            >
              Cancel
            </button>
          </div>
          <p className="hint">
            Copy it now: after saving, envforge only shows its last characters. Use it as <span className="mono">Authorization: Bearer &lt;key&gt;</span> when you call
            the endpoint yourself. The new key applies the next time vLLM starts here.
          </p>
        </div>
      )}
    </div>
  );
}

function CheckFacts({ box }: { box: GpuBox }) {
  const r = box.check_result;
  if (!r) return <p className="hint">Not checked yet. A check connects over SSH and reports the GPU, driver, disk and whether vLLM is ready.</p>;
  const gpu = r.gpus[0];
  const reached = r.checks[0]?.ok;  // the first check is ssh itself: without it nothing below is known
  const vllm = !reached ? "?" : r.runtime === "pod" ? (r.vllm_python ? `vLLM ${r.vllm_python}` : "not importable") : r.vllm_image ? "image pulled" : "image not pulled yet";
  const facts = [
    { label: "GPU", value: gpu ? `${r.gpus.length > 1 ? `${r.gpus.length}× ` : ""}${gpu.name}` : "?" },
    { label: "VRAM", value: gpu ? gib(gpu.memory_total_mib) : "?" },
    { label: "Driver / CUDA", value: `${r.driver ?? "?"} / ${r.cuda ?? "?"}` },
    { label: "Free disk", value: r.disk_free_gib !== null ? `${r.disk_free_gib} GB` : "?" },
    { label: "vLLM", value: vllm },
    { label: "Runtime", value: r.runtime },
  ];
  return (
    <div className="space-y-3">
      <dl className="grid grid-cols-2 gap-x-4 gap-y-2 sm:grid-cols-3 xl:grid-cols-6">
        {facts.map((x) => (
          <div key={x.label} className="min-w-0">
            <dt className="label mb-0">{x.label}</dt>
            <dd className="[overflow-wrap:anywhere]">{x.value}</dd>
          </div>
        ))}
      </dl>
      <ul className="flex flex-wrap gap-2">
        {r.checks.map((c) => (
          <li key={c.name} className={`pill ${c.ok ? "pill-completed" : "pill-failed"}`} title={c.detail}>
            {c.ok ? "✓" : "×"} {c.name}
          </li>
        ))}
      </ul>
      {r.checks.filter((c) => !c.ok && !c.detail.startsWith("skipped")).map((c) => (
        <p key={c.name} className="hint [overflow-wrap:anywhere]" style={{ color: "var(--bad)" }}>
          {c.name}: {c.detail}
        </p>
      ))}
    </div>
  );
}

function BoxCard({ box, wsId, defaultModel, canManage, reload }: { box: GpuBox; wsId: string; defaultModel: string; canManage: boolean; reload: () => void }) {
  const toast = useToast();
  const confirm = useConfirm();
  const [model, setModel] = useState(box.served_model || defaultModel);
  const busy = !!box.job;

  async function act(what: string, fn: () => Promise<unknown>) {
    try {
      await fn();
      reload();
    } catch (ex) {
      toast.error(`Couldn't ${what}`, ex);
    }
  }

  async function remove() {
    const ok = await confirm({
      title: `Remove GPU box “${box.name}”?`,
      body: "envforge connects to the box, stops vLLM on it, then forgets the box and its key. Adapters and files on the machine stay; delete the rental to remove them.",
      confirmLabel: "Stop vLLM and remove",
      tone: "danger",
    });
    if (ok) await act("remove the box", () => Compute.deleteBox(wsId, box.id));
  }

  async function forget() {
    const ok = await confirm({
      title: `Forget “${box.name}” without stopping vLLM?`,
      body: "envforge couldn't reach the box to stop vLLM. Forgetting it removes the box and its key from envforge only: if vLLM is still running there, stop it on the machine or delete the rental.",
      confirmLabel: "Forget anyway",
      tone: "danger",
    });
    if (ok) await act("forget the box", () => Compute.deleteBox(wsId, box.id, true));
  }

  return (
    <section className="panel min-w-0 space-y-4 p-5" aria-label={`GPU box ${box.name}`}>
      <div className="flex flex-wrap items-start gap-x-4 gap-y-2">
        <div className="min-w-0 flex-1">
          <div className="flex flex-wrap items-center gap-2">
            <h2 className="text-lg font-semibold">{box.name}</h2>
            <span className={`pill ${STATUS_PILL[box.status]}`}>{box.status === "unknown" ? "not checked" : box.status}</span>
            <span className={`pill ${SERVE_PILL[box.serve_state]}`}>vLLM {box.serve_state}</span>
          </div>
          <div className="mono hint mt-1 [overflow-wrap:anywhere]">
            {box.user}@{box.host}:{box.port} · {box.ssh_key_source === "workspace" ? "workspace key" : "key"} {box.ssh_key_fingerprint}
            {box.vllm_access === "tunnel" && " · vLLM via SSH tunnel"} · checked {ago(box.checked_at)}
            {box.hourly_usd !== null && ` · $${box.hourly_usd}/h`}
          </div>
        </div>
        {canManage && (
          <div className="flex flex-wrap gap-2">
            <button className="btn-ghost" disabled={busy} onClick={() => act("queue a check", () => Compute.checkBox(wsId, box.id))}>
              Check
            </button>
            <button className="btn-danger" disabled={busy} onClick={remove}>
              Remove
            </button>
            {box.delete_failed && (
              <button className="btn-danger" disabled={busy} onClick={forget}>
                Forget anyway
              </button>
            )}
          </div>
        )}
      </div>

      {box.job && (
        <div className="inset-card px-3 py-2 text-sm" role="status">
          <span style={{ color: "var(--accent)" }}>{JOB_LABEL[box.job.kind] ?? box.job.kind}…</span> {box.job.status === "queued" ? "waiting for the compute worker" : box.job.progress}
        </div>
      )}

      <CheckFacts box={box} />

      <div className="space-y-3 border-t pt-4" style={{ borderColor: "var(--line)" }}>
        <div className="font-semibold">Serving</div>
        <VllmKey box={box} wsId={wsId} canManage={canManage} reload={reload} />
        {box.serve_state === "serving" || box.serve_state === "stopping" ? (
          <div className="space-y-2">
            <div>
              <span className="mono">{box.served_model}</span>
              {box.served_adapters.length > 0 && <span className="hint"> + LoRA {box.served_adapters.join(", ")}</span>}
            </div>
            <div className="mono text-sm [overflow-wrap:anywhere]">
              {box.endpoint_url}
              {box.vllm_access === "tunnel" && <span className="hint"> on the box, reached through an SSH tunnel</span>}
            </div>
            <p className="hint">
              In the model picker as <span className="mono">vllm:{box.served_model}</span>
              {box.served_adapters.map((a) => (
                <span key={a}>
                  , <span className="mono">vllm:{a}</span>
                </span>
              ))}
              . Runs use this box&apos;s endpoint and key; no vLLM key of your own needed.{" "}
              <Link href="/runs#new-run" className="underline">
                Start a run
              </Link>
            </p>
            {canManage && (
              <button className="btn-ghost" disabled={busy} onClick={() => act("stop vLLM", () => Compute.stop(wsId, box.id))}>
                Stop vLLM
              </button>
            )}
          </div>
        ) : (
          <div className="space-y-2">
            {box.serve_state === "failed" && box.serve_error && (
              <pre className="code-block max-h-48 overflow-auto" style={{ color: "var(--bad)" }}>
                {box.serve_error}
              </pre>
            )}
            {canManage ? (
              <div className="flex flex-wrap items-end gap-2">
                <div className="min-w-0 flex-1 basis-72">
                  <label className="label">Open model (Hugging Face id)</label>
                  <HubModelPicker value={model} onChange={setModel} disabled={busy} />
                </div>
                <button className="btn btn-field" disabled={busy || !model} onClick={() => act("start vLLM", () => Compute.serve(wsId, box.id, { model }).then(invalidateModelCache))}>
                  Serve
                </button>
              </div>
            ) : (
              <p className="hint">Not serving. An admin can start vLLM here.</p>
            )}
            <p className="hint">
              Starts vLLM on port {box.vllm_port}; the first start downloads the weights, which can take a while. Trained adapters are served from the{" "}
              <Link href="/models" className="underline">
                Models
              </Link>{" "}
              page.
            </p>
          </div>
        )}
      </div>
    </section>
  );
}

export default function GpuBoxesPage() {
  const { current: ws } = useWorkspace();
  const [data, setData] = useState<BoxesResponse | null>(null);
  const [adding, setAdding] = useState(false);
  const toast = useToast();

  const load = useCallback(() => {
    if (!ws) return;
    Compute.boxes(ws.id)
      .then((d) => {
        setData((prev) => {
          // a box just started (or stopped) serving: the model picker list changed
          if (prev && JSON.stringify(prev.boxes.map((b) => b.serve_state)) !== JSON.stringify(d.boxes.map((b) => b.serve_state))) invalidateModelCache();
          return d;
        });
      })
      .catch((e) => toast.error("Couldn't load GPU boxes", e));
  }, [ws, toast]);
  useEffect(() => {
    setData(null);
    load();
  }, [load]);
  const active = !!data?.boxes.some((b) => b.job || b.serve_state === "starting" || b.serve_state === "stopping");
  usePoll(load, active, 3000);

  if (!ws) return null;
  const boxes = data?.boxes ?? [];
  const canManage = !!data?.can_manage;

  return (
    <Page>
      <PageHeader
        title="Workspace settings"
        description={
          <>
            GPU boxes are the machines <span className="mono">{ws.name}</span> serves open models and trains adapters on. Rent one (RunPod, Vast, Lambda…), reach
            it over Tailscale or plain SSH, and add it here. docs/COMPUTE.md has the details.
          </>
        }
        actions={
          canManage && boxes.length > 0 && !adding ? (
            <button className="btn" onClick={() => setAdding(true)}>
              Add a GPU box
            </button>
          ) : undefined
        }
      />
      <SettingsTabs active="gpu" />

      {data && !canManage && (
        <div className="inset-card px-4 py-3 text-sm" role="note">
          GPU boxes are admin-only on this server{data.admin_only ? " (GPU_BOXES_ADMIN_ONLY)" : ""}: you can see the boxes in this workspace and run models they
          serve, but only an admin can add, check, serve on or train on them.
        </div>
      )}

      {!data ? (
        <div className="hint">Loading…</div>
      ) : boxes.length === 0 && !adding ? (
        <Panel title="No GPU box yet">
          <div className="space-y-3">
            <p style={{ color: "var(--muted)" }}>
              Add a GPU box first: it&apos;s where open models are served with vLLM (so you can evaluate them like any other model) and where LoRA adapters are
              trained. A 24 GB+ GPU (L4, A10G, L40S, A100) running Docker with the NVIDIA toolkit, or a RunPod/Vast pod with the envforge GPU image, works.
            </p>
            {canManage ? (
              <button className="btn" onClick={() => setAdding(true)}>
                Add a GPU box
              </button>
            ) : (
              <p className="hint">Ask an admin of this server to add one.</p>
            )}
          </div>
        </Panel>
      ) : null}

      {data && <WorkspaceKeyPanel wsId={ws.id} canManage={canManage} />}

      {adding && canManage && (
        <Panel title="Add a GPU box" hint="Host, SSH access and (optionally) what it costs per hour, for training cost estimates.">
          <AddBoxForm
            wsId={ws.id}
            onDone={() => {
              setAdding(false);
              load();
            }}
            onCancel={() => setAdding(false)}
          />
        </Panel>
      )}

      {boxes.map((b) => (
        <BoxCard key={b.id} box={b} wsId={ws.id} defaultModel={ws.train_base_model} canManage={canManage} reload={load} />
      ))}
      {boxes.length > 0 && (
        <Panel title="What next" flush>
          <Empty>
            Serve an open model and score it on the{" "}
            <Link href="/runs#new-run" className="underline">
              Runs
            </Link>{" "}
            page, then build a{" "}
            <Link href="/datasets" className="underline">
              dataset
            </Link>{" "}
            from a strong model&apos;s solutions and{" "}
            <Link href="/training" className="underline">
              fine-tune
            </Link>{" "}
            on this box.
          </Empty>
        </Panel>
      )}
    </Page>
  );
}

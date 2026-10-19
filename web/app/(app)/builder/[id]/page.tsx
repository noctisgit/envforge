"use client";

// One builder project: chat with the builder on the left, the live project on the right
// (readiness, spec, tasks, data, revisions, preview). The builder runs in the background
// worker; this page polls for new messages while it works.

import Link from "next/link";
import { useParams, useRouter } from "next/navigation";
import { useCallback, useEffect, useRef, useState } from "react";
import { useConfirm } from "@/components/ConfirmDialog";
import { DiagramView } from "@/components/DiagramView";
import { Markdown } from "@/components/Markdown";
import {
  Api,
  ApiError,
  uploadBuilderFile,
  type BuilderMessage,
  type BuilderProject,
  type PackDetail,
  type RevisionDiff,
  type RevisionSummary,
  type TaskPreview,
  type TaskRow,
} from "@/lib/api";
import { useWorkspace } from "@/lib/workspace";

const msgOf = (e: unknown) => (e instanceof ApiError ? e.message : String(e));

const TOOL_LABEL: Record<string, string> = {
  get_state: "Read the project",
  update_spec: "Updated the spec",
  sample_conversations: "Read sample conversations",
  estimate_extraction: "Estimated extraction cost",
  extract_tasks: "Extracted tasks",
  edit_tasks: "Edited tasks",
  validate: "Validated",
  preview_task: "Previewed a task",
};

type Tab = "overview" | "spec" | "tasks" | "data" | "revisions" | "preview";

export default function BuilderProjectPage() {
  const { id } = useParams<{ id: string }>();
  const { current: ws } = useWorkspace();
  const [project, setProject] = useState<BuilderProject | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [tab, setTab] = useState<Tab>("overview");
  const [revKey, setRevKey] = useState(0); // bump to refresh right-hand panels

  const loadProject = useCallback(async () => {
    if (!ws) return;
    try {
      setProject(await Api.project(ws.id, id));
    } catch (e) {
      setErr(msgOf(e));
    }
  }, [ws, id]);

  useEffect(() => {
    loadProject();
  }, [loadProject]);

  if (err) return <div style={{ color: "var(--bad)" }}>{err}</div>;
  if (!ws || !project) return <div style={{ color: "var(--muted)" }}>Loading…</div>;

  const refresh = async () => {
    await loadProject();
    setRevKey((k) => k + 1);
  };

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-center gap-3">
        <Link href="/builder" style={{ color: "var(--muted)" }}>
          ← Builder
        </Link>
        <h1 className="text-lg font-semibold">{project.title || project.name}</h1>
        {project.head && <span className="pill pill-queued">r{project.head.number}</span>}
        {project.versions.length > 0 && (
          <span className="pill pill-completed">published {project.versions[project.versions.length - 1].pack_name}</span>
        )}
      </div>
      <div className="grid grid-cols-1 gap-4 lg:grid-cols-[minmax(0,5fr)_minmax(0,7fr)]">
        <Chat wsId={ws.id} project={project} onChange={refresh} onOpenRevisions={() => setTab("revisions")} />
        <div className="panel min-w-0 p-0">
          <div className="flex flex-wrap gap-1 border-b px-3 pt-3" style={{ borderColor: "var(--line)" }}>
            {(["overview", "spec", "tasks", "data", "revisions", "preview"] as Tab[]).map((t) => (
              <button
                key={t}
                onClick={() => setTab(t)}
                className="rounded-t-md px-3 py-1.5 text-sm capitalize"
                style={{
                  color: tab === t ? "var(--fg)" : "var(--muted)",
                  borderBottom: `2px solid ${tab === t ? "var(--accent)" : "transparent"}`,
                }}
              >
                {t}
                {t === "data" && project.uploads.length > 0 && ` (${project.uploads.length})`}
              </button>
            ))}
          </div>
          <div className="p-4">
            {tab === "overview" && <Overview wsId={ws.id} project={project} onChange={refresh} />}
            {tab === "spec" && <SpecTab key={revKey} wsId={ws.id} project={project} onChange={refresh} />}
            {tab === "tasks" && <TasksTab key={revKey} wsId={ws.id} project={project} />}
            {tab === "data" && <DataTab wsId={ws.id} project={project} onChange={refresh} />}
            {tab === "revisions" && <RevisionsTab key={revKey} wsId={ws.id} project={project} onChange={refresh} />}
            {tab === "preview" && <PreviewTab key={revKey} wsId={ws.id} project={project} />}
          </div>
        </div>
      </div>
    </div>
  );
}

// ---- chat ------------------------------------------------------------------------------

function Chat({ wsId, project, onChange, onOpenRevisions }: { wsId: string; project: BuilderProject; onChange: () => void; onOpenRevisions: () => void }) {
  const [messages, setMessages] = useState<BuilderMessage[]>([]);
  const [job, setJob] = useState<{ status: string; progress: string } | null>(project.job);
  const [draft, setDraft] = useState("");
  const [sendErr, setSendErr] = useState<string | null>(null);
  const [uploading, setUploading] = useState<string | null>(null);
  const lastSeq = useRef(0);
  const wasBusy = useRef(!!project.job);
  const scroller = useRef<HTMLDivElement>(null);
  const fileInput = useRef<HTMLInputElement>(null);

  useEffect(() => {
    const e = sessionStorage.getItem(`builder-error-${project.id}`);
    if (e) {
      setSendErr(e);
      sessionStorage.removeItem(`builder-error-${project.id}`);
    }
  }, [project.id]);

  const poll = useCallback(async () => {
    try {
      const r = await Api.builderMessages(wsId, project.id, lastSeq.current);
      if (r.messages.length) {
        lastSeq.current = r.messages[r.messages.length - 1].seq;
        setMessages((m) => [...m, ...r.messages]);
      }
      setJob(r.job);
      if (wasBusy.current && !r.job) onChange(); // the builder finished: refresh the project
      if (r.messages.some((m) => m.role === "tool" && m.meta?.revision)) onChange();
      wasBusy.current = !!r.job;
    } catch {
      /* transient */
    }
  }, [wsId, project.id, onChange]);

  useEffect(() => {
    poll();
  }, [poll]);
  useEffect(() => {
    const t = setInterval(poll, job ? 1500 : 8000);
    return () => clearInterval(t);
  }, [poll, job]);
  useEffect(() => {
    scroller.current?.scrollTo({ top: scroller.current.scrollHeight, behavior: "smooth" });
  }, [messages.length, job?.progress]);

  async function send() {
    const text = draft.trim();
    if (!text) return;
    setSendErr(null);
    try {
      await Api.sendBuilderMessage(wsId, project.id, text);
      setDraft("");
      wasBusy.current = true;
      setJob({ status: "queued", progress: "Thinking" });
      poll();
    } catch (e) {
      setSendErr(msgOf(e));
    }
  }

  async function upload(files: FileList | null) {
    if (!files?.length) return;
    for (const f of Array.from(files)) {
      setUploading(f.name);
      try {
        await uploadBuilderFile(wsId, project.id, f);
      } catch (e) {
        setSendErr(`${f.name}: ${msgOf(e)}`);
      }
    }
    setUploading(null);
    onChange();
  }

  const busy = !!job;
  return (
    <div className="panel flex h-[calc(100vh-170px)] min-h-[520px] flex-col p-0">
      <div ref={scroller} className="flex-1 space-y-3 overflow-y-auto p-4">
        {messages.length === 0 && !busy && (
          <div className="text-sm" style={{ color: "var(--muted)" }}>
            Tell the builder what the agent should learn, and attach your chat logs with 📎.
          </div>
        )}
        {messages.map((m) => (
          <ChatMessage key={m.id} m={m} onOpenRevisions={onOpenRevisions} />
        ))}
        {busy && (
          <div className="flex items-center gap-2 text-sm" style={{ color: "var(--muted)" }}>
            <span className="inline-block h-2 w-2 animate-pulse rounded-full" style={{ background: "var(--accent)" }} />
            {job?.status === "queued" ? "Queued…" : `${job?.progress || "Working"}…`}
          </div>
        )}
      </div>
      <div className="space-y-2 border-t p-3" style={{ borderColor: "var(--line)" }}>
        {sendErr && (
          <div className="text-xs" style={{ color: "var(--bad)" }}>
            {sendErr}
          </div>
        )}
        {uploading && (
          <div className="text-xs" style={{ color: "var(--muted)" }}>
            Uploading {uploading}…
          </div>
        )}
        <div className="flex items-end gap-2">
          <button
            type="button"
            className="btn-ghost px-2.5"
            title="Upload chat logs (CSV, TXT, JSON) — personal data is redacted before the builder sees it"
            onClick={() => fileInput.current?.click()}
          >
            📎
          </button>
          <input ref={fileInput} type="file" multiple accept=".csv,.txt,.json,.jsonl,.log" className="hidden" onChange={(e) => upload(e.target.files)} />
          <textarea
            className="input max-h-40 min-h-[44px] flex-1 resize-y"
            placeholder={busy ? "The builder is working…" : "Message the builder (Enter to send, Shift+Enter for a new line)"}
            value={draft}
            maxLength={20000}
            onChange={(e) => setDraft(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                if (!busy) send();
              }
            }}
          />
          <button className="btn" onClick={send} disabled={busy || !draft.trim()}>
            Send
          </button>
        </div>
        <div className="text-[11px]" style={{ color: "var(--muted)" }}>
          {project.builder_model} · every change becomes a revision you can roll back
        </div>
      </div>
    </div>
  );
}

function ChatMessage({ m, onOpenRevisions }: { m: BuilderMessage; onOpenRevisions: () => void }) {
  if (m.role === "user") {
    const system = m.content.startsWith("(") && m.content.endsWith(")");
    if (system)
      return (
        <div className="text-center text-xs italic" style={{ color: "var(--muted)" }}>
          {m.content.slice(1, -1)}
        </div>
      );
    return (
      <div className="flex justify-end">
        <div className="max-w-[85%] whitespace-pre-wrap rounded-lg px-3 py-2 text-sm" style={{ background: "#173b36" }}>
          {m.content}
        </div>
      </div>
    );
  }
  if (m.role === "tool") {
    const rev = m.meta?.revision;
    let error = false;
    try {
      error = !!JSON.parse(m.content)?.error;
    } catch {
      error = m.content.includes('"error"');
    }
    return (
      <div className="flex items-center gap-2 text-xs" style={{ color: "var(--muted)" }}>
        <span style={{ color: error ? "var(--warn)" : "var(--good)" }}>{error ? "!" : "✓"}</span>
        {TOOL_LABEL[m.name ?? ""] ?? m.name}
        {rev && (
          <button className="underline" onClick={onOpenRevisions}>
            → revision {rev}
          </button>
        )}
        {error && <span style={{ color: "var(--warn)" }}>(needs attention)</span>}
      </div>
    );
  }
  if (!m.content.trim()) return null;
  return (
    <div className="max-w-[95%] text-sm">
      <Markdown text={m.content} />
    </div>
  );
}

// ---- overview ----------------------------------------------------------------------------

function Overview({ wsId, project, onChange }: { wsId: string; project: BuilderProject; onChange: () => void }) {
  const confirm = useConfirm();
  const router = useRouter();
  const [msg, setMsg] = useState<string | null>(null);
  const h = project.head;
  const v = h?.validation;
  const canPublish = !!v?.spec_ok && (v?.tasks_ok ?? 0) > 0;
  const next = project.versions.length + 1;

  return (
    <div className="space-y-5 text-sm">
      {project.purpose && (
        <div>
          <div className="label">purpose</div>
          <p className="whitespace-pre-wrap">{project.purpose}</p>
        </div>
      )}
      <div>
        <div className="label mb-1">readiness</div>
        <ul className="space-y-1">
          {project.readiness.map((r) => (
            <li key={r.item} className="flex items-center gap-2">
              <span style={{ color: r.ok ? "var(--good)" : "var(--muted)" }}>{r.ok ? "✓" : "○"}</span>
              {r.item}
              {r.detail !== undefined && r.detail !== null && typeof r.detail !== "object" && (
                <span style={{ color: "var(--muted)" }}>· {String(r.detail)}</span>
              )}
            </li>
          ))}
        </ul>
      </div>
      <div>
        <div className="label mb-1">current revision</div>
        {h ? (
          <div className="space-y-1">
            <div>
              r{h.number} · {h.n_tasks} task{h.n_tasks === 1 ? "" : "s"} ({v?.tasks_ok ?? 0} valid) · spec{" "}
              <span style={{ color: v?.spec_ok ? "var(--good)" : "var(--bad)" }}>{v?.spec_ok ? "valid" : "has problems"}</span>
            </div>
            {!!v?.spec_problems?.length && (
              <ul className="list-disc pl-5 text-xs" style={{ color: "var(--warn)" }}>
                {v.spec_problems.slice(0, 8).map((p, i) => (
                  <li key={i}>{p}</li>
                ))}
              </ul>
            )}
          </div>
        ) : (
          <div style={{ color: "var(--muted)" }}>No revision yet — the builder creates one when it drafts the spec.</div>
        )}
      </div>
      <div className="flex flex-wrap items-center gap-3" hidden={!h}>
        <button
          className="btn"
          disabled={!canPublish}
          title={canPublish ? "" : "needs a valid spec and at least one valid task"}
          onClick={async () => {
            const ok = await confirm({
              title: `Publish revision ${h!.number} as ${project.name}@v${next}?`,
              body: `The spec and its ${v!.tasks_ok} valid task(s) are frozen as a runnable environment in this workspace. Tasks with problems are left out. You can keep editing and publish again later.`,
              confirmLabel: "Publish",
            });
            if (!ok) return;
            try {
              const r = await Api.publish(wsId, project.id);
              setMsg(`Published ${r.pack_name}.`);
              onChange();
            } catch (e) {
              setMsg(msgOf(e));
            }
          }}
        >
          Publish r{h?.number ?? "–"} as v{next}
        </button>
        {msg && <span style={{ color: "var(--accent)" }}>{msg}</span>}
      </div>
      {project.versions.length > 0 && (
        <div>
          <div className="label mb-1">published versions</div>
          <table className="data">
            <tbody>
              {[...project.versions].reverse().map((ver) => (
                <tr key={ver.version}>
                  <td className="mono">{ver.pack_name}</td>
                  <td style={{ color: "var(--muted)" }}>{new Date(ver.created_at).toLocaleString()}</td>
                  <td>
                    <Link href={`/environments/${encodeURIComponent(ver.pack_name)}`} className="underline">
                      view
                    </Link>
                  </td>
                  <td>
                    <Link href={`/runs?pack=${encodeURIComponent(ver.pack_name)}`} className="underline">
                      run
                    </Link>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
      <div className="border-t pt-4" style={{ borderColor: "var(--line)" }}>
        <button
          className="btn-ghost"
          style={{ color: "var(--bad)" }}
          onClick={async () => {
            const ok = await confirm({
              title: "Delete this project?",
              body: "The chat, revisions and uploads are removed from the builder. Published versions stay runnable, because runs may reference them.",
              confirmLabel: "Delete project",
              tone: "danger",
              requireText: project.name,
            });
            if (!ok) return;
            await Api.deleteProject(wsId, project.id);
            router.replace("/builder");
          }}
        >
          Delete project
        </button>
      </div>
    </div>
  );
}

// ---- spec ---------------------------------------------------------------------------------

function SpecTab({ wsId, project, onChange }: { wsId: string; project: BuilderProject; onChange: () => void }) {
  const spec = project.head?.spec;
  const [editing, setEditing] = useState(false);
  const [text, setText] = useState(spec ? JSON.stringify(spec, null, 2) : "");
  const [message, setMessage] = useState("");
  const [err, setErr] = useState<string | null>(null);

  if (!spec && !editing)
    return (
      <div className="text-sm" style={{ color: "var(--muted)" }}>
        No spec yet. The builder drafts one once it understands your goal.
      </div>
    );

  async function save() {
    setErr(null);
    let parsed: unknown;
    try {
      parsed = JSON.parse(text);
    } catch (e) {
      setErr(`Not valid JSON: ${String(e)}`);
      return;
    }
    try {
      await Api.editSpec(wsId, project.id, parsed, message || "Edited the spec by hand");
      setEditing(false);
      onChange();
    } catch (e) {
      setErr(msgOf(e));
    }
  }

  return (
    <div className="space-y-3">
      <div className="flex items-center gap-2 text-sm">
        <span style={{ color: "var(--muted)" }}>revision {project.head?.number} · EnvSpec JSON</span>
        {!editing ? (
          <button className="btn-ghost ml-auto py-1" onClick={() => setEditing(true)}>
            Edit by hand
          </button>
        ) : (
          <div className="ml-auto flex gap-2">
            <input className="input !w-64 py-1" placeholder="what changed (optional)" value={message} onChange={(e) => setMessage(e.target.value)} />
            <button className="btn py-1" onClick={save}>
              Save as new revision
            </button>
            <button className="btn-ghost py-1" onClick={() => setEditing(false)}>
              Cancel
            </button>
          </div>
        )}
      </div>
      {err && <div className="text-sm" style={{ color: "var(--bad)" }}>{err}</div>}
      {editing ? (
        <textarea className="input mono h-[60vh] text-xs" value={text} onChange={(e) => setText(e.target.value)} spellCheck={false} />
      ) : (
        <pre className="mono max-h-[60vh] overflow-auto rounded-md p-3 text-xs" style={{ background: "#0d1014" }}>
          {JSON.stringify(spec, null, 2)}
        </pre>
      )}
    </div>
  );
}

// ---- tasks --------------------------------------------------------------------------------

function TasksTab({ wsId, project }: { wsId: string; project: BuilderProject }) {
  const [rows, setRows] = useState<TaskRow[] | null>(null);
  const [open, setOpen] = useState<string | null>(null);
  useEffect(() => {
    if (project.head) Api.revisionTasks(wsId, project.id, project.head.id).then(setRows).catch(() => setRows([]));
    else setRows([]);
  }, [wsId, project.id, project.head]);
  if (!rows) return <div style={{ color: "var(--muted)" }}>Loading…</div>;
  if (!rows.length)
    return (
      <div className="text-sm" style={{ color: "var(--muted)" }}>
        No tasks yet. Upload chat logs and ask the builder to extract a few.
      </div>
    );
  const valid = rows.filter((r) => !r.problems.length).length;
  return (
    <div className="space-y-2">
      <div className="text-sm" style={{ color: "var(--muted)" }}>
        {rows.length} task(s) in r{project.head?.number}: {valid} valid, {rows.length - valid} with problems (left out when publishing).
      </div>
      <div className="overflow-x-auto">
        <table className="data">
          <thead>
            <tr>
              <th>task</th>
              <th>tier</th>
              <th>customer&apos;s message</th>
              <th>checks</th>
              <th>status</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((r) => (
              <tr key={r.id} className="cursor-pointer" onClick={() => setOpen(open === r.id ? null : r.id)}>
                <td className="mono text-xs">{r.id}</td>
                <td>{r.tier}</td>
                <td className="max-w-md">
                  <div className={open === r.id ? "" : "truncate"}>{r.goal}</div>
                  {open === r.id && r.problems.length > 0 && (
                    <ul className="mt-1 list-disc pl-4 text-xs" style={{ color: "var(--warn)" }}>
                      {r.problems.map((p, i) => (
                        <li key={i}>{p}</li>
                      ))}
                    </ul>
                  )}
                </td>
                <td>
                  {r.n_success} / {r.n_reference} steps
                </td>
                <td style={{ color: r.problems.length ? "var(--warn)" : "var(--good)" }}>{r.problems.length ? `${r.problems.length} problem(s)` : "valid"}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

// ---- data ---------------------------------------------------------------------------------

function DataTab({ wsId, project, onChange }: { wsId: string; project: BuilderProject; onChange: () => void }) {
  const confirm = useConfirm();
  const processing = project.uploads.some((u) => u.status === "uploaded" || u.status === "processing");
  useEffect(() => {
    if (!processing) return;
    const t = setInterval(onChange, 2500);
    return () => clearInterval(t);
  }, [processing, onChange]);

  return (
    <div className="space-y-3 text-sm">
      <p style={{ color: "var(--muted)" }}>
        Uploaded files are parsed and <strong>redacted</strong> (emails, phone numbers, card numbers, addresses, names after
        cues like “my name is”…) by the builder worker, then the original file is deleted. Only redacted text is stored or shown
        to the builder model. Redaction is best-effort: review samples before extracting at scale. Attach files with 📎 in the
        chat (CSV, TXT, JSON; up to 95 MB each).
      </p>
      {project.uploads.length === 0 && <div style={{ color: "var(--muted)" }}>No uploads yet.</div>}
      {project.uploads.map((u) => (
        <div key={u.id} className="rounded-md border p-3" style={{ borderColor: "var(--line)" }}>
          <div className="flex flex-wrap items-center gap-2">
            <span className="font-medium">{u.filename}</span>
            <span style={{ color: "var(--muted)" }}>
              {u.size_bytes < 1024 ? `${u.size_bytes} B` : u.size_bytes < 2 ** 20 ? `${(u.size_bytes / 1024).toFixed(0)} KB` : `${(u.size_bytes / 2 ** 20).toFixed(1)} MB`}
            </span>
            <span className={`pill ${u.status === "ready" ? "pill-completed" : u.status === "failed" ? "pill-failed" : "pill-running"}`}>{u.status}</span>
            {u.status === "ready" && <span>{u.n_conversations} conversations</span>}
            {u.format?.format && <span style={{ color: "var(--muted)" }}>· {u.format.format}</span>}
            <button
              className="ml-auto text-xs underline"
              style={{ color: "var(--muted)" }}
              disabled={u.status === "processing"}
              onClick={async () => {
                const ok = await confirm({
                  title: `Delete ${u.filename}?`,
                  body: "The redacted conversations from this file are removed. Tasks already extracted from it stay in their revisions.",
                  confirmLabel: "Delete upload",
                  tone: "danger",
                });
                if (!ok) return;
                await Api.deleteUpload(wsId, project.id, u.id);
                onChange();
              }}
            >
              delete
            </button>
          </div>
          {u.error && <div className="mt-1 text-xs" style={{ color: "var(--bad)" }}>{u.error}</div>}
          {u.redactions && Object.keys(u.redactions).length > 0 && (
            <div className="mt-2 flex flex-wrap gap-1 text-xs">
              <span style={{ color: "var(--muted)" }}>redacted:</span>
              {Object.entries(u.redactions).map(([k, n]) => (
                <span key={k} className="rounded px-1.5" style={{ background: "var(--line)" }}>
                  {k} ×{n}
                </span>
              ))}
            </div>
          )}
          {u.status === "ready" && u.redactions && Object.keys(u.redactions).length === 0 && (
            <div className="mt-2 text-xs" style={{ color: "var(--muted)" }}>
              No personal data patterns found.
            </div>
          )}
          {!!u.format?.warnings?.length && (
            <ul className="mt-2 list-disc pl-5 text-xs" style={{ color: "var(--warn)" }}>
              {u.format.warnings.slice(0, 5).map((w, i) => (
                <li key={i}>{w}</li>
              ))}
            </ul>
          )}
        </div>
      ))}
    </div>
  );
}

// ---- revisions -----------------------------------------------------------------------------

function RevisionsTab({ wsId, project, onChange }: { wsId: string; project: BuilderProject; onChange: () => void }) {
  const confirm = useConfirm();
  const [revs, setRevs] = useState<RevisionSummary[] | null>(null);
  const [diffFor, setDiffFor] = useState<string | null>(null);
  const [diff, setDiff] = useState<RevisionDiff | null>(null);

  useEffect(() => {
    Api.revisions(wsId, project.id).then(setRevs).catch(() => setRevs([]));
  }, [wsId, project.id, project.head?.id]);
  useEffect(() => {
    setDiff(null);
    if (diffFor) Api.revisionDiff(wsId, project.id, diffFor).then(setDiff).catch(() => {});
  }, [wsId, project.id, diffFor]);

  if (!revs) return <div style={{ color: "var(--muted)" }}>Loading…</div>;
  if (!revs.length) return <div className="text-sm" style={{ color: "var(--muted)" }}>No revisions yet.</div>;

  // depth in the revision graph, so branches are visible
  const byNumber = new Map(revs.map((r) => [r.number, r]));
  const depth = (r: RevisionSummary): number => {
    let d = 0;
    let p = r.parent_number;
    let prev = r.number;
    while (p !== null && p !== undefined) {
      if (p !== prev - 1) d = Math.max(d, 1);
      prev = p;
      p = byNumber.get(p)?.parent_number ?? null;
    }
    return d;
  };

  return (
    <div className="space-y-2">
      <p className="text-xs" style={{ color: "var(--muted)" }}>
        Every change is an immutable revision. <strong>Restore</strong> makes a new revision copying an old one (history is kept).{" "}
        <strong>Check out</strong> moves the head: the next change branches from there.
      </p>
      {[...revs].reverse().map((r) => (
        <div key={r.id} className="rounded-md border p-3 text-sm" style={{ borderColor: r.is_head ? "var(--accent)" : "var(--line)", marginLeft: depth(r) * 16 }}>
          <div className="flex flex-wrap items-center gap-2">
            <span className="mono font-semibold">r{r.number}</span>
            {r.parent_number !== null && r.parent_number !== r.number - 1 && (
              <span className="text-xs" style={{ color: "var(--warn)" }}>
                branched from r{r.parent_number}
              </span>
            )}
            <span className="min-w-0 flex-1 truncate">{r.message}</span>
            {r.is_head && <span className="pill pill-running">head</span>}
            {r.published_version && <span className="pill pill-completed">v{r.published_version}</span>}
          </div>
          <div className="mt-1 flex flex-wrap items-center gap-3 text-xs" style={{ color: "var(--muted)" }}>
            <span>{r.author === "user" ? "you" : "builder"}</span>
            <span>{new Date(r.created_at).toLocaleString()}</span>
            <span>
              {r.n_tasks} tasks ({r.tasks_ok} valid)
            </span>
            <span style={{ color: r.spec_ok ? "var(--good)" : "var(--warn)" }}>{r.spec_ok ? "spec valid" : "spec has problems"}</span>
            <span className="ml-auto flex gap-3">
              <button className="underline" onClick={() => setDiffFor(diffFor === r.id ? null : r.id)}>
                {diffFor === r.id ? "hide changes" : "changes"}
              </button>
              {!r.is_head && (
                <>
                  <button
                    className="underline"
                    onClick={async () => {
                      await Api.checkoutRevision(wsId, project.id, r.id);
                      onChange();
                    }}
                  >
                    check out
                  </button>
                  <button
                    className="underline"
                    onClick={async () => {
                      const ok = await confirm({
                        title: `Restore revision ${r.number}?`,
                        body: `A new revision is created with r${r.number}'s spec and tasks and becomes the head. Nothing is deleted — you can return to the current state any time.`,
                        confirmLabel: "Restore",
                      });
                      if (!ok) return;
                      await Api.restoreRevision(wsId, project.id, r.id);
                      onChange();
                    }}
                  >
                    restore
                  </button>
                </>
              )}
              {r.spec_ok && r.tasks_ok > 0 && !r.published_version && (
                <button
                  className="underline"
                  onClick={async () => {
                    const ok = await confirm({
                      title: `Publish revision ${r.number}?`,
                      body: `Frozen as ${project.name}@v${project.versions.length + 1} with its ${r.tasks_ok} valid task(s), runnable in this workspace.`,
                      confirmLabel: "Publish",
                    });
                    if (!ok) return;
                    await Api.publish(wsId, project.id, r.id);
                    onChange();
                  }}
                >
                  publish
                </button>
              )}
            </span>
          </div>
          {diffFor === r.id && (
            <div className="mt-2 space-y-2">
              {!diff ? (
                <div className="text-xs" style={{ color: "var(--muted)" }}>
                  Loading…
                </div>
              ) : (
                <>
                  <div className="text-xs" style={{ color: "var(--muted)" }}>
                    {diff.from ? `r${diff.from} → r${diff.to}` : `first revision`}
                    {diff.tasks_added.length > 0 && ` · +${diff.tasks_added.length} task(s)`}
                    {diff.tasks_removed.length > 0 && ` · −${diff.tasks_removed.length} task(s)`}
                  </div>
                  {diff.spec_diff ? (
                    <pre className="mono max-h-80 overflow-auto rounded-md p-2 text-[11px]" style={{ background: "#0d1014" }}>
                      {diff.spec_diff.split("\n").map((l, i) => (
                        <div key={i} style={{ color: l.startsWith("+") && !l.startsWith("+++") ? "var(--good)" : l.startsWith("-") && !l.startsWith("---") ? "var(--bad)" : l.startsWith("@@") ? "var(--accent)" : "var(--muted)" }}>
                          {l || " "}
                        </div>
                      ))}
                    </pre>
                  ) : (
                    <div className="text-xs" style={{ color: "var(--muted)" }}>
                      Spec unchanged.
                    </div>
                  )}
                </>
              )}
            </div>
          )}
        </div>
      ))}
    </div>
  );
}

// ---- preview -----------------------------------------------------------------------------

function PreviewTab({ wsId, project }: { wsId: string; project: BuilderProject }) {
  const [seed, setSeed] = useState(0);
  const [tier, setTier] = useState<number | undefined>(undefined);
  const [data, setData] = useState<{ overview: PackDetail; preview: TaskPreview } | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [view, setView] = useState<"before" | "after">("before");

  const load = useCallback(
    async (s: number, t?: number) => {
      setErr(null);
      try {
        setData(await Api.builderPreview(wsId, project.id, s, t));
        setView("before");
      } catch (e) {
        setErr(msgOf(e));
      }
    },
    [wsId, project.id],
  );
  useEffect(() => {
    load(0);
  }, [load]);

  if (err) return <div className="text-sm" style={{ color: "var(--muted)" }}>Preview not available yet: {err}</div>;
  if (!data) return <div style={{ color: "var(--muted)" }}>Loading…</div>;
  const { overview, preview } = data;
  const diagram = view === "after" && preview.diagram_after ? preview.diagram_after : preview.diagram_before;

  return (
    <div className="space-y-4 text-sm">
      <div className="flex flex-wrap items-end gap-3">
        <div>
          <label className="label">Seed</label>
          <input className="input !w-28" type="number" min={0} value={seed} onChange={(e) => setSeed(Math.max(0, Number(e.target.value)))} />
        </div>
        <div>
          <label className="label">Tier</label>
          <select className="input !w-24" value={tier ?? overview.tiers[0]} onChange={(e) => setTier(Number(e.target.value))}>
            {overview.tiers.map((t) => (
              <option key={t} value={t}>
                {t}
              </option>
            ))}
          </select>
        </div>
        <button className="btn" onClick={() => load(seed, tier)}>
          Generate
        </button>
        <span className="text-xs" style={{ color: "var(--muted)" }}>
          head revision r{project.head?.number} · no model called
        </span>
      </div>
      <div>
        <div className="label">the agent is told</div>
        <p>{preview.task.goal}</p>
      </div>
      {diagram && (
        <div className="space-y-2">
          {preview.diagram_after && (
            <div className="flex gap-1">
              {(["before", "after"] as const).map((v) => (
                <button
                  key={v}
                  onClick={() => setView(v)}
                  className="rounded-md border px-2.5 py-1"
                  style={{ borderColor: view === v ? "var(--accent)" : "var(--line)", color: view === v ? "var(--accent)" : "var(--muted)" }}
                >
                  {v === "before" ? "Start" : "After the reference solution"}
                </button>
              ))}
            </div>
          )}
          <DiagramView diagram={diagram} height={380} />
        </div>
      )}
      <div>
        <div className="label">reference solution{preview.verdict && ` → reward ${preview.verdict.reward.toFixed(1)}`}</div>
        <ol className="list-decimal space-y-1 pl-5">
          {preview.solution.map((c, i) => (
            <li key={i}>
              <span className="mono text-xs" style={{ color: "var(--accent)" }}>
                {c.name}({Object.entries(c.args).map(([k, v]) => `${k}=${JSON.stringify(v)}`).join(", ")})
              </span>
            </li>
          ))}
        </ol>
      </div>
      <details>
        <summary className="label cursor-pointer">dimensions, tools and system prompt</summary>
        <table className="data mt-2">
          <tbody>
            {overview.dimensions.map(([k, v]) => (
              <tr key={k}>
                <td style={{ color: "var(--muted)" }}>{k}</td>
                <td>{v}</td>
              </tr>
            ))}
          </tbody>
        </table>
        <pre className="mono mt-2 whitespace-pre-wrap text-xs" style={{ color: "var(--muted)" }}>
          {preview.system_prompt}
        </pre>
      </details>
    </div>
  );
}

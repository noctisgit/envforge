"use client";

// One builder project: chat with the builder on the left, the live project on the right
// (readiness, spec, tasks, data, revisions, preview). The builder runs in the background
// worker; this page polls for new messages while it works.

import Link from "next/link";
import { useParams, useRouter } from "next/navigation";
import { useCallback, useEffect, useRef, useState } from "react";
import { useConfirm } from "@/components/ConfirmDialog";
import { useDeleteEnvironment } from "@/components/DeleteEnvironment";
import { DiagramView } from "@/components/DiagramView";
import { Markdown } from "@/components/Markdown";
import { Empty, Page, PageHeader } from "@/components/Page";
import { CorpusPanel, PrivacyPanel } from "@/components/PrivacyPanel";
import { Tabs, tabPanelProps, useHashTab, type TabItem } from "@/components/Tabs";
import { errorText, useToast } from "@/components/Toast";
import {
  Api,
  importBuilderUrl,
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

const TAB_IDS = ["overview", "spec", "tasks", "data", "revisions", "preview"] as const;
const P = "proj";

/** Publish a revision (the head by default) after a confirmation; toasts the result. */
function usePublish(wsId: string, project: BuilderProject, onChange: () => void) {
  const confirm = useConfirm();
  const toast = useToast();
  return async (rev: { id?: string; number: number; tasksOk: number }) => {
    const next = project.versions.length + 1;
    const ok = await confirm({
      title: `Publish revision ${rev.number} as ${project.name}@v${next}?`,
      body: `The spec and its ${rev.tasksOk} valid task${rev.tasksOk === 1 ? "" : "s"} are frozen as an environment anyone in this workspace can run. Tasks with problems are left out. Earlier versions stay runnable, and you can keep editing and publish again later.`,
      confirmLabel: "Publish",
    });
    if (!ok) return;
    try {
      const r = await Api.publish(wsId, project.id, rev.id);
      toast.success(`Published ${r.pack_name}`, "It's listed under Environments, ready to run.");
      onChange();
    } catch (e) {
      toast.error("Couldn't publish", e);
    }
  };
}

export default function BuilderProjectPage() {
  const { id } = useParams<{ id: string }>();
  const { current: ws } = useWorkspace();
  const [project, setProject] = useState<BuilderProject | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [tab, setTab] = useHashTab(TAB_IDS, "overview");
  const [revKey, setRevKey] = useState(0); // bump to refresh right-hand panels

  const loadProject = useCallback(async () => {
    if (!ws) return;
    try {
      setProject(await Api.project(ws.id, id));
    } catch (e) {
      setErr(errorText(e));
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

  const latest = project.versions[project.versions.length - 1];
  const tabs: TabItem[] = [
    { id: "overview", label: "Overview" },
    { id: "spec", label: "Spec" },
    { id: "tasks", label: "Tasks", ...(project.head ? { count: project.head.n_tasks } : {}) },
    { id: "data", label: "Data", ...(project.uploads.length ? { count: project.uploads.length } : {}) },
    { id: "revisions", label: "Revisions" },
    { id: "preview", label: "Preview" },
  ];

  return (
    <Page>
      <PageHeader
        back={{ href: "/builder", label: "Builder" }}
        title={
          <>
            {project.title || project.name}
            {project.title && project.title !== project.name && (
              <span className="mono ml-2 text-sm font-normal" style={{ color: "var(--muted)" }}>
                {project.name}
              </span>
            )}
          </>
        }
        meta={
          <>
            {project.head && <span className="pill pill-tag">r{project.head.number} · head</span>}
            {latest && <span className="pill pill-completed">published {latest.pack_name}</span>}
          </>
        }
        actions={
          <>
            {latest && (
              <Link href={`/environments/${encodeURIComponent(latest.pack_name)}`} className="btn-ghost">
                View as environment
              </Link>
            )}
            {project.head && <PublishButton wsId={ws.id} project={project} onChange={refresh} />}
          </>
        }
      />
      <div className="grid grid-cols-1 gap-5 lg:grid-cols-[minmax(0,5fr)_minmax(0,7fr)]">
        <Chat wsId={ws.id} project={project} onChange={refresh} onOpenRevisions={() => setTab("revisions")} />
        <section className="panel flex min-w-0 flex-col" aria-label="Project">
          <div className="px-4 pt-2">
            <Tabs idPrefix={P} label="Project sections" tabs={tabs} active={tab} onChange={setTab} />
          </div>
          <div {...tabPanelProps(P, tab)} className={`min-w-0 outline-none ${tab === "tasks" ? "" : "p-4"}`}>
            {tab === "overview" && <Overview wsId={ws.id} project={project} onChange={loadProject} />}
            {tab === "spec" && <SpecTab key={revKey} wsId={ws.id} project={project} onChange={refresh} />}
            {tab === "tasks" && <TasksTab key={revKey} wsId={ws.id} project={project} />}
            {tab === "data" && <DataTab wsId={ws.id} project={project} onChange={refresh} />}
            {tab === "revisions" && <RevisionsTab key={revKey} wsId={ws.id} project={project} onChange={refresh} />}
            {tab === "preview" && <PreviewTab key={revKey} wsId={ws.id} project={project} />}
          </div>
        </section>
      </div>
    </Page>
  );
}

function PublishButton({ wsId, project, onChange }: { wsId: string; project: BuilderProject; onChange: () => void }) {
  const publish = usePublish(wsId, project, onChange);
  const h = project.head!;
  const v = h.validation;
  const canPublish = !!v?.spec_ok && (v?.tasks_ok ?? 0) > 0;
  return (
    <button
      className="btn"
      disabled={!canPublish}
      title={canPublish ? "" : "needs a valid spec and at least one valid task"}
      onClick={() => publish({ number: h.number, tasksOk: v?.tasks_ok ?? 0 })}
    >
      Publish r{h.number} as v{project.versions.length + 1}
    </button>
  );
}

// ---- chat ------------------------------------------------------------------------------

function Chat({ wsId, project, onChange, onOpenRevisions }: { wsId: string; project: BuilderProject; onChange: () => void; onOpenRevisions: () => void }) {
  const toast = useToast();
  const [messages, setMessages] = useState<BuilderMessage[]>([]);
  const [job, setJob] = useState<{ status: string; progress: string } | null>(project.job);
  const [draft, setDraft] = useState("");
  const [uploading, setUploading] = useState<string | null>(null);
  const lastSeq = useRef(0);
  const wasBusy = useRef(!!project.job);
  const scroller = useRef<HTMLDivElement>(null);
  const fileInput = useRef<HTMLInputElement>(null);

  useEffect(() => {
    // an error from the first message, sent by the "Start building" form before it navigated here
    const e = sessionStorage.getItem(`builder-error-${project.id}`);
    if (e) {
      toast.error("Couldn't send your first message", e);
      sessionStorage.removeItem(`builder-error-${project.id}`);
    }
  }, [project.id, toast]);

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
    try {
      await Api.sendBuilderMessage(wsId, project.id, text);
      setDraft("");
      wasBusy.current = true;
      setJob({ status: "queued", progress: "Thinking" });
      poll();
    } catch (e) {
      toast.error("Message not sent", e);
    }
  }

  async function upload(files: FileList | null) {
    if (!files?.length) return;
    let done = 0;
    for (const f of Array.from(files)) {
      setUploading(f.name);
      try {
        await uploadBuilderFile(wsId, project.id, f, (sent, total) =>
          setUploading(`${f.name} — ${Math.floor((100 * sent) / Math.max(1, total))}% of ${(total / 2 ** 20).toFixed(total > 2 ** 30 ? 0 : 1)} MB`),
        );
        done++;
      } catch (e) {
        toast.error(`Upload failed: ${f.name}`, e);
      }
    }
    setUploading(null);
    if (fileInput.current) fileInput.current.value = "";
    if (done) toast.success(done === 1 ? "File uploaded" : `${done} files uploaded`, "Redacting now — progress is under Data.");
    onChange();
  }

  const busy = !!job;
  return (
    <section className="panel flex h-[calc(100vh-200px)] min-h-[560px] min-w-0 flex-col" aria-label="Builder chat">
      <div className="flex flex-wrap items-center gap-x-2 gap-y-1 border-b px-4 py-3" style={{ borderColor: "var(--line)" }}>
        <h2 className="text-lg font-semibold">Builder</h2>
        <span className="hint mono">{project.builder_model}</span>
        <span className="hint">· every change becomes a revision</span>
      </div>
      <div ref={scroller} className="flex-1 space-y-3 overflow-y-auto p-4">
        {messages.length === 0 && !busy && (
          <div style={{ color: "var(--muted)" }}>Tell the builder what the agent should learn, and attach your chat logs.</div>
        )}
        {messages.map((m) => (
          <ChatMessage key={m.id} m={m} onOpenRevisions={onOpenRevisions} />
        ))}
        {busy && (
          <div className="flex items-center gap-2 text-xs" style={{ color: "var(--muted)" }}>
            <span className="inline-block h-2 w-2 animate-pulse rounded-full" style={{ background: "var(--accent)" }} />
            {job?.status === "queued" ? "Queued…" : `${job?.progress || "Working"}…`}
          </div>
        )}
      </div>
      <div className="space-y-2 border-t px-4 py-3" style={{ borderColor: "var(--line)" }}>
        {uploading && <div className="hint">Uploading {uploading}… (you can leave and re-pick the same file later to resume)</div>}
        <div className="flex items-end gap-2">
          <button
            type="button"
            className="btn-ghost btn-field"
            aria-label="Attach chat logs (.csv .txt .json .jsonl .log, optionally .gz) — redacted before the model sees them"
            title="Attach chat logs (CSV, TXT, JSON) — personal data is redacted before the builder sees it"
            onClick={() => fileInput.current?.click()}
          >
            <svg aria-hidden width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M21.44 11.05l-9.19 9.19a6 6 0 0 1-8.49-8.49l9.19-9.19a4 4 0 0 1 5.66 5.66l-9.2 9.19a2 2 0 0 1-2.83-2.83l8.49-8.48" />
            </svg>
            Attach
          </button>
          <input ref={fileInput} type="file" multiple accept=".csv,.txt,.json,.jsonl,.log,.gz" className="hidden" onChange={(e) => upload(e.target.files)} />
          <textarea
            className="input max-h-40 min-h-[44px] flex-1 resize-y"
            aria-label="Message the builder"
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
          <button className="btn btn-field" onClick={send} disabled={busy || !draft.trim()}>
            Send
          </button>
        </div>
      </div>
    </section>
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
        <div
          className="max-w-[85%] border px-3 py-2 whitespace-pre-wrap [overflow-wrap:anywhere]"
          style={{ background: "var(--input-bg)", borderColor: "var(--line)", borderRadius: "10px 10px 2px 10px" }}
        >
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
      <div className="flex flex-wrap items-center gap-2 text-xs" style={{ color: "var(--muted)" }}>
        <span style={{ color: error ? "var(--warn)" : "var(--good)" }}>{error ? "!" : "✓"}</span>
        {TOOL_LABEL[m.name ?? ""] ?? m.name}
        {rev && (
          <button className="btn-ghost btn-sm !py-0" onClick={onOpenRevisions}>
            → revision {rev}
          </button>
        )}
        {error && <span style={{ color: "var(--warn)" }}>(needs attention)</span>}
      </div>
    );
  }
  if (!m.content.trim()) return null;
  return (
    <div className="max-w-[92%]">
      <Markdown text={m.content} />
    </div>
  );
}

// ---- overview ----------------------------------------------------------------------------

function Overview({ wsId, project, onChange }: { wsId: string; project: BuilderProject; onChange: () => void }) {
  const confirm = useConfirm();
  const removeEnv = useDeleteEnvironment();
  const toast = useToast();
  const router = useRouter();
  const h = project.head;
  const v = h?.validation;
  const revOf = (revisionId: string) => (h && h.id === revisionId ? `r${h.number}` : null);

  return (
    <div className="space-y-4">
      {project.purpose && (
        <div>
          <div className="label">purpose</div>
          <p className="whitespace-pre-wrap">{project.purpose}</p>
        </div>
      )}
      <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
        <div className="space-y-1.5">
          <div className="label">readiness</div>
          <ul className="space-y-1.5">
            {project.readiness.map((r) => (
              <li key={r.item} className="flex gap-2">
                <span style={{ color: r.ok ? "var(--good)" : "var(--muted)" }}>{r.ok ? "✓" : "○"}</span>
                <span>
                  {r.item}
                  {r.detail !== undefined && r.detail !== null && typeof r.detail !== "object" && <span style={{ color: "var(--muted)" }}> · {String(r.detail)}</span>}
                </span>
              </li>
            ))}
          </ul>
        </div>
        <div className="space-y-1.5">
          <div className="label">current revision</div>
          {h ? (
            <>
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
            </>
          ) : (
            <div style={{ color: "var(--muted)" }}>No revision yet — the builder creates one when it drafts the spec.</div>
          )}
        </div>
      </div>
      {project.versions.length > 0 && (
        <div className="border-t pt-3" style={{ borderColor: "var(--line)" }}>
          <div className="label mb-1.5">published versions</div>
          <div className="overflow-x-auto">
            <table className="data inset">
              <thead>
                <tr>
                  <th>pack</th>
                  <th>published</th>
                  <th></th>
                </tr>
              </thead>
              <tbody>
                {[...project.versions].reverse().map((ver) => (
                  <tr key={ver.version}>
                    <td className="mono">
                      {ver.pack_name}
                      {revOf(ver.revision_id) && <span style={{ color: "var(--muted)" }}> · from {revOf(ver.revision_id)}</span>}
                    </td>
                    <td style={{ color: "var(--muted)" }}>{new Date(ver.created_at).toLocaleString()}</td>
                    <td>
                      <span className="flex justify-end gap-1.5">
                        <Link href={`/environments/${encodeURIComponent(ver.pack_name)}`} className="btn-ghost btn-sm">
                          view
                        </Link>
                        <Link href={`/runs?pack=${encodeURIComponent(ver.pack_name)}`} className="btn-ghost btn-sm">
                          run
                        </Link>
                        <button
                          className="btn-ghost btn-sm"
                          style={{ color: "var(--bad)" }}
                          onClick={async () => {
                            if (await removeEnv(wsId, ver.pack_name)) onChange();
                          }}
                        >
                          delete
                        </button>
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
      <div className="flex flex-wrap items-center justify-between gap-3 border-t pt-3" style={{ borderColor: "var(--line)" }}>
        <span className="hint">Deleting removes the chat, every revision and upload, and the environments published from it.</span>
        <button
          className="btn-danger"
          onClick={async () => {
            const ok = await confirm({
              title: "Delete this project?",
              body: `The chat, revisions and uploads are removed from the builder${project.versions.length ? `, and ${project.versions.map((v) => v.pack_name).join(", ")} leave${project.versions.length === 1 ? "s" : ""} the Environments list` : ""}. Runs already made keep their results and replays.`,
              confirmLabel: "Delete project",
              tone: "danger",
              requireText: project.name,
            });
            if (!ok) return;
            try {
              await Api.deleteProject(wsId, project.id);
              toast.success("Project deleted");
              router.replace("/builder");
            } catch (e) {
              toast.error("Couldn't delete the project", e);
            }
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
  const toast = useToast();
  const spec = project.head?.spec;
  const [editing, setEditing] = useState(false);
  const [text, setText] = useState(spec ? JSON.stringify(spec, null, 2) : "");
  const [message, setMessage] = useState("");
  const [err, setErr] = useState<string | null>(null);

  if (!spec && !editing)
    return (
      <div style={{ color: "var(--muted)" }}>
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
      toast.success("Spec saved as a new revision");
      onChange();
    } catch (e) {
      toast.error("Couldn't save the spec", e);
    }
  }

  return (
    <div className="space-y-3">
      <div className="flex flex-wrap items-center gap-2">
        <span style={{ color: "var(--muted)" }}>
          revision {project.head?.number} · <span className="mono text-xs">EnvSpec</span> JSON
        </span>
        {!editing ? (
          <button className="btn-ghost ml-auto" onClick={() => setEditing(true)}>
            Edit by hand
          </button>
        ) : (
          <div className="ml-auto flex flex-wrap gap-2">
            <input className="input !w-64" aria-label="What changed" placeholder="what changed (optional)" value={message} onChange={(e) => setMessage(e.target.value)} />
            <button className="btn" onClick={save}>
              Save as new revision
            </button>
            <button className="btn-ghost" onClick={() => setEditing(false)}>
              Cancel
            </button>
          </div>
        )}
      </div>
      {err && <div style={{ color: "var(--bad)" }}>{err}</div>}
      {editing ? (
        <textarea className="input mono h-[60vh] text-xs" aria-label="EnvSpec JSON" value={text} onChange={(e) => setText(e.target.value)} spellCheck={false} />
      ) : (
        <pre className="code-block max-h-[60vh] overflow-auto">{JSON.stringify(spec, null, 2)}</pre>
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
  if (!rows) return <Empty>Loading…</Empty>;
  if (!rows.length) return <Empty>No tasks yet. Upload chat logs and ask the builder to extract a few.</Empty>;
  const valid = rows.filter((r) => !r.problems.length).length;
  return (
    <div>
      <div className="border-b px-4 py-3" style={{ borderColor: "var(--line)", color: "var(--muted)" }}>
        {rows.length} task{rows.length === 1 ? "" : "s"} in r{project.head?.number}: <span style={{ color: "var(--good)" }}>{valid} valid</span>
        {rows.length - valid > 0 && (
          <>
            , <span style={{ color: "var(--warn)" }}>{rows.length - valid} with problems</span> (left out when publishing)
          </>
        )}
        . Click a row to read it in full.
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
                <td className="num">{r.tier}</td>
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
                <td className="num whitespace-nowrap">
                  {r.n_success} / {r.n_reference} steps
                </td>
                <td className="whitespace-nowrap" style={{ color: r.problems.length ? "var(--warn)" : "var(--good)" }}>
                  {r.problems.length ? `${r.problems.length} problem${r.problems.length === 1 ? "" : "s"}` : "valid"}
                </td>
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
  const toast = useToast();
  const [url, setUrl] = useState("");
  const processing = project.uploads.some((u) => ["receiving", "uploaded", "processing"].includes(u.status));
  useEffect(() => {
    if (!processing) return;
    const t = setInterval(onChange, 2500);
    return () => clearInterval(t);
  }, [processing, onChange]);
  return (
    <div className="space-y-3">
      <p style={{ color: "var(--muted)" }}>
        Uploaded files are parsed and <strong style={{ color: "var(--fg)" }}>redacted</strong> (emails, phone numbers, card numbers, addresses,
        names after cues like “my name is”…) by the builder worker, then the original file is deleted. Only redacted text is stored or shown
        to the builder model. Redaction is best-effort: run a privacy review before extracting at scale. Attach files in the chat (CSV, TXT,
        JSON/JSONL, optionally .gz; large files upload in resumable chunks).
      </p>
      <PrivacyPanel wsId={wsId} project={project} onChange={onChange} />
      <CorpusPanel wsId={wsId} project={project} />

      <div className="flex flex-wrap items-end gap-2">
        <div className="min-w-[16rem] flex-1">
          <label className="label" htmlFor="import-url">
            Import from a URL (large archives — e.g. a presigned S3/R2 https link)
          </label>
          <input id="import-url" className="input mono" placeholder="https://…/helpdesk-2025.jsonl" value={url} onChange={(e) => setUrl(e.target.value)} />
        </div>
        <button
          className="btn-ghost btn-field"
          disabled={!url.trim().startsWith("https://")}
          onClick={async () => {
            try {
              await importBuilderUrl(wsId, project.id, url.trim());
              setUrl("");
              toast.success("Import started", "The worker downloads and redacts the file; progress shows below.");
              onChange();
            } catch (e) {
              toast.error("Couldn't import from that URL", e);
            }
          }}
        >
          Import
        </button>
      </div>

      {project.uploads.length === 0 && <div style={{ color: "var(--muted)" }}>No uploads yet.</div>}
      {project.uploads.map((u) => (
        <div key={u.id} className="inset-card space-y-1.5 p-4">
          <div className="flex flex-wrap items-center gap-2">
            <span className="mono">{u.filename}</span>
            <span className="hint">
              {u.size_bytes < 1024 ? `${u.size_bytes} B` : u.size_bytes < 2 ** 20 ? `${(u.size_bytes / 1024).toFixed(0)} KB` : `${(u.size_bytes / 2 ** 20).toFixed(1)} MB`}
            </span>
            <span className={`pill ${u.status === "ready" ? "pill-completed" : u.status === "failed" ? "pill-failed" : "pill-running"}`}>{u.status}</span>
            {u.status === "ready" && (
              <span style={{ color: "var(--muted)" }}>
                {u.n_conversations.toLocaleString()} conversations{u.format?.format && ` · ${u.format.format}`}
              </span>
            )}
            {u.status !== "ready" && u.format?.format && <span style={{ color: "var(--muted)" }}>{u.format.format}</span>}
            <button
              className="btn-danger btn-sm ml-auto"
              disabled={u.status === "processing"}
              onClick={async () => {
                const ok = await confirm({
                  title: `Delete ${u.filename}?`,
                  body: "The redacted conversations from this file are removed. Tasks already extracted from it stay in their revisions.",
                  confirmLabel: "Delete upload",
                  tone: "danger",
                });
                if (!ok) return;
                try {
                  await Api.deleteUpload(wsId, project.id, u.id);
                  toast.success(`Deleted ${u.filename}`);
                  onChange();
                } catch (e) {
                  toast.error(`Couldn't delete ${u.filename}`, e);
                }
              }}
            >
              Delete
            </button>
          </div>
          {u.status === "receiving" && (
            <Progress value={u.bytes_received} total={u.size_bytes} label="receiving (resume by re-picking the same file)" />
          )}
          {(u.status === "processing" || u.status === "uploaded") && u.progress && (
            <Progress
              value={u.progress.bytes ?? 0}
              total={u.progress.total ?? u.size_bytes}
              label={`${u.progress.stage ?? "processing"}${u.progress.conversations ? ` · ${u.progress.conversations.toLocaleString()} conversations` : ""}`}
            />
          )}
          {u.error && <div className="text-xs" style={{ color: "var(--bad)" }}>{u.error}</div>}
          {u.redactions && Object.keys(u.redactions).length > 0 && (
            <div className="flex flex-wrap items-center gap-1.5">
              <span className="hint">redacted:</span>
              {Object.entries(u.redactions).map(([k, n]) => (
                <span key={k} className="pill pill-cancelled">
                  {k} ×{n.toLocaleString()}
                </span>
              ))}
            </div>
          )}
          {u.status === "ready" && u.redactions && Object.keys(u.redactions).length === 0 && <div className="hint">No personal data patterns found.</div>}
          {!!u.format?.warnings?.length && (
            <ul className="list-disc pl-5 text-xs" style={{ color: "var(--warn)" }}>
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

function Progress({ value, total, label }: { value: number; total: number; label: string }) {
  const pct = total ? Math.min(100, Math.floor((100 * value) / total)) : null;
  return (
    <div className="space-y-1 pt-1">
      <div className="h-1.5 w-full overflow-hidden rounded-full" style={{ background: "var(--line)" }}>
        <div className="h-full rounded-full transition-all" style={{ width: `${pct ?? 30}%`, background: "var(--accent)" }} />
      </div>
      <div className="hint">
        {label}
        {pct !== null && ` · ${pct}%`}
      </div>
    </div>
  );
}

// ---- revisions -----------------------------------------------------------------------------

function RevisionsTab({ wsId, project, onChange }: { wsId: string; project: BuilderProject; onChange: () => void }) {
  const confirm = useConfirm();
  const toast = useToast();
  const publish = usePublish(wsId, project, onChange);
  const act = async (title: string, fn: () => Promise<unknown>, done?: string) => {
    try {
      await fn();
      if (done) toast.success(done);
      onChange();
    } catch (e) {
      toast.error(title, e);
    }
  };
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
  if (!revs.length) return <div style={{ color: "var(--muted)" }}>No revisions yet.</div>;

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
    <div className="space-y-3">
      <p className="hint">
        Every change is an immutable revision, newest first. <strong>Restore</strong> makes a new revision copying an old one (history is kept).{" "}
        <strong>Check out</strong> moves the head: the next change branches from there.
      </p>
      {[...revs].reverse().map((r) => (
        <div
          key={r.id}
          className="panel space-y-2 p-4"
          style={{ background: "var(--input-bg)", borderColor: r.is_head ? "var(--accent)" : "var(--line)", marginLeft: depth(r) * 16 }}
        >
          <div className="flex flex-wrap items-center gap-2">
            <span className="mono font-semibold">r{r.number}</span>
            {r.parent_number !== null && r.parent_number !== r.number - 1 && (
              <span className="text-xs" style={{ color: "var(--warn)" }}>
                branched from r{r.parent_number}
              </span>
            )}
            {r.is_head && <span className="pill pill-running">head</span>}
            {r.published_version && <span className="pill pill-completed">v{r.published_version}</span>}
            <span className="min-w-0 flex-1 truncate" style={{ color: "var(--muted)" }}>
              {r.message}
            </span>
          </div>
          <div className="flex flex-wrap items-center gap-3 text-xs" style={{ color: "var(--muted)" }}>
            <span>{r.author === "user" ? "you" : "builder"}</span>
            <span>{new Date(r.created_at).toLocaleString()}</span>
            <span>
              {r.n_tasks} tasks ({r.tasks_ok} valid)
            </span>
            <span style={{ color: r.spec_ok ? "var(--good)" : "var(--warn)" }}>{r.spec_ok ? "spec valid" : "spec has problems"}</span>
            <span className="ml-auto flex flex-wrap gap-1.5">
              <button className="btn-ghost btn-sm" aria-expanded={diffFor === r.id} onClick={() => setDiffFor(diffFor === r.id ? null : r.id)}>
                {diffFor === r.id ? "hide changes" : "changes"}
              </button>
              {!r.is_head && (
                <>
                  <button
                    className="btn-ghost btn-sm"
                    onClick={() => act(`Couldn't check out r${r.number}`, () => Api.checkoutRevision(wsId, project.id, r.id), `Checked out r${r.number}`)}
                  >
                    check out
                  </button>
                  <button
                    className="btn-ghost btn-sm"
                    onClick={async () => {
                      const ok = await confirm({
                        title: `Restore revision ${r.number}?`,
                        body: `A new revision is created with r${r.number}'s spec and tasks and becomes the head. Nothing is deleted — you can return to the current state any time.`,
                        confirmLabel: "Restore",
                      });
                      if (ok) await act(`Couldn't restore r${r.number}`, () => Api.restoreRevision(wsId, project.id, r.id), `Restored r${r.number} as the new head`);
                    }}
                  >
                    restore
                  </button>
                </>
              )}
              {r.spec_ok && r.tasks_ok > 0 && !r.published_version && (
                <button className="btn-ghost btn-sm" onClick={() => publish({ id: r.id, number: r.number, tasksOk: r.tasks_ok })}>
                  publish
                </button>
              )}
            </span>
          </div>
          {diffFor === r.id && (
            <div className="space-y-2">
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
                    <pre className="code-block max-h-80 overflow-auto" style={{ whiteSpace: "pre" }}>
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
        setErr(errorText(e));
      }
    },
    [wsId, project.id],
  );
  useEffect(() => {
    load(0);
  }, [load]);

  if (err) return <div style={{ color: "var(--muted)" }}>Preview not available yet: {err}</div>;
  if (!data) return <div style={{ color: "var(--muted)" }}>Loading…</div>;
  const { overview, preview } = data;
  const diagram = view === "after" && preview.diagram_after ? preview.diagram_after : preview.diagram_before;

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-end gap-3">
        <div>
          <label className="label" htmlFor="pv-seed">
            Seed
          </label>
          <input id="pv-seed" className="input mono !w-28" type="number" min={0} value={seed} onChange={(e) => setSeed(Math.max(0, Number(e.target.value)))} />
        </div>
        <div>
          <label className="label" htmlFor="pv-tier">
            Tier
          </label>
          <select id="pv-tier" className="input !w-24" value={tier ?? overview.tiers[0]} onChange={(e) => setTier(Number(e.target.value))}>
            {overview.tiers.map((t) => (
              <option key={t} value={t}>
                {t}
              </option>
            ))}
          </select>
        </div>
        <button className="btn btn-field" onClick={() => load(seed, tier)}>
          Generate
        </button>
        <span className="hint pb-2.5">head revision r{project.head?.number} · no model called</span>
      </div>
      <div>
        <div className="label">the agent is told</div>
        <p>{preview.task.goal}</p>
      </div>
      {diagram && (
        <div className="space-y-2">
          {preview.diagram_after && (
            <Tabs
              idPrefix="pv-world"
              variant="pills"
              label="World state"
              tabs={[
                { id: "before", label: "Start" },
                { id: "after", label: "After the reference solution" },
              ]}
              active={view}
              onChange={(id) => setView(id as "before" | "after")}
            />
          )}
          <div {...(preview.diagram_after ? tabPanelProps("pv-world", view) : {})} className="outline-none">
            <DiagramView diagram={diagram} height={380} />
          </div>
        </div>
      )}
      <div>
        <div className="label">reference solution{preview.verdict && ` → reward ${preview.verdict.reward.toFixed(1)}`}</div>
        <ol className="mono mt-1 list-decimal space-y-1 pl-5 text-xs">
          {preview.solution.map((c, i) => (
            <li key={i}>
              <span style={{ color: "var(--accent)" }}>
                {c.name}({Object.entries(c.args).map(([k, v]) => `${k}=${JSON.stringify(v)}`).join(", ")})
              </span>
            </li>
          ))}
        </ol>
      </div>
      <details>
        <summary className="label cursor-pointer">dimensions, tools and system prompt</summary>
        <table className="data inset mt-2">
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

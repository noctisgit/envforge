"use client";

// Builder projects: each one is a conversation with the builder agent that produces a
// versioned environment. Starting a project sends your description as the first message.

import Link from "next/link";
import { useRouter } from "next/navigation";
import { useEffect, useState } from "react";
import { Api, ApiError, type BuilderProjectSummary } from "@/lib/api";
import { useWorkspace } from "@/lib/workspace";

const EXAMPLES = [
  "Train an agent to resolve home-internet support tickets from our chat logs: outages, slow speeds, billing credits.",
  "An agent that handles SaaS subscription support: upgrades, refunds within policy, seat changes, cancellations.",
  "Evaluate how well models troubleshoot printer and device issues for an office IT helpdesk.",
];

export default function BuilderPage() {
  const { current: ws } = useWorkspace();
  const router = useRouter();
  const [projects, setProjects] = useState<BuilderProjectSummary[] | null>(null);
  const [title, setTitle] = useState("");
  const [purpose, setPurpose] = useState("");
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    if (ws) Api.projects(ws.id).then(setProjects).catch(() => setProjects([]));
  }, [ws]);

  if (!ws) return null;

  async function start(e: React.FormEvent) {
    e.preventDefault();
    setBusy(true);
    setErr(null);
    try {
      const p = await Api.createProject(ws!.id, title.trim() || purpose.trim().slice(0, 60), purpose.trim());
      try {
        await Api.sendBuilderMessage(ws!.id, p.id, purpose.trim());
      } catch (ex) {
        // e.g. no key for the builder model: open the project anyway and show the error there
        sessionStorage.setItem(`builder-error-${p.id}`, ex instanceof ApiError ? ex.message : String(ex));
      }
      router.push(`/builder/${p.id}`);
    } catch (ex) {
      setErr(ex instanceof ApiError ? ex.message : String(ex));
      setBusy(false);
    }
  }

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-lg font-semibold">Environment builder</h1>
        <p className="max-w-3xl text-sm" style={{ color: "var(--muted)" }}>
          Describe what you want an agent to learn, upload your data (CSV, TXT or JSON chat logs — personal data is redacted
          before any model sees it), and chat with the builder. It asks questions until it can build a verifiable environment,
          and every change it makes is a revision you can review, diff and roll back. Publish a revision to run and train on it.
        </p>
      </div>

      <form onSubmit={start} className="panel space-y-4 p-5">
        <div className="text-sm font-semibold">New environment</div>
        <div>
          <label className="label">What should the agent learn to do?</label>
          <textarea
            className="input min-h-[96px]"
            value={purpose}
            maxLength={4000}
            onChange={(e) => setPurpose(e.target.value)}
            placeholder="e.g. Resolve customer support tickets for our ISP using the chat logs I'll upload…"
          />
          <div className="mt-2 flex flex-wrap gap-2">
            {EXAMPLES.map((ex) => (
              <button key={ex} type="button" className="rounded-full border px-2.5 py-1 text-left text-xs" style={{ borderColor: "var(--line)", color: "var(--muted)" }} onClick={() => setPurpose(ex)}>
                {ex.slice(0, 70)}…
              </button>
            ))}
          </div>
        </div>
        <div className="flex flex-wrap items-end gap-3">
          <div className="min-w-[16rem] flex-1">
            <label className="label">Name (optional)</label>
            <input className="input" value={title} maxLength={200} onChange={(e) => setTitle(e.target.value)} placeholder="ISP support desk" />
          </div>
          <button className="btn" disabled={busy || purpose.trim().length < 10}>
            {busy ? "Starting…" : "Start building"}
          </button>
        </div>
        <div className="text-xs" style={{ color: "var(--muted)" }}>
          Builder model: <span className="mono">{ws.builder_model}</span> (change it in{" "}
          <Link href="/settings" className="underline">
            Workspace settings
          </Link>
          ). Each message costs a little; extraction of more than 5 conversations always asks you first.
        </div>
        {err && <div className="text-sm" style={{ color: "var(--bad)" }}>{err}</div>}
      </form>

      <div className="panel overflow-x-auto p-2">
        {!projects ? (
          <div className="p-4 text-sm" style={{ color: "var(--muted)" }}>
            Loading…
          </div>
        ) : projects.length === 0 ? (
          <div className="p-4 text-sm" style={{ color: "var(--muted)" }}>
            No projects yet.
          </div>
        ) : (
          <table className="data">
            <thead>
              <tr>
                <th>project</th>
                <th>purpose</th>
                <th>revision</th>
                <th>tasks</th>
                <th>published</th>
                <th>created</th>
              </tr>
            </thead>
            <tbody>
              {projects.map((p) => (
                <tr key={p.id}>
                  <td>
                    <Link href={`/builder/${p.id}`} style={{ color: "var(--accent)" }}>
                      {p.title || p.name}
                    </Link>
                    <div className="mono text-xs" style={{ color: "var(--muted)" }}>
                      {p.name}
                    </div>
                  </td>
                  <td className="max-w-md truncate" style={{ color: "var(--muted)" }}>
                    {p.purpose}
                  </td>
                  <td>{p.head_number ? `r${p.head_number}` : "—"}</td>
                  <td>{p.n_tasks}</td>
                  <td>{p.versions ? `${p.versions} version${p.versions > 1 ? "s" : ""}` : "—"}</td>
                  <td style={{ color: "var(--muted)" }}>{new Date(p.created_at).toLocaleDateString()}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </div>
  );
}

"use client";

// Builder projects: each one is a conversation with the builder agent that produces a
// versioned environment. Starting a project sends your description as the first message.

import Link from "next/link";
import { useRouter } from "next/navigation";
import { useEffect, useState } from "react";
import { Empty, Page, PageHeader, Panel } from "@/components/Page";
import { useToast } from "@/components/Toast";
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
  const toast = useToast();

  useEffect(() => {
    if (ws)
      Api.projects(ws.id)
        .then(setProjects)
        .catch(() => setProjects([]));
  }, [ws]);

  if (!ws) return null;

  async function start(e: React.FormEvent) {
    e.preventDefault();
    setBusy(true);
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
      toast.error("Couldn't start the project", ex);
      setBusy(false);
    }
  }

  return (
    <Page>
      <PageHeader
        title="Environment builder"
        description="Describe what the agent should learn, upload real chat logs (CSV, TXT, JSON), and the builder drafts a declarative spec and extracts tasks from redacted conversations. Every change is a revision you can roll back; publishing makes it a pack you can run."
      />

      <div className="grid grid-cols-1 gap-6 xl:grid-cols-[minmax(0,5fr)_minmax(0,3fr)]">
        <Panel title="New environment">
          <form onSubmit={start} className="space-y-4">
            <div>
              <label className="label" htmlFor="purpose">
                What should the agent learn to do?
              </label>
              <textarea
                id="purpose"
                className="input min-h-[96px]"
                rows={4}
                value={purpose}
                maxLength={4000}
                onChange={(e) => setPurpose(e.target.value)}
                placeholder="e.g. Resolve customer support tickets for our ISP using the chat logs I'll upload…"
              />
              <div className="mt-2 flex flex-wrap gap-1.5">
                {EXAMPLES.map((ex) => (
                  <button
                    key={ex}
                    type="button"
                    className="rounded-full border px-2.5 py-1 text-left text-xs hover:text-[var(--fg)]"
                    style={{
                      borderColor: purpose === ex ? "var(--accent)" : "var(--line)",
                      color: purpose === ex ? "var(--accent)" : "var(--muted)",
                      background: purpose === ex ? "var(--accent-tint)" : undefined,
                    }}
                    aria-pressed={purpose === ex}
                    onClick={() => setPurpose(ex)}
                  >
                    {ex.slice(0, 70)}…
                  </button>
                ))}
              </div>
            </div>
            <div className="flex flex-wrap items-end gap-4">
              <div className="min-w-[15rem] flex-1">
                <label className="label" htmlFor="title">
                  Name (optional)
                </label>
                <input id="title" className="input" value={title} maxLength={200} onChange={(e) => setTitle(e.target.value)} placeholder="ISP support desk" />
              </div>
              <button className="btn btn-field" disabled={busy || purpose.trim().length < 10}>
                {busy ? "Starting…" : "Start building"}
              </button>
            </div>
            <div className="hint border-t pt-3" style={{ borderColor: "var(--line)" }}>
              Builder model: <span className="mono">{ws.builder_model}</span> — change it in{" "}
              <Link href="/settings" className="underline">
                Workspace settings
              </Link>
              . Each message costs a little; extracting more than 5 conversations always asks you first.
            </div>
          </form>
        </Panel>

        <Panel title="From chat logs to a pack">
          <ol className="list-decimal space-y-2 pl-5" style={{ color: "var(--muted)" }}>
            <li>
              <span style={{ color: "var(--fg)" }}>Upload</span> — files are redacted before any model sees them (names, emails, phones, card numbers).
            </li>
            <li>
              <span style={{ color: "var(--fg)" }}>Draft</span> — the builder writes an <span className="mono text-xs">EnvSpec</span>: state, tools, rules, system
              prompt.
            </li>
            <li>
              <span style={{ color: "var(--fg)" }}>Extract</span> — each conversation becomes a seeded task with checks the verifier can score.
            </li>
            <li>
              <span style={{ color: "var(--fg)" }}>Preview</span> — generate tasks and the reference solution; no model is called.
            </li>
            <li>
              <span style={{ color: "var(--fg)" }}>Publish</span> — <span className="mono text-xs">name@vN</span> appears under Environments, ready to run.
            </li>
          </ol>
        </Panel>
      </div>

      <Panel title="Projects" hint={projects ? `${projects.length} in this workspace` : undefined} flush>
        {!projects ? (
          <Empty>Loading…</Empty>
        ) : projects.length === 0 ? (
          <Empty>No projects yet. Describe an environment above to start one.</Empty>
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
                    <Link href={`/builder/${p.id}`} className="font-medium" style={{ color: "var(--accent)" }}>
                      {p.title || p.name}
                    </Link>
                    <div className="mono text-xs" style={{ color: "var(--muted)" }}>
                      {p.name}
                    </div>
                  </td>
                  <td className="max-w-md truncate" style={{ color: "var(--muted)" }}>
                    {p.purpose}
                  </td>
                  <td className="mono">{p.head_number ? `r${p.head_number}` : "—"}</td>
                  <td className="num">{p.n_tasks}</td>
                  <td>{p.versions ? `${p.versions} version${p.versions > 1 ? "s" : ""}` : "—"}</td>
                  <td style={{ color: "var(--muted)" }}>{new Date(p.created_at).toLocaleDateString()}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </Panel>
    </Page>
  );
}

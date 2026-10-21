"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import { useDeleteEnvironment } from "@/components/DeleteEnvironment";
import { Page, PageHeader, Panel } from "@/components/Page";
import { Tabs, type TabItem } from "@/components/Tabs";
import { Api, type Pack } from "@/lib/api";
import { useWorkspace } from "@/lib/workspace";

const kindLabel = (k: string) => (k === "sim" ? "simulation" : "sandbox");

export default function EnvironmentsPage() {
  const { current: ws } = useWorkspace();
  const [packs, setPacks] = useState<Pack[] | null>(null);
  const [kind, setKind] = useState("all");
  const removeEnv = useDeleteEnvironment();
  useEffect(() => {
    if (ws)
      Api.wsPacks(ws.id)
        .then(setPacks)
        .catch(() => setPacks([]));
  }, [ws]);

  const all = packs ?? [];
  const count = (f: (p: Pack) => boolean) => all.filter(f).length;
  const kinds: TabItem[] = [
    { id: "all", label: "all", count: all.length },
    { id: "sim", label: "simulation", count: count((p) => p.kind === "sim") },
    { id: "sandbox", label: "sandbox", count: count((p) => p.kind !== "sim") },
    { id: "custom", label: "built here", count: count((p) => !!p.custom) },
  ].filter((k) => k.id === "all" || k.count);
  const shown = all.filter((p) => kind === "all" || (kind === "custom" ? p.custom : kind === "sim" ? p.kind === "sim" : p.kind !== "sim"));

  return (
    <Page>
      <PageHeader
        title="Environments"
        description="A pack is a world the agent acts on: its state, the tools it may call, a seeded task generator, a verifier that scores the outcome, and the system prompt. Anything that supplies those five things runs through the same loop."
        actions={
          <Link href="/builder" className="btn">
            Build a new one
          </Link>
        }
      />

      {!packs ? (
        <div style={{ color: "var(--muted)" }}>Loading…</div>
      ) : (
        <>
          <div className="flex flex-wrap items-center gap-3">
            <Tabs idPrefix="kind" label="Kind" variant="pills" tabs={kinds} active={kind} onChange={setKind} />
            <span className="hint ml-auto">
              {shown.length} of {all.length} environment{all.length === 1 ? "" : "s"}
            </span>
          </div>

          <div className="grid grid-cols-1 gap-4 md:grid-cols-2 xl:grid-cols-3 min-[2000px]:grid-cols-4">
            {shown.map((p) => (
              // the title link stretches over the whole card; the delete button sits above it
              <div key={p.name} className="panel relative flex min-h-[196px] flex-col gap-3 p-5 hover:border-[var(--accent)]">
                <div className="flex flex-wrap items-center gap-2">
                  <Link
                    href={`/environments/${encodeURIComponent(p.name)}`}
                    className="text-lg font-semibold break-all after:absolute after:inset-0 after:content-['']"
                    style={{ color: "var(--fg)" }}
                  >
                    {p.name}
                  </Link>
                  <span className="pill pill-tag">{kindLabel(p.kind)}</span>
                  {p.custom && <span className="pill pill-running">built in this workspace</span>}
                </div>
                <p className="line-clamp-4 flex-1" style={{ color: "var(--muted)" }}>
                  {p.description}
                </p>
                <div className="hint flex flex-wrap items-center justify-between gap-3 border-t pt-3" style={{ borderColor: "var(--line)" }}>
                  <span>
                    tiers {p.tiers.join(", ")} · up to {p.max_steps} turns
                  </span>
                  {p.custom && ws && (
                    <button
                      type="button"
                      className="btn-ghost btn-sm relative z-10"
                      style={{ color: "var(--bad)" }}
                      onClick={async () => {
                        if (await removeEnv(ws.id, p.name)) setPacks((ps) => (ps ?? []).filter((x) => x.name !== p.name));
                      }}
                    >
                      Delete
                    </button>
                  )}
                </div>
              </div>
            ))}
          </div>
        </>
      )}

      <Panel>
        <div className="flex flex-wrap items-center gap-6">
          <div className="min-w-0 flex-1 basis-[360px] space-y-1">
            <div className="text-lg font-semibold">Writing a pack in code</div>
            <div style={{ color: "var(--muted)" }}>
              Copy <span className="mono text-xs">docs/PACKS.md</span>, satisfy the contract in <span className="mono text-xs">envforge/core/env.py</span>, and
              ship a <span className="mono text-xs">reference_solve()</span> so the scripted model and tests run offline. Packs under{" "}
              <span className="mono text-xs">envforge/packs/</span> are discovered automatically.
            </div>
          </div>
          <Link href="/builder" className="btn-ghost">
            Or describe one in the Builder
          </Link>
        </div>
      </Panel>
    </Page>
  );
}

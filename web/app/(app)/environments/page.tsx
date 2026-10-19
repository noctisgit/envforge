"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import { Page, PageHeader } from "@/components/Page";
import { Api, type Pack } from "@/lib/api";
import { useWorkspace } from "@/lib/workspace";

export default function EnvironmentsPage() {
  const { current: ws } = useWorkspace();
  const [packs, setPacks] = useState<Pack[] | null>(null);
  useEffect(() => {
    if (ws)
      Api.wsPacks(ws.id)
        .then(setPacks)
        .catch(() => setPacks([]));
  }, [ws]);

  return (
    <Page>
      <PageHeader
        title="Environments"
        description={
          <>
            An environment (a <em>pack</em>) is a world an agent acts in: its state, the tools it may call, a generator of seeded tasks at difficulty tiers, and
            a verifier that scores the end state in code. Every pack below is described by its own code, so new domains appear here automatically — including
            environments you publish from the{" "}
            <Link href="/builder" className="underline">
              Builder
            </Link>
            .
          </>
        }
      />
      {!packs ? (
        <div style={{ color: "var(--muted)" }}>Loading…</div>
      ) : (
        <div className="grid grid-cols-1 gap-6 md:grid-cols-2 xl:grid-cols-3 2xl:grid-cols-4 min-[2200px]:grid-cols-5">
          {packs.map((p) => (
            <Link key={p.name} href={`/environments/${encodeURIComponent(p.name)}`} className="panel block space-y-2 p-5 hover:border-[var(--accent)]">
              <div className="flex items-center gap-2">
                <span className="font-semibold">{p.name}</span>
                <span className="pill pill-queued">{p.kind === "sim" ? "simulation" : "sandbox"}</span>
                {p.custom && <span className="pill pill-running">built in this workspace</span>}
              </div>
              <p className="text-sm" style={{ color: "var(--muted)" }}>
                {p.description}
              </p>
              <div className="text-xs" style={{ color: "var(--muted)" }}>
                tiers {p.tiers.join(", ")} · up to {p.max_steps} turns
              </div>
            </Link>
          ))}
        </div>
      )}
    </Page>
  );
}

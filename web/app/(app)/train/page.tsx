"use client";

// Train: the distillation guide on its own page — teacher runs → dataset → LoRA on a GPU →
// test → use — whatever goal Home is set to. Steps 4–6 run on the command line for now;
// the guide fills the commands in with this workspace's runs.

import { Journey, useJourneyData } from "@/components/Journey";
import { Page, PageHeader } from "@/components/Page";
import { useWorkspace } from "@/lib/workspace";

export default function TrainPage() {
  const { current: ws } = useWorkspace();
  const { data, reload } = useJourneyData(ws);
  if (!ws) return null;
  return (
    <Page>
      <PageHeader
        title="Train an open model"
        description={
          <>
            A strong model (the <em>teacher</em>) solves tasks; its clean solves become training data; a small open model (the <em>student</em>) is
            fine-tuned with a LoRA adapter to copy them; the same tasks measure it before and after. Training runs on a rented GPU and is driven
            from the command line for now: each step shows the exact commands.
          </>
        }
      />
      {data ? <Journey goal="distill" ws={ws} data={data} onChange={() => reload().catch(() => {})} /> : <div className="hint">Loading…</div>}
    </Page>
  );
}

"use client";

// "Export training data": rejection sampling as a download. Keeps the episodes that
// scored at least `min reward` from the chosen runs and saves them as JSONL.

import { useState } from "react";
import { useToast } from "@/components/Toast";
import { downloadExport } from "@/lib/api";

export function ExportButton({ wsId, runIds }: { wsId: string; runIds: string[] }) {
  const [open, setOpen] = useState(false);
  const [minReward, setMinReward] = useState(1);
  const [format, setFormat] = useState<"chat" | "trajectory">("chat");
  const toast = useToast();
  const [busy, setBusy] = useState(false);

  async function go() {
    setBusy(true);
    try {
      const n = await downloadExport(wsId, runIds, minReward, format);
      toast.success("Training data exported", `${n} episode${n === 1 ? "" : "s"} with reward ≥ ${minReward}.`);
      setOpen(false);
    } catch (e) {
      toast.error("Export failed", e);
    } finally {
      setBusy(false);
    }
  }

  if (!open)
    return (
      <button className="btn-ghost" onClick={() => setOpen(true)}>
        Export training data
      </button>
    );
  return (
    <div className="flex flex-wrap items-center gap-2 text-sm">
      <label className="flex items-center gap-1">
        <span style={{ color: "var(--muted)" }}>min reward</span>
        <select className="input !w-auto py-1" value={minReward} onChange={(e) => setMinReward(Number(e.target.value))}>
          <option value={1}>1.0 (clean solves)</option>
          <option value={0.3}>0.3 (incl. collateral)</option>
          <option value={0}>0 (everything)</option>
        </select>
      </label>
      <select className="input !w-auto py-1" value={format} onChange={(e) => setFormat(e.target.value as "chat" | "trajectory")}>
        <option value="chat">chat JSONL (for SFT)</option>
        <option value="trajectory">raw trajectories</option>
      </select>
      <button className="btn" onClick={go} disabled={busy}>
        Download
      </button>
      <button className="btn-ghost" onClick={() => setOpen(false)}>
        Cancel
      </button>
    </div>
  );
}

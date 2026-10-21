"use client";

// "Build a training dataset": opens the Datasets page with these runs preselected,
// where the filters, a live preview (split, overlap with evaluation runs) and saving
// live. Runs on evaluation seeds are listed there as not usable, with the reason.
// (The raw JSONL export stays available as GET /api/workspaces/{id}/runs/export.)

import Link from "next/link";

export function ExportButton({ runIds }: { wsId?: string; runIds: string[] }) {
  const qs = runIds.length ? `?runs=${runIds.map(encodeURIComponent).join(",")}` : "";
  return (
    <Link href={`/datasets${qs}`} className="btn-ghost">
      Build a training dataset
    </Link>
  );
}

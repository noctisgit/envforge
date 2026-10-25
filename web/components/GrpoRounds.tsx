"use client";

// GRPO's result, round by round (Fine-tune job detail and the Models page): how well the
// policy did on its own training rollouts each round, and how much of it carried signal.

import Link from "next/link";
import type { GrpoRound } from "@/lib/api";

/** GRPO's result, round by round: the policy's reward on its own training rollouts. */
export function GrpoRounds({ rounds }: { rounds: GrpoRound[] }) {
  if (!rounds.length) return <p className="hint">No round finished yet.</p>;
  return (
    <div className="overflow-x-auto">
      <table className="data inset">
        <thead>
          <tr>
            <th>Round</th>
            <th className="text-right">Mean reward</th>
            <th className="text-right">Solved</th>
            <th className="text-right">Groups with signal</th>
            <th className="text-right">Trained on</th>
            <th>Rollouts</th>
          </tr>
        </thead>
        <tbody>
          {rounds.map((r) => (
            <tr key={r.round}>
              <td className="num">{r.round}</td>
              <td className="num text-right">{r.mean_reward.toFixed(2)}</td>
              <td className="num text-right">{Math.round(r.solve_rate * 100)}%</td>
              <td className="num text-right">
                {r.informative_groups} / {r.groups}
              </td>
              <td className="num text-right">{r.trained_on ? `${r.trained_on} attempts` : "skipped"}</td>
              <td>
                <Link href={`/runs/${encodeURIComponent(r.run_id)}`} className="underline">
                  run
                </Link>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

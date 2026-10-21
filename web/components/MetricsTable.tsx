import { rateTone, toneColor } from "@/components/Page";
import { fmtCost, fmtTokens, type Metrics } from "@/lib/api";

const pct = (x: number) => `${(x * 100).toFixed(1)}%`;

/** Overall metrics, then one row per tier. Solve rate is coloured: ≥ 80% good, 40–80% warn, below bad. */
export function MetricsTable({ overall, byTier }: { overall: Metrics; byTier: Record<string, Metrics> }) {
  const rows: [string, Metrics][] = [["overall", overall], ...Object.entries(byTier).map(([t, m]) => [`tier ${t}`, m] as [string, Metrics])];
  return (
    <table className="data">
      <thead>
        <tr>
          <th></th>
          {["n", "solve", "reward", "steps", "calls", "tool err", "collateral", "budget out", "tokens", "cost"].map((h) => (
            <th key={h} className="text-right">
              {h}
            </th>
          ))}
        </tr>
      </thead>
      <tbody>
        {rows.map(([label, m]) => {
          const tone = rateTone(m.solve_rate);
          return (
            <tr key={label} className={label === "overall" ? "font-semibold" : ""}>
              <td>{label}</td>
              <td className="num text-right">{m.n}</td>
              <td className="num text-right" style={{ color: toneColor(tone) }}>
                {pct(m.solve_rate)}
              </td>
              <td className="num text-right">{m.mean_reward.toFixed(2)}</td>
              <td className="num text-right">{m.mean_steps.toFixed(1)}</td>
              <td className="num text-right">{m.mean_tool_calls.toFixed(1)}</td>
              <td className="num text-right">{pct(m.tool_error_rate)}</td>
              <td className="num text-right" style={{ color: m.collateral_rate > 0 ? "var(--warn)" : undefined }}>
                {pct(m.collateral_rate)}
              </td>
              <td className="num text-right">{pct(m.max_steps_rate)}</td>
              <td className="num text-right">{fmtTokens(m.input_tokens + m.output_tokens)}</td>
              <td className="num mono text-right">{fmtCost(m.cost_usd)}</td>
            </tr>
          );
        })}
      </tbody>
    </table>
  );
}

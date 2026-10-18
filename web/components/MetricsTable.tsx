import type { Metrics } from "@/lib/api";

const pct = (x: number) => `${Math.round(x * 100)}%`;

export function MetricsTable({ overall, byTier }: { overall: Metrics; byTier: Record<string, Metrics> }) {
  const rows: [string, Metrics][] = [["overall", overall], ...Object.entries(byTier).map(([t, m]) => [`tier ${t}`, m] as [string, Metrics])];
  return (
    <table className="data">
      <thead>
        <tr>
          <th></th>
          <th>n</th>
          <th>solve</th>
          <th>reward</th>
          <th>steps</th>
          <th>calls</th>
          <th>tool err</th>
          <th>collateral</th>
          <th>budget out</th>
          <th>tokens</th>
        </tr>
      </thead>
      <tbody>
        {rows.map(([label, m]) => (
          <tr key={label} className={label === "overall" ? "font-semibold" : ""}>
            <td>{label}</td>
            <td>{m.n}</td>
            <td style={{ color: m.solve_rate >= 0.8 ? "var(--good)" : m.solve_rate >= 0.4 ? "var(--warn)" : "var(--bad)" }}>{pct(m.solve_rate)}</td>
            <td>{m.mean_reward.toFixed(2)}</td>
            <td>{m.mean_steps.toFixed(1)}</td>
            <td>{m.mean_tool_calls.toFixed(1)}</td>
            <td>{pct(m.tool_error_rate)}</td>
            <td style={{ color: m.collateral_rate > 0 ? "var(--warn)" : undefined }}>{pct(m.collateral_rate)}</td>
            <td>{pct(m.max_steps_rate)}</td>
            <td>{(m.input_tokens + m.output_tokens).toLocaleString()}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

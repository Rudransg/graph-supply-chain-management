// frontend/src/components/AnalyticsPanel.jsx
import { AreaChart, Area, XAxis, YAxis, Tooltip, ResponsiveContainer } from "recharts";

export default function AnalyticsPanel({
  title = "Supply chain analytics",
  kpis = [],
  trend = [],
}) {
  // kpis: [{ label, value, unit, delta }]
  // trend: [{ x, y }]
  return (
    <div style={S.card}>
      <div style={S.header}>
        <h2 style={S.title}>{title}</h2>
        <p style={S.subtitle}>Key KPIs and recent trend</p>
      </div>

      <div style={S.kpiRow}>
        {kpis.map((kpi) => (
          <div key={kpi.label} style={S.kpiCard}>
            <p style={S.kpiLabel}>{kpi.label}</p>
            <p style={S.kpiValue}>
              {kpi.value}
              {kpi.unit ? <span style={S.kpiUnit}> {kpi.unit}</span> : null}
            </p>
            {kpi.delta != null && (
              <p
                style={{
                  ...S.kpiDelta,
                  color: kpi.delta >= 0 ? "#4ade80" : "#f97373",
                }}
              >
                {kpi.delta >= 0 ? "+" : ""}
                {kpi.delta}%
              </p>
            )}
          </div>
        ))}
        {kpis.length === 0 && (
          <p style={S.empty}>No KPIs yet – wire this to your summary API.</p>
        )}
      </div>

      <div style={S.chartWrap}>
        <ResponsiveContainer width="100%" height={140}>
          <AreaChart data={trend}>
            <defs>
              <linearGradient id="analyticsArea" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor="#38bdf8" stopOpacity={0.9} />
                <stop offset="100%" stopColor="#38bdf8" stopOpacity={0} />
              </linearGradient>
            </defs>
            <XAxis dataKey="x" hide />
            <YAxis hide />
            <Tooltip />
            <Area
              type="monotone"
              dataKey="y"
              stroke="#38bdf8"
              fill="url(#analyticsArea)"
              strokeWidth={2}
            />
          </AreaChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}

const S = {
  card: {
    background: "#020617",
    borderRadius: 16,
    padding: 20,
    border: "1px solid #1f2937",
    display: "flex",
    flexDirection: "column",
  },
  header: { marginBottom: 16 },
  title: { margin: 0, color: "#e5e7eb", fontSize: 16, fontWeight: 600 },
  subtitle: { margin: 0, marginTop: 4, color: "#9ca3af", fontSize: 12 },
  kpiRow: {
    display: "flex",
    gap: 16,
    flexWrap: "wrap",
    marginBottom: 16,
  },
  kpiCard: {
    flex: "1 1 120px",
    minWidth: 120,
    padding: 10,
    borderRadius: 12,
    background: "#020617",
    border: "1px solid #111827",
  },
  kpiLabel: { margin: 0, color: "#9ca3af", fontSize: 11 },
  kpiValue: { margin: "6px 0 0", color: "#e5e7eb", fontSize: 18, fontWeight: 600 },
  kpiUnit: { fontSize: 11, color: "#9ca3af", marginLeft: 2 },
  kpiDelta: { margin: "4px 0 0", fontSize: 11 },
  empty: { color: "#6b7280", fontSize: 13 },
  chartWrap: { width: "100%", height: 140 },
};

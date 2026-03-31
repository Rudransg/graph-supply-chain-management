// frontend/src/components/ForecastPanel.jsx
import { LineChart, Line, XAxis, YAxis, Tooltip, CartesianGrid, ResponsiveContainer } from "recharts";

export default function ForecastPanel({ title = "Demand forecast", data = [] }) {
  return (
    <div style={S.card}>
      <div style={S.header}>
        <h2 style={S.title}>{title}</h2>
        <p style={S.subtitle}>Historical vs predicted</p>
      </div>

      <div style={S.chartWrap}>
        <ResponsiveContainer width="100%" height={260}>
          <LineChart data={data} margin={{ top: 10, right: 20, left: -10, bottom: 0 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
            <XAxis dataKey="date" stroke="#64748b" />
            <YAxis stroke="#64748b" />
            <Tooltip />
            <Line type="monotone" dataKey="actual" stroke="#38bdf8" dot={false} strokeWidth={2} />
            <Line type="monotone" dataKey="predicted" stroke="#a855f7" dot={false} strokeWidth={2} />
          </LineChart>
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
  header: { marginBottom: 12 },
  title: { margin: 0, color: "#e5e7eb", fontSize: 16, fontWeight: 600 },
  subtitle: { margin: 0, marginTop: 4, color: "#9ca3af", fontSize: 12 },
  chartWrap: { width: "100%", height: 260 },
};

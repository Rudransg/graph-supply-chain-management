import { useEffect, useState } from "react";
import { fetchMetrics, fetchHealth, fetchModelInfo } from "../api";
import MetricCard from "./MetricCard";

export default function Overview() {
  const [metrics,   setMetrics]   = useState(null);
  const [health,    setHealth]    = useState(null);
  const [modelInfo, setModelInfo] = useState(null);
  const [error,     setError]     = useState("");

  useEffect(() => {
    fetchMetrics()
      .then(r => setMetrics(r.data))
      .catch(() => setError("Backend unreachable — is it running on :8000?"));
    fetchHealth().then(r => setHealth(r.data)).catch(() => {});
    fetchModelInfo().then(r => setModelInfo(r.data)).catch(() => {});
  }, []);

  return (
    <div>
      <h2 style={S.heading}>Overview</h2>

      {/* status bar */}
      <div style={S.statusBar}>
        <span style={{ color: health?.status === "ok" ? "#4ade80" : "#f87171", fontSize: 13 }}>
          ● Backend {health?.status === "ok" ? "Online" : "Offline"}
        </span>
        <span style={S.pill}>Device: {health?.device ?? "—"}</span>
        <span style={S.pill}>Nodes: {health?.num_nodes ?? "—"}</span>
        <span style={S.pill}>Conv: {health?.conv_type ?? "—"}</span>
        <span style={S.pill}>Layers: {health?.layers ?? "—"}</span>
      </div>

      {error && <p style={{ color: "#f87171", marginBottom: 16 }}>{error}</p>}

      {/* metric cards */}
      <div style={S.cards}>
        <MetricCard label="MAE"              value={metrics?.mae}                  color="#7dd3fc" />
        <MetricCard label="MSE"              value={metrics?.mse}                  color="#a78bfa" />
        <MetricCard label="RMSE"             value={metrics?.rmse}                 color="#fb923c" />
        <MetricCard label="R²"              value={metrics?.r2}                   color="#4ade80" />
        <MetricCard label="Asymmetric Loss"  value={metrics?.test_asymmetric_loss} color="#f472b6" />
      </div>

      {/* model config table */}
      {modelInfo && (
        <div style={{ marginTop: 32 }}>
          <h3 style={S.subheading}>Model Configuration</h3>
          <table style={S.table}>
            <tbody>
              {Object.entries(modelInfo).map(([k, v]) => (
                <tr key={k}>
                  <td style={S.tdKey}>{k}</td>
                  <td style={S.tdVal}>
                    {Array.isArray(v) ? v.join(", ") : String(v)}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}

const S = {
  heading:    { color: "#e2e8f0", marginBottom: 20, fontSize: 22, fontWeight: 700 },
  subheading: { color: "#94a3b8", marginBottom: 12, fontSize: 14, fontWeight: 600 },
  statusBar:  { display: "flex", gap: 12, alignItems: "center", flexWrap: "wrap", marginBottom: 24 },
  pill: {
    background: "#1e293b", color: "#94a3b8",
    borderRadius: 20, padding: "3px 12px", fontSize: 12,
  },
  cards: { display: "flex", gap: 14, flexWrap: "wrap" },
  table: { borderCollapse: "collapse", width: "100%", maxWidth: 520 },
  tdKey: {
    padding: "8px 16px 8px 0", color: "#64748b",
    fontSize: 13, borderBottom: "1px solid #1e293b", width: 200,
  },
  tdVal: {
    padding: "8px 0", color: "#e2e8f0",
    fontSize: 13, borderBottom: "1px solid #1e293b",
  },
};

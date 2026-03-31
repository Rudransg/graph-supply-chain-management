import { useEffect, useState } from "react";
import { fetchModelInfo } from "../api";

const SECTION = ({ title, rows }) => (
  <div style={S.section}>
    <p style={S.sectionTitle}>{title}</p>
    <table style={S.table}>
      <tbody>
        {rows.map(([k, v]) => (
          <tr key={k}>
            <td style={S.tdKey}>{k}</td>
            <td style={S.tdVal}>{Array.isArray(v) ? v.join(", ") : String(v)}</td>
          </tr>
        ))}
      </tbody>
    </table>
  </div>
);

export default function ModelInfoPanel() {
  const [info,  setInfo]  = useState(null);
  const [error, setError] = useState("");

  useEffect(() => {
    fetchModelInfo()
      .then(r => setInfo(r.data))
      .catch(() => setError("Could not load model info"));
  }, []);

  if (error) return <p style={{ color: "#f87171" }}>{error}</p>;
  if (!info) return <p style={{ color: "#64748b" }}>Loading…</p>;

  return (
    <div>
      <h2 style={S.heading}>Model Info</h2>
      <p style={S.sub}>
        Architecture and training configuration for the deployed SupplyGraphModel.
      </p>

      <div style={S.grid}>
        <SECTION
          title="Architecture"
          rows={[
            ["node_type",       info.node_type],
            ["num_nodes",       info.num_nodes],
            ["conv_type",       info.conv_type],
            ["layers",          info.layers],
            ["in_channels",     info.in_channels],
            ["hidden_channels", info.hidden_channels],
            ["out_channels",    info.out_channels],
            ["aggregation",     info.aggregation],
          ]}
        />
        <SECTION
          title="Training"
          rows={[
            ["target_signal",   info.target_signal],
            ["rolling_window",  info.rolling_window],
            ["epochs",          info.epochs],
            ["learning_rate",   info.learning_rate],
            ["loss_alpha",      info.loss_alpha],
          ]}
        />
        <SECTION
          title="Edge Relations"
          rows={info.edge_relations.map((r, i) => [`relation_${i + 1}`, r])}
        />
      </div>

      {/* MLflow run badge */}
      <div style={S.badge}>
        <span style={{ color: "#64748b", fontSize: 12 }}>MLflow Run ID</span>
      </div>
    </div>
  );
}

const S = {
  heading:      { color: "#e2e8f0", fontSize: 22, fontWeight: 700, marginBottom: 8 },
  sub:          { color: "#64748b", fontSize: 13, marginBottom: 24 },
  grid:         { display: "flex", gap: 16, flexWrap: "wrap" },
  section: {
    background: "#0d1424", border: "1px solid #1e293b",
    borderRadius: 12, padding: "18px 20px", flex: "1 1 220px",
  },
  sectionTitle: { color: "#94a3b8", fontSize: 12, fontWeight: 600, textTransform: "uppercase", letterSpacing: "0.06em", marginBottom: 12 },
  table:        { borderCollapse: "collapse", width: "100%" },
  tdKey: {
    padding: "7px 12px 7px 0", color: "#64748b",
    fontSize: 12, borderBottom: "1px solid #1e293b", width: 150,
  },
  tdVal: {
    padding: "7px 0", color: "#e2e8f0",
    fontSize: 12, borderBottom: "1px solid #1e293b",
  },
  badge: {
    marginTop: 24, background: "#0d1424",
    border: "1px solid #1e293b", borderRadius: 10,
    padding: "12px 18px", display: "inline-flex", alignItems: "center",
  },
};

export default function MetricCard({ label, value, color = "#7dd3fc", prefix = "", suffix = "" }) {
  const display = value !== undefined && value !== null
    ? `${prefix}${Number(value).toFixed(4)}${suffix}`
    : "—";

  return (
    <div style={S.card}>
      <p style={S.label}>{label}</p>
      <h2 style={{ ...S.value, color }}>{display}</h2>
    </div>
  );
}

const S = {
  card: {
    background: "#0d1424",
    border: "1px solid #1e293b",
    borderRadius: 12,
    padding: "18px 22px",
    minWidth: 155,
    flex: "1 1 140px",
  },
  label: { color: "#64748b", fontSize: 12, fontWeight: 500, letterSpacing: "0.05em", textTransform: "uppercase" },
  value: { fontSize: 24, fontWeight: 700, marginTop: 8 },
};

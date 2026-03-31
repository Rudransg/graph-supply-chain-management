// frontend/src/components/FactoryGraph.jsx
export default function FactoryGraph({ title = "Factory graph", nodes = [], edges = [] }) {
  return (
    <div style={S.card}>
      <div style={S.header}>
        <h2 style={S.title}>{title}</h2>
        <p style={S.subtitle}>High‑level view of factories and links</p>
      </div>

      <div style={S.body}>
        <p style={S.hint}>
          Graph visualization placeholder. Later: render supply graph (nodes: {nodes.length}, edges: {edges.length}).
        </p>
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
    minHeight: 260,
    display: "flex",
    flexDirection: "column",
  },
  header: { marginBottom: 12 },
  title: { margin: 0, color: "#e5e7eb", fontSize: 16, fontWeight: 600 },
  subtitle: { margin: 0, marginTop: 4, color: "#9ca3af", fontSize: 12 },
  body: {
    flex: 1,
    borderRadius: 12,
    border: "1px dashed #1f2937",
    background: "radial-gradient(circle at top, #020617 0, #020617 40%, #020617)",
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    padding: 16,
  },
  hint: {
    color: "#6b7280",
    fontSize: 13,
    textAlign: "center",
    maxWidth: 260,
    lineHeight: 1.5,
  },
};

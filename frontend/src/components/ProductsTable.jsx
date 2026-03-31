// frontend/src/components/ProductsTable.jsx
export default function ProductsTable({ title = "Products", products = [] }) {
  return (
    <div style={S.card}>
      <div style={S.header}>
        <h2 style={S.title}>{title}</h2>
        <p style={S.subtitle}>Top products by risk / opportunity</p>
      </div>

      <table style={S.table}>
        <thead>
          <tr>
            <th style={S.th}>#</th>
            <th style={S.th}>Product</th>
            <th style={S.th}>Factory</th>
            <th style={S.th}>Score</th>
          </tr>
        </thead>
        <tbody>
          {products.length === 0 && (
            <tr>
              <td style={S.empty} colSpan={4}>No data</td>
            </tr>
          )}
          {products.map((row, idx) => (
            <tr key={row.id ?? `${row.name}-${idx}`} style={S.tr}>
              <td style={S.tdIndex}>{idx + 1}</td>
              <td style={S.tdMain}>{row.name}</td>
              <td style={S.td}>{row.factory}</td>
              <td style={S.tdScore}>{row.score?.toFixed?.(2) ?? row.score}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

const S = {
  card: {
    background: "#020617",
    borderRadius: 16,
    padding: 20,
    border: "1px solid #1f2937",
  },
  header: { marginBottom: 12 },
  title: { margin: 0, color: "#e5e7eb", fontSize: 16, fontWeight: 600 },
  subtitle: { margin: 0, marginTop: 4, color: "#9ca3af", fontSize: 12 },
  table: {
    width: "100%",
    borderCollapse: "collapse",
    fontSize: 13,
  },
  th: {
    textAlign: "left",
    padding: "8px 10px",
    color: "#9ca3af",
    borderBottom: "1px solid #1f2937",
    fontWeight: 500,
  },
  tr: {
    borderBottom: "1px solid #020617",
  },
  tdIndex: {
    padding: "8px 10px",
    color: "#6b7280",
    width: 40,
  },
  tdMain: {
    padding: "8px 10px",
    color: "#e5e7eb",
    fontWeight: 500,
  },
  td: {
    padding: "8px 10px",
    color: "#9ca3af",
  },
  tdScore: {
    padding: "8px 10px",
    color: "#f97316",
    fontVariantNumeric: "tabular-nums",
  },
  empty: {
    padding: "12px 10px",
    textAlign: "center",
    color: "#6b7280",
  },
};

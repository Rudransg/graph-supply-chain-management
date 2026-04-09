const NAV = [
  { id: "dashboard",   label: "Dashboard",   icon: "◈" },
  { id: "analytics",   label: "Analytics",   icon: "📈" },
  { id: "prediction",  label: "Prediction",  icon: "⬡" },
  { id: "whatif",      label: "What-If",     icon: "❓" },
  { id: "experiments", label: "Experiments", icon: "⌬" },
  { id: "model",       label: "Model Info",  icon: "⬢" },
  { id: "pipeline",    label: "Pipeline",    icon: "⟳" },
];

export default function Sidebar({ active, setActive }) {
  return (
    <aside style={S.aside}>
      <div style={S.brand}>
        <span style={S.brandIcon}>⬡</span>
        <div>
          <div style={S.brandTitle}>Supply Graph</div>
          <div style={S.brandSub}>HeteroSAGE · MLflow · Docker</div>
        </div>
      </div>

      <nav style={{ marginTop: 32 }}>
        {NAV.map((item) => (
          <button
            key={item.id}
            onClick={() => setActive(item.id)}
            style={{
              ...S.navBtn,
              background: active === item.id ? "#1e3a5f" : "transparent",
              color: active === item.id ? "#7dd3fc" : "#64748b",
              borderLeft:
                active === item.id
                  ? "3px solid #3b82f6"
                  : "3px solid transparent",
            }}
          >
            <span style={{ marginRight: 10, fontSize: 16 }}>{item.icon}</span>
            {item.label}
          </button>
        ))}
      </nav>

      <div style={S.footer}>
        <div style={S.footerDot} />
        <span style={{ color: "#475569", fontSize: 11 }}>
          node_type: rolled_prod
        </span>
      </div>
    </aside>
  );
}

const S = {
  aside: {
    width: 230,
    minHeight: "100vh",
    background: "#0d1424",
    borderRight: "1px solid #1e293b",
    display: "flex",
    flexDirection: "column",
    padding: "24px 0",
    position: "sticky",
    top: 0,
  },
  brand: {
    display: "flex",
    alignItems: "center",
    gap: 12,
    padding: "0 20px 24px",
    borderBottom: "1px solid #1e293b",
  },
  brandIcon: { fontSize: 28, color: "#3b82f6" },
  brandTitle: { fontSize: 15, fontWeight: 700, color: "#e2e8f0" },
  brandSub: { fontSize: 10, color: "#475569", marginTop: 2 },
  navBtn: {
    display: "flex",
    alignItems: "center",
    width: "100%",
    padding: "11px 20px",
    border: "none",
    cursor: "pointer",
    fontSize: 13,
    fontWeight: 500,
    textAlign: "left",
    transition: "all 0.15s",
  },
  footer: {
    marginTop: "auto",
    padding: "16px 20px",
    display: "flex",
    alignItems: "center",
    gap: 8,
  },
  footerDot: {
    width: 7,
    height: 7,
    borderRadius: "50%",
    background: "#4ade80",
  },
};


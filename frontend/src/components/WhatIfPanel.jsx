import { useState } from "react";
import { runWhatIf } from "../api";

const SIGNAL_COLORS = {
  production_unit:  "#38bdf8",
  delivery_unit:    "#34d399",
  sales_order_unit: "#f59e0b",
};

const SIGNAL_LABELS = {
  production_unit:  "Production",
  delivery_unit:    "Delivery",
  sales_order_unit: "Sales Order",
};

export default function WhatIfPanel({ products = [], factories = [] }) {
  const [selectedProduct,   setProduct]   = useState(products[0] ?? "");
  const [zeroProd,          setZeroProd]  = useState([]);
  const [zeroFact,          setZeroFact]  = useState([]);
  const [result,            setResult]    = useState(null);
  const [loading,           setLoading]   = useState(false);
  const [error,             setError]     = useState("");

  async function runScenario() {
    setLoading(true); setError(""); setResult(null);
    try {
      const res = await runWhatIf({
        product_name:      selectedProduct,
        zeroed_products:   zeroProd,
        zeroed_factories:  zeroFact,
        capacity_overrides: {},
        dropped_relations: [],
      });
      setResult(res.data);
    } catch (e) {
      setError(e.response?.data?.detail ?? e.message);
    } finally {
      setLoading(false);
    }
  }

  function toggleItem(list, setList, item) {
    setList(list.includes(item) ? list.filter(x => x !== item) : [...list, item]);
  }

  return (
    <div>
      <h2 style={S.heading}>What-If Scenario Builder</h2>
      <p style={S.sub}>Simulate disruptions and see their impact on demand forecasts.</p>

      {/* product selector */}
      <div style={S.section}>
        <p style={S.label}>Forecast For</p>
        <select value={selectedProduct} onChange={e => setProduct(e.target.value)} style={S.select}>
          {products.map(p => <option key={p} value={p}>{p}</option>)}
        </select>
      </div>

      {/* stop products */}
      <div style={S.section}>
        <p style={S.label}>Stop Producing These Products</p>
        <div style={S.chipRow}>
          {products.filter(p => p !== selectedProduct).map(p => (
            <div
              key={p}
              onClick={() => toggleItem(zeroProd, setZeroProd, p)}
              style={{ ...S.chip, ...(zeroProd.includes(p) ? S.chipActive : {}) }}
            >
              {p}
            </div>
          ))}
        </div>
      </div>

      {/* offline factories */}
      <div style={S.section}>
        <p style={S.label}>Take Factories Offline</p>
        <div style={S.chipRow}>
          {factories.map(f => (
            <div
              key={f}
              onClick={() => toggleItem(zeroFact, setZeroFact, f)}
              style={{ ...S.chip, ...(zeroFact.includes(f) ? S.chipDanger : {}) }}
            >
              {f}
            </div>
          ))}
        </div>
      </div>

      <button onClick={runScenario} disabled={loading} style={S.btn}>
        {loading ? "Running…" : "Run Scenario"}
      </button>

      {error && <p style={{ color: "#f87171", marginTop: 12 }}>{error}</p>}

      {/* results */}
      {result && (
        <div style={{ marginTop: 28 }}>
          <div style={S.grid}>
            {Object.keys(result.baseline).map(signal => {
              const base  = result.baseline[signal];
              const scen  = result.scenario[signal];
              const delta = result.delta[signal];
              const pct   = result.delta_pct[signal];
              const drop  = delta < 0;

              return (
                <div key={signal} style={S.card}>
                  <p style={S.cardLabel}>{SIGNAL_LABELS[signal]}</p>

                  {/* baseline vs scenario */}
                  <div style={S.compareRow}>
                    <div>
                      <p style={S.tiny}>Baseline</p>
                      <p style={{ ...S.big, color: SIGNAL_COLORS[signal] }}>{base.toFixed(0)}</p>
                    </div>
                    <div style={S.arrow}>→</div>
                    <div>
                      <p style={S.tiny}>Scenario</p>
                      <p style={{ ...S.big, color: drop ? "#f87171" : "#4ade80" }}>
                        {scen.toFixed(0)}
                      </p>
                    </div>
                  </div>

                  {/* delta badge */}
                  <div style={{ ...S.badge, background: drop ? "#2d1a1a" : "#1a2d1a",
                                            color: drop ? "#f87171" : "#4ade80",
                                            border: `1px solid ${drop ? "#7f1d1d" : "#14532d"}` }}>
                    {drop ? "▼" : "▲"} {Math.abs(delta).toFixed(0)} ({Math.abs(pct)}%)
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
}

const S = {
  heading:    { color: "#e2e8f0", fontSize: 22, fontWeight: 700, marginBottom: 8 },
  sub:        { color: "#64748b", fontSize: 13, marginBottom: 24 },
  section:    { marginBottom: 20 },
  label:      { color: "#64748b", fontSize: 11, textTransform: "uppercase",
                letterSpacing: "0.05em", marginBottom: 8 },
  select:     { background: "#0d1424", color: "#e2e8f0", border: "1px solid #334155",
                borderRadius: 8, padding: "10px 14px", fontSize: 13, minWidth: 220 },
  chipRow:    { display: "flex", flexWrap: "wrap", gap: 8 },
  chip:       { background: "#0d1424", border: "1px solid #334155", borderRadius: 999,
                padding: "6px 14px", fontSize: 12, color: "#94a3b8", cursor: "pointer" },
  chipActive: { background: "#1e3a5f", border: "1px solid #38bdf8", color: "#38bdf8" },
  chipDanger: { background: "#3d1a1a", border: "1px solid #f87171", color: "#f87171" },
  btn:        { background: "#1d4ed8", color: "#fff", border: "none", borderRadius: 8,
                padding: "12px 28px", fontSize: 14, fontWeight: 600, cursor: "pointer",
                marginTop: 8 },
  grid:       { display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(200px, 1fr))", gap: 16 },
  card:       { background: "#0d1424", border: "1px solid #1e293b", borderRadius: 12, padding: "18px 20px" },
  cardLabel:  { color: "#64748b", fontSize: 11, textTransform: "uppercase",
                letterSpacing: "0.05em", marginBottom: 12 },
  compareRow: { display: "flex", alignItems: "center", gap: 12, marginBottom: 12 },
  tiny:       { color: "#475569", fontSize: 10, marginBottom: 2 },
  big:        { fontSize: 24, fontWeight: 700 },
  arrow:      { color: "#475569", fontSize: 18 },
  badge:      { display: "inline-block", borderRadius: 999, padding: "4px 12px",
                fontSize: 12, fontWeight: 600 },
};
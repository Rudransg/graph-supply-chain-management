import { useEffect, useState } from "react";
import { fetchProducts, runPredict } from "../api";
import {
  BarChart, Bar, XAxis, YAxis, Tooltip,
  CartesianGrid, ResponsiveContainer, Cell,
} from "recharts";

export default function PredictionPanel() {
  const [products,     setProducts]     = useState({});
  const [selectedName, setSelectedName] = useState("");
  const [result,       setResult]       = useState(null);
  const [loading,      setLoading]      = useState(false);
  const [error,        setError]        = useState("");

  useEffect(() => {
    fetchProducts()
      .then(r => {
        setProducts(r.data.products);
        const first = Object.values(r.data.products)[0];
        if (first) setSelectedName(first);
      })
      .catch(() => setError("Could not load product list"));
  }, []);

  const handlePredict = async () => {
    if (!selectedName) return;
    setLoading(true); setError(""); setResult(null);
    try {
      const res = await runPredict({ product_name: selectedName });
      setResult(res.data);
    } catch (e) {
      setError(e.response?.data?.detail ?? e.message);
    } finally {
      setLoading(false);
    }
  };

  const chartData = result
    ? [{ name: result.product_name, value: parseFloat(result.prediction.toFixed(4)) }]
    : [];

  return (
    <div>
      <h2 style={S.heading}>Demand Prediction</h2>
      <p style={S.sub}>
        Select a product node and run the HeteroSAGE model for a
        one-step-ahead production weight forecast.
      </p>

      {/* controls */}
      <div style={S.row}>
        <select
          value={selectedName}
          onChange={e => setSelectedName(e.target.value)}
          style={S.select}
        >
          {Object.entries(products).map(([idx, name]) => (
            <option key={idx} value={name}>{name}</option>
          ))}
        </select>

        <button onClick={handlePredict} disabled={loading} style={S.btn}>
          {loading ? "Running inference…" : "Run Predict"}
        </button>
      </div>

      {error && <p style={{ color: "#f87171", marginTop: 12 }}>{error}</p>}

      {result && (
        <div style={{ marginTop: 28 }}>

          {/* result card */}
          <div style={S.resultCard}>
            <div style={S.resultRow}>
              <div>
                <p style={S.resultLabel}>Product</p>
                <p style={S.resultValue}>{result.product_name}</p>
              </div>
              <div>
                <p style={S.resultLabel}>Node Index</p>
                <p style={S.resultValue}>{result.product_idx}</p>
              </div>
              <div>
                <p style={S.resultLabel}>Predicted Demand</p>
                <p style={{ ...S.resultValue, color: "#4ade80", fontSize: 28 }}>
                  {result.prediction.toFixed(4)}
                </p>
              </div>
              <div>
                <p style={S.resultLabel}>Model Version</p>
                <p style={S.resultValue}>{result.model_version}</p>
              </div>
            </div>
            <p style={{ color: "#475569", fontSize: 11, marginTop: 4 }}>
              
            </p>
          </div>


          {/* bar chart */}
          <div style={{ marginTop: 24 }}>
            <p style={S.chartTitle}>Predicted Production Weight</p>
            <ResponsiveContainer width="100%" height={260}>
              <BarChart data={chartData} margin={{ top: 10, right: 20, left: 0, bottom: 10 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                <XAxis dataKey="name" stroke="#475569" tick={{ fill: "#94a3b8", fontSize: 12 }} />
                <YAxis stroke="#475569" tick={{ fill: "#94a3b8", fontSize: 12 }} />
                <Tooltip
                  contentStyle={{ background: "#0d1424", border: "1px solid #1e293b", borderRadius: 8 }}
                  labelStyle={{ color: "#94a3b8" }}
                  itemStyle={{ color: "#4ade80" }}
                />
                <Bar dataKey="value" radius={[6, 6, 0, 0]}>
                  <Cell fill="#3b82f6" />
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}
    </div>
  );
}

const S = {
  heading:     { color: "#e2e8f0", fontSize: 22, fontWeight: 700, marginBottom: 8 },
  sub:         { color: "#64748b", fontSize: 13, marginBottom: 24 },
  row:         { display: "flex", gap: 12, alignItems: "center", flexWrap: "wrap" },
  select: {
    background: "#0d1424", color: "#e2e8f0",
    border: "1px solid #334155", borderRadius: 8,
    padding: "10px 14px", fontSize: 13, minWidth: 200,
    cursor: "pointer",
  },
  btn: {
    background: "#3b82f6", color: "#fff",
    border: "none", borderRadius: 8,
    padding: "10px 22px", fontSize: 13,
    fontWeight: 600, cursor: "pointer",
  },
  resultCard: {
    background: "#0d1424",
    border: "1px solid #1e293b",
    borderRadius: 12,
    padding: "20px 24px",
  },
  resultRow:   { display: "flex", gap: 32, flexWrap: "wrap", marginBottom: 12 },
  resultLabel: { color: "#64748b", fontSize: 11, textTransform: "uppercase", letterSpacing: "0.05em", margin: 0 },
  resultValue: { color: "#e2e8f0", fontSize: 18, fontWeight: 600, marginTop: 4 },
  runId:       { color: "#475569", fontSize: 11, marginTop: 4 },
  chartTitle:  { color: "#94a3b8", fontSize: 13, marginBottom: 8 },
};

import { useEffect, useState } from "react";
import { fetchProducts, runPredict } from "../api";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid,
  ResponsiveContainer,
  Cell,
} from "recharts";

const SIGNAL_LABELS = {
  production_unit: "Production",
  delivery_unit: "Delivery",
  sales_order_unit: "Sales Order",
};

function normalizePrediction(prediction) {
  if (prediction && typeof prediction === "object" && !Array.isArray(prediction)) {
    return Object.entries(prediction).map(([key, value]) => ({
      key,
      label: SIGNAL_LABELS[key] || key,
      value: Number(value ?? 0),
    }));
  }

  return [
    {
      key: "production_unit",
      label: "Production",
      value: Number(prediction ?? 0),
    },
  ];
}

export default function PredictionPanelV2() {
  const [products, setProducts] = useState([]);
  const [selectedName, setSelectedName] = useState("");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    async function loadProducts() {
      try {
        const response = await fetchProducts();
        const productList = response?.data?.products || [];
        setProducts(productList);
        if (productList[0]) {
          setSelectedName(productList[0]);
        }
      } catch {
        setError("Could not load product list");
      }
    }

    loadProducts();
  }, []);

  useEffect(() => {
    async function loadPrediction() {
      if (!selectedName) return;

      setLoading(true);
      setError("");

      try {
        const res = await runPredict({ product_name: selectedName });
        setResult(res.data);
      } catch (e) {
        setResult(null);
        setError(e.response?.data?.detail ?? e.message);
      } finally {
        setLoading(false);
      }
    }

    loadPrediction();
  }, [selectedName]);

  const chartData = result ? normalizePrediction(result.prediction) : [];
  const topForecast = chartData[0]?.value ?? 0;
  const horizonDays = result?.forecast_horizon_days ?? 7;

  return (
    <div>
      <h2 style={S.heading}>Demand Prediction</h2>
      <p style={S.sub}>
        Choose a product and the panel recalculates total {horizonDays}-day
        production, delivery, and sales order forecasts.
      </p>

      <div style={S.row}>
        <select
          value={selectedName}
          onChange={(e) => setSelectedName(e.target.value)}
          style={S.select}
        >
          {products.map((name) => (
            <option key={name} value={name}>
              {name}
            </option>
          ))}
        </select>

        <div style={S.statusPill}>
          {loading ? "Refreshing forecast..." : "Live forecast ready"}
        </div>
      </div>

      {error && <p style={{ color: "#f87171", marginTop: 12 }}>{error}</p>}

      {result && (
        <div style={{ marginTop: 28 }}>
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
                <p style={S.resultLabel}>Highest Weekly Total</p>
                <p style={{ ...S.resultValue, color: "#4ade80", fontSize: 28 }}>
                  {topForecast.toFixed(2)}
                </p>
              </div>
              <div>
                <p style={S.resultLabel}>Forecast Horizon</p>
                <p style={S.resultValue}>{horizonDays} days</p>
              </div>
              <div>
                <p style={S.resultLabel}>Model Version</p>
                <p style={S.resultValue}>{result.model_version}</p>
              </div>
            </div>

            <div style={S.signalGrid}>
              {chartData.map((item) => (
                  <div key={item.key} style={S.signalCard}>
                    <div style={S.signalLabel}>{item.label}</div>
                    <div style={S.signalValue}>{item.value.toFixed(2)}</div>
                    <div style={S.signalSubtext}>
                      Next day: {(result?.next_day_prediction?.[item.key] ?? 0).toFixed(2)}
                    </div>
                  </div>
                ))}
              </div>
          </div>

          <div style={{ marginTop: 24 }}>
            <p style={S.chartTitle}>Total {horizonDays}-Day Forecast By Signal</p>
            <ResponsiveContainer width="100%" height={280}>
              <BarChart data={chartData} margin={{ top: 10, right: 20, left: 0, bottom: 10 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                <XAxis dataKey="label" stroke="#475569" tick={{ fill: "#94a3b8", fontSize: 12 }} />
                <YAxis stroke="#475569" tick={{ fill: "#94a3b8", fontSize: 12 }} />
                <Tooltip
                  contentStyle={{ background: "#0d1424", border: "1px solid #1e293b", borderRadius: 8 }}
                  labelStyle={{ color: "#94a3b8" }}
                />
                <Bar dataKey="value" radius={[6, 6, 0, 0]}>
                  {chartData.map((entry) => (
                    <Cell
                      key={entry.key}
                      fill={
                        entry.key === "production_unit"
                          ? "#38bdf8"
                          : entry.key === "delivery_unit"
                            ? "#34d399"
                            : "#f59e0b"
                      }
                    />
                  ))}
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
  heading: { color: "#e2e8f0", fontSize: 22, fontWeight: 700, marginBottom: 8 },
  sub: { color: "#64748b", fontSize: 13, marginBottom: 24 },
  row: { display: "flex", gap: 12, alignItems: "center", flexWrap: "wrap" },
  select: {
    background: "#0d1424",
    color: "#e2e8f0",
    border: "1px solid #334155",
    borderRadius: 8,
    padding: "10px 14px",
    fontSize: 13,
    minWidth: 220,
    cursor: "pointer",
  },
  statusPill: {
    background: "#10203a",
    color: "#93c5fd",
    border: "1px solid #1d4ed8",
    borderRadius: 999,
    padding: "8px 12px",
    fontSize: 12,
    fontWeight: 600,
  },
  resultCard: {
    background: "#0d1424",
    border: "1px solid #1e293b",
    borderRadius: 12,
    padding: "20px 24px",
  },
  resultRow: { display: "flex", gap: 32, flexWrap: "wrap", marginBottom: 18 },
  resultLabel: { color: "#64748b", fontSize: 11, textTransform: "uppercase", letterSpacing: "0.05em", margin: 0 },
  resultValue: { color: "#e2e8f0", fontSize: 18, fontWeight: 600, marginTop: 4 },
  signalGrid: {
    display: "grid",
    gridTemplateColumns: "repeat(auto-fit, minmax(160px, 1fr))",
    gap: 12,
  },
  signalCard: {
    background: "#0b1120",
    border: "1px solid #1e293b",
    borderRadius: 10,
    padding: "14px 16px",
  },
  signalLabel: { color: "#94a3b8", fontSize: 12, marginBottom: 6 },
  signalValue: { color: "#f8fafc", fontSize: 22, fontWeight: 700 },
  signalSubtext: { color: "#94a3b8", fontSize: 12, marginTop: 6 },
  chartTitle: { color: "#94a3b8", fontSize: 13, marginBottom: 8 },
};

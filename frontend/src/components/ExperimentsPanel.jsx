import { useEffect, useState } from "react";
import { fetchAllPreds } from "../api";
import {
  ScatterChart, Scatter, XAxis, YAxis, Tooltip,
  CartesianGrid, ResponsiveContainer, ReferenceLine,
  LineChart, Line, Legend,
} from "recharts";

export default function ExperimentsPanel() {
  const [points,  setPoints]  = useState([]);
  const [loading, setLoading] = useState(true);
  const [error,   setError]   = useState("");

  useEffect(() => {
    fetchAllPreds()
      .then(r => setPoints(r.data.points))
      .catch(() => setError("Could not load precomputed predictions"))
      .finally(() => setLoading(false));
  }, []);

  // residuals for line chart
  const residuals = points.map((p, i) => ({
    idx:      i,
    residual: parseFloat((p.predicted - p.actual).toFixed(4)),
  }));

  // axis domain
  const allVals  = points.flatMap(p => [p.predicted, p.actual]);
  const domainMin = allVals.length ? Math.floor(Math.min(...allVals))  : 0;
  const domainMax = allVals.length ? Math.ceil(Math.max(...allVals))   : 1;

  if (loading) return <p style={{ color: "#64748b" }}>Loading experiment data…</p>;
  if (error)   return <p style={{ color: "#f87171" }}>{error}</p>;

  return (
    <div>
      <h2 style={S.heading}>Experiments — Predictions vs Actuals</h2>
      <p style={S.sub}>
        Precomputed predictions from the last training run on the test split.
        Points on the diagonal line = perfect prediction.
      </p>

      {/* scatter */}
      <div style={S.chartBox}>
        <p style={S.chartTitle}>Scatter: Predicted vs Actual (production weight)</p>
        <ResponsiveContainer width="100%" height={320}>
          <ScatterChart margin={{ top: 10, right: 20, bottom: 30, left: 10 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
            <XAxis
              type="number" dataKey="actual" name="Actual"
              domain={[domainMin, domainMax]}
              stroke="#475569" tick={{ fill: "#94a3b8", fontSize: 11 }}
              label={{ value: "Actual", position: "insideBottom", offset: -15, fill: "#64748b", fontSize: 12 }}
            />
            <YAxis
              type="number" dataKey="predicted" name="Predicted"
              domain={[domainMin, domainMax]}
              stroke="#475569" tick={{ fill: "#94a3b8", fontSize: 11 }}
              label={{ value: "Predicted", angle: -90, position: "insideLeft", fill: "#64748b", fontSize: 12 }}
            />
            <Tooltip
              cursor={{ strokeDasharray: "3 3" }}
              contentStyle={{ background: "#0d1424", border: "1px solid #1e293b", borderRadius: 8 }}
              itemStyle={{ color: "#7dd3fc" }}
            />
            {/* perfect prediction line */}
            <ReferenceLine
              segment={[
                { x: domainMin, y: domainMin },
                { x: domainMax, y: domainMax },
              ]}
              stroke="#4ade80" strokeDasharray="5 5" strokeWidth={1.5}
            />
            <Scatter data={points} fill="#3b82f6" opacity={0.55} r={3} />
          </ScatterChart>
        </ResponsiveContainer>
      </div>

      {/* residuals */}
      <div style={{ ...S.chartBox, marginTop: 28 }}>
        <p style={S.chartTitle}>Residuals over test samples (Predicted − Actual)</p>
        <ResponsiveContainer width="100%" height={220}>
          <LineChart data={residuals} margin={{ top: 10, right: 20, bottom: 10, left: 10 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
            <XAxis dataKey="idx" stroke="#475569" tick={{ fill: "#94a3b8", fontSize: 10 }} />
            <YAxis stroke="#475569" tick={{ fill: "#94a3b8", fontSize: 10 }} />
            <Tooltip
              contentStyle={{ background: "#0d1424", border: "1px solid #1e293b", borderRadius: 8 }}
              itemStyle={{ color: "#fb923c" }}
            />
            <ReferenceLine y={0} stroke="#4ade80" strokeDasharray="4 4" />
            <Line
              type="monotone" dataKey="residual"
              stroke="#fb923c" strokeWidth={1.5}
              dot={false} name="Residual"
            />
          </LineChart>
        </ResponsiveContainer>
      </div>

      <p style={{ color: "#475569", fontSize: 11, marginTop: 12 }}>
        Showing {points.length} test samples · Target signal: production_weight · Rolling window: 30
      </p>
    </div>
  );
}

const S = {
  heading:    { color: "#e2e8f0", fontSize: 22, fontWeight: 700, marginBottom: 8 },
  sub:        { color: "#64748b", fontSize: 13, marginBottom: 24 },
  chartBox: {
    background: "#0d1424",
    border: "1px solid #1e293b",
    borderRadius: 12, padding: "20px 16px",
  },
  chartTitle: { color: "#94a3b8", fontSize: 13, marginBottom: 12 },
};

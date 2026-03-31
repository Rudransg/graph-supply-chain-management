// top of Dashboard.jsx
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid,
  ResponsiveContainer,
} from "recharts";

import { useState, useEffect } from "react";

import {
  fetchDashboardStats,
  fetchAtRiskProducts,
  fetchFactoryLoad,
  fetchProductTrend,
  fetchProducts,
  fetch_Products
  // fetchAllPreds, // keep this only if used somewhere else
} from "../api";

export default function Dashboard() {
  const [stats, setStats] = useState(null);
  const [atRisk, setAtRisk] = useState([]);
  const [factoryLoad, setFactoryLoad] = useState([]);
  const [loading, setLoading] = useState(true);
  const [trendPoints, setTrendPoints] = useState([]);
  const [products, setProducts] = useState([]);
  const [selectedProduct, setSelectedProduct] = useState(null);

  useEffect(() => {
    async function load() {
      try {
        const [statsRes, atRiskRes, factoryRes, productsRes] = await Promise.all([
          fetchDashboardStats(),
          fetchAtRiskProducts(6),
          fetchFactoryLoad(),
          fetchProducts(),
        ]);

        setStats(statsRes.data);
        setAtRisk(atRiskRes.data);
        setFactoryLoad(factoryRes.data);

        const productList = productsRes.data.products || [];
        setProducts(productList);

        const initialProduct =
          atRiskRes.data[0]?.product || productList[0] || null;

        if (initialProduct) {
          setSelectedProduct(initialProduct);
          const trendRes = await fetchProductTrend(initialProduct);
          // const pts = trendRes.data.points || [];
          // setTrendPoints(
          //   pts.map((p) => ({
          //     time: p.timestamp,
          //     actual: p.actual,
          //   }))
          // );
          // const pts = trendRes.data.points || [];
          setTrendPoints(
            pts.map((p) => ({
              time: p.timestamp,
              actual: p.actual,
              predicted: p.predicted,
            }))
          );
        }
      } finally {
        setLoading(false);
      }
    }
    load();
  }, []);

  async function handleProductChange(e) {
    const product = e.target.value;
    setSelectedProduct(product);
    const trendRes = await fetchProductTrend(product);
    const pts = trendRes.data.points || [];
    setTrendPoints(
      pts.map((p) => ({
        time: p.timestamp,
        actual: p.actual,
        predicted: p.predicted,
      }))
    );
  }

  if (loading) return <div style={S.loading}>Loading dashboard...</div>;

  return (
    <div style={S.container}>
      {/* Header */}
      <div style={S.header}>
        <div style={S.greeting}>
          Good Morning 👋 Here's your supply chain overview
        </div>
        <div style={S.subtitle}>
          Monday, 24 March 2026  •  Last updated: just now
        </div>
      </div>

      {/* Stats Cards */}
      <div style={S.stats}>
        <StatCard
          icon="📦"
          label="Total Products"
          value={stats?.total_products || 0}
        />
        <StatCard
          icon="⚠️"
          label="At Risk"
          value={stats?.at_risk || 0}
          subtext={`+${stats?.new_at_risk_today || 0} today`}
          danger
        />
        <StatCard
          icon="✅"
          label="On Track"
          value={stats?.on_track || 0}
        />
        <StatCard
          icon="🏭"
          label="Active Factories"
          value={stats?.active_factories || 0}
        />
        <StatCard
          icon="🎯"
          label="Forecast Accuracy"
          value={`${stats?.forecast_accuracy || 0}%`}
          subtext="(R² score)"
        />
      </div>

      {/* Production Weight Trend */}
      <div style={S.card}>
        <div
          style={{
            ...S.cardTitle,
            display: "flex",
            justifyContent: "space-between",
            alignItems: "center",
          }}
        >
          <span>
            📊 Production Weight Trend{" "}
            {selectedProduct ? `— ${selectedProduct}` : ""}
          </span>
          <select
            value={selectedProduct || ""}
            onChange={handleProductChange}
            style={{
              background: "#020617",
              color: "#e2e8f0",
              borderRadius: 6,
              padding: "4px 8px",
              border: "1px solid #1e293b",
            }}
          >
            <option value="" disabled>
              Select product
            </option>
            {products.map((p) => (
              <option key={p} value={p}>
                {p}
              </option>
            ))}
          </select>
        </div>

        <div style={S.chart}>
          {trendPoints.length === 0 ? (
            <div style={S.chartPlaceholder}>
              No trend data yet.
              <br />
              <span style={{ fontSize: 12, color: "#64748b" }}>
                Ensure /forecast/trend/{selectedProduct || "<product>"} returns
                points.
              </span>
            </div>
          ) : (
            <ResponsiveContainer width="100%" height={200}>
              <LineChart data={trendPoints}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" />
              <XAxis dataKey="time" />
              <YAxis />
              <Tooltip />
              <Line
                type="monotone"
                dataKey="actual"
                stroke="#38bdf8"
                strokeWidth={2}
                dot={false}
              />
              <Line
                type="monotone"
                dataKey="predicted"   // exactly this key
                stroke="#a855f7"
                strokeWidth={2}
                dot={false}
              />
              </LineChart>

            </ResponsiveContainer>
          )}
        </div>
      </div>

      {/* Bottom Row */}
      <div style={S.row}>
        {/* At Risk */}
        <div style={{ ...S.card, flex: 1 }}>
          <div style={S.cardTitle}>⚠️ Products At Risk</div>
          <div style={S.list}>
            {atRisk.length === 0 && (
              <div style={S.empty}>No products at risk</div>
            )}
            {atRisk.map((p) => (
              <div key={p.product} style={S.listItem}>
                <span style={S.productName}>{p.product}</span>
                <span style={{ ...S.trend, color: "#ef4444" }}>
                  ↓ {Math.abs(p.trend_pct)}%
                </span>
              </div>
            ))}
          </div>
        </div>

        {/* Factory Load */}
        <div style={{ ...S.card, flex: 1 }}>
          <div style={S.cardTitle}>🏭 Factory Load</div>
          <div style={S.list}>
            {factoryLoad.map((f) => (
              <div key={f.factory} style={S.listItem}>
                <span style={S.productName}>{f.factory}</span>
                <div style={S.barContainer}>
                  <div
                    style={{
                      ...S.bar,
                      width: `${f.load_pct}%`,
                      background:
                        f.load_pct > 85 ? "#ef4444" : "#3b82f6",
                    }}
                  />
                </div>
                <span style={S.pct}>{f.load_pct}%</span>
                {f.load_pct > 85 && <span style={{ marginLeft: 4 }}>⚠️</span>}
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}

// StatCard and S unchanged...


function StatCard({ icon, label, value, subtext, danger }) {
  return (
    <div style={S.statCard}>
      <div style={S.statIcon}>{icon}</div>
      <div style={S.statLabel}>{label}</div>
      <div
        style={{
          ...S.statValue,
          color: danger ? "#ef4444" : "#e2e8f0",
        }}
      >
        {value}
      </div>
      {subtext && <div style={S.statSubtext}>{subtext}</div>}
    </div>
  );
}

const S = {
  container: { display: "flex", flexDirection: "column", gap: 24 },
  loading: { color: "#94a3b8", fontSize: 14 },
  header: { marginBottom: 8 },
  greeting: {
    fontSize: 20,
    fontWeight: 600,
    color: "#e2e8f0",
    marginBottom: 6,
  },
  subtitle: { fontSize: 13, color: "#64748b" },
  stats: {
    display: "grid",
    gridTemplateColumns: "repeat(auto-fit, minmax(180px, 1fr))",
    gap: 16,
  },
  statCard: {
    background: "#0d1424",
    border: "1px solid #1e293b",
    borderRadius: 12,
    padding: "20px 18px",
    display: "flex",
    flexDirection: "column",
    gap: 6,
  },
  statIcon: { fontSize: 24 },
  statLabel: { fontSize: 13, color: "#94a3b8", fontWeight: 500 },
  statValue: { fontSize: 28, fontWeight: 700, lineHeight: 1 },
  statSubtext: { fontSize: 12, color: "#64748b" },
  card: {
    background: "#0d1424",
    border: "1px solid #1e293b",
    borderRadius: 12,
    padding: 24,
  },
  cardTitle: {
    fontSize: 15,
    fontWeight: 600,
    color: "#e2e8f0",
    marginBottom: 16,
  },
  chart: {
    height: 200,
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
  },
  chartPlaceholder: {
    color: "#64748b",
    fontSize: 14,
    textAlign: "center",
    border: "1px dashed #1e293b",
    borderRadius: 8,
    padding: "40px 20px",
    width: "100%",
  },
  row: { display: "grid", gridTemplateColumns: "1fr 1fr", gap: 24 },
  list: { display: "flex", flexDirection: "column", gap: 10 },
  listItem: { display: "flex", alignItems: "center", gap: 12 },
  productName: {
    fontSize: 14,
    color: "#cbd5e1",
    fontWeight: 500,
    minWidth: 110,
  },
  trend: { fontSize: 14, fontWeight: 600 },
  barContainer: {
    flex: 1,
    height: 8,
    background: "#1e293b",
    borderRadius: 4,
    overflow: "hidden",
  },
  bar: { height: "100%", borderRadius: 4, transition: "width 0.3s" },
  pct: {
    fontSize: 13,
    color: "#94a3b8",
    minWidth: 40,
    textAlign: "right",
  },
  empty: {
    fontSize: 13,
    color: "#64748b",
    fontStyle: "italic",
  },
};

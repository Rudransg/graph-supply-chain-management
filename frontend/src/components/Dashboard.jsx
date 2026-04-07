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
  fetchLiveProductTrend,
  fetchProducts,
  fetchForecastCategory,
  runPredict,
} from "../api";

const CATEGORY_TO_SIGNAL = {
  production: "production_unit",
  delivery: "delivery_unit",
  supply_order: "sales_order_unit",
};

export default function Dashboard() {
  const [stats, setStats] = useState(null);
  const [atRisk, setAtRisk] = useState([]);
  const [factoryLoad, setFactoryLoad] = useState([]);
  const [loading, setLoading] = useState(true);
  const [trendPoints, setTrendPoints] = useState([]);
  const [products, setProducts] = useState([]);
  const [selectedProduct, setSelectedProduct] = useState("");
  const [selectedCategory, setSelectedCategory] = useState("production");
  const [forecastSummary, setForecastSummary] = useState(null);

  async function loadTrendAndForecast(product, category) {
    if (!product) {
      setTrendPoints([]);
      setForecastSummary(null);
      return;
    }

    const [trendRes, predictRes] = await Promise.all([
      fetchLiveProductTrend(product, CATEGORY_TO_SIGNAL[category], 30),
      runPredict({ product_name: product }),
    ]);

    const pts = trendRes.data.points || [];
    setTrendPoints(
      pts.map((p) => ({
        time: p.timestamp,
        actual: p.actual,
        predicted: p.predicted,
      }))
    );

    setForecastSummary(predictRes.data?.prediction || null);
  }

  useEffect(() => {
    async function load() {
      try {
        const [statsRes, atRiskRes, factoryRes, productsRes, categoryRes] =
          await Promise.all([
            fetchDashboardStats(),
            fetchAtRiskProducts(6),
            fetchFactoryLoad(),
            fetchProducts(),
            fetchForecastCategory("production"),
          ]);

        setStats(statsRes.data);
        setAtRisk(atRiskRes.data || []);
        setFactoryLoad(factoryRes.data || []);

        const productList =
          categoryRes?.data?.products?.map((p) => p.product) ||
          productsRes?.data?.products ||
          [];

        setProducts(productList);

        const initialProduct =
          atRiskRes.data?.[0]?.product || productList[0] || "";

        if (initialProduct) {
          setSelectedProduct(initialProduct);
          await loadTrendAndForecast(initialProduct, "production");
        }
      } catch (err) {
        console.error("Dashboard load failed:", err);
      } finally {
        setLoading(false);
      }
    }

    load();
  }, []);

  async function handleProductChange(e) {
    const product = e.target.value;
    setSelectedProduct(product);

    try {
      await loadTrendAndForecast(product, selectedCategory);
    } catch (err) {
      console.error("Trend load failed:", err);
      setTrendPoints([]);
      setForecastSummary(null);
    }
  }

  async function handleCategoryChange(e) {
    const category = e.target.value;
    setSelectedCategory(category);
    setSelectedProduct("");
    setTrendPoints([]);

    try {
      const categoryRes = await fetchForecastCategory(category);
      const categoryProducts =
        categoryRes?.data?.products?.map((p) => p.product) || [];

      setProducts(categoryProducts);

      const firstProduct = categoryProducts[0] || "";
      setSelectedProduct(firstProduct);

      if (firstProduct) {
        await loadTrendAndForecast(firstProduct, category);
      }
    } catch (err) {
      console.error("Category load failed:", err);
      setProducts([]);
      setTrendPoints([]);
      setForecastSummary(null);
    }
  }

  if (loading) return <div style={S.loading}>Loading dashboard...</div>;

  return (
    <div style={S.container}>
      <div style={S.header}>
        <div style={S.greeting}>
          Good Morning 👋 Here's your supply chain overview
        </div>
        <div style={S.subtitle}>
          Monday, 24 March 2026 • Last updated: just now
        </div>
      </div>

      <div style={S.stats}>
        <StatCard icon="📦" label="Total Products" value={stats?.total_products || 0} />
        <StatCard
          icon="⚠️"
          label="At Risk"
          value={stats?.at_risk || 0}
          subtext={`+${stats?.new_at_risk_today || 0} today`}
          danger
        />
        <StatCard icon="✅" label="On Track" value={stats?.on_track || 0} />
        <StatCard icon="🏭" label="Active Factories" value={stats?.active_factories || 0} />
        <StatCard
          icon="🎯"
          label="Forecast Accuracy"
          value={`${stats?.forecast_accuracy || 0}%`}
          subtext="(R² score)"
        />
      </div>

      <div style={S.card}>
        <div style={S.filtersRow}>
          <div>
            <div style={S.cardTitle}>
              📊 {selectedCategory.replace("_", " ").toUpperCase()} Trend
              {selectedProduct ? ` — ${selectedProduct}` : ""}
            </div>
            <div style={S.chartMeta}>Recent actuals plus live 7-day model forecast</div>
            {forecastSummary && (
              <div style={S.forecastRow}>
                <span style={S.forecastChip}>
                  Production: {(forecastSummary.production_unit ?? 0).toFixed(2)}
                </span>
                <span style={S.forecastChip}>
                  Delivery: {(forecastSummary.delivery_unit ?? 0).toFixed(2)}
                </span>
                <span style={S.forecastChip}>
                  Sales Order: {(forecastSummary.sales_order_unit ?? 0).toFixed(2)}
                </span>
              </div>
            )}
          </div>

          <div style={S.selectGroup}>
            <select
              value={selectedCategory}
              onChange={handleCategoryChange}
              style={S.select}
            >
              <option value="production">Production</option>
              <option value="delivery">Delivery</option>
              <option value="supply_order">Supply Order</option>
            </select>

            <select
              value={selectedProduct || ""}
              onChange={handleProductChange}
              style={S.select}
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
        </div>

        <div style={S.chart}>
          {trendPoints.length === 0 ? (
            <div style={S.chartPlaceholder}>
              No trend data yet.
              <br />
              <span style={{ fontSize: 12, color: "#64748b" }}>
                Ensure the forecast trend API returns points for this category.
              </span>
            </div>
          ) : (
            <ResponsiveContainer width="100%" height={200}>
              <LineChart data={trendPoints}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" />
                <XAxis dataKey="time" stroke="#94a3b8" />
                <YAxis stroke="#94a3b8" />
                <Tooltip />
                <Line
                  type="linear"
                  dataKey="actual"
                  stroke="#38bdf8"
                  strokeWidth={2}
                  dot={false}
                />
                <Line
                  type="linear"
                  dataKey="predicted"
                  stroke="#a855f7"
                  strokeWidth={2}
                  dot={false}
                />
              </LineChart>
            </ResponsiveContainer>
          )}
        </div>
      </div>

      <div style={S.row}>
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
                  ↓ {Math.abs(p.trend_pct || 0)}%
                </span>
              </div>
            ))}
          </div>
        </div>

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
                      background: f.load_pct > 85 ? "#ef4444" : "#3b82f6",
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
  },
  chartMeta: {
    color: "#64748b",
    fontSize: 12,
    marginTop: 6,
  },
  filtersRow: {
    display: "flex",
    justifyContent: "space-between",
    alignItems: "center",
    gap: 16,
    marginBottom: 16,
    flexWrap: "wrap",
  },
  forecastRow: {
    display: "flex",
    gap: 8,
    flexWrap: "wrap",
    marginTop: 10,
  },
  forecastChip: {
    background: "#08101f",
    border: "1px solid #1e293b",
    color: "#cbd5e1",
    borderRadius: 999,
    padding: "6px 10px",
    fontSize: 12,
    fontWeight: 600,
  },
  selectGroup: {
    display: "flex",
    gap: 10,
    flexWrap: "wrap",
  },
  select: {
    background: "#020617",
    color: "#e2e8f0",
    borderRadius: 6,
    padding: "8px 10px",
    border: "1px solid #1e293b",
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

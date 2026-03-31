import { useEffect, useState } from "react";
import { fetchDashboardStats, fetchFactoryLoad } from "../api";
import AnalyticsPanel from "./AnalyticsPanel";

export default function AnalyticsView() {
  const [kpis, setKpis] = useState([]);
  const [trend, setTrend] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  useEffect(() => {
    async function load() {
      try {
        const [statsRes, factoryRes] = await Promise.all([
          fetchDashboardStats(),
          fetchFactoryLoad(),
        ]);

        const stats = statsRes.data;
        const factories = factoryRes.data;

        const kpisData = [
          { label: "Total products",    value: stats.total_products },
          { label: "At risk",           value: stats.at_risk, delta: stats.new_at_risk_today ?? 0 },
          { label: "Active factories",  value: stats.active_factories },
          { label: "Forecast accuracy", value: stats.forecast_accuracy, unit: "%" },
        ];

        const trendData = factories.map((f, idx) => ({
          x: f.factory ?? `F${idx + 1}`,
          y: f.load_pct ?? 0,
        }));

        setKpis(kpisData);
        setTrend(trendData);
      } catch (e) {
        console.error(e);
        setError("Could not load analytics data");
      } finally {
        setLoading(false);
      }
    }

    load();
  }, []);

  if (loading) {
    return <p style={{ color: "#64748b", fontSize: 14 }}>Loading analytics…</p>;
  }

  if (error) {
    return <p style={{ color: "#f87171", fontSize: 14 }}>{error}</p>;
  }

  return (
    <AnalyticsPanel
      title="Supply chain analytics"
      kpis={kpis}
      trend={trend}
    />
  );
}

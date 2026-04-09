import { Routes, Route, useLocation, useNavigate } from "react-router-dom";
import { useEffect, useState } from "react";                          // ← add
import { fetchProducts, fetchFactoryLoad } from "./api";             // ← add
import Sidebar from "./components/Sidebar";
import Dashboard from "./components/Dashboard";
import PredictionPanel from "./components/PredictionPanelV2";
import ExperimentsPanel from "./components/ExperimentsPanel";
import ModelInfoPanel from "./components/ModelInfoPanel";
import PipelinePanel from "./components/PipelinePanel";
import AnalyticsView from "./components/AnalyticsView";
import WhatIfPanel from "./components/WhatIfPanel";

const ROUTES = {
  "/": "dashboard",
  "/analytics": "analytics",
  "/prediction": "prediction",
  "/experiments": "experiments",
  "/whatif": "whatif",
  "/model": "model",
  "/pipeline": "pipeline",
};

const S = {
  main: {
    flex: 1,
    display: "flex",
    flexDirection: "column",
    minHeight: "100vh",
    background: "#0a0f1e",
    overflowY: "auto",
  },
  topBar: {
    display: "flex",
    alignItems: "center",
    justifyContent: "space-between",
    padding: "18px 44px",
    borderBottom: "1px solid #1e293b",
    background: "#0d1424",
    position: "sticky",
    top: 0,
    zIndex: 10,
  },
  topBarTitle: {
    color: "#e2e8f0",
    fontSize: 16,
    fontWeight: 600,
  },
  topBarBadge: {
    background: "#1e293b",
    color: "#64748b",
    fontSize: 11,
    padding: "4px 12px",
    borderRadius: 20,
    fontWeight: 500,
  },
  content: {
    flex: 1,
    padding: "36px 44px",
    maxWidth: 1200,
    width: "100%",
  },
};

export default function App() {
  const location = useLocation();
  const navigate = useNavigate();
  const [products,  setProducts]  = useState([]);
  const [factories, setFactories] = useState([]);

  useEffect(() => {
    fetchProducts()
      .then(res => setProducts(res?.data?.products || []))
      .catch(() => {});

    fetchFactoryLoad()
      .then(res => setFactories((res?.data || []).map(f => f.factory)))
      .catch(() => {});
  }, []);
  const activeKey = ROUTES[location.pathname] || "dashboard";
  const title = activeKey.charAt(0).toUpperCase() + activeKey.slice(1);

  const handleSetActive = (key) => {
    const path =
      Object.entries(ROUTES).find(([, value]) => value === key)?.[0] || "/";
    navigate(path);
  };

  return (
    <div style={{ display: "flex", minHeight: "100vh", background: "#0a0f1e" }}>
      <Sidebar active={activeKey} setActive={handleSetActive} />
      <main style={S.main}>
        <div style={S.topBar}>
          <span style={S.topBarTitle}>{title}</span>
          <span style={S.topBarBadge}>Supply Graph — DeepGCNGRU v2</span>
        </div>
        <div style={S.content}>
          <Routes>
            <Route path="/" element={<Dashboard />} />
            <Route path="/analytics" element={<AnalyticsView />} />
            <Route path="/prediction" element={<PredictionPanel />} />
            <Route path="/experiments" element={<ExperimentsPanel />} />
            <Route path="/whatif"      element={
              <WhatIfPanel products={products} factories={factories} />  // ← pass here
            } />
            <Route path="/model" element={<ModelInfoPanel />} />
            <Route path="/pipeline" element={<PipelinePanel />} />
            
          </Routes>
        </div>
      </main>
    </div>
  );
}

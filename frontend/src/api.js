import axios from "axios";

const API = axios.create({
  baseURL: "",
  timeout: 15000,
});

export const fetchHealth        = ()         => API.get("/api/health");
export const fetchMetrics       = ()         => API.get("/api/metrics");
export const fetch_Products     = ()         => API.get("/api/products");
export const fetchAllPreds      = ()         => API.get("/api/predictions/all");
export const fetchModelInfo     = ()         => API.get("/api/model/info");
export const runPredict         = (payload)  => API.post("/api/predict", payload);

// dashboard routes
export const fetchDashboardStats = () =>
  API.get("/api/dashboard/stats");
export const fetchAtRiskProducts = (topN = 6) =>
  API.get("/api/products/at-risk", { params: { top_n: topN } });
export const fetchFactoryLoad = () =>
  API.get("/api/factory/load");

// forecast routes
export const fetchProductTrend = async (product) => {
  try {
    return await API.get(`/api/forecast/trend/${encodeURIComponent(product)}`);
  } catch (err) {
    if (err.response?.status === 404) {
      return { data: { product, points: [] } };
    }
    throw err;
  }
};

export const fetchProducts = () =>
  API.get("/api/forecast/products");




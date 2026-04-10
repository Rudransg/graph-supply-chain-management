import axios from "axios";

const API = axios.create({
  baseURL: import.meta.env.VITE_API_URL || "http://localhost:8000",
  timeout: 15000,
});

export const runWhatIf = (payload) => API.post("/api/predict/whatif", payload);
export const fetchHealth = () => API.get("/api/health");
export const fetchMetrics = () => API.get("/api/metrics");
export const fetchProductsApi = () => API.get("/api/products");
export const fetchAllPreds = () => API.get("/api/predictions/all");
export const fetchModelInfo = () => API.get("/api/model/info");
export const runPredict = (payload) => API.post("/api/predict", payload);

// dashboard routes
export const fetchDashboardStats = () => API.get("/api/dashboard/stats");

export const fetchAtRiskProducts = (topN = 6) =>
  API.get("/api/products/at-risk", { params: { top_n: topN } });

export const fetchFactoryLoad = () => API.get("/api/factory/load");

// forecast routes
export const fetchProductTrend = async (
  product,
  signalType = "production_unit",
  limit = 30
) => {
  try {
    return await API.get(`/api/forecast/trend/${encodeURIComponent(product)}`, {
      params: {
        signal_type: signalType,
        limit,
      },
    });
  } catch (err) {
    if (err.response?.status === 404) {
      return {
        data: {
          product,
          signal_type: signalType,
          points: [],
        },
      };
    }
    throw err;
  }
};

export const fetchLiveProductTrend = async (
  product,
  signalType = "production_unit",
  historyPoints = 30
) => {
  try {
    return await API.get(`/api/forecast/live/${encodeURIComponent(product)}`, {
      params: {
        signal_type: signalType,
        history_points: historyPoints,
      },
    });
  } catch (err) {
    if (err.response?.status === 404) {
      return {
        data: {
          product,
          signal_type: signalType,
          points: [],
        },
      };
    }
    throw err;
  }
};

export const fetchProducts = () => API.get("/api/forecast/products");

export const fetchForecastCategory = (category) =>
  API.get(`/api/forecast/category/${encodeURIComponent(category)}`);

import axios from "axios";

const API = axios.create({
  baseURL: "http://localhost:8000",
  timeout: 30000,
});
export const fetchGraphEdges = () => API.get("/graph-edges");
export const runWhatIf = (payload) => API.post("/predict/whatif", payload);
export const fetchHealth = () => API.get("/health");
export const fetchMetrics = () => API.get("/metrics");
export const fetchProductsApi = () => API.get("/products");
export const fetchAllPreds = () => API.get("/predictions/all");
export const fetchModelInfo = () => API.get("/model/info");
export const runPredict = (payload) => API.post("/predict", payload);

// dashboard routes
export const fetchDashboardStats = () => API.get("/dashboard/stats");

export const fetchAtRiskProducts = (topN = 6) =>
  API.get("/products/at-risk", { params: { top_n: topN } });

export const fetchFactoryLoad = () => API.get("/factory/load");

// forecast routes
export const fetchProductTrend = async (
  product,
  signalType = "production_unit",
  limit = 30
) => {
  try {
    return await API.get(`/forecast/trend/${encodeURIComponent(product)}`, {
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
    return await API.get(`/forecast/live/${encodeURIComponent(product)}`, {
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

export const fetchProducts = () => API.get("/forecast/products");

export const fetchForecastCategory = (category) =>
  API.get(`/forecast/category/${encodeURIComponent(category)}`);

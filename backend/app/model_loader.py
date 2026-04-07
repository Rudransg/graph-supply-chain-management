"""
Loads SupplyGraphModel (state_dict) + all artifacts needed for inference.

Key insight from model_training.py:
    x_cat = torch.cat([static_x, temporal_x], dim=-1)
    x_dict = {NODE_TYPE: x_cat}
    y_hat  = model(x_dict, edge_index_dict)

- static_x  : hetero_data["rolled_prod"].x  shape [41, IN_CHANNELS=17]  (zeros)
- temporal_x: last snapshot of X.npy        shape [41, 1]
- cat result : shape [41, 18]  → in_channels for the model = 18
- out        : shape [41, 1]   → prediction per node
"""

import json
import sys
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from pathlib import Path
from typing import cast

# ── resolve project root so src/ and config/ are importable ──────────────────
ROOT = Path(__file__).resolve().parents[2]          # Supply Graph/
sys.path.insert(0, str(ROOT))

from torch_geometric.nn import HeteroConv, SAGEConv, GCNConv
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.typing import EdgeType

from src.logger import get_logger
from src.custom_exception import CustomException
from config.path_config import *


logger = get_logger(__name__)

# ── hardcoded from config.yaml (safe inside Docker) ──────────────────────────
NODE_TYPE        = "rolled_prod"
EDGE_RELATIONS   = ["same_plant", "same_storage", "same_product_group", "same_product_subgroup"]
IN_CHANNELS      = 17          # static_x dim
HIDDEN_CHANNELS  = 32
OUT_CHANNELS     = 3           # production_unit, delivery_unit, sales_order_unit
CONV_TYPE        = "sage"
AGGREGATION      = "sum"
LAYERS           = 2
NUM_NODES        = 41

# ── artifact paths ────────────────────────────────────────────────────────────
ARTIFACTS        = ROOT / "artifacts"
MODEL_PATH       = ARTIFACTS / "models"  / "hetero_sage_model.pt"
METRICS_PATH     = ARTIFACTS / "models"  / "metrics.json"
PREDICTIONS_PATH = ARTIFACTS / "models"  / "predictions.npy"
TARGETS_PATH     = ARTIFACTS / "models"  / "targets.npy"
HETERO_DATA_PATH = ARTIFACTS / "processed" / "hetero_data.pt"
X_PATH           = ARTIFACTS / "processed" / "X.npy"
IDX_TO_PROD_PATH = ARTIFACTS / "processed" / "idx_to_product.json"
PROD_TO_IDX_PATH = ARTIFACTS / "processed" / "product_to_idx.json"


# ── model architecture (mirrors src/model_training.py exactly) ────────────────
class DeepGCNGRUCell(nn.Module):
    def __init__(self, in_channels, hidden_channels, dropout=0.2):
        super().__init__()
        self.gcn1 = GCNConv(in_channels, hidden_channels)
        self.gcn2 = GCNConv(hidden_channels, hidden_channels)
        self.gru1 = nn.GRUCell(hidden_channels, hidden_channels)
        self.gru2 = nn.GRUCell(hidden_channels, hidden_channels)
        self.gru3 = nn.GRUCell(hidden_channels, hidden_channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, h1, h2, h3):
        x = F.relu(self.gcn1(x, edge_index))
        x = self.dropout(x)
        x = F.relu(self.gcn2(x, edge_index))
        x = self.dropout(x)
        h1 = self.gru1(x, h1)
        h2 = self.gru2(h1, h2)
        h3 = self.gru3(h2, h3)
        return h1, h2, h3


class MultiStepGCNGRU(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, forecast_horizon):
        super().__init__()
        self.cell = DeepGCNGRUCell(in_channels, hidden_channels)
        self.proj = nn.Linear(hidden_channels, out_channels)
        self.forecast_horizon = forecast_horizon

    def forward(self, x_seq, edge_index):
        B, P, N, C = x_seq.shape
        H = self.proj.in_features
        device = x_seq.device

        h1 = torch.zeros(B * N, H, device=device)
        h2 = torch.zeros(B * N, H, device=device)
        h3 = torch.zeros(B * N, H, device=device)

        for t in range(P):
            xt = x_seq[:, t].reshape(B * N, C)
            h1, h2, h3 = self.cell(xt, edge_index, h1, h2, h3)

        preds = []
        x_dec = x_seq[:, -1]

        for _ in range(self.forecast_horizon):
            xt = x_dec.reshape(B * N, C)
            h1, h2, h3 = self.cell(xt, edge_index, h1, h2, h3)
            y_t = self.proj(h3).reshape(B, N, -1)
            preds.append(y_t)
            x_dec = y_t

        return torch.stack(preds, dim=1)

# ── service singleton ─────────────────────────────────────────────────────────
class MultiStepGCNGRUModelService:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")

        self._load_model()
        self._load_graph_data()
        self._load_mappings()
        self._load_metrics()
        self._load_precomputed()

    # ── loaders ───────────────────────────────────────────────────────────────
    def _load_model(self):
        try:
            logger.info("Rebuilding MultiStepGCNGRU architecture")
            self.model = MultiStepGCNGRU(
                in_channels=TEMPORAL_FEATURE_DIM,
                hidden_channels=HIDDEN_CHANNELS,
                out_channels=3,
                forecast_horizon=PREDICTION_HORIZON,
            ).to(self.device)

            state_dict = torch.load(MODEL_OUTPUT_PATH, map_location=self.device, weights_only=True)
            self.model.load_state_dict(state_dict)
            self.model.eval()
            logger.info("Model loaded and set to eval mode")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise CustomException("Failed to load model", e)

    def _load_graph_data(self):
        try:
            logger.info("Loading hetero_data.pt")
            self.hetero_data = torch.load(
                HETERO_DATA_PATH,
                map_location=self.device,
                weights_only=False,
            )

            edges = list(self.hetero_data.edge_index_dict.values())
            self.edge_index = torch.cat(edges, dim=1).to(self.device)

            logger.info(
                f"Graph loaded | edge_index shape: {tuple(self.edge_index.shape)}"
            )
        except Exception as e:
            logger.error(f"Failed to load graph data: {e}")
            raise CustomException("Failed to load graph data", e)
    def _load_mappings(self):
        try:
            with open(IDX_TO_PROD_PATH, "r") as f:
                # keys are int-string "0","1",... values are product names
                self.idx_to_product: dict[str, str] = json.load(f)
            with open(PROD_TO_IDX_PATH, "r") as f:
                # keys are product names, values are int indices
                self.product_to_idx: dict[str, int] = json.load(f)
            logger.info(f"Loaded mappings for {len(self.idx_to_product)} products")
        except Exception as e:
            logger.error(f"Failed to load mappings: {e}")
            raise CustomException("Failed to load mappings", e)

    def _load_metrics(self):
        try:
            if METRICS_PATH.exists():
                with open(METRICS_PATH, "r") as f:
                    self.metrics: dict = json.load(f)
                logger.info(f"Metrics loaded: {self.metrics}")
            else:
                self.metrics = {}
                logger.warning("metrics.json not found")
        except Exception as e:
            logger.error(f"Failed to load metrics: {e}")
            raise CustomException("Failed to load metrics", e)

    def _load_precomputed(self):
        try:
            # X shape: [num_snapshots, num_nodes, 1]
            self.X = np.load(X_PATH)
            # last snapshot used as temporal_x for live inference
            self.last_temporal_x = torch.tensor(
                self.X[-1],                        # shape [41, 1]
                dtype=torch.float32,
                device=self.device
            )
            if PREDICTIONS_PATH.exists() and TARGETS_PATH.exists():
                preds_raw = np.load(PREDICTIONS_PATH)  # shape (T, 41*3*7) or (T, 123)
                targets_raw = np.load(TARGETS_PATH)    # shape (T, 41*3*7) or (T, 123)
                
                # Reshape predictions and targets to extract first horizon step
                # Input: [T, 861] (multi-step 7-day forecasts) -> Output: [T, 123] (single-step)
                T = preds_raw.shape[0]
                n_products = 41
                n_signals = 3
                horizon = 7
                
                if preds_raw.shape[1] == n_products * n_signals * horizon:
                    # Multi-step predictions: reshape to [T, 7, 41, 3] then take first step
                    self.predictions = preds_raw.reshape(T, horizon, n_products, n_signals)[:, 0, :, :].reshape(T, -1)
                elif preds_raw.shape[1] == n_products * n_signals:
                    # Already in single-step format [T, 123]
                    self.predictions = preds_raw.reshape(T, n_products, n_signals).reshape(T, -1)
                else:
                    # Fallback: assume it's [T, 41] (single target) and expand to 3 targets
                    self.predictions = np.tile(preds_raw, (1, 3))
                
                # Reshape targets similarly
                if targets_raw.shape[1] == n_products * n_signals * horizon:
                    self.targets = targets_raw.reshape(T, horizon, n_products, n_signals)[:, 0, :, :].reshape(T, -1)
                elif targets_raw.shape[1] == n_products * n_signals:
                    self.targets = targets_raw.reshape(T, n_products, n_signals).reshape(T, -1)
                else:
                    self.targets = np.tile(targets_raw, (1, 3))
                
            else:
                self.predictions = np.array([])
                self.targets     = np.array([])
            logger.info(f"Precomputed data loaded | snapshots: {self.X.shape[0]}, preds shape: {self.predictions.shape}")
        except Exception as e:
            logger.error(f"Failed to load precomputed data: {e}")
            raise CustomException("Failed to load precomputed data", e)

    # ── inference ─────────────────────────────────────────────────────────────
    def predict(self, product_name: str) -> dict:
        try:
            if product_name not in self.product_to_idx:
                raise ValueError(f"Product '{product_name}' not found in mapping")

            product_idx = int(self.product_to_idx[product_name])

            with torch.no_grad():
                # forward: [1, Q, N, C]
                out = self.model(self.last_input_window, self.edge_index)

            # take first forecast step
            pred_vec = out[0, 0, product_idx].cpu().numpy()

            if np.ndim(pred_vec) == 0:
                prediction = float(pred_vec)
            else:
                prediction = pred_vec.tolist()

            return {
                    "product_name": product_name,
                    "product_idx":  product_idx,
                    "prediction":   prediction,
                }
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise CustomException("Prediction failed", e)    

    def get_all_products(self) -> dict[str, str]:
        # returns {idx_str: product_name}
        return self.idx_to_product

    def get_scatter_data(self, limit: int = 300) -> list[dict]:
        if self.predictions.size == 0 or self.targets.size == 0:
            return []
        preds   = self.predictions.flatten()[:limit]
        targets = self.targets.flatten()[:limit]
        return [
            {"predicted": round(float(p), 4), "actual": round(float(t), 4)}
            for p, t in zip(preds, targets)
        ]


# singleton — loaded once on startup
model_service = MultiStepGCNGRUModelService()

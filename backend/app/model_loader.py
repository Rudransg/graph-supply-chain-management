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

logger = get_logger(__name__)

# ── hardcoded from config.yaml (safe inside Docker) ──────────────────────────
NODE_TYPE        = "rolled_prod"
EDGE_RELATIONS   = ["same_plant", "same_storage", "same_product_group", "same_product_subgroup"]
IN_CHANNELS      = 17          # static_x dim
HIDDEN_CHANNELS  = 32
OUT_CHANNELS     = 1
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
class SupplyGraphModel(nn.Module):
    def __init__(
        self,
        node_type: str,
        relations: list,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        conv_type: str = "sage",
        aggregation: str = "sum",
        layers: int = 2,
    ) -> None:
        super().__init__()
        self.node_type = node_type
        self.relations = relations
        self.layers    = layers

        self.convs = nn.ModuleList()

        first_layer: dict[EdgeType, MessagePassing] = {
            (node_type, rel, node_type): self._make_conv(conv_type, in_channels, hidden_channels)
            for rel in relations
        }
        self.convs.append(HeteroConv(first_layer, aggr=aggregation))

        for _ in range(layers - 1):
            hidden_layer: dict[EdgeType, MessagePassing] = {
                (node_type, rel, node_type): self._make_conv(conv_type, hidden_channels, hidden_channels)
                for rel in relations
            }
            self.convs.append(HeteroConv(hidden_layer, aggr=aggregation))

        self.lin = nn.Linear(hidden_channels, out_channels)

    @staticmethod
    def _make_conv(conv_type: str, in_ch: int, out_ch: int) -> MessagePassing:
        if conv_type.lower() == "sage":
            return cast(MessagePassing, SAGEConv(in_ch, out_ch))
        if conv_type.lower() == "gcn":
            return cast(MessagePassing, GCNConv(in_ch, out_ch))
        raise ValueError(f"Unsupported conv_type: {conv_type}")

    def forward(self, x_dict, edge_index_dict):
        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {k: F.relu(v) for k, v in x_dict.items()}
        return self.lin(x_dict[self.node_type])


# ── service singleton ─────────────────────────────────────────────────────────
class SupplyGraphModelService:
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
            logger.info(f"Rebuilding SupplyGraphModel architecture")
            # actual in_channels = static_x_dim(17) + temporal_dim(1) = 18
            self.model = SupplyGraphModel(
                node_type=NODE_TYPE,
                relations=EDGE_RELATIONS,
                in_channels=IN_CHANNELS + 1,   # 18
                hidden_channels=HIDDEN_CHANNELS,
                out_channels=OUT_CHANNELS,
                conv_type=CONV_TYPE,
                aggregation=AGGREGATION,
                layers=LAYERS,
            ).to(self.device)

            state_dict = torch.load(MODEL_PATH, map_location=self.device, weights_only=True)
            self.model.load_state_dict(state_dict)
            self.model.eval()
            logger.info("Model loaded and set to eval mode")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise CustomException("Failed to load model", e)

    def _load_graph_data(self):
        try:
            logger.info("Loading hetero_data.pt")
            self.hetero_data = torch.load(HETERO_DATA_PATH, map_location=self.device, weights_only=False)
            # static_x: shape [41, 17] — zeros from preprocessing
            self.static_x = self.hetero_data[NODE_TYPE].x.to(self.device)

            # build normalized edge_index_dict exactly as ModelTrainer does
            self.edge_index_dict = {
                (NODE_TYPE, rel, NODE_TYPE): edge_index.to(self.device)
                for (src, rel, dst), edge_index in self.hetero_data.edge_index_dict.items()
            }
            logger.info(f"Graph loaded | nodes: {self.static_x.shape[0]} | edge types: {len(self.edge_index_dict)}")
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
                self.predictions = np.load(PREDICTIONS_PATH)
                self.targets     = np.load(TARGETS_PATH)
            else:
                self.predictions = np.array([])
                self.targets     = np.array([])
            logger.info(f"Precomputed data loaded | snapshots: {self.X.shape[0]}")
        except Exception as e:
            logger.error(f"Failed to load precomputed data: {e}")
            raise CustomException("Failed to load precomputed data", e)

    # ── inference ─────────────────────────────────────────────────────────────
    def predict(self, product_name: str) -> dict:
        try:
            if product_name not in self.product_to_idx:
                raise ValueError(f"Product '{product_name}' not found in mapping")

            product_idx = self.product_to_idx[product_name]

            with torch.no_grad():
                # replicate exact forward logic from ModelTrainer.evaluate()
                x_cat  = torch.cat([self.static_x, self.last_temporal_x], dim=-1)  # [41, 18]
                x_dict = {NODE_TYPE: x_cat}
                out    = self.model(x_dict, self.edge_index_dict)                   # [41, 1]

            prediction = float(out[product_idx].cpu().item())

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
model_service = SupplyGraphModelService()

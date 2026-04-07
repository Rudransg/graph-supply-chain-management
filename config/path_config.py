import os
from pathlib import Path
import yaml


# ============================================================
# PROJECT ROOT
# ============================================================
ROOT_DIR = Path(__file__).resolve().parent.parent


# ============================================================
# CONFIG
# ============================================================
CONFIG_PATH = ROOT_DIR / "config" / "config.yaml"

with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)


# ============================================================
# DATA INGESTION CONFIG VALUES
# ============================================================
_ingestion = config["data_ingestion"]

BUCKET_NAME = _ingestion["bucket_name"]
BUCKET_FOLDER_NAME = _ingestion["bucket_folder_name"]
TRAIN_RATIO = _ingestion["train_ratio"]

_nodes = _ingestion["nodes"]
_edges = _ingestion["edges"]
_unit = _ingestion["temporal_data"]["unit"]
_weight = _ingestion["temporal_data"]["weight"]


# ============================================================
# LOCAL ARTIFACT DIRECTORIES
# ============================================================
ARTIFACTS_DIR = ROOT_DIR / "artifacts"

DATA_INGESTION_DIR = ARTIFACTS_DIR / "data_ingestion"
NODES_DIR = DATA_INGESTION_DIR / "nodes"
EDGES_DIR = DATA_INGESTION_DIR / "edges"
TEMPORAL_DIR = DATA_INGESTION_DIR / "temporal"
TEMPORAL_UNIT_DIR = TEMPORAL_DIR / "unit"
TEMPORAL_WEIGHT_DIR = TEMPORAL_DIR / "weight"

PROCESSED_DIR = ARTIFACTS_DIR / "processed"
MODELS_DIR = ARTIFACTS_DIR / "models"


# ============================================================
# LOCAL NODE FILE PATHS
# ============================================================
NODES_INDEX_FILE = NODES_DIR / "nodes_index.csv"
NODES_FILE = NODES_DIR / "nodes.csv"
NODES_PLANT_FILE = NODES_DIR / "nodes_type_plant_storage.csv"
NODES_GROUP_FILE = NODES_DIR / "node_types_product_group_subgroup.csv"

NODES_ID_COLUMN = _nodes[0]["id_column"]
NUM_NODES = _nodes[0]["num_nodes"]


# ============================================================
# LOCAL EDGE FILE PATHS
# ============================================================
EDGES_PLANT_FILE = EDGES_DIR / "same_plant.csv"
EDGES_STORAGE_FILE = EDGES_DIR / "same_storage.csv"
EDGES_GROUP_FILE = EDGES_DIR / "same_product_group.csv"
EDGES_SUBGROUP_FILE = EDGES_DIR / "same_product_subgroup.csv"

EDGE_SRC_COL = _edges[0]["src_col"]
EDGE_DST_COL = _edges[0]["dst_col"]
EDGE_UNDIRECTED = _edges[0]["undirected"]


# ============================================================
# LOCAL TEMPORAL UNIT FILE PATHS
# ============================================================
PRODUCTION_UNIT_FILE = TEMPORAL_UNIT_DIR / "production_unit.csv"
DELIVERY_UNIT_FILE = TEMPORAL_UNIT_DIR / "delivery_unit.csv"
FACTORY_ISSUE_UNIT_FILE = TEMPORAL_UNIT_DIR / "factory_issue_unit.csv"
SALES_ORDER_UNIT_FILE = TEMPORAL_UNIT_DIR / "sales_order_unit.csv"


# ============================================================
# LOCAL TEMPORAL WEIGHT FILE PATHS
# ============================================================
PRODUCTION_WEIGHT_FILE = TEMPORAL_WEIGHT_DIR / "production_weight.csv"
DELIVERY_WEIGHT_FILE = TEMPORAL_WEIGHT_DIR / "delivery_weight.csv"
FACTORY_ISSUE_WEIGHT_FILE = TEMPORAL_WEIGHT_DIR / "factory_issue_weight.csv"
SALES_ORDER_WEIGHT_FILE = TEMPORAL_WEIGHT_DIR / "sales_order_weight.csv"

TEMPORAL_DATE_COLUMN = _unit[0]["date_column"]


# ============================================================
# DATA PROCESSING CONFIG
# ============================================================
_processing = config["data_processing"]
_model = config["model"]

NODE_TYPE = _processing["node_type"]
NUM_NODES = _processing["num_nodes"]

ROLLING_WINDOW = _processing["temporal"]["rolling_window"]
DROP_FIRST_N_ROWS = _processing["temporal"]["drop_first_n_rows"]
TARGET_SIGNALS = _processing["temporal"]["target_signals"]
PREDICTION_HORIZON = _processing["temporal"]["prediction_horizon"]
HISTORY_STEPS = _processing["temporal"]["history_steps"]

EDGE_RELATIONS = _processing["edge_relations"]


# ============================================================
# MODEL CONFIG
# ============================================================
STATIC_FEATURE_DIM = _model["static_feature_dim"]
TEMPORAL_FEATURE_DIM = _model["temporal_feature_dim"]
IN_CHANNELS = _model["in_channels"]
HIDDEN_CHANNELS = _model["hidden_channels"]
OUT_CHANNELS = _model["out_channels"]
CONV_TYPE = _model["conv_type"]
AGGREGATION = _model["aggregation"]
LAYERS = _model["layers"]
MODEL_TRAIN_RATIO = _model["train_ratio"]
LEARNING_RATE = _model.get("learning_rate", 0.01)
EPOCHS = _model.get("epochs", 50)
LOSS_ALPHA = _model.get("loss_alpha", 2.0)
BATCH_SIZE = _model.get("batch_size", 8)


# ============================================================
# PROCESSED ARTIFACT PATHS  ← all missing paths were here
# ============================================================
ROLLED_SIGNAL_PATH = PROCESSED_DIR / "rolled_signals.csv"
ROLLED_SIGNAL_WITH_PREDS_PATH = PROCESSED_DIR / "rolled_signals_with_preds.csv"
PRODUCT_TREND_LONG_PATH = PROCESSED_DIR / "product_trend_long.csv"

VALUES_NUMPY_PATH = PROCESSED_DIR / "values.npy"        # [T, N, C] — replaces X/Y
EDGE_WEIGHT_PATH = PROCESSED_DIR / "edge_weight.npy"
EDGE_INDEX_HOMO_PATH = PROCESSED_DIR / "edge_index_homo.pt"
HETERO_DATA_PATH = PROCESSED_DIR / "hetero_data.pt"

PROCESSED_TRAIN_DATA_PATH = PROCESSED_DIR / "train_data.csv"
PROCESSED_TEST_DATA_PATH = PROCESSED_DIR / "test_data.csv"

COLUMN_MAPPING_PATH = PROCESSED_DIR / "column_mapping.json"
REVERSE_MAPPING_PATH = PROCESSED_DIR / "reverse_mapping.json"
PRODUCT_ORDER_PATH = PROCESSED_DIR / "product_order.json"
PRODUCT_NAME_TO_IDX_PATH = PROCESSED_DIR / "product_name_to_idx.json"
PRODUCT_IDX_TO_NAME_PATH = PROCESSED_DIR / "product_idx_to_name.json"
PRODUCT_TO_IDX_PATH = PROCESSED_DIR / "product_to_idx.json"
IDX_TO_PRODUCT_PATH = PROCESSED_DIR / "idx_to_product.json"
METADATA_PATH = PROCESSED_DIR / "metadata.json"
X_NUMPY_PATH = PROCESSED_DIR / "X.npy"
Y_NUMPY_PATH = PROCESSED_DIR / "Y.npy"

# ============================================================
# MODEL OUTPUT PATHS  ← all missing paths were here
# ============================================================
MODEL_OUTPUT_PATH = MODELS_DIR / "model.pt"
METRICS_OUTPUT_PATH = MODELS_DIR / "metrics.json"
PREDICTIONS_OUTPUT_PATH = MODELS_DIR / "predictions.npy"
TARGETS_OUTPUT_PATH = MODELS_DIR / "targets.npy"

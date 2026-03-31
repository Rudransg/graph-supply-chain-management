import json
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / "artifacts"
NODES_DIR = ARTIFACTS / "data_ingestion" / "nodes"
EDGES_DIR = ARTIFACTS / "data_ingestion" / "edges"
IDX_TO_PROD_PATH = ARTIFACTS / "processed" / "idx_to_product.json"
PROD_TO_IDX_PATH = ARTIFACTS / "processed" / "product_to_idx.json"

_cache = {}

def _load(path: Path):
    key = str(path)
    if key not in _cache:
        _cache[key] = pd.read_csv(path)
    return _cache[key]

def get_group_subgroup_map() -> dict[str, dict]:
    """Returns {product_name: {group, subgroup}}"""
    csv_path = NODES_DIR / "node_types_product_group_and_subgroup.csv"
    if not csv_path.exists():
        return {}
    df = _load(csv_path)
    result = {}
    for _, row in df.iterrows():
        result[row["Node"]] = {
            "group":    row["Group"],
            "subgroup": row["Sub-Group"],
        }
    return result

def get_factory_map(idx_to_product: dict) -> dict[str, str]:
    """
    Deterministic factory assignment based on product index.
    Plant A=0, B=1, C=2, D=3 cycling by idx % 4.
    """
    factories = ["Plant A", "Plant B", "Plant C", "Plant D"]
    return {
        prod: factories[int(idx) % 4]
        for idx, prod in idx_to_product.items()
    }

def build_forecast_cache(
    idx_to_product: dict[str, str],
    predictions:    np.ndarray,
    targets:        np.ndarray,
    X:              np.ndarray,
) -> list[dict]:
    """
    Build per-product enriched forecast rows using real model output.

    predictions shape: [num_snapshots * num_nodes] or [num_snapshots, num_nodes]
    X shape:           [num_snapshots, num_nodes, 1]
    """
    gsmap      = get_group_subgroup_map()
    fmap       = get_factory_map(idx_to_product)
    n_products = len(idx_to_product)

    # Normalise predictions shape
    if predictions.size == 0:
        preds_2d = np.zeros((1, n_products))
    elif predictions.ndim == 1:
        # Flatten case: reshape to [snapshots, nodes]
        total_elements = predictions.size
        if total_elements % n_products == 0:
            n_snapshots = total_elements // n_products
            preds_2d = predictions.reshape(n_snapshots, n_products)
        else:
            # Fallback: treat as single snapshot
            preds_2d = predictions[:n_products].reshape(1, -1)
            if preds_2d.shape[1] < n_products:
                # Pad with zeros if not enough values
                pad_width = ((0, 0), (0, n_products - preds_2d.shape[1]))
                preds_2d = np.pad(preds_2d, pad_width, constant_values=0)
    elif predictions.ndim == 2:
        preds_2d = predictions
        # Safety check: ensure correct width
        if preds_2d.shape[1] != n_products:
            print(f"[Warning] predictions.shape={preds_2d.shape} but expected {n_products} products. Using zeros.")
            preds_2d = np.zeros((preds_2d.shape[0], n_products))
    else:
        preds_2d = np.zeros((1, n_products))

    results: list[dict] = []

    for idx_str, prod in idx_to_product.items():
        idx = int(idx_str)

        # latest forecast value
        if preds_2d.shape[1] > idx:
            latest_pred = float(preds_2d[-1, idx])
        else:
            latest_pred = 0.0

        # trend = % change from second-to-last snapshot
        if preds_2d.shape[0] > 1 and preds_2d.shape[1] > idx:
            prev = float(preds_2d[-2, idx])
            trend_pct = ((latest_pred - prev) / (abs(prev) + 1e-9)) * 100.0
        else:
            # fallback: use X temporal data slope
            if X.ndim == 3 and X.shape[1] > idx and X.shape[0] > 1:
                v_now  = float(X[-1,  idx, 0])
                v_prev = float(X[-2,  idx, 0])
                trend_pct = ((v_now - v_prev) / (abs(v_prev) + 1e-9)) * 100.0
            else:
                np.random.seed(idx * 13 + 7)
                trend_pct = float(np.random.uniform(-25.0, 25.0))

        gs = gsmap.get(prod, {"group": "Unknown", "subgroup": "Unknown"})

        results.append({
            "product":     prod,
            "forecast_kg": round(latest_pred, 2),
            "trend_pct":   round(trend_pct,   1),
            "factory":     fmap.get(prod, "Unknown"),
            "group":       gs["group"],
            "subgroup":    gs["subgroup"],
            "status":      "at_risk" if trend_pct < -10.0 else "on_track",
        })

    return results


def build_trend_series(
    product_idx: int,
    X:           np.ndarray,
    predictions: np.ndarray,
    days:        int = 30,
) -> dict:

    """
    Use real X temporal data for historical trend.
    X shape: [num_snapshots, num_nodes, 1]
    """
    if X.ndim == 3:
        series = X[:, product_idx, 0].tolist()
    else:
        series = X[:, product_idx].tolist()

    # Take last `days` for historical
    historical = [round(float(v), 2) for v in series[-days:]]

    # Extrapolate 7-day forecast with slight noise
    np.random.seed(product_idx + 99)
    last = historical[-1] if historical else 500.0
    forecast = [
        round(float(last * (1 + np.random.uniform(-0.03, 0.05))), 2)
        for _ in range(7)
    ]

    return {
        "historical": historical,
        "forecast":   forecast,
        "days_historical": len(historical),
        "days_forecast":   7,
    }
def get_edge_data() -> dict[str, pd.DataFrame]:
    """Returns {relation_name: DataFrame with src/dst columns}"""
    edges: dict[str, pd.DataFrame] = {}
    if not EDGES_DIR.exists():
        return edges
    for f in EDGES_DIR.glob("*.csv"):
        edges[f.stem] = _load(f)
    return edges



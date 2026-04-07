from pathlib import Path
import pandas as pd
import json

BASE = Path(__file__).resolve().parents[2] / "artifacts"
PROCESSED = BASE / "processed"

with open(PROCESSED / "product_to_idx.json", "r") as f:
    product_to_idx = json.load(f)

with open(PROCESSED / "idx_to_product.json", "r") as f:
    idx_to_product = json.load(f)

meta = pd.read_csv(PROCESSED / "rolled_signal_with_preds.csv")

KNOWN_SIGNALS = [
    "production_unit",
    "delivery_unit",
    "sales_order_unit",
]

def split_signal_product(signal_product: str):
    for signal in KNOWN_SIGNALS:
        prefix = f"{signal}_"
        if signal_product.startswith(prefix):
            return signal, signal_product[len(prefix):]
    return None, signal_product

def build_signal_product_map() -> dict[str, dict[str, str]]:
    result = {}
    for signal_product, idx in product_to_idx.items():
        signal_type, product_id = split_signal_product(signal_product)
        if signal_type is None:
            continue
        if product_id not in result:
            result[product_id] = {}
        result[product_id][signal_type] = idx
    return result

signal_product_map = build_signal_product_map()
unique_products = set(signal_product_map.keys())

def get_trend_for_product(
    product_id: str,
    signal_type: str = "produced_units",
    limit: int = 30,
):
    signal_product_key = f"{signal_type}_{product_id}"
    if signal_product_key not in product_to_idx:
        return []

    idx = int(product_to_idx[signal_product_key])
    actual_col = str(idx)
    pred_col = f"{signal_product_key}_pred"

    if actual_col not in meta.columns or pred_col not in meta.columns:
        return []

    df = meta.reset_index().rename(columns={"index": "time_step"})
    df = df[["time_step", actual_col, pred_col]].sort_values("time_step").tail(limit)

    points = [
        {
            "timestamp": int(row["time_step"]),
            "actual": float(row[actual_col]),
            "predicted": float(row[pred_col]),
        }
        for _, row in df.iterrows()
    ]
    return points

def list_forecast_products() -> list[str]:
    return sorted(unique_products)

def get_products_by_category(category: str) -> list[str]:
    cat_signals = {
        "production": ["production_unit"],
        "delivery": ["delivery_unit"],
        "supply_order": ["sales_order_unit"],
    }
    prefixes = cat_signals.get(category.lower(), [])

    cat_products = [
        product_id
        for product_id, signals in signal_product_map.items()
        if any(signal in signals for signal in prefixes)
    ]
    return sorted(cat_products)

def get_category_forecast_summary(category: str, limit: int = 20) -> list[dict]:
    category_default_signal = {
        "production": "production_unit",
        "delivery": "delivery_unit",
        "supply_order": "sales_order_unit",
    }

    signal_type = category_default_signal.get(category.lower())
    if not signal_type:
        return []

    products = get_products_by_category(category)[:limit]
    summaries = []

    for prod in products:
        trend_data = get_trend_for_product(prod, signal_type=signal_type)
        if trend_data:
            latest = trend_data[-1]
            prev_pred = trend_data[-2]["predicted"] if len(trend_data) > 1 else latest["predicted"]
            trend_pct = round(
                ((latest["predicted"] - prev_pred) / (abs(prev_pred) + 1e-9)) * 100,
                1,
            )

            summaries.append({
                "product": prod,
                "signal_type": signal_type,
                "latest_actual": latest["actual"],
                "latest_pred": latest["predicted"],
                "trend_pct": trend_pct,
                "status": "at_risk" if trend_pct < -10.0 else "on_track",
            })

    return summaries
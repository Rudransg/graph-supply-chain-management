from pathlib import Path
import pandas as pd
import json

BASE = Path(__file__).resolve().parents[2] / "artifacts"
PROCESSED = BASE / "processed"
with open(PROCESSED / "product_to_idx.json", "r") as f:
    product_to_idx = json.load(f)
print("known products example:", list(product_to_idx.keys())[:10])
print("has ATPPCH5X5K:", "ATPPCH5X5K" in product_to_idx)
if "ATPPCH5X5K" in product_to_idx:
    print("idx for ATPPCH5X5K:", product_to_idx["ATPPCH5X5K"])
meta = pd.read_csv(PROCESSED / "rolled_signal_with_preds.csv")
print("columns in meta:", meta.columns.tolist())




def get_trend_for_product(product_id: str, limit: int = 30):
    if product_id not in product_to_idx:
        return []

    idx = int(product_to_idx[product_id])
    actual_col = str(idx)
    pred_col = f"{product_id}_pred"

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
    # product_to_idx is already loaded
    return sorted(product_to_idx.keys())

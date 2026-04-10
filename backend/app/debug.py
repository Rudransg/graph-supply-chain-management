import pandas as pd
import json
from pathlib import Path
import string



df = pd.read_csv("D:\\Supply Graph\\artifacts\\data_ingestion\\edges\\same_plant.csv")
with open("D:\\Supply Graph\\artifacts\\processed\\idx_to_product.json") as f:
    idx_to_product = json.load(f)

def get_plant_label(i):
    letters = string.ascii_uppercase
    return f"Plant {letters[i]}" if i < 26 else f"Plant {letters[i//26-1]}{letters[i%26]}"

NUM_SIGNALS = 3
product_idx_to_name = {}
for idx_str, signal_product in idx_to_product.items():
    product_idx = int(idx_str) // NUM_SIGNALS
    if product_idx not in product_idx_to_name:
        for sig in ["production_unit_", "delivery_unit_", "sales_order_unit_"]:
            if signal_product.startswith(sig):
                product_idx_to_name[product_idx] = signal_product[len(sig):]
                break

unique_plants = sorted(df["Plant"].unique())
plant_rename = {str(int(p)): get_plant_label(i) for i, p in enumerate(unique_plants)}

print(f"Total plants in edges: {len(unique_plants)}")
print(f"Products in idx_to_product: {len(product_idx_to_name)}\n")

missing_plants = []
for i, plant_id in enumerate(unique_plants):
    label = get_plant_label(i)
    plant_nodes = set(
        df[df["Plant"] == plant_id]["node1"].tolist() +
        df[df["Plant"] == plant_id]["node2"].tolist()
    )
    matched = [product_idx_to_name[n] for n in plant_nodes if n in product_idx_to_name]
    if not matched:
        missing_plants.append((label, plant_id, sorted(plant_nodes)))
        print(f"❌ {label} (ID {plant_id}): node indices {sorted(plant_nodes)} → NO matched products")
    else:
        print(f"✅ {label} (ID {plant_id}): {matched}")

print(f"\nTotal missing plants: {len(missing_plants)}")
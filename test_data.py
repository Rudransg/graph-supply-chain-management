import pandas as pd
import json

csv_path = 'artifacts/processed/rolled_signal_with_preds.csv'
meta = pd.read_csv(csv_path)
print(f'CSV shape: {meta.shape}')
print(f'Columns: {len(meta.columns)}')

# Check for prediction columns
pred_cols = [c for c in meta.columns if 'pred' in c]
print(f'Prediction columns: {len(pred_cols)}')
print('Sample pred columns:', pred_cols[:3])

# Check a specific product
idx_map = json.load(open('artifacts/processed/product_to_idx.json'))
sample_key = 'production_unit_AT5X5K'
if sample_key in idx_map:
    idx = int(idx_map[sample_key])
    print(f'\nSample check for {sample_key} (idx={idx}):')
    print(f'  Column {idx} exists: {idx < len(meta.columns) or str(idx) in meta.columns}')
    print(f'  Pred column exists: {f"{sample_key}_pred" in meta.columns}')
    print(f'  Available columns sample: {list(meta.columns)[:5]}')

# Check if predictions file has the data
preds_file = 'artifacts/models/predictions.npy'
import os
print(f'\nPredictions file exists: {os.path.exists(preds_file)}')

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv,HeteroConv, SAGEConv

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from config.path_config import *
from src.custom_exception import CustomException
from src.logger import get_logger


logger = get_logger(__name__)


def split_signal_product(signal_product: str) -> tuple[str | None, str]:
    for signal in TARGET_SIGNALS:
        prefix = f"{signal}_"
        if signal_product.startswith(prefix):
            return signal, signal_product[len(prefix):]
    return None, signal_product


class DeepGCNGRUCell(nn.Module):
    def __init__(self, in_channels, hidden_channels, edge_relations, node_type, dropout=0.2):
        super().__init__()
        self.node_type = node_type
        self.hetero_conv = HeteroConv({
            (node_type, rel, node_type): SAGEConv((-1, -1), hidden_channels)
            for rel in edge_relations
        }, aggr='sum')
        self.gru1 = nn.GRUCell(hidden_channels, hidden_channels)
        self.gru2 = nn.GRUCell(hidden_channels, hidden_channels)
        self.gru3 = nn.GRUCell(hidden_channels, hidden_channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index_dict, h1, h2, h3):
        x_dict = {self.node_type: x}
        x_dict = self.hetero_conv(x_dict, edge_index_dict)
        x = self.dropout(x_dict[self.node_type].relu())
        h1 = self.gru1(x, h1)
        h2 = self.gru2(h1, h2)
        h3 = self.gru3(h2, h3)
        return h1, h2, h3


class MultiStepGCNGRU(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels,
                 forecast_horizon, edge_relations, node_type):
        super().__init__()
        self.node_type = node_type
        self.cell = DeepGCNGRUCell(in_channels, hidden_channels, edge_relations, node_type)
        self.proj = nn.Linear(hidden_channels, out_channels)
        self.forecast_horizon = forecast_horizon

    def batch_edge_index_dict(self, edge_index_dict, B, N):
        batched = {}
        for key, ei in edge_index_dict.items():
            offsets = torch.arange(B, device=ei.device) * N
            batched[key] = torch.cat([ei + off for off in offsets], dim=1)
        return batched

    def forward(self, x_seq, edge_index_dict):
        B, P, N, C = x_seq.shape
        H = self.proj.in_features
        device = x_seq.device

        h1 = torch.zeros(B * N, H, device=device)
        h2 = torch.zeros(B * N, H, device=device)
        h3 = torch.zeros(B * N, H, device=device)

        edge_index_dict_batched = self.batch_edge_index_dict(edge_index_dict, B, N)

        for t in range(P):
            xt = x_seq[:, t].reshape(B * N, C)
            h1, h2, h3 = self.cell(xt, edge_index_dict_batched, h1, h2, h3)

        preds = []
        x_dec = x_seq[:, -1]
        for _ in range(self.forecast_horizon):
            xt = x_dec.reshape(B * N, C)
            h1, h2, h3 = self.cell(xt, edge_index_dict_batched, h1, h2, h3)
            y_t = self.proj(h3).reshape(B, N, -1)
            preds.append(y_t)
            x_dec = y_t

        return torch.stack(preds, dim=1)

# ── service singleton ─────────────────────────────────────────────────────────
class MultiStepGCNGRUModelService:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        self.model:              MultiStepGCNGRU
        self.edge_index_dict:    dict        # ← fixes the Pylance erro
        self.product_positions:  dict
        self.metrics:            dict
        self.last_input_window:  torch.Tensor
        self.y_mean:             np.ndarray
        self.y_std:              np.ndarray
        self.Y_mean:             np.ndarray
        self.Y_std:              np.ndarray
        self.predictions:        np.ndarray
        self.targets:            np.ndarray

        self._load_model()
        self._load_graph_data()
        self._load_mappings()
        self._load_metrics()
        self._load_precomputed()

    def _load_model(self):
        try:
            self.model = MultiStepGCNGRU(
                in_channels=TEMPORAL_FEATURE_DIM,
                hidden_channels=HIDDEN_CHANNELS,
                out_channels=OUT_CHANNELS,
                forecast_horizon=PREDICTION_HORIZON,
                edge_relations=EDGE_RELATIONS,
                node_type=NODE_TYPE,
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
            self.hetero_data = torch.load(
                HETERO_DATA_PATH,
                map_location=self.device,
                weights_only=False,
            )
            self.edge_index_dict = {
            key: ei.to(self.device)
            for key, ei in self.hetero_data.edge_index_dict.items()
        }
            logger.info(f"Graph loaded | relations: {list(self.edge_index_dict.keys())}")
        except Exception as e:
            logger.error(f"Failed to load graph data: {e}")
            raise CustomException("Failed to load graph data", e)

    def _load_mappings(self):
        try:
            with open(REVERSE_MAPPING_PATH, "r") as f:
                self.idx_to_product: dict[str, str] = json.load(f)
            with open(COLUMN_MAPPING_PATH, "r") as f:
                self.product_to_idx: dict[str, int] = json.load(f)
            if PRODUCT_ORDER_PATH.exists():
                with open(PRODUCT_ORDER_PATH, "r") as f:
                    product_ids = json.load(f)
            else:
                product_ids = []
                for _, signal_product in sorted(self.idx_to_product.items(), key=lambda item: int(item[0])):
                    signal_type, product_id = split_signal_product(signal_product)
                    if signal_type is None:
                        continue
                    if product_id not in product_ids:
                        product_ids.append(product_id)

            product_positions = {product_id: idx for idx, product_id in enumerate(product_ids)}
            signal_index_map = {}

            sorted_entries = sorted(self.idx_to_product.items(), key=lambda item: int(item[0]))
            for idx_str, signal_product in sorted_entries:
                signal_type, product_id = split_signal_product(signal_product)
                if signal_type is None:
                    continue

                signal_index_map.setdefault(product_id, {})[signal_type] = int(idx_str)

            self.product_ids = product_ids
            self.product_positions = product_positions
            self.signal_index_map = signal_index_map

            logger.info(
                f"Loaded mappings | signal-product entries: {len(self.idx_to_product)} | products: {len(self.product_ids)}"
            )
        except Exception as e:
            logger.error(f"Failed to load mappings: {e}")
            raise CustomException("Failed to load mappings", e)

    def _load_metrics(self):
        try:
            if METRICS_OUTPUT_PATH.exists():
                with open(METRICS_OUTPUT_PATH, "r") as f:
                    self.metrics: dict = json.load(f)
            else:
                self.metrics = {}
            logger.info("Metrics loaded")
        except Exception as e:
            logger.error(f"Failed to load metrics: {e}")
            raise CustomException("Failed to load metrics", e)

    def _reshape_forecast_arrays(self, arr: np.ndarray) -> np.ndarray:
        if arr.size == 0:
            return np.array([])

        if arr.ndim == 4:
            return arr[:, 0, :, :].reshape(arr.shape[0], -1)

        if arr.ndim != 2:
            return arr.reshape(arr.shape[0], -1)

        expected_step = len(self.product_ids) * len(TARGET_SIGNALS)
        expected_full = expected_step * PREDICTION_HORIZON

        if arr.shape[1] == expected_full:
            reshaped = arr.reshape(
                arr.shape[0],
                PREDICTION_HORIZON,
                len(self.product_ids),
                len(TARGET_SIGNALS),
            )
            return reshaped[:, 0, :, :].reshape(arr.shape[0], -1)

        return arr

    def _load_norm_array(self, path: Path, shape: tuple[int, ...], default_value: float):
        if path.exists():
            arr = np.load(path)
            if arr.shape == shape:
                return arr.astype(np.float32)

            logger.warning(
                "Normalization artifact %s has shape %s; expected %s. Using fallback values.",
                path.name,
                arr.shape,
                shape,
            )
        else:
            logger.warning(f"Normalization artifact missing: {path.name}; using fallback values")
        return np.full(shape, default_value, dtype=np.float32)

    def _build_input_norm_from_values(self, values: np.ndarray):
        total_sequences = values.shape[0] - HISTORY_STEPS - PREDICTION_HORIZON + 1
        if total_sequences <= 0:
            raise ValueError(
                f"Not enough timesteps in values.npy to build input normalization: {values.shape[0]}"
            )

        ratio = MODEL_TRAIN_RATIO if MODEL_TRAIN_RATIO is not None else TRAIN_RATIO
        split_idx = int(total_sequences * ratio)
        if split_idx <= 0:
            raise ValueError("Training split produced zero sequences; cannot build input normalization")

        x_sequences = np.stack(
            [values[t : t + HISTORY_STEPS] for t in range(total_sequences)],
            axis=0,
        )
        x_train = x_sequences[:split_idx]
        x_mean = x_train.mean(axis=(0, 1), keepdims=True)
        x_std = x_train.std(axis=(0, 1), keepdims=True) + 1e-8
        return x_mean.astype(np.float32), x_std.astype(np.float32)

    def _load_precomputed(self):
        try:
            self.values = np.load(VALUES_NUMPY_PATH)
            self.X = self.values

            if self.values.shape[0] < HISTORY_STEPS:
                raise ValueError(
                    f"values.npy has only {self.values.shape[0]} timesteps, but HISTORY_STEPS={HISTORY_STEPS}"
                )

            num_products = self.values.shape[1]
            feature_dim = self.values.shape[2]
            norm_shape = (1, 1, num_products, feature_dim)
            self.y_mean = self._load_norm_array(MODELS_DIR / "Y_mean.npy", norm_shape, 0.0)
            self.y_std = self._load_norm_array(MODELS_DIR / "Y_std.npy", norm_shape, 1.0)

            x_mean_path = MODELS_DIR / "X_mean.npy"
            x_std_path = MODELS_DIR / "X_std.npy"
            if x_mean_path.exists() and x_std_path.exists():
                self.x_mean = self._load_norm_array(x_mean_path, norm_shape, 0.0)
                self.x_std = self._load_norm_array(x_std_path, norm_shape, 1.0)
            else:
                logger.warning("X_mean.npy / X_std.npy missing; reconstructing input normalization from values.npy")
                self.x_mean, self.x_std = self._build_input_norm_from_values(self.values)

            last_window = self.values[-HISTORY_STEPS:]
            normalized_window = (last_window - self.x_mean.squeeze()) / self.x_std.squeeze()
            self.last_input_window = torch.tensor(
                normalized_window[np.newaxis, ...],
                dtype=torch.float32,
                device=self.device,
            )

            if PREDICTIONS_OUTPUT_PATH.exists() and TARGETS_OUTPUT_PATH.exists():
                self.predictions = self._reshape_forecast_arrays(np.load(PREDICTIONS_OUTPUT_PATH))
                self.targets = self._reshape_forecast_arrays(np.load(TARGETS_OUTPUT_PATH))
            else:
                self.predictions = np.array([])
                self.targets = np.array([])

            logger.info(
                f"Precomputed data loaded | values shape: {self.values.shape} | preds shape: {self.predictions.shape}"
            )
        except Exception as e:
            logger.error(f"Failed to load precomputed data: {e}")
            raise CustomException("Failed to load precomputed data", e)
    

    def predict(self, product_name: str) -> dict:
        try:
            product_id = product_name
            requested_signal = None

            if product_name in self.product_to_idx:
                requested_signal, product_id = split_signal_product(product_name)

            if product_id not in self.product_positions:
                raise ValueError(f"Product '{product_name}' not found in mapping")

            product_idx = self.product_positions[product_id]

            with torch.no_grad():
                out = self.model(self.last_input_window, self.edge_index_dict)

            y_mean = torch.tensor(self.y_mean, dtype=torch.float32, device=self.device)
            y_std = torch.tensor(self.y_std, dtype=torch.float32, device=self.device)
            out = out * y_std + y_mean

            product_forecast = out[0, :, product_idx, :].detach().cpu().numpy()
            pred_mean = product_forecast.mean(axis=0)   # [C] — mean across Q days
            pred_std  = product_forecast.std(axis=0)    # [C] — spread across Q days
            
            weekly_totals = product_forecast.sum(axis=0)
            next_day = product_forecast[0]

            ### lower bound 
            lower_bound = {
            signal: round(float(max(0, weekly_totals[i] - 7*pred_std[i])), 0)
            for i, signal in enumerate(TARGET_SIGNALS)}
            ## upper bound
            upper_bound = {
            signal: round(float(weekly_totals[i] + 7*pred_std[i]), 0)
            for i, signal in enumerate(TARGET_SIGNALS)}

            prediction_by_signal = {
                signal: round(float(weekly_totals[i]), 0)
                for i, signal in enumerate(TARGET_SIGNALS)
            }
            next_day_prediction = {
                signal: round(float(next_day[i]), 0)
                for i, signal in enumerate(TARGET_SIGNALS)
            }
            daily_forecast = {
                signal: [round(float(day[i]), 0) for day in product_forecast]
                for i, signal in enumerate(TARGET_SIGNALS)
            }

            prediction = prediction_by_signal
            if requested_signal is not None:
                prediction = round(float(prediction_by_signal[requested_signal]), 0)

            return {
                "product_name": product_id,
                "product_idx": product_idx,
                "prediction": prediction,
                "next_day_prediction": next_day_prediction,
                "daily_forecast": daily_forecast,
                "forecast_horizon_days": int(product_forecast.shape[0]),
                "lower_bound": lower_bound,
                "upper_bound": upper_bound,
            }
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise CustomException("Prediction failed", e)

    def predict_whatif(
        self,
        product_name: str,
        zeroed_products: list[str] = [],
        zeroed_factories: list[str] = [],
        capacity_overrides: dict[str, float] = {},
        dropped_relations: list[str] = [],
    ) -> dict:
        try:
            # ── 1. resolve product ──────────────────────────────────────────
            product_id = product_name
            requested_signal = None
            if product_name in self.product_to_idx:
                requested_signal, product_id = split_signal_product(product_name)
            if product_id not in self.product_positions:
                raise ValueError(f"Product '{product_name}' not found")

            # ── 2. clone input — never mutate the original ──────────────────
            input_window = self.last_input_window.clone()  # [1, HISTORY, N, C]

            # ── 3. build product→factory map from cache ─────────────────────
            from .data_service import build_forecast_cache
            cache = build_forecast_cache(
                idx_to_product=self.idx_to_product,
                predictions=self.predictions,
                targets=self.targets,
                values=self.values,
            )
            product_factory_map = {p["product"]: p["factory"] for p in cache}

            # ── 4. collect indices to zero out ──────────────────────────────
            indices_to_zero = set()

            for p in zeroed_products:
                if p in self.product_positions:
                    indices_to_zero.add(self.product_positions[p])

            for p in self.product_ids:
                if product_factory_map.get(p) in zeroed_factories:
                    if p in self.product_positions:
                        indices_to_zero.add(self.product_positions[p])

            # ── 5. apply zeroing ────────────────────────────────────────────
            for idx in indices_to_zero:
                input_window[:, :, idx, :] = 0.0

            # ── 6. apply capacity scaling ───────────────────────────────────
            for p, scale in capacity_overrides.items():
                if p in self.product_positions:
                    idx = self.product_positions[p]
                    input_window[:, :, idx, :] *= float(scale)

            # ── 7. optionally drop edge relations ───────────────────────────
            edge_dict = {
                k: v for k, v in self.edge_index_dict.items()
                if k[1] not in dropped_relations
            }

            # ── 8. run inference ────────────────────────────────────────────
            with torch.no_grad():
                out = self.model(input_window, edge_dict)

            y_mean = torch.tensor(self.y_mean, dtype=torch.float32, device=self.device)
            y_std  = torch.tensor(self.y_std,  dtype=torch.float32, device=self.device)
            while y_mean.dim() < 4: y_mean = y_mean.unsqueeze(0)
            while y_std.dim()  < 4: y_std  = y_std.unsqueeze(0)
            out = out * y_std + y_mean

            product_idx      = self.product_positions[product_id]
            product_forecast = out[0, :, product_idx, :].detach().cpu().numpy()
            weekly_totals    = product_forecast.sum(axis=0)
            next_day         = product_forecast[0]

            prediction_by_signal = {
                signal: round(float(weekly_totals[i]), 0)
                for i, signal in enumerate(TARGET_SIGNALS)
            }
            next_day_prediction = {
                signal: round(float(next_day[i]), 0)
                for i, signal in enumerate(TARGET_SIGNALS)
            }
            daily_forecast = {
                signal: [round(float(day[i]), 0) for day in product_forecast]
                for i, signal in enumerate(TARGET_SIGNALS)
            }

            return {
                "product_name":         product_id,
                "prediction":           prediction_by_signal,
                "next_day_prediction":  next_day_prediction,
                "daily_forecast":       daily_forecast,
                "forecast_horizon_days": int(product_forecast.shape[0]),
                "scenario": {
                    "zeroed_products":    zeroed_products,
                    "zeroed_factories":   zeroed_factories,
                    "capacity_overrides": capacity_overrides,
                    "dropped_relations":  dropped_relations,
                }
            }
        except Exception as e:
            logger.error(f"What-if prediction failed: {e}")
            raise CustomException("What-if prediction failed", e)

    def get_live_forecast_series(
        self,
        product_name: str,
        signal_type: str,
        history_points: int = 30,
    ) -> dict:
        try:
            if signal_type not in TARGET_SIGNALS:
                raise ValueError(f"Unknown signal type '{signal_type}'")

            if product_name not in self.product_positions:
                raise ValueError(f"Product '{product_name}' not found in mapping")

            product_idx = self.product_positions[product_name]
            signal_idx = TARGET_SIGNALS.index(signal_type)

            prediction = self.predict(product_name)
            forecast_values = prediction["daily_forecast"][signal_type]

            actual_history = self.values[:, product_idx, signal_idx]
            history = actual_history[-history_points:]
            history_start = len(actual_history) - len(history)

            points = []
            for offset, value in enumerate(history):
                points.append(
                    {
                        "timestamp": int(history_start + offset),
                        "actual": float(value),
                        "predicted": None,
                    }
                )

            forecast_start = len(actual_history)
            for offset, value in enumerate(forecast_values, start=1):
                points.append(
                    {
                        "timestamp": int(forecast_start + offset - 1),
                        "actual": None,
                        "predicted": float(value),
                    }
                )

            return {
                "product": product_name,
                "signal_type": signal_type,
                "history_points": len(history),
                "forecast_horizon_days": len(forecast_values),
                "points": points,
            }
        except Exception as e:
            logger.error(f"Live forecast series failed: {e}")
            raise CustomException("Live forecast series failed", e)

    def get_all_products(self) -> dict[str, str]:
        return {str(idx): product_id for idx, product_id in enumerate(self.product_ids)}

    def get_scatter_data(self, limit: int = 300) -> list[dict]:
        if self.predictions.size == 0 or self.targets.size == 0:
            return []

        preds = self.predictions.flatten()[:limit]
        targets = self.targets.flatten()[:limit]
        return [
            {"predicted": round(float(p), 4), "actual": round(float(t), 4)}
            for p, t in zip(preds, targets)
        ]


model_service = MultiStepGCNGRUModelService()

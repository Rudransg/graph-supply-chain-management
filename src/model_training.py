import json

import mlflow
import mlflow.pytorch as mlflow_pytorch
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from torch.utils.data import Dataset, DataLoader
from torch_geometric.nn import GCNConv,HeteroConv, SAGEConv

from src.logger import get_logger
from src.custom_exception import CustomException
from config.path_config import *

logger = get_logger(__name__)


# ── CHANGE 1: Removed SupplyGraphModel (no temporal memory).
#    Replaced with GCN + 3-layer GRU sequence model from notebook. ──

class SupplySequenceDataset(Dataset):
    def __init__(self, X_seq, Y_seq):
        self.X = torch.tensor(X_seq, dtype=torch.float32)  # [S, P, N, C]
        self.Y = torch.tensor(Y_seq, dtype=torch.float32)  # [S, Q, N, C]

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


class DeepGCNGRUCell(nn.Module):
    def __init__(self, in_channels, hidden_channels,edge_relations, node_type, dropout=0.2):
        super().__init__()
        self.edge_relations = edge_relations
        self.node_type = node_type
        self.hetero_conv = HeteroConv({
            (self.node_type, rel, self.node_type): SAGEConv((-1, -1), hidden_channels)
            for rel in self.edge_relations
        }, aggr='sum')
        self.gru1 = nn.GRUCell(hidden_channels, hidden_channels)
        self.gru2 = nn.GRUCell(hidden_channels, hidden_channels)
        self.gru3 = nn.GRUCell(hidden_channels, hidden_channels)
        self.dropout = nn.Dropout(dropout)

    
        # x: [B*N, C] → wrap as dict for HeteroConv
    def forward(self, x, edge_index_dict, h1, h2, h3):
        x_dict = {self.node_type: x}
        x_dict = self.hetero_conv(x_dict, edge_index_dict)
        x = self.dropout(x_dict[self.node_type].relu())
        h1 = self.gru1(x, h1)
        h2 = self.gru2(h1, h2)
        h3 = self.gru3(h2, h3)
        return h1, h2, h3
    # def forward(self, x, edge_index, h1, h2, h3):
    #     # x: [B*N, C]
    #     x = F.relu(self.gcn1(x, edge_index))
    #     x = self.dropout(x)
    #     x = F.relu(self.gcn2(x, edge_index))
    #     x = self.dropout(x)
    #     h1 = self.gru1(x, h1)
    #     h2 = self.gru2(h1, h2)
    #     h3 = self.gru3(h2, h3)
    #     return h1, h2, h3


class MultiStepGCNGRU(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, forecast_horizon, edge_relations, node_type):
        super().__init__()
        self.cell = DeepGCNGRUCell(in_channels, hidden_channels, edge_relations, node_type)
        self.proj = nn.Linear(hidden_channels, out_channels)
        self.forecast_horizon = forecast_horizon
    def batch_edge_index_dict(self, edge_index_dict, B, N):
        """Tile edge_index_dict across B batches with correct node offsets."""
        batched = {}
        for key, ei in edge_index_dict.items():         # ei: [2, E]
            offsets = torch.arange(B, device=ei.device) * N  # [B]
            ei_batched = torch.cat([ei + off for off in offsets], dim=1)  # [2, B*E]
            batched[key] = ei_batched
        return batched
    def forward(self, x_seq, edge_index_dict):
        """
        x_seq      : [B, P, N, C]
        edge_index : [2, E]
        returns    : [B, Q, N, C]
        """
        B, P, N, C = x_seq.shape
        H = self.proj.in_features
        device = x_seq.device

        h1 = torch.zeros(B * N, H, device=device)
        h2 = torch.zeros(B * N, H, device=device)
        h3 = torch.zeros(B * N, H, device=device)
        edge_index_dict_batched = self.batch_edge_index_dict(edge_index_dict, B, N)
        # encode history
        for t in range(P):
            xt = x_seq[:, t].reshape(B * N, C)
            h1, h2, h3 = self.cell(xt, edge_index_dict_batched, h1, h2, h3)

        # autoregressive decode
        preds = []
        x_dec = x_seq[:, -1]  # [B, N, C]

        preds = []
        x_dec = x_seq[:, -1]
        for _ in range(self.forecast_horizon):
            xt = x_dec.reshape(B * N, C)
            h1, h2, h3 = self.cell(xt, edge_index_dict_batched, h1, h2, h3)
            y_t = self.proj(h3).reshape(B, N, -1)
            preds.append(y_t)
            x_dec = y_t

        return torch.stack(preds, dim=1)  # [B, Q, N, C]


class ModelTrainer:
    def __init__(self):
        try:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            MODELS_DIR.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            logger.error("Error while initializing ModelTrainer")
            raise CustomException("Error while initializing ModelTrainer", e)

    # ── CHANGE 2: load values.npy [T, N, C] instead of X.npy / Y.npy ──
    def load_artifacts(self):
        try:
            logger.info("Loading processed artifacts")
            hetero_data = torch.load(HETERO_DATA_PATH, weights_only=False)
            values = np.load(VALUES_NUMPY_PATH)  # [T, N, C]
            logger.info(f"Loading values.npy = {values}")
            logger.info(f"values shape: {values.shape}")
            logger.info(f"Target signals: {TARGET_SIGNALS}")

            return hetero_data, values
        except Exception as e:
            logger.error("Error while loading artifacts")
            raise CustomException("Error while loading artifacts", e)

    # ── CHANGE 3: build sliding-window sequences in the trainer ──
    def build_sequences(self, values: np.ndarray):
        try:
            T, N, C = values.shape
            P, Q = HISTORY_STEPS, PREDICTION_HORIZON

            X_list, Y_list = [], []
            for t in range(T - P - Q + 1):
                X_list.append(values[t : t + P])
                Y_list.append(values[t + P : t + P + Q])

            X_seq = np.stack(X_list)  # [S, P, N, C]
            Y_seq = np.stack(Y_list)  # [S, Q, N, C]

            logger.info(f"Sequences: X={X_seq.shape}  Y={Y_seq.shape}")
            return X_seq, Y_seq
        except Exception as e:
            logger.error("Error while building sequences")
            raise CustomException("Error while building sequences", e)

    # ── CHANGE 4: normalize using train split only ──
    def normalize(self, X_seq: np.ndarray, Y_seq: np.ndarray):
        try:
            S = len(X_seq)
            ratio = MODEL_TRAIN_RATIO if MODEL_TRAIN_RATIO is not None else TRAIN_RATIO
            split_idx = int(S * ratio)

            X_tr_raw, X_te_raw = X_seq[:split_idx], X_seq[split_idx:]
            Y_tr_raw, Y_te_raw = Y_seq[:split_idx], Y_seq[split_idx:]

            # Normalize per product and per signal using only the training split.
            # This preserves product-specific scale while staying time-causal.
            X_mean = X_tr_raw.mean(axis=(0, 1), keepdims=True)
            X_std  = X_tr_raw.std(axis=(0, 1),  keepdims=True) + 1e-8
            Y_mean = Y_tr_raw.mean(axis=(0, 1), keepdims=True)
            Y_std  = Y_tr_raw.std(axis=(0, 1),  keepdims=True) + 1e-8

            X_train = (X_tr_raw - X_mean) / X_std
            Y_train = (Y_tr_raw - Y_mean) / Y_std
            X_test  = (X_te_raw - X_mean) / X_std
            Y_test  = (Y_te_raw - Y_mean) / Y_std

            norm_stats = {
                "X_mean": X_mean,
                "X_std": X_std,
                "Y_mean": Y_mean,
                "Y_std": Y_std,
            }

            logger.info(f"Train size: {split_idx}  Test size: {S - split_idx}")
            return X_train, Y_train, X_test, Y_test, norm_stats
        except Exception as e:
            logger.error("Error during normalization")
            raise CustomException("Error during normalization", e)

    # ── CHANGE 5: merge hetero edge indices into one homo edge_index ──
    def build_edge_index(self, hetero_data):
        try:
            edges = list(hetero_data.edge_index_dict.values())
            edge_index_homo = torch.cat(edges, dim=1).to(self.device)
            return edge_index_homo
        except Exception as e:
            logger.error("Error while building edge index")
            raise CustomException("Error while building edge index", e)

    def asymmetric_loss(self, y_hat, y_true, alpha=2.0):
        try:
            diff = y_hat - y_true
            under = (diff < 0).float()
            over  = (diff >= 0).float()
            return ((diff ** 2) * (alpha * under + over)).mean()
        except Exception as e:
            logger.error("Error while computing asymmetric loss")
            raise CustomException("Error while computing asymmetric loss", e)

    # ── CHANGE 6: build_model now returns MultiStepGCNGRU ──
    def build_model(self):
        try:
            model = MultiStepGCNGRU(
                in_channels=TEMPORAL_FEATURE_DIM,   # 3 (no static_x needed)
                hidden_channels=HIDDEN_CHANNELS,
                out_channels=OUT_CHANNELS,
                forecast_horizon=PREDICTION_HORIZON,
                edge_relations=EDGE_RELATIONS,
                node_type=NODE_TYPE,
            ).to(self.device)
            return model
        except Exception as e:
            logger.error("Error while building model")
            raise CustomException("Error while building model", e)

    # ── CHANGE 7: train on DataLoader batches [B, P, N, C] ──
    def train_one_epoch(self, model, optimizer, train_loader, edge_index):
        try:
            model.train()
            total_loss, total_count = 0.0, 0

            for x_batch, y_batch in train_loader:
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                optimizer.zero_grad()
                y_hat = model(x_batch, edge_index)
                loss = self.asymmetric_loss(y_hat, y_batch, alpha=LOSS_ALPHA)
                loss.backward()
                optimizer.step()

                total_loss  += loss.item() * x_batch.size(0)
                total_count += x_batch.size(0)

            return total_loss / total_count
        except Exception as e:
            logger.error("Error during training epoch")
            raise CustomException("Error during training epoch", e)

    # ── CHANGE 8: evaluate on batches; denormalize before metrics ──
    def evaluate(self, model, test_loader, edge_index, norm_stats):
        try:
            model.eval()
            preds_all, targets_all, loss_list = [], [], []

            Y_mean = torch.tensor(norm_stats["Y_mean"], dtype=torch.float32, device=self.device)
            Y_std  = torch.tensor(norm_stats["Y_std"],  dtype=torch.float32, device=self.device)

            with torch.no_grad():
                for x_batch, y_batch in test_loader:
                    x_batch = x_batch.to(self.device)
                    y_batch = y_batch.to(self.device)

                    y_hat = model(x_batch, edge_index)
                    loss  = self.asymmetric_loss(y_hat, y_batch, alpha=LOSS_ALPHA)
                    loss_list.append(loss.item())

                    y_hat_d  = y_hat  * Y_std + Y_mean
                    y_true_d = y_batch * Y_std + Y_mean

                    preds_all.append(y_hat_d.cpu().numpy())
                    targets_all.append(y_true_d.cpu().numpy())

            preds_arr   = np.concatenate(preds_all,   axis=0)  # [S_test, Q, N, C]
            targets_arr = np.concatenate(targets_all, axis=0)

            preds_flat   = preds_arr.reshape(-1)
            targets_flat = targets_arr.reshape(-1)

            mse  = mean_squared_error(targets_flat, preds_flat)
            mae  = mean_absolute_error(targets_flat, preds_flat)
            rmse = float(np.sqrt(mse))
            r2   = r2_score(targets_flat, preds_flat)

            per_signal = {}
            for i, name in enumerate(TARGET_SIGNALS):
                sp = preds_arr[..., i].reshape(-1)
                st = targets_arr[..., i].reshape(-1)
                sm = mean_squared_error(st, sp)
                per_signal[name] = {
                    "mse":  float(sm),
                    "mae":  float(mean_absolute_error(st, sp)),
                    "rmse": float(np.sqrt(sm)),
                    "r2":   float(r2_score(st, sp)),
                }

            metrics = {
                "test_asymmetric_loss": float(np.mean(loss_list)),
                "mse": float(mse), "mae": float(mae),
                "rmse": rmse, "r2": float(r2),
                "per_signal": per_signal,
            }
            return metrics, preds_arr, targets_arr
        except Exception as e:
            logger.error("Error during evaluation")
            raise CustomException("Error during evaluation", e)

    # ── CHANGE 9: also save norm stats for inference ──
    def save_outputs(self, model, metrics, preds_all, targets_all, norm_stats):
        try:
            torch.save(model.state_dict(), MODEL_OUTPUT_PATH)

            with open(METRICS_OUTPUT_PATH, "w") as f:
                json.dump(metrics, f, indent=2)

            np.save(PREDICTIONS_OUTPUT_PATH, preds_all.reshape(preds_all.shape[0], -1))
            np.save(TARGETS_OUTPUT_PATH,     targets_all.reshape(targets_all.shape[0], -1))
            np.save(MODELS_DIR / "X_mean.npy", norm_stats["X_mean"])
            np.save(MODELS_DIR / "X_std.npy",  norm_stats["X_std"])
            np.save(MODELS_DIR / "Y_mean.npy", norm_stats["Y_mean"])
            np.save(MODELS_DIR / "Y_std.npy",  norm_stats["Y_std"])

            logger.info(f"Saved predictions shape: {preds_all.shape}")
        except Exception as e:
            logger.error("Error while saving outputs")
            raise CustomException("Error while saving outputs", e)

    def run(self):
        try:
            with mlflow.start_run():
                logger.info("Starting model training pipeline")

                hetero_data, values = self.load_artifacts()

                if NODE_TYPE not in hetero_data.node_types:
                    raise ValueError(f"Node type '{NODE_TYPE}' not found in hetero_data")

                edge_index_dict = {
                key: ei.to(self.device)
                for key, ei in hetero_data.edge_index_dict.items()
            }
                X_seq, Y_seq = self.build_sequences(values)
                X_train, Y_train, X_test, Y_test, norm_stats = self.normalize(X_seq, Y_seq)

                train_loader = DataLoader(
                    SupplySequenceDataset(X_train, Y_train),
                    batch_size=BATCH_SIZE, shuffle=True,
                )
                test_loader = DataLoader(
                    SupplySequenceDataset(X_test, Y_test),
                    batch_size=BATCH_SIZE, shuffle=False,
                )

                model     = self.build_model()
                optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

                mlflow.log_params({
                    "node_type": NODE_TYPE,
                    "relations": EDGE_RELATIONS,
                    "target_signals": TARGET_SIGNALS,
                    "history_steps": HISTORY_STEPS,
                    "prediction_horizon": PREDICTION_HORIZON,
                    "hidden_channels": HIDDEN_CHANNELS,
                    "conv_type": CONV_TYPE,
                    "learning_rate": LEARNING_RATE,
                    "epochs": EPOCHS,
                    "loss_alpha": LOSS_ALPHA,
                    "batch_size": BATCH_SIZE,
                })

                logger.info("Starting training")
                for epoch in range(EPOCHS):
                    train_loss = self.train_one_epoch(model, optimizer, train_loader, edge_index_dict)
                    logger.info(f"Epoch {epoch + 1}/{EPOCHS} - train loss: {train_loss:.4f}")
                    mlflow.log_metric("train_loss", train_loss, step=epoch)

                logger.info("Starting evaluation")
                metrics, preds_all, targets_all = self.evaluate(
                    model, test_loader, edge_index_dict, norm_stats
                )

                mlflow.log_metrics({
                    "test_asymmetric_loss": metrics["test_asymmetric_loss"],
                    "mse": metrics["mse"], "mae": metrics["mae"],
                    "rmse": metrics["rmse"], "r2": metrics["r2"],
                })

                mlflow_pytorch.log_model(model, name="model")
                self.save_outputs(model, metrics, preds_all, targets_all, norm_stats)

                logger.info(f"Final metrics: {metrics}")
                return metrics

        except Exception as e:
            logger.error("Error in trainer.run()")
            raise CustomException("Error in trainer.run()", e)


if __name__ == "__main__":
    trainer = ModelTrainer()
    metrics = trainer.run()
    print(metrics)

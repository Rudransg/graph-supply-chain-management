import json
from typing import cast
from src.logger import get_logger
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import mlflow
import mlflow.pytorch as mlflow_pytorch

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from torch_geometric.nn import HeteroConv, SAGEConv, GCNConv
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.typing import EdgeType
from torch_geometric_temporal.signal import StaticGraphTemporalSignal,temporal_signal_split


from src.custom_exception import CustomException
from config.path_config import *


logger =get_logger(__name__)

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
    )-> None: 
        try:
            super().__init__()
            self.node_type = node_type
            self.relations = relations
            self.layers = layers

            if conv_type.lower() == "sage":
                conv_layer = SAGEConv
            elif conv_type.lower() == "gcn":
                conv_layer = GCNConv
            else:
                raise ValueError(f"Unsupported conv_type: {conv_type}")

            self.convs = nn.ModuleList()

            first_layer: dict[EdgeType, MessagePassing] = {
            (node_type, rel, node_type): SupplyGraphModel._make_conv(
                conv_type, in_channels, hidden_channels
            )
            for rel in relations
        }
            self.convs.append(HeteroConv(first_layer, aggr=aggregation))

            for _ in range(layers - 1):
                hidden_layer: dict[EdgeType, MessagePassing] = {
                    (node_type, rel, node_type): SupplyGraphModel._make_conv(
                        conv_type, hidden_channels, hidden_channels
                    )
                    for rel in relations
                }
                self.convs.append(HeteroConv(hidden_layer, aggr=aggregation))

            self.lin = nn.Linear(hidden_channels, out_channels)

        except Exception as e:
            logger.error("Error while initializing SupplyGraphModel")
            raise CustomException("error found",e)
    @staticmethod
    def _make_conv(conv_type: str, in_ch: int, out_ch: int) -> MessagePassing:
        if conv_type.lower() == "sage":
            return cast(MessagePassing, SAGEConv(in_ch, out_ch))
        if conv_type.lower() == "gcn":
            return cast(MessagePassing, GCNConv(in_ch, out_ch))
        raise ValueError(f"Unsupported conv_type: {conv_type}")

    def forward(self, x_dict, edge_index_dict):
        try:
            for conv in self.convs:
                x_dict = conv(x_dict, edge_index_dict)
                x_dict = {k: F.relu(v) for k, v in x_dict.items()}
            out = self.lin(x_dict[self.node_type])
            return out

        except Exception as e:
            logger.error("Error in forward pass")
            raise CustomException("error found",e)


class ModelTrainer:
    def __init__(self):
        try:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            MODELS_DIR.mkdir(parents=True, exist_ok=True)

        except Exception as e:
            logger.error("Error while initializing ModelTrainer")
            raise CustomException("error found",e)
    def load_artifacts(self):
        try:
            logger.info("Loading processed artifacts")
            hetero_data = torch.load(HETERO_DATA_PATH, weights_only=False)
            X = np.load(X_NUMPY_PATH)
            Y = np.load(Y_NUMPY_PATH)
            return hetero_data, X, Y

        except Exception as e:
            logger.error("Error while loading artifacts")
            raise CustomException("error found",e)
    def build_temporal_dataset(self, hetero_data, X, Y):
        try:
            logger.info("Building temporal dataset")

            edge_indices = []
            for _, edge_index in hetero_data.edge_index_dict.items():
                edge_indices.append(edge_index)

            edge_index_homo = torch.cat(edge_indices, dim=1)
            edge_index_np = edge_index_homo.cpu().numpy()
            edge_weight = np.ones(edge_index_np.shape[1], dtype=np.float32)

            dataset = StaticGraphTemporalSignal(
                edge_index=edge_index_np,
                edge_weight=edge_weight,
                features=[x for x in X],
                targets=[y for y in Y],
            )

            split_ratio = MODEL_TRAIN_RATIO if MODEL_TRAIN_RATIO is not None else TRAIN_RATIO
            train_dataset, test_dataset = temporal_signal_split(dataset, train_ratio=split_ratio)

            return dataset, train_dataset, test_dataset

        except Exception as e:
            logger.error("Error while building temporal dataset")
            raise CustomException("error found",e)

    def normalize_edge_index_dict(self, edge_index_dict):
        try:
            normalized = {}
            for (src_type, rel_type, dst_type), edge_index_tensor in edge_index_dict.items():
                normalized[(NODE_TYPE, rel_type, NODE_TYPE)] = edge_index_tensor
            return normalized

        except Exception as e:
            logger.error("Error while normalizing edge_index_dict")
            raise CustomException("error found",e)

    def asymmetric_loss(self, y_hat, y_true, alpha=2.0):
        try:
            diff = y_hat - y_true
            under_mask = (diff < 0).float()
            over_mask = (diff >= 0).float()

            under_loss = (diff ** 2) * alpha * under_mask
            over_loss = (diff ** 2) * over_mask

            return (under_loss + over_loss).mean()

        except Exception as e:
            logger.error("Error while computing asymmetric loss")
            raise CustomException("error found",e)
    def build_model(self, static_x_dim):
        try:
            expected_in_channels = static_x_dim + 1

            if IN_CHANNELS != expected_in_channels:
                logger.warning(
                    f"Config in_channels={IN_CHANNELS} does not match static_x_dim + 1 = {expected_in_channels}. "
                    f"Using computed value {expected_in_channels}."
                )

            model = SupplyGraphModel(
                node_type=NODE_TYPE,
                relations=EDGE_RELATIONS,
                in_channels=expected_in_channels,
                hidden_channels=HIDDEN_CHANNELS,
                out_channels=OUT_CHANNELS,
                conv_type=CONV_TYPE,
                aggregation=AGGREGATION,
                layers=LAYERS,
            ).to(self.device)

            return model

        except Exception as e:
            logger.exception("Error while building model")
            raise CustomException("error found",e)

    def train_one_epoch(self, model, optimizer, train_dataset, static_x, edge_index_dict):
        try:
            model.train()
            total_loss = 0.0

            for snapshot in train_dataset:
                temporal_x = torch.as_tensor(snapshot.x, dtype=torch.float32, device=self.device)
                y_true = torch.as_tensor(snapshot.y, dtype=torch.float32, device=self.device)

                x_cat = torch.cat([static_x, temporal_x], dim=-1)
                x_dict = {NODE_TYPE: x_cat}

                y_hat = model(x_dict, edge_index_dict)
                loss = self.asymmetric_loss(y_hat, y_true, alpha=LOSS_ALPHA)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            return total_loss / len(train_dataset.features)

        except Exception as e:
            logger.error("Error during training epoch")
            raise CustomException("error found",e)

    def evaluate(self, model, test_dataset, static_x, edge_index_dict):
        
        try:
            model.eval()

            preds_all = []
            targets_all = []
            loss_list = []

            with torch.no_grad():
                for snapshot in test_dataset:
                    temporal_x = torch.as_tensor(snapshot.x, dtype=torch.float32, device=self.device)
                    y_true = torch.as_tensor(snapshot.y, dtype=torch.float32, device=self.device)

                    x_cat = torch.cat([static_x, temporal_x], dim=-1)
                    x_dict = {NODE_TYPE: x_cat}

                    y_hat = model(x_dict, edge_index_dict)  # [num_nodes, 1]
                    loss = self.asymmetric_loss(y_hat, y_true, alpha=LOSS_ALPHA)

                    # keep full node dimension per time step
                    preds_all.append(y_hat.cpu().numpy())   # each: (num_nodes, 1)
                    targets_all.append(y_true.cpu().numpy())
                    loss_list.append(loss.item())

            # stack along time dimension: (T, num_nodes, 1)
            preds_all_arr = np.stack(preds_all, axis=0)
            targets_all_arr = np.stack(targets_all, axis=0)

            # for metrics, flatten back to 1D
            preds_flat = preds_all_arr.reshape(-1)
            targets_flat = targets_all_arr.reshape(-1)

            mse = mean_squared_error(targets_flat, preds_flat)
            mae = mean_absolute_error(targets_flat, preds_flat)
            rmse = float(np.sqrt(mse))
            r2 = r2_score(targets_flat, preds_flat)

            metrics = {
                "test_asymmetric_loss": float(np.mean(loss_list)),
                "mse": float(mse),
                "mae": float(mae),
                "rmse": rmse,
                "r2": float(r2),
            }

            # return both: 3D arrays for saving + metrics
            return metrics, preds_all_arr, targets_all_arr

        except Exception as e:
            logger.error("Error during evaluation")
            raise CustomException("error found", e)

    def save_outputs(self, model, metrics, preds_all, targets_all):
        try:
            torch.save(model.state_dict(), MODEL_OUTPUT_PATH)

            with open(METRICS_OUTPUT_PATH, "w") as f:
                json.dump(metrics, f, indent=2)

            # preds_all: (T, num_nodes, 1)
            preds_per_product = preds_all[..., 0]    # (T, num_nodes)
            targets_per_product = targets_all[..., 0]

            np.save(PREDICTIONS_OUTPUT_PATH, preds_per_product)
            np.save(TARGETS_OUTPUT_PATH, targets_per_product)

            logger.info(
                f"Saved per-product predictions with shape {preds_per_product.shape}"
            )

        except Exception as e:
            logger.error("Error while saving outputs")
            raise CustomException("error found", e)


    def run(self):
        try:
            with mlflow.start_run():
                logger.info("Starting the model saving pipline")
                logger.info("Starting our MLFLOW experimentation")
                logger.info("Logging the training and testing dataset to MLFLOW")
                hetero_data, X, Y = self.load_artifacts()

                if NODE_TYPE not in hetero_data.node_types:
                    raise ValueError(f"Node type '{NODE_TYPE}' not found in hetero_data")

                static_x = hetero_data[NODE_TYPE].x.to(self.device)
                hetero_data = hetero_data.to(self.device)

                _, train_dataset, test_dataset = self.build_temporal_dataset(hetero_data, X, Y)
                edge_index_dict = self.normalize_edge_index_dict(hetero_data.edge_index_dict)

                model = self.build_model(static_x_dim=static_x.size(1))
                optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

                logger.info("Starting training")
                for epoch in range(EPOCHS):
                    train_loss = self.train_one_epoch(
                        model=model,
                        optimizer=optimizer,
                        train_dataset=train_dataset,
                        static_x=static_x,
                        edge_index_dict=edge_index_dict,
                    )
                    logger.info(f"Epoch {epoch + 1}/{EPOCHS} - train loss: {train_loss:.4f}")

                logger.info("Starting evaluation")
                metrics, preds_all, targets_all = self.evaluate(
                    model=model,
                    test_dataset=test_dataset,
                    static_x=static_x,
                    edge_index_dict=edge_index_dict,
                )
                logger.info("logging metrics and params in mlfow")
                # 3) Save the model itself
                mlflow_pytorch.log_model(model, name="model")

                mlflow.log_metrics(metrics)
                self.save_outputs(model, metrics, preds_all, targets_all)
                logger.info(f"Final metrics: {metrics}")

                return metrics

        except Exception as e:
            logger.error("Error in trainer.run()")
            raise CustomException("error found",e)


if __name__ == "__main__":
    try:
        trainer = ModelTrainer()
        metrics = trainer.run()
        print(metrics)
    except Exception as e:
        logger.error("Fatal error in main execution")
        raise CustomException("error found",e)


import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.logger import get_logger
from src.custom_exception import CustomException
from config.path_config import *
from utils.common_functions import read_yaml

logger = get_logger(__name__)

try:
    logger.info("Importing torch and torch-geometric libraries")
    import torch
    from torch_geometric.data import HeteroData
    from torch_geometric_temporal.signal import (
        StaticGraphTemporalSignal,
        temporal_signal_split,
    )
except Exception as e:
    logger.error(f"Error occurred during torch imports: {e}")
    raise CustomException(f"Error occurred during torch imports: {e}")


class SupplyGraphDataProcessor:
    def __init__(self, config_path=CONFIG_PATH, processed_dir=PROCESSED_DIR):
        try:
            self.config_path = Path(config_path)
            self.config = read_yaml(self.config_path)

            self.processed_dir = Path(processed_dir)
            self.processed_dir.mkdir(parents=True, exist_ok=True)

            self._load_config_values()

            logger.info("SupplyGraphDataProcessor initialized successfully")

        except Exception as e:
            logger.error(f"Error during processor initialization: {e}")
            raise CustomException(f"Error during processor initialization: {e}")

    def _load_config_values(self):
        try:
            self.train_ratio = TRAIN_RATIO
            self.node_type = NODE_TYPE
            self.num_nodes = NUM_NODES
            self.rolling_window = ROLLING_WINDOW
            self.drop_first_n_rows = DROP_FIRST_N_ROWS
            self.target_signal = TARGET_SIGNAL
            self.edge_relations = EDGE_RELATIONS

            self.nodes_index_file = Path(NODES_INDEX_FILE)
            self.nodes_id_column = NODES_ID_COLUMN

            self.edge_src_col = EDGE_SRC_COL
            self.edge_dst_col = EDGE_DST_COL
            self.edge_undirected = EDGE_UNDIRECTED
            self.temporal_date_column = TEMPORAL_DATE_COLUMN

            self.edge_files = {
                "same_plant": Path(EDGES_PLANT_FILE),
                "same_storage": Path(EDGES_STORAGE_FILE),
                "same_product_group": Path(EDGES_GROUP_FILE),
                "same_product_subgroup": Path(EDGES_SUBGROUP_FILE),
            }

            self.unit_signal_files = {
                "production_unit": Path(PRODUCTION_UNIT_FILE),
                "delivery_unit": Path(DELIVERY_UNIT_FILE),
                "factory_issue_unit": Path(FACTORY_ISSUE_UNIT_FILE),
                "sales_order_unit": Path(SALES_ORDER_UNIT_FILE),
            }

            self.weight_signal_files = {
                "production_weight": Path(PRODUCTION_WEIGHT_FILE),
                "delivery_weight": Path(DELIVERY_WEIGHT_FILE),
                "factory_issue_weight": Path(FACTORY_ISSUE_WEIGHT_FILE),
                "sales_order_weight": Path(SALES_ORDER_WEIGHT_FILE),
            }

        except Exception as e:
            logger.error(f"Error while loading config values: {e}")
            raise CustomException(f"Error while loading config values: {e}")

    def load_csv(self, path) -> pd.DataFrame:
        try:
            path = Path(path)
            logger.info(f"Loading CSV: {path}")

            if not path.exists():
                raise CustomException(f"CSV file not found: {path.resolve()}")

            return pd.read_csv(path)

        except Exception as e:
            logger.error(f"Error loading CSV {path}: {e}")
            raise CustomException(f"Error loading CSV {path}: {e}")

    def get_target_file(self) -> Path:
        try:
            if self.target_signal in self.weight_signal_files:
                return self.weight_signal_files[self.target_signal]

            if self.target_signal in self.unit_signal_files:
                return self.unit_signal_files[self.target_signal]

            raise CustomException(f"Unknown target_signal: {self.target_signal}")

        except Exception as e:
            logger.error(f"Error while getting target file: {e}")
            raise CustomException(f"Error while getting target file: {e}")

    def preprocess_temporal_signal(self, df: pd.DataFrame):
        try:
            logger.info("Starting temporal preprocessing")

            if self.temporal_date_column in df.columns:
                dates = df[self.temporal_date_column].copy()
                df = df.drop(columns=[self.temporal_date_column])
            else:
                dates = None

            logger.info("Creating product-to-index mapping")
            column_mapping = {col: idx for idx, col in enumerate(df.columns)}
            reverse_mapping = {idx: col for col, idx in column_mapping.items()}

            logger.info("Renaming product columns to integer node ids")
            df = df.rename(columns=column_mapping)

            logger.info(f"Applying rolling mean with window={self.rolling_window}")
            rolled_df = df.rolling(window=self.rolling_window).mean()

            logger.info(f"Dropping first {self.drop_first_n_rows} rows")
            rolled_df = rolled_df.drop(index=list(range(self.drop_first_n_rows)))
            rolled_df = rolled_df.reset_index(drop=True)

            values = rolled_df.values.astype(np.float32)

            logger.info("Creating one-step-ahead forecasting pairs")
            X = values[:-1]
            Y = values[1:]

            X = X.reshape((X.shape[0], X.shape[1], 1))
            Y = Y.reshape((Y.shape[0], Y.shape[1], 1))

            return {
                "dates": dates,
                "rolled_df": rolled_df,
                "X": X,
                "Y": Y,
                "column_mapping": column_mapping,
                "reverse_mapping": reverse_mapping,
            }

        except Exception as e:
            logger.error(f"Error during temporal preprocessing: {e}")
            raise CustomException(f"Error during temporal preprocessing: {e}")

    def load_edge_index(self, path: Path) -> torch.Tensor:
        try:
            df = self.load_csv(path)

            src = torch.tensor(df[self.edge_src_col].values, dtype=torch.long)
            dst = torch.tensor(df[self.edge_dst_col].values, dtype=torch.long)

            if self.edge_undirected:
                edge_index = torch.stack(
                    [torch.cat([src, dst]), torch.cat([dst, src])],
                    dim=0
                )
            else:
                edge_index = torch.stack([src, dst], dim=0)

            return edge_index

        except Exception as e:
            logger.error(f"Error while loading edge index from {path}: {e}")
            raise CustomException(f"Error while loading edge index from {path}: {e}")

    def build_hetero_graph(self):
        try:
            logger.info("Building HeteroData graph")

            nodes_idx_df = self.load_csv(self.nodes_index_file)
            num_prod = len(nodes_idx_df)

            data = HeteroData()
            data[self.node_type].x = torch.zeros((num_prod, IN_CHANNELS), dtype=torch.float32)

            edge_index_dict = {}
            for relation in self.edge_relations:
                if relation not in self.edge_files:
                    raise CustomException(f"Relation {relation} not found in edge_files")

                path = self.edge_files[relation]
                edge_index = self.load_edge_index(path)

                data[(self.node_type, relation, self.node_type)].edge_index = edge_index
                edge_index_dict[relation] = edge_index

            return data, edge_index_dict

        except Exception as e:
            logger.error(f"Error while building hetero graph: {e}")
            raise CustomException(f"Error while building hetero graph: {e}")

    def build_temporal_dataset(self, X: np.ndarray, Y: np.ndarray, edge_index_dict: dict):
        try:
            logger.info("Merging relation edge indices into one temporal graph")

            edge_indices = [edge_index for edge_index in edge_index_dict.values()]
            edge_index_homo = torch.cat(edge_indices, dim=1)

            edge_index_np = edge_index_homo.cpu().numpy()
            edge_weight = np.ones(edge_index_np.shape[1], dtype=np.float32)

            features = [x for x in X]
            targets = [y for y in Y]

            dataset = StaticGraphTemporalSignal(
                edge_index=edge_index_np,
                edge_weight=edge_weight,
                features=features,
                targets=targets
            )

            train_dataset, test_dataset = temporal_signal_split(
                dataset,
                train_ratio=self.train_ratio
            )

            return dataset, train_dataset, test_dataset, edge_index_homo, edge_weight

        except Exception as e:
            logger.error(f"Error while building temporal dataset: {e}")
            raise CustomException(f"Error while building temporal dataset: {e}")

    def save_artifacts(
        self,
        rolled_df: pd.DataFrame,
        X: np.ndarray,
        Y: np.ndarray,
        hetero_data: HeteroData,
        edge_index_homo: torch.Tensor,
        edge_weight: np.ndarray,
        column_mapping: dict,
        reverse_mapping: dict,
    ):
        try:
            logger.info("Saving processed artifacts")

            self.processed_dir.mkdir(parents=True, exist_ok=True)

            rolled_df.to_csv(self.processed_dir / "rolled_signal.csv", index=False)
            np.save(self.processed_dir / "X.npy", X)
            np.save(self.processed_dir / "Y.npy", Y)
            np.save(self.processed_dir / "edge_weight.npy", edge_weight)
            torch.save(edge_index_homo, self.processed_dir / "edge_index_homo.pt")
            torch.save(hetero_data, self.processed_dir / "hetero_data.pt")

            split_idx = int(len(rolled_df) * self.train_ratio)
            train_df = rolled_df.iloc[:split_idx].reset_index(drop=True)
            test_df = rolled_df.iloc[split_idx:].reset_index(drop=True)

            train_df.to_csv(PROCESSED_TRAIN_DATA_PATH, index=False)
            test_df.to_csv(PROCESSED_TEST_DATA_PATH, index=False)

            with open(self.processed_dir / "product_to_idx.json", "w") as f:
                json.dump(column_mapping, f, indent=2)

            with open(self.processed_dir / "idx_to_product.json", "w") as f:
                json.dump(reverse_mapping, f, indent=2)

            metadata = {
                "bucket_name": BUCKET_NAME,
                "bucket_folder": BUCKET_FOLDER_NAME,
                "node_type": self.node_type,
                "num_nodes": self.num_nodes,
                "rolling_window": self.rolling_window,
                "drop_first_n_rows": self.drop_first_n_rows,
                "target_signal": self.target_signal,
                "train_ratio": self.train_ratio,
                "num_snapshots": int(X.shape[0]),
                "feature_dim": int(X.shape[2]),
                "edge_relations": self.edge_relations,
                "nodes_index_file": str(self.nodes_index_file),
            }

            with open(self.processed_dir / "metadata.json", "w") as f:
                json.dump(metadata, f, indent=2)

        except Exception as e:
            logger.error(f"Error while saving artifacts: {e}")
            raise CustomException(f"Error while saving artifacts: {e}")
    def build_product_trend_file(self, rolled_df: pd.DataFrame, column_mapping: dict):
        """
        Create a long-format time series table:
        time_step, product_id, value
        One row per product per time step.
        """
        try:
            logger.info("Building product trend file for dashboard")

            # invert mapping: idx -> product_id (original column name)
            idx_to_product = {idx: col for col, idx in column_mapping.items()}

            # create a time_step axis (0..T-1)
            rolled_df = rolled_df.reset_index().rename(columns={"index": "time_step"})

            records = []
            value_cols = [c for c in rolled_df.columns if isinstance(c, int)]

            for _, row in rolled_df.iterrows():
                t = int(row["time_step"])
                for idx in value_cols:
                    product_id = idx_to_product[idx]
                    value = float(row[idx])
                    records.append(
                        {
                            "time_step": t,
                            "product_id": product_id,
                            "value": value,
                        }
                    )

            trend_df = pd.DataFrame(records)

            out_path = self.processed_dir / "product_trend_long.csv"
            trend_df.to_csv(out_path, index=False)
            logger.info(f"Saved product trend file to {out_path}")

        except Exception as e:
            logger.error(f"Error while building product trend file: {e}")
            raise CustomException(f"Error while building product trend file: {e}")


    def process(self):
        try:
            logger.info("Starting full preprocessing pipeline")

            target_file = self.get_target_file()
            temporal_df = self.load_csv(target_file)

            temporal_out = self.preprocess_temporal_signal(temporal_df)
            rolled_df = temporal_out["rolled_df"]

            X = temporal_out["X"]
            Y = temporal_out["Y"]
            column_mapping = temporal_out["column_mapping"]
            reverse_mapping = temporal_out["reverse_mapping"]

            hetero_data, edge_index_dict = self.build_hetero_graph()
            

            assert X.shape[1] == hetero_data[self.node_type].x.size(0), \
                "Mismatch between temporal nodes and graph nodes"

            dataset, train_dataset, test_dataset, edge_index_homo, edge_weight = self.build_temporal_dataset(
                X, Y, edge_index_dict
            )
            PRED_PATH = Path("artifacts") /"models"/ "predictions.npy"

            preds = np.load(PRED_PATH)  # shape (T, num_products)

            # align length with rolled_df
            T = min(len(rolled_df), preds.shape[0])
            rolled_df = rolled_df.iloc[:T].copy()
            preds = preds[:T, :]

            # column_mapping: original product_id -> integer index
            # rolled_df columns are already renamed to 0..N-1
            for product_id, idx in column_mapping.items():
                idx_int = int(idx)
                col_name = f"{product_id}_pred"   # use readable product id
                rolled_df[col_name] = preds[:, idx_int]


            # save extended CSV
            rolled_df.to_csv(self.processed_dir / "rolled_signal_with_preds.csv", index=False)
            self.build_product_trend_file(rolled_df, column_mapping)

            self.save_artifacts(
                rolled_df=rolled_df,
                X=X,
                Y=Y,
                hetero_data=hetero_data,
                edge_index_homo=edge_index_homo,
                edge_weight=edge_weight,
                column_mapping=column_mapping,
                reverse_mapping=reverse_mapping,
            )

            logger.info("Pipeline completed successfully")

            return {
                "rolled_df": rolled_df,
                "X_shape": X.shape,
                "Y_shape": Y.shape,
                "hetero_data": hetero_data,
                "edge_index_dict": edge_index_dict,
                "dataset": dataset,
                "train_dataset": train_dataset,
                "test_dataset": test_dataset,
                "target_file": str(target_file),
            }

        except Exception as e:
            logger.error(f"Error during preprocessing pipeline: {e}")
            raise CustomException("Error during preprocessing pipeline:",e)


if __name__ == "__main__":
    try:
        processor = SupplyGraphDataProcessor()
        artifacts = processor.process()
        logger.info(f"Processing completed. Target file used: {artifacts['target_file']}")
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise CustomException(f"Pipeline failed: {e}")


import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import HeteroData

from src.logger import get_logger
from src.custom_exception import CustomException
from config.path_config import *

logger = get_logger(__name__)


class SupplyGraphDataProcessor:
    def __init__(self, processed_dir=PROCESSED_DIR):
        try:
            self.processed_dir = Path(processed_dir)
            self.processed_dir.mkdir(parents=True, exist_ok=True)

            self.train_ratio = TRAIN_RATIO
            self.node_type = NODE_TYPE
            self.num_nodes = NUM_NODES
            self.rolling_window = ROLLING_WINDOW
            self.drop_first_n_rows = DROP_FIRST_N_ROWS
            self.target_signals = TARGET_SIGNALS
            self.prediction_horizon = PREDICTION_HORIZON
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

            logger.info("SupplyGraphDataProcessor initialized successfully")

        except Exception as e:
            logger.error(f"Error during processor initialization: {e}")
            raise CustomException(f"Error during processor initialization: {e}")

    def make_unique_labels(self, labels: list[str]) -> list[str]:
        try:
            counts = {}
            unique_labels = []
            for label in labels:
                if label not in counts:
                    counts[label] = 0
                    unique_labels.append(label)
                else:
                    counts[label] += 1
                    unique_labels.append(f"{label}.{counts[label]}")
            return unique_labels
        except Exception as e:
            logger.error(f"Error while uniquifying labels: {e}")
            raise CustomException(f"Error while uniquifying labels: {e}")


    def load_csv(self, path) -> pd.DataFrame:
        try:
            path = Path(path)
            if not path.exists():
                raise CustomException(f"CSV file not found: {path.resolve()}")
            return pd.read_csv(path)
        except Exception as e:
            logger.error(f"Error loading CSV {path}: {e}")
            raise CustomException(f"Error loading CSV {path}: {e}")


    def get_target_files(self) -> dict:
        try:
            target_files = {}
            for signal in self.target_signals:
                if signal in self.unit_signal_files:
                    target_files[signal] = self.unit_signal_files[signal]
                elif signal in self.weight_signal_files:
                    target_files[signal] = self.weight_signal_files[signal]
                else:
                    raise CustomException(f"Unknown target_signal: {signal}")
            return target_files
        except Exception as e:
            logger.error(f"Error while getting target files: {e}")
            raise CustomException(f"Error while getting target files: {e}")


    def preprocess_temporal_signals(self, signal_files: dict):
        try:
            logger.info("Starting temporal preprocessing")

            processed_signals = {}
            combined_df = None
            dates = None

            signal_products = {}
            for signal_name, file_path in signal_files.items():
                df = self.load_csv(file_path)
                if self.temporal_date_column in df.columns:
                    df = df.drop(columns=[self.temporal_date_column])
                signal_products[signal_name] = set(df.columns)

            common_products = set.intersection(*signal_products.values()) if signal_products else set()
            common_products = sorted(common_products)
            logger.info(f"Found {len(common_products)} common products")

            for signal_name, file_path in signal_files.items():
                df = self.load_csv(file_path)

                valid_cols = [c for c in df.columns if c in common_products or c == self.temporal_date_column]
                df = df[valid_cols]

                if self.temporal_date_column in df.columns:
                    if dates is None:
                        dates = df[self.temporal_date_column].copy()
                    df = df.drop(columns=[self.temporal_date_column])

                rolled_df = df.rolling(window=self.rolling_window).mean()
                rolled_df = rolled_df.drop(index=list(range(self.drop_first_n_rows)))
                rolled_df = rolled_df.reset_index(drop=True)
                rolled_df = rolled_df.rename(columns=lambda x: f"{signal_name}_{x}")

                if combined_df is None:
                    combined_df = rolled_df
                else:
                    combined_df = pd.concat([combined_df, rolled_df], axis=1)

                processed_signals[signal_name] = {
                    "original_df": df,
                    "rolled_df": rolled_df,
                }

            column_mapping = {}
            reverse_mapping = {}
            idx = 0

            for signal_name in self.target_signals:
                original_cols = sorted(processed_signals[signal_name]["original_df"].columns)
                for col in original_cols:
                    column_mapping[f"{signal_name}_{col}"] = idx
                    reverse_mapping[idx] = f"{signal_name}_{col}"
                    idx += 1

            combined_df = combined_df.rename(columns=column_mapping)

            # ── CHANGE 1: save raw [T, N, C] tensor; sequence windows built in trainer ──
            raw = combined_df.values.astype(np.float32)
            values = raw.reshape(
                combined_df.shape[0],
                len(common_products),
                len(self.target_signals),
            )
            logger.info(f"values tensor shape: {values.shape}")  # [T, N, C]

            return {
                "dates": dates,
                "combined_rolled_df": combined_df,
                "values": values,
                "column_mapping": column_mapping,
                "reverse_mapping": reverse_mapping,
                "processed_signals": processed_signals,
                "common_products": common_products,
            }

        except Exception as e:
            logger.error(f"Error during temporal preprocessing: {e}")
            raise CustomException(f"Error during temporal preprocessing: {e}")


    def build_named_artifacts(self, processed_signals: dict, product_order: list[str]):
        try:
            values = np.stack(
                [
                    processed_signals[signal_name]["rolled_df"][
                        [f"{signal_name}_{product_name}" for product_name in product_order]
                    ].values.astype(np.float32)
                    for signal_name in self.target_signals
                ],
                axis=-1,
            )

            flattened_columns = {}
            reverse_mapping = {}
            idx = 0
            for product_name in product_order:
                for signal_name in self.target_signals:
                    key = f"{signal_name}_{product_name}"
                    flattened_columns[key] = idx
                    reverse_mapping[idx] = key
                    idx += 1

            combined_df = pd.DataFrame(
                values.reshape(values.shape[0], -1),
                columns=list(range(values.shape[1] * values.shape[2])),
            )

            product_name_to_idx = {product_name: idx for idx, product_name in enumerate(product_order)}
            product_idx_to_name = {
                idx: product_name for product_name, idx in product_name_to_idx.items()
            }

            return {
                "values": values,
                "combined_df": combined_df,
                "column_mapping": flattened_columns,
                "reverse_mapping": reverse_mapping,
                "product_name_to_idx": product_name_to_idx,
                "product_idx_to_name": product_idx_to_name,
            }

        except Exception as e:
            logger.error(f"Error while building named artifacts: {e}")
            raise CustomException(f"Error while building named artifacts: {e}")


    def load_edge_index(self, path: Path) -> torch.Tensor:
        try:
            df = self.load_csv(path)

            if hasattr(self, "old_to_new"):
                df = df.copy()
                df[self.edge_src_col] = df[self.edge_src_col].map(self.old_to_new)
                df[self.edge_dst_col] = df[self.edge_dst_col].map(self.old_to_new)
                df = df.dropna(subset=[self.edge_src_col, self.edge_dst_col])

            src = torch.tensor(df[self.edge_src_col].values.astype(int), dtype=torch.long)
            dst = torch.tensor(df[self.edge_dst_col].values.astype(int), dtype=torch.long)

            if self.edge_undirected:
                edge_index = torch.stack(
                    [torch.cat([src, dst]), torch.cat([dst, src])],
                    dim=0,
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

            data = HeteroData()
            data[self.node_type].x = torch.zeros((self.num_nodes, STATIC_FEATURE_DIM), dtype=torch.float32)

            edge_index_dict = {}
            for relation in self.edge_relations:
                if relation not in self.edge_files:
                    raise CustomException(f"Relation {relation} not found in edge_files")

                edge_index = self.load_edge_index(self.edge_files[relation])
                data[(self.node_type, relation, self.node_type)].edge_index = edge_index
                edge_index_dict[relation] = edge_index

            return data, edge_index_dict

        except Exception as e:
            logger.error(f"Error while building hetero graph: {e}")
            raise CustomException(f"Error while building hetero graph: {e}")


    def build_edge_index_homo(self, edge_index_dict: dict):
        # ── helper used by trainer; also called here when saving ──
        try:
            edge_indices = list(edge_index_dict.values())
            edge_index_homo = torch.cat(edge_indices, dim=1)
            edge_weight = np.ones(edge_index_homo.shape[1], dtype=np.float32)
            return edge_index_homo, edge_weight
        except Exception as e:
            logger.error(f"Error while building homo edge index: {e}")
            raise CustomException(f"Error while building homo edge index: {e}")


    def build_product_trend_file(self, rolled_df: pd.DataFrame, column_mapping: dict):
        try:
            idx_to_product_signal = {idx: col for col, idx in column_mapping.items()}
            rolled_df = rolled_df.reset_index().rename(columns={"index": "time_step"})

            records = []
            value_cols = [c for c in rolled_df.columns if isinstance(c, int)]

            for _, row in rolled_df.iterrows():
                t = int(row["time_step"])
                for idx in value_cols:
                    product_signal = idx_to_product_signal[idx]

                    # ── CHANGE 2: prefix-match against target_signals list ──
                    matched_signal = next(
                        (sig for sig in self.target_signals if product_signal.startswith(f"{sig}_")),
                        None,
                    )
                    if matched_signal:
                        signal_type = matched_signal
                        product_id = product_signal[len(matched_signal) + 1:]
                    else:
                        signal_type = "unknown"
                        product_id = product_signal

                    records.append(
                        {
                            "time_step": t,
                            "product_id": product_id,
                            "signal_type": signal_type,
                            "value": float(row[idx]),
                        }
                    )

            trend_df = pd.DataFrame(records)
            trend_df.to_csv(PRODUCT_TREND_LONG_PATH, index=False)

        except Exception as e:
            logger.error(f"Error while building product trend file: {e}")
            raise CustomException(f"Error while building product trend file: {e}")


    def save_artifacts(
        self,
        rolled_df,
        values,              # ── CHANGE 3: values replaces X / Y ──
        hetero_data,
        edge_index_homo,
        edge_weight,
        column_mapping,
        reverse_mapping,
        product_order,
        product_name_to_idx,
        product_idx_to_name,
    ):
        try:
            rolled_df.to_csv(ROLLED_SIGNAL_PATH, index=False)
            np.save(VALUES_NUMPY_PATH, values)          # [T, N, C]
            np.save(EDGE_WEIGHT_PATH, edge_weight)
            torch.save(edge_index_homo, EDGE_INDEX_HOMO_PATH)
            torch.save(hetero_data, HETERO_DATA_PATH)

            split_idx = int(len(rolled_df) * self.train_ratio)
            rolled_df.iloc[:split_idx].reset_index(drop=True).to_csv(PROCESSED_TRAIN_DATA_PATH, index=False)
            rolled_df.iloc[split_idx:].reset_index(drop=True).to_csv(PROCESSED_TEST_DATA_PATH, index=False)

            with open(COLUMN_MAPPING_PATH, "w") as f:
                json.dump(column_mapping, f, indent=2)

            with open(REVERSE_MAPPING_PATH, "w") as f:
                json.dump(reverse_mapping, f, indent=2)

            with open(PRODUCT_ORDER_PATH, "w") as f:
                json.dump(product_order, f, indent=2)

            with open(PRODUCT_NAME_TO_IDX_PATH, "w") as f:
                json.dump(product_name_to_idx, f, indent=2)

            with open(PRODUCT_IDX_TO_NAME_PATH, "w") as f:
                json.dump(product_idx_to_name, f, indent=2)

            with open(PRODUCT_TO_IDX_PATH, "w") as f:
                json.dump(column_mapping, f, indent=2)

            with open(IDX_TO_PRODUCT_PATH, "w") as f:
                json.dump(reverse_mapping, f, indent=2)

            metadata = {
                "node_type": self.node_type,
                "num_nodes": self.num_nodes,
                "rolling_window": self.rolling_window,
                "drop_first_n_rows": self.drop_first_n_rows,
                "target_signals": self.target_signals,
                "train_ratio": self.train_ratio,
                "num_timesteps": int(values.shape[0]),   # T
                "num_nodes": int(values.shape[1]),        # N
                "feature_dim": int(values.shape[2]),      # C
                "edge_relations": self.edge_relations,
                "nodes_index_file": str(self.nodes_index_file),
                "product_order": product_order,
                "bucket_name": BUCKET_NAME,
                "bucket_folder": BUCKET_FOLDER_NAME,
            }

            with open(METADATA_PATH, "w") as f:
                json.dump(metadata, f, indent=2)

        except Exception as e:
            logger.error(f"Error while saving artifacts: {e}")
            raise CustomException(f"Error while saving artifacts: {e}")


    def process(self):
        try:
            logger.info("Starting preprocessing pipeline")

            target_files = self.get_target_files()
            temporal_out = self.preprocess_temporal_signals(target_files)

            rolled_df = temporal_out["combined_rolled_df"]
            values = temporal_out["values"]              # ── CHANGE 4: values, not X/Y ──
            column_mapping = temporal_out["column_mapping"]
            reverse_mapping = temporal_out["reverse_mapping"]
            common_products = temporal_out["common_products"]
            processed_signals = temporal_out["processed_signals"]

            nodes_idx_df = self.load_csv(self.nodes_index_file)
            nodes_idx_df["original_index"] = nodes_idx_df.index
            nodes_idx_df["_product_key"] = self.make_unique_labels(
                nodes_idx_df[self.nodes_id_column].astype(str).tolist()
            )

            filtered_nodes = (
                nodes_idx_df[nodes_idx_df["_product_key"].isin(common_products)]
                .sort_values("_product_key")
                .reset_index(drop=True)
            )

            product_order = filtered_nodes["_product_key"].tolist()
            self.old_to_new = {row["original_index"]: idx for idx, row in filtered_nodes.iterrows()}
            self.num_nodes = len(filtered_nodes)

            hetero_data, edge_index_dict = self.build_hetero_graph()

            named_artifacts = self.build_named_artifacts(
                processed_signals=processed_signals,
                product_order=product_order,
            )
            rolled_df = named_artifacts["combined_df"]
            values = named_artifacts["values"]
            column_mapping = named_artifacts["column_mapping"]
            reverse_mapping = named_artifacts["reverse_mapping"]
            product_name_to_idx = named_artifacts["product_name_to_idx"]
            product_idx_to_name = named_artifacts["product_idx_to_name"]

            # ── CHANGE 5: assert uses values.shape[1] ──
            expected_nodes = hetero_data[self.node_type].x.size(0)
            assert values.shape[1] == expected_nodes, \
                f"Mismatch between temporal nodes ({values.shape[1]}) and graph nodes ({expected_nodes})"

            edge_index_homo, edge_weight = self.build_edge_index_homo(edge_index_dict)

            if PREDICTIONS_OUTPUT_PATH.exists():
                preds = np.load(PREDICTIONS_OUTPUT_PATH)
                T = min(len(rolled_df), preds.shape[0])
                rolled_df = rolled_df.iloc[:T].copy()
                preds = preds[:T, :]

                for idx in range(preds.shape[1]):
                    if idx in reverse_mapping:
                        rolled_df[f"{reverse_mapping[idx]}_pred"] = preds[:, idx]

                rolled_df.to_csv(ROLLED_SIGNAL_WITH_PREDS_PATH, index=False)

            self.build_product_trend_file(rolled_df, column_mapping)

            self.save_artifacts(
                rolled_df=rolled_df,
                values=values,
                hetero_data=hetero_data,
                edge_index_homo=edge_index_homo,
                edge_weight=edge_weight,
                column_mapping=column_mapping,
                reverse_mapping=reverse_mapping,
                product_order=product_order,
                product_name_to_idx=product_name_to_idx,
                product_idx_to_name=product_idx_to_name,
            )

            logger.info("Pipeline completed successfully")

            return {
                "rolled_df": rolled_df,
                "values_shape": values.shape,
                "hetero_data": hetero_data,
                "edge_index_dict": edge_index_dict,
                "target_file": str(target_files),
            }

        except Exception as e:
            logger.error(f"Error during preprocessing pipeline: {e}")
            raise CustomException(f"Error during preprocessing pipeline: {e}")


if __name__ == "__main__":
    processor = SupplyGraphDataProcessor()
    artifacts = processor.process()
    logger.info(f"Processing completed. Target file used: {artifacts['target_file']}")

import os
import pandas as pd
import re
from google.cloud import storage
from src.logger import get_logger
from src.custom_exception import CustomException
from config.path_config import *
from utils.common_functions import read_yaml

logger = get_logger(__name__)

class DataIngestion:
    def __init__(self, config):
        try:
            self.config = config["data_ingestion"]

            self.bucket_name = self.config["bucket_name"]
            self.folder_name = self.config["bucket_folder_name"]
            self.train_ratio = self.config["train_ratio"]

            self.nodes_cfg = self.config.get("nodes", [])
            self.edges_cfg = self.config.get("edges", [])
            self.temporal_cfg = self.config.get("temporal_data", {})
            self.unit_cfg = self.temporal_cfg.get("unit", [])
            self.weight_cfg = self.temporal_cfg.get("weight", [])

            self.artifacts_dir = Path("artifacts") / "data_ingestion"
            self.nodes_dir = self.artifacts_dir / "nodes"
            self.edges_dir = self.artifacts_dir / "edges"
            self.temporal_unit_dir = self.artifacts_dir / "temporal" / "unit"
            self.temporal_weight_dir = self.artifacts_dir / "temporal" / "weight"

            for directory in [
                self.nodes_dir,
                self.edges_dir,
                self.temporal_unit_dir,
                self.temporal_weight_dir,
            ]:
                directory.mkdir(parents=True, exist_ok=True)

            self.node_files = self._prepare_node_file_map()
            self.edge_files = self._prepare_edge_file_map()
            self.unit_files = self._prepare_temporal_file_map(
                temporal_cfg=self.unit_cfg,
                target_dir=self.temporal_unit_dir
            )
            self.weight_files = self._prepare_temporal_file_map(
                temporal_cfg=self.weight_cfg,
                target_dir=self.temporal_weight_dir
            )

            logger.info(
                f"Data ingestion initialized | "
                f"bucket: {self.bucket_name} | "
                f"folder: {self.folder_name} | "
                f"train_ratio: {self.train_ratio}"
            )

        except Exception as e:
            logger.error(f"Error during DataIngestion initialization: {e}")
            raise CustomException(f"Error during DataIngestion initialization: {e}")

    def _sanitize_name(self, file_name: str) -> str:
        try:
            file_stem = Path(file_name).stem.strip()
            file_stem = re.sub(r"[^\w]+", "_", file_stem)
            return file_stem.strip("_").lower()
        except Exception as e:
            logger.error(f"Error while sanitizing file name {file_name}: {e}")
            raise CustomException(f"Error while sanitizing file name {file_name}: {e}")

    def _download_file(self, bucket_file_name: str, local_path: Path):
        try:
            client = storage.Client()
            bucket = client.bucket(self.bucket_name)

            blob_path = f"{self.folder_name}/{bucket_file_name}"
            blob = bucket.blob(blob_path)

            blob.download_to_filename(str(local_path))

            logger.info(f"Downloaded: gs://{self.bucket_name}/{blob_path} -> {local_path}")

        except Exception as e:
            logger.error(f"Failed to download {bucket_file_name}: {e}")
            raise CustomException(f"Failed to download {bucket_file_name}: {e}")

    def _prepare_node_file_map(self):
        try:
            node_file_map = {}

            for node_cfg in self.nodes_cfg:
                bucket_file_name = node_cfg["bucket_file_name"]

                if "id_column" in node_cfg:
                    local_file_name = "nodes_index.csv"
                    key = "nodes_index"
                else:
                    sanitized = self._sanitize_name(bucket_file_name)

                    if sanitized == "nodes":
                        local_file_name = "nodes.csv"
                        key = "nodes"
                    else:
                        local_file_name = f"{sanitized}.csv"
                        key = sanitized

                node_file_map[key] = {
                    "bucket_file_name": bucket_file_name,
                    "local_path": self.nodes_dir / local_file_name,
                    "meta": node_cfg,
                }

            return node_file_map

        except Exception as e:
            logger.error(f"Error while preparing node file map: {e}")
            raise CustomException(f"Error while preparing node file map: {e}")

    def _prepare_edge_file_map(self):
        try:
            edge_file_map = {}

            for edge_cfg in self.edges_cfg:
                relation = edge_cfg["relation"]

                edge_file_map[relation] = {
                    "bucket_file_name": edge_cfg["bucket_file_name"],
                    "local_path": self.edges_dir / f"{relation}.csv",
                    "meta": edge_cfg,
                }

            return edge_file_map

        except Exception as e:
            logger.error(f"Error while preparing edge file map: {e}")
            raise CustomException(f"Error while preparing edge file map: {e}")

    def _prepare_temporal_file_map(self, temporal_cfg, target_dir: Path):
        try:
            temporal_file_map = {}

            for file_cfg in temporal_cfg:
                signal_name = file_cfg["signal_name"]

                temporal_file_map[signal_name] = {
                    "bucket_file_name": file_cfg["bucket_file_name"],
                    "local_path": target_dir / f"{signal_name}.csv",
                    "meta": file_cfg,
                }

            return temporal_file_map

        except Exception as e:
            logger.error(f"Error while preparing temporal file map: {e}")
            raise CustomException(f"Error while preparing temporal file map: {e}")

    def download_nodes(self):
        try:
            logger.info("Downloading node files...")

            for node_name, node_info in self.node_files.items():
                self._download_file(
                    bucket_file_name=node_info["bucket_file_name"],
                    local_path=node_info["local_path"]
                )
                logger.info(f"Node file saved | {node_name}: {node_info['local_path']}")

            logger.info("All node files downloaded successfully.")

        except Exception as e:
            logger.error(f"Error while downloading node files: {e}")
            raise CustomException(f"Error while downloading node files: {e}")

    def download_edges(self):
        try:
            logger.info("Downloading edge files...")

            for relation, edge_info in self.edge_files.items():
                self._download_file(
                    bucket_file_name=edge_info["bucket_file_name"],
                    local_path=edge_info["local_path"]
                )
                logger.info(f"Edge file saved | {relation}: {edge_info['local_path']}")

            logger.info("All edge files downloaded successfully.")

        except Exception as e:
            logger.error(f"Error while downloading edge files: {e}")
            raise CustomException(f"Error while downloading edge files: {e}")

    def download_temporal(self):
        try:
            logger.info("Downloading temporal unit files...")

            for signal_name, file_info in self.unit_files.items():
                self._download_file(
                    bucket_file_name=file_info["bucket_file_name"],
                    local_path=file_info["local_path"]
                )
                logger.info(f"Temporal unit file saved | {signal_name}: {file_info['local_path']}")

            logger.info("Downloading temporal weight files...")

            for signal_name, file_info in self.weight_files.items():
                self._download_file(
                    bucket_file_name=file_info["bucket_file_name"],
                    local_path=file_info["local_path"]
                )
                logger.info(f"Temporal weight file saved | {signal_name}: {file_info['local_path']}")

            logger.info("All temporal files downloaded successfully.")

        except Exception as e:
            logger.error(f"Error while downloading temporal files: {e}")
            raise CustomException(f"Error while downloading temporal files: {e}")

    def get_artifact_paths(self):
        try:
            return {
                "nodes": {k: str(v["local_path"]) for k, v in self.node_files.items()},
                "edges": {k: str(v["local_path"]) for k, v in self.edge_files.items()},
                "temporal_unit": {k: str(v["local_path"]) for k, v in self.unit_files.items()},
                "temporal_weight": {k: str(v["local_path"]) for k, v in self.weight_files.items()},
                "train_ratio": self.train_ratio,
            }
        except Exception as e:
            logger.error(f"Error while collecting artifact paths: {e}")
            raise CustomException(f"Error while collecting artifact paths: {e}")

    def run(self):
        try:
            logger.info("Starting data ingestion pipeline...")

            self.download_nodes()
            self.download_edges()
            self.download_temporal()

            artifact_paths = self.get_artifact_paths()

            logger.info("Data ingestion process completed successfully.")
            return artifact_paths

        except CustomException as ce:
            logger.error(f"CustomException in run(): {ce}")
            raise ce

        except Exception as e:
            logger.error(f"Unexpected error in run(): {e}")
            raise CustomException(f"Unexpected error in run(): {e}")

        finally:
            logger.info("Data ingestion process finished.")


if __name__ == "__main__":
    try:
        config = read_yaml(CONFIG_PATH)
        data_ingestion = DataIngestion(config)
        artifacts = data_ingestion.run()
        logger.info(f"Downloaded artifact summary: {artifacts}")

    except CustomException as ce:
        logger.error(f"Pipeline failed with CustomException: {ce}")
        raise ce

    except Exception as e:
        logger.error(f"Pipeline failed with unexpected error: {e}")
        raise CustomException(f"Pipeline failed with unexpected error: {e}")


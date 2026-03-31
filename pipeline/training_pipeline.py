from src.data_ingestion import DataIngestion
from src.datapreprocessing import SupplyGraphDataProcessor
from src.model_training import ModelTrainer
from utils.common_functions import read_yaml
from config.path_config import *
from src.custom_exception import CustomException
from src.logger import get_logger

if __name__ == "__main__":
    logger = get_logger(__name__)
    ## 1. data ingestion
    data_ingestion = DataIngestion(read_yaml(CONFIG_PATH))
    data_ingestion.run()
    ## 2. Data Processing
    try:
        processor = SupplyGraphDataProcessor()
        artifacts = processor.process()
        logger.info(f"Processing completed. Target file used: {artifacts['target_file']}")
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise CustomException(f"Pipeline failed: {e}")
    ## Model training
    try:
        trainer = ModelTrainer()
        metrics = trainer.run()
        print(metrics)
    except Exception as e:
        logger.error("Fatal error in main execution")
        raise CustomException("error found",e)
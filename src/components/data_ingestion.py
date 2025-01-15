import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd

from sklearn.model_selection import train_test_split

from src.entity.config_entity import DataIngestionConfig
from src.entity.artifact_entity import DataIngestionArtifact

class DataIngestion:
    def __init__(self, data_ingestion_config: DataIngestionConfig):
        self.ingestion_config = data_ingestion_config

    def initiate_data_ingestion(self, source_data_path: str):
        logging.info("Entered the data ingestion method or component")
        try:
            # Step 1 - Check if the source data exists
            if not os.path.exists(source_data_path):
                raise FileNotFoundError(f"Source data file not found at {source_data_path}")

            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok=True)
            
            # Step 2 - Read the original data and save it in the artifacts
            logging.info("Reading the dataset from source")
            df = pd.read_csv(source_data_path)
            logging.info("Dataset successfully read into a DataFrame")
            
            # Step 3 - Rename the target column
            df.rename(columns={'Crop_Yield (kg/ha)':'cy'}, inplace=True)
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

            data_ingestion_artifact = DataIngestionArtifact(self.ingestion_config.raw_data_path)
            return data_ingestion_artifact

        except Exception as e:
            logging.error(f"An error occurred during data ingestion: {e}")
            raise CustomException(e, sys)

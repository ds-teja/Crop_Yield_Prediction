import os
import sys
import numpy as np

"""
Defining common constant variables for the training pipeline
"""

# General Constants
SOURCE_FILE_DIR: str = "data"
SOURCE_FILE_NAME: str = "data.csv"
TARGET_COLUMN: str = "cy"
PIPELINE_NAME: str = "src"
ARTIFACT_DIR: str = "artifacts"

RAW_FILE_NAME: str = "data.csv"
TRAIN_FILE_NAME: str = "train.csv"
TEST_FILE_NAME: str = "test.csv"

SAVED_MODEL_DIR: str = os.path.join("saved_models")
MODEL_FILE_NAME: str = "model.pkl"

# 1. Data Ingestion related constants start with DATA_INGESTION
DATA_INGESTION_DIR_NAME: str = "data_ingestion"
DATA_INGESTION_INGESTED_DIR: str = "ingested"
DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO: float = 0.1

# 2. Data Transformation related constants start with DATA_TRANSFORMATION
DATA_TRANSFORMATION_DIR_NAME: str = "data_transformation"
DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR: str = "transformed_data"
DATA_TRANSFORMATION_TRANSFORMED_FILE_NAME: str = "transformed_data.csv"
DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR: str = "transformed_object"
ONE_HOT_ENCODER_PATH: str = "one_hot_encoder.pkl"
YEAR_ENCODER_PATH: str = "year_encoder.pkl"

# 3. Model Trainer related constants start with MODEL_TRAINER
MODEL_TRAINER_DIR_NAME: str = "model_trainer"
MODEL_TRAINER_TRAINED_MODEL_DIR: str = "trained_model"
MODEL_TRAINER_TRAINED_MODEL_NAME: str = "model.pkl"
MODEL_TRAINER_EXPECTED_SCORE: float = 0.6
MODEL_TRAINER_OVER_FITTING_UNDER_FITTING_THRESHOLD: float = 0.05

# Cloud Storage Constants
TRAINING_BUCKET_NAME: str = "networksecurity"

# Preprocessor related constants
PREPROCESSOR_DIR: str = "preprocessor"
PREPROCESSOR_FILE_NAME: str = "preprocessor.pkl"

# Logging constants
LOG_FILE_NAME: str = "training_pipeline.log"
LOGGING_LEVEL: str = "INFO"

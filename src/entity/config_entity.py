from datetime import datetime
import os
import src.constant as constants

class TrainingPipelineConfig:
    def __init__(self):
        timestamp = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")
        self.source_data_path = os.path.join(constants.SOURCE_FILE_DIR,constants.SOURCE_FILE_NAME)
        self.pipeline_name = constants.PIPELINE_NAME
        self.artifact_name = constants.ARTIFACT_DIR
        self.artifact_dir = os.path.join(self.artifact_name, timestamp)
        self.model_dir = os.path.join("final_model")
        self.timestamp: str = timestamp

class PredictPipelineConfig:
    def __init__(self):
        self.data_transformation_dir: str = os.path.join(
            constants.ARTIFACT_DIR, 
            constants.DATA_TRANSFORMATION_DIR_NAME
        )
        self.model_trainer_dir: str = os.path.join(
            constants.ARTIFACT_DIR, 
            constants.MODEL_TRAINER_DIR_NAME
        )
        self.trained_model_path: str = os.path.join(
            self.model_trainer_dir, 
            constants.MODEL_TRAINER_TRAINED_MODEL_DIR,
            constants.MODEL_FILE_NAME
        )
        self.one_hot_encoder_path: str = os.path.join(
            self.data_transformation_dir, 
            constants.DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR,
            constants.ONE_HOT_ENCODER_PATH,
        )
        self.year_encoder_path: str = os.path.join(
            self.data_transformation_dir, 
            constants.DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR,
            constants.YEAR_ENCODER_PATH,
        )
        

class DataIngestionConfig:
    def __init__(self):
        self.raw_data_path: str = os.path.join(constants.ARTIFACT_DIR, constants.DATA_INGESTION_DIR_NAME, constants.RAW_FILE_NAME)

class DataTransformationConfig:
    def __init__(self):
        self.target_column = constants.TARGET_COLUMN
        self.data_transformation_dir: str = os.path.join(
            constants.ARTIFACT_DIR, 
            constants.DATA_TRANSFORMATION_DIR_NAME
        )
        self.transformed_data_path: str = os.path.join(
            self.data_transformation_dir, 
            constants.DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR,
            constants.DATA_TRANSFORMATION_TRANSFORMED_FILE_NAME,
        )
        self.one_hot_encoder_path: str = os.path.join(
            self.data_transformation_dir, 
            constants.DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR,
            constants.ONE_HOT_ENCODER_PATH,
        )
        self.year_encoder_path: str = os.path.join(
            self.data_transformation_dir, 
            constants.DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR,
            constants.YEAR_ENCODER_PATH,
        )

class ModelTrainerConfig:
    def __init__(self):
        self.model_trainer_dir: str = os.path.join(
            constants.ARTIFACT_DIR, 
            constants.MODEL_TRAINER_DIR_NAME
        )
        self.trained_model_path: str = os.path.join(
            self.model_trainer_dir, 
            constants.MODEL_TRAINER_TRAINED_MODEL_DIR,
            constants.MODEL_FILE_NAME
        )

import os
import sys
import pandas as pd
import joblib
from src.exception import CustomException
from src.logger import logging
from datetime import datetime
from src.components.data_transformation import DataTransformation  
from src.entity.config_entity import PredictPipelineConfig
from src.entity.artifact_entity import DataTransformationArtifact

class PredictPipeline:
    def __init__(self):
        self.predict_pipeline_config = PredictPipelineConfig()


    def predict(self, custom_data=None, input_data_path=None):
        logging.info("Entered the prediction method")

        try:
            # Step 1: Load the model and preprocessor
            logging.info("Loading model and preprocessor")
            model = joblib.load(self.predict_pipeline_config.trained_model_path)

            logging.info("Processing single data input")
            preprocessed_data = self.apply_transformations(custom_data)
            prediction = model.predict(preprocessed_data)
            return prediction[0] 
        
        except Exception as e:
            raise CustomException(f"Error during prediction: {str(e)}", sys)

    def apply_transformations(self, data_frame):
        """Apply necessary transformations on the data."""
        logging.info("Applying custom transformations to the input data")
        try:
            one_hot_encoder = joblib.load(self.predict_pipeline_config.one_hot_encoder_path)
            year_encoder = joblib.load(self.predict_pipeline_config.year_encoder_path)

            # Step 1 - Apply label encoding to the 'Year' column
            data_frame['Year'] = year_encoder.transform(data_frame['Year'])

            # Step 2 - Transform categorical columns using one-hot encoder
            categorical_cols = ['Crop_Type', 'Soil_Type']
            categorical_data = one_hot_encoder.transform(data_frame[categorical_cols])

            # Combine transformed categorical data with other columns
            categorical_columns = one_hot_encoder.get_feature_names_out(categorical_cols)
            categorical_df = pd.DataFrame(categorical_data, columns=categorical_columns, index=data_frame.index)

            # Drop original categorical columns and concatenate transformed data
            data_frame.drop(columns=categorical_cols, inplace=True)
            transformed_data = pd.concat([data_frame, categorical_df], axis=1)
            print(transformed_data)
            logging.info("Data successfully transformed")
            return transformed_data

        except Exception as e:
            raise CustomException(f"Error during data transformation: {str(e)}", sys)
        

class CustomData:
    def __init__(self, rainfall: float, year: int, irrigation_area: float, crop_type: str, soil_type: str):
        self.year = year
        self.rainfall = rainfall
        self.irrigation_area = irrigation_area
        self.crop_type = crop_type
        self.soil_type = soil_type

    def get_data_as_data_frame(self):
        """Convert the custom data to a DataFrame."""
        try:
            custom_data_input_dict = {
                "Year": [self.year],
                "Rainfall": [self.rainfall],
                "Irrigation_Area": [self.irrigation_area],
                "Crop_Type": [self.crop_type],
                "Soil_Type": [self.soil_type]
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)

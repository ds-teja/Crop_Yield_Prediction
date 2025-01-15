import os
import sys
import joblib  # For saving and loading the preprocessor
from dataclasses import dataclass
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from datetime import datetime

import pandas as pd
from src.exception import CustomException
from src.logger import logging
from src.utils import read_data

from src.entity.config_entity import DataTransformationConfig
from src.entity.artifact_entity import DataTransformationArtifact

class DataTransformation:
    def __init__(self,data_transformation_config: DataTransformationConfig,data_ingestion_artifact: DataTransformationArtifact):
        self.data_transformation_config = data_transformation_config
        self.data_ingestion_artifact = data_ingestion_artifact    

    def get_individual_transformers(self):
        try:
            onehot_encoder = OneHotEncoder(sparse_output=False, drop='first')
            year_encoder = LabelEncoder()
            return onehot_encoder, year_encoder
        
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self):
        logging.info("Entered the data transformation method or component")
        try:
            # Step 1 - Load train and test data
            data = read_data(self.data_ingestion_artifact.file_path)
            logging.info("Loaded data for transformation")
            
            # Step 2 - Replace missing crop yield (cy) values
            target = str(self.data_transformation_config.target_column)
            avg_cy = data.loc[data['Irrigation_Area'] < 4, target].mean()
            data.loc[data[target] == 0, target] = int(avg_cy)

            # Step 3 - Identify categorical and year columns
            categorical_cols = ['Crop_Type', 'Soil_Type']
            year_col = ['Year']

            # Step 4 - Create transformer object
            one_hot_encoder, year_encoder = self.get_individual_transformers()

            # Step 5 - Fit the transformers to the data
            one_hot_encoder.fit(data[categorical_cols])
            year_encoder.fit(data[year_col])

            # Step 6 - Transform the data
            categorical_data = one_hot_encoder.transform(data[categorical_cols])
            data['Year'] = year_encoder.transform(data['Year'])

            # Step 7 - Combine transformed categorical data with other columns
            categorical_columns = one_hot_encoder.get_feature_names_out(categorical_cols)
            categorical_df = pd.DataFrame(categorical_data, columns=categorical_columns)

            # Drop original categorical columns and add transformed data
            data.reset_index(drop=True, inplace=True)
            transformed_data = pd.concat([data.drop(columns=categorical_cols), categorical_df], axis=1)
            transformed_data.drop(columns=['id','State'],inplace=True)
            
            # Step 8 - Save the transformers
            os.makedirs(os.path.dirname(self.data_transformation_config.one_hot_encoder_path), exist_ok=True)
            joblib.dump(one_hot_encoder, self.data_transformation_config.one_hot_encoder_path)
            joblib.dump(year_encoder, self.data_transformation_config.year_encoder_path)

            # Step 9 - Save the transformed data
            os.makedirs(os.path.dirname(self.data_transformation_config.transformed_data_path), exist_ok=True)
            transformed_data.to_csv(self.data_transformation_config.transformed_data_path, index=False)
            logging.info(f"Transformed data saved")

            return DataTransformationArtifact(
                self.data_transformation_config.transformed_data_path,
                self.data_transformation_config.one_hot_encoder_path,
                self.data_transformation_config.year_encoder_path
            )
        except Exception as e:
            logging.error(e)
            raise CustomException(e, sys)

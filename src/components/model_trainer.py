import os
import sys
import joblib
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from src.exception import CustomException
from src.logger import logging
from src.utils import read_data

from src.entity.config_entity import ModelTrainerConfig
from src.entity.artifact_entity import DataTransformationArtifact, ModelTrainerArtifact

class EnsembleModel:
    """
    Custom class to encapsulate ensemble logic.
    """
    def __init__(self, rf_model, gb_model, rf_weight=0.8, gb_weight=0.2):
        self.rf_model = rf_model
        self.gb_model = gb_model
        self.rf_weight = rf_weight
        self.gb_weight = gb_weight

    def predict(self, X):
        rf_predictions = self.rf_model.predict(X)
        gb_predictions = self.gb_model.predict(X)
        return self.rf_weight * rf_predictions + self.gb_weight * gb_predictions

class ModelTrainer:
    def __init__(self, model_trainer_config:ModelTrainerConfig, data_transformation_artifact:DataTransformationArtifact):
        self.model_trainer_config = model_trainer_config
        self.data_transformation_artifact = data_transformation_artifact
        
    def initiate_model_trainer(self):
        logging.info("Entered the model training method or component")
        try:
            # Step 1 - Load train and test data
            data = read_data(self.data_transformation_artifact.transformed_file_path)

            # Step 2 - Separate features and target
            X, y = data.drop(columns=['cy']), data['cy']
            print(X)
            logging.info("Separated features and target")

            # Step 3 - Train the models
            logging.info("Training the RandomForest model")
            rf_model = RandomForestRegressor(n_estimators=37, max_depth=7, random_state=2)
            rf_model.fit(X, y)

            logging.info("Training the GradientBoosting model")
            gb_model = GradientBoostingRegressor(
                n_estimators=300, learning_rate=0.2, max_depth=4, 
                min_samples_split=3, min_samples_leaf=4
            )
            gb_model.fit(X, y)

            # Step 4 - Create ensemble model
            logging.info("Creating the ensemble model")
            ensemble_model = EnsembleModel(rf_model, gb_model, rf_weight=0.8, gb_weight=0.2)

            # Step 6 - Save the ensemble model
            os.makedirs(os.path.dirname(self.model_trainer_config.trained_model_path), exist_ok=True)
            joblib.dump(ensemble_model, self.model_trainer_config.trained_model_path)
            logging.info(f"Ensemble model saved at {self.model_trainer_config.trained_model_path}")
            logging.info("Training Succesfully completed!")
            # Step 7 - Return Model Trainer Artifact
            model_trainer_artifact = ModelTrainerArtifact(self.model_trainer_config.trained_model_path)
            return model_trainer_artifact

        except Exception as e:
            logging.error(e)
            raise CustomException(e, sys)

from dataclasses import dataclass

@dataclass
class DataIngestionArtifact:
    file_path:str

@dataclass
class DataTransformationArtifact:
    transformed_file_path: str
    one_hot_encoder_path: str
    year_encoder_path: str
    
@dataclass
class ModelTrainerArtifact:
    trained_model_file_path: str

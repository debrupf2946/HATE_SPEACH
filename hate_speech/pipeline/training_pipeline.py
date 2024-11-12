import sys
from hate_speech.logger import logging
from hate_speech.components.data_ingestion import DataIngestion,DataValidation
from hate_speech.components.data_transformation import DataTransformation
from hate_speech.entity.config_entity import (DataIngestionConfig,
                                              DataTransformationConfig,
                                              ModelTrainerConfig)
from hate_speech.entity.artifact_entity import (DataIngestionArtifacts,
                                                DataTransformationArtifacts,
                                                ModelTrainerArtifacts)
from hate_speech.components.model_trainer import ModelTrainer


class TrainPipeline:
    def __init__(self):
        self.data_ingestion_config=DataIngestionConfig()
        self.data_transformation_config=DataTransformationConfig()
        self.model_trainer_config=ModelTrainerConfig()
        
    
    def start_data_ingestion(self)->DataIngestionArtifacts:
        
        logging.info("Entered the start_data_ingestion of training pipeline")
        data_ingestion=DataIngestion(data_ingestion_config=self.data_ingestion_config)
        data_ingestion_artifacts=data_ingestion.initiate_data_ingestion()
        
        logging.info("Got train and Valid  from GCloud Storage")
        
        logging.info("Exited the start_data_ingestion of training pipeline")
        
        return data_ingestion_artifacts
    
    def start_data_validation(self):
        logging.info("Entered the start_data_validation of training pipeline")
        data_validation=DataValidation(data_ingestion_config=self.data_ingestion_config)
        data_validation.initiate_data_validation()
        logging.info("Exited the start_data_validation of training pipeline")
        
    def start_data_transformation(self,data_ingestion_artifacts=DataIngestionArtifacts)-> DataTransformationArtifacts:
        
        data_transformation=DataTransformation(data_transformation_config=self.data_transformation_config,
                                               data_ingestion_artifacts=data_ingestion_artifacts)
        
        data_transformation_artifacts=data_transformation.initiate_data_transformation()
        
        logging.info("Exited the start_data_transformation method of TrainPipeline class")
        
        return data_transformation_artifacts
        
    def start_model_trainer(self,data_transformation_artifacts=DataTransformationArtifacts)->ModelTrainerArtifacts:
        
        model_trainer=ModelTrainer(data_transformation_artifacts=data_transformation_artifacts,
                                   model_trainer_config=self.model_trainer_config)
        
        model_trainer_artifacts=model_trainer.initiate_model_trainer()
        
        logging.info("Exited the start_model_trainer method of TrainPipeline class")
        return model_trainer_artifacts
    
    def run_pipeline(self):
        logging.info("Entered the run_pipeline of training pipeline")
        
        data_ingestion_artifacts=self.start_data_ingestion()
        self.start_data_validation()
        data_transformation_artifacts=self.start_data_transformation(
            data_ingestion_artifacts=data_ingestion_artifacts
        )
        
        model_trainer_artifacts=self.start_model_trainer(data_transformation_artifacts=data_transformation_artifacts)
        
        
        logging.info("Exited the run_pipeline of training pipeline")

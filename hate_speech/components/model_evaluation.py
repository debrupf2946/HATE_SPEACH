import os
import sys
import keras
import pickle
import numpy as np
import pandas as pd
from hate_speech.constants import *
from hate_speech.configuration.gcloud_syncer import GCloudSync
from keras.utils import pad_sequences
from sklearn.metrics import confusion_matrix
from hate_speech.entity.config_entity import ModelEvaluationConfig
from hate_speech.entity.artifact_entity import  ModelTrainerArtifacts,ModelEvaluationArtifacts,DataTransformationArtifacts

class ModelEvaluation:
    def __init__(self,model_evaluation_config: ModelEvaluationConfig,
                 model_trainer_artifacts: ModelTrainerArtifacts,
                 data_transformation_artifacts: DataTransformationArtifacts):
        self.model_evaluation_config=model_evaluation_config
        self.model_trainer_artifacts=model_trainer_artifacts
        self.data_transformation_artifacts=data_transformation_artifacts
        self.gcloud=GCloudSync()
        
    def get_best_model_from_gcloud(self):
        
        
        logging.info("Entered the get_best_model_from_gcloud method of Model Evaluation class")
        
        os.makedirs(self.model_evaluation_config.BEST_MODEL_DIR_PATH,exist_ok=True)
        self.gcloud.sync_folder_from_gcloud(self.model_evaluation_config.BUCKET_NAME,
                                            self.model_evaluation_config.MODEL_NAME,
                                            self.model_evaluation_config.BEST_MODEL_DIR_PATH)
        best_model_path=os.path.join(self.model_evaluation_config.BEST_MODEL_DIR_PATH,
                                     self.model_evaluation_config.BUCKET_NAME)
        
        logging.info("Exited the get_best_model_from_gcloud method of Model Evaluation class")
        
        return best_model_path
    
    def evaluate(self):
        logging.info("Entering into to the evaluate function of Model Evaluation class")
        print(self.model_trainer_artifacts.x_test_path)

        x_test = pd.read_csv(self.model_trainer_artifacts.x_test_path,index_col=0)
        print(x_test)
        y_test = pd.read_csv(self.model_trainer_artifacts.y_test_path,index_col=0)

        with open('tokenizer.pickle', 'rb') as handle:
            tokenizer = pickle.load(handle)

        load_model=keras.models.load_model(self.model_trainer_artifacts.trained_model_path)

        x_test = x_test[TWEET].astype(str)

        x_test = x_test.squeeze()
        y_test = y_test.squeeze()
        
        test_sequences = tokenizer.texts_to_sequences(x_test)
        test_sequences_matrix = pad_sequences(test_sequences,maxlen=MAX_LEN)
        print(f"----------{test_sequences_matrix}------------------")

        print(f"-----------------{x_test.shape}--------------")
        print(f"-----------------{y_test.shape}--------------")
        accuracy = load_model.evaluate(test_sequences_matrix,y_test)
        logging.info(f"the test accuracy is {accuracy}")
        
        lstm_prediction = load_model.predict(test_sequences_matrix)
        res = []
        for prediction in lstm_prediction:
            if prediction[0] < 0.5:
                res.append(0)
            else:
                res.append(1)
        print(confusion_matrix(y_test,res))
        logging.info(f"the confusion_matrix is {confusion_matrix(y_test,res)} ")
        return accuracy
        
        
    def initiate_model_evaluation(self)->ModelEvaluationArtifacts:
        
        logging.info("Initiate Model Evaluation")
        
        logging.info("Loading currently trained model")
        
        trained_model=keras.models.load_model(self.model_trainer_artifacts.trained_model_path)
        
        with open('tokenizer.pickle', 'rb') as handle:
            load_tokenizer = pickle.load(handle)
        
        trained_model_accuracy=self.evaluate()
        
        logging.info("Fetch best model from gcloud storage")
        best_model_path = self.get_best_model_from_gcloud()
        
        logging.info("Check is best model present in the gcloud storage or not ?")
        
        if os.path.isfile(best_model_path) is False:
            is_model_accepted=True
            logging.info("glcoud storage model is false and currently trained model accepted is true")
            
        else:
            logging.info("Load best model fetched from gcloud storage")
            best_model=keras.models.load_model(best_model_path)
            
        
    
import os
import sys
from zipfile import ZipFile
import pandas as pd
from hate_speech.logger import logging
from hate_speech.configuration.gcloud_syncer import GCloudSync
from hate_speech.entity.config_entity import DataIngestionConfig
from hate_speech.entity.artifact_entity import DataIngestionArtifacts
# from hate_speech.exception import 

class DataIngestion:
    def __init__(self,data_ingestion_config: DataIngestionConfig):
        self.data_ingestion_config=data_ingestion_config
        self.gcloud=GCloudSync()
        
    def get_data_from_gcloud(self)-> None:
        logging.info('Entered get_data_from_gcloud method of DataIngestion class')
        os.makedirs(self.data_ingestion_config.DATA_INGESTION_ARTIFACTS_DIR,exist_ok=True)
        
        self.gcloud.sync_folder_from_gcloud(self.data_ingestion_config.BUCKET_NAME, 
                                            self.data_ingestion_config.ZIP_FILE_NAME, 
                                            self.data_ingestion_config.DATA_INGESTION_ARTIFACTS_DIR)
        
        logging.info("Exited get_data_from_gcloud of DataIngestion class")
        
        
    def unzip_and_clean(self):
        logging.info('Entered the unzip_and_clean method of data ingestion class')
        
        with ZipFile(self.data_ingestion_config.ZIP_FILE_PATH,'r') as zip_ref:
            zip_ref.extractall(self.data_ingestion_config.ZIP_FILE_DIR)
            
        logging.info('Exited unzip_and_clean of DataIngestion class')
        
        return self.data_ingestion_config.DATA_ARTIFACTS_DIR,self.data_ingestion_config.NEW_DATA_ARTIFACTS_DIR
            
    def initiate_data_ingestion(self)-> DataIngestionArtifacts:
        logging.info('Entered the initiate_data_ingestio method of data ingestion class')
        
        self.get_data_from_gcloud()
        logging.info('Fetched data from gcloud bucket')
        imbalance_data_file_path,raw_data_file_path=self.unzip_and_clean()
        logging.info('unzipped file and split into train and valid')
        
        data_ingestion_artifacts=DataIngestionArtifacts(imbalance_data_file_path=imbalance_data_file_path,
                                                        raw_data_file_path=raw_data_file_path)
        
        logging.info('Exited the initiate_data_ingestio method of data ingestion class')
        
        logging.info(f"Data Ingestion artifacts : {data_ingestion_artifacts}")
        
        return data_ingestion_artifacts
    


class DataValidation:
    def __init__(self,data_ingestion_config: DataIngestionConfig):
        self.data_ingestion_config=data_ingestion_config
        
    def get_downloaded_data(self):
        logging.info("Entered get_downloaded_data  method of DataValidation Class")
        imbalance_data_path=self.data_ingestion_config.DATA_ARTIFACTS_DIR
        raw_data_path=self.data_ingestion_config.NEW_DATA_ARTIFACTS_DIR
        
        self.imbalance_df=pd.read_csv(imbalance_data_path)
        self.raw_data_df=pd.read_csv(raw_data_path)
        logging.info("Built imbalance_df and raw_data_df of DataValidation  Class")
        logging.info("exited get_downloaded_data  method of DataValidation Class")
        
    
    def validate_imbalance_data(self):
        logging.info("Entered validate_imbalance_data  method of DataValidation Class")
        imbalance_df_columns=self.imbalance_df.columns
        if len(imbalance_df_columns)== 3 and  set(imbalance_df_columns)==set(['id', 'label', 'tweet']):
            logging.info("Exiting validate_imbalance_data  method of DataValidation Class with validation")
            return True
        logging.info("Exiting validate_imbalance_data  method of DataValidation Class with no validation")
        
    def validate_raw_data(self):
        logging.info("Entered validate_raw_data  method of DataValidation Class")
        raw_data_columns=self.raw_data_df.columns
        if (len(raw_data_columns)==7) and (set(raw_data_columns)==set(['Unnamed: 0', 'count', 'hate_speech', 'offensive_language', 'neither',
       'class', 'tweet'])):
            logging.info("Exiting validate_raw_data  method of DataValidation Class with validation")
            return True
        logging.info("Exiting validate_raw_data  method of DataValidation Class with no validation")
        
        
    
    def initiate_data_validation(self):
        
        logging.info("Entered initiate_data_validation  method of DataValidation Class")
        self.get_downloaded_data()
        imbalance_data_validation=self.validate_imbalance_data()
        raw_data_validation=self.validate_raw_data()
        
        logging.info(f"""Exiting initiate_data_validation  method of DataValidation Class with \n
                     imbalance_data_validation : {imbalance_data_validation} \n
                     raw_data_validation : {raw_data_validation}""")
        
        
        
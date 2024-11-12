from datetime import datetime
import os


#Common Constants

TIMESTAMP: str=datetime.now().strftime('%m_%d_%Y_%H_%M_%S')

ARTIFACTS_DIR=os.path.join('artifacts',TIMESTAMP)

BUCKET_NAME ='hate_speech_2026'
ZIP_FILE_NAME="dataset.zip"

LABEL='label'
TWEET='tweet'

#Data Ingestion Constants

DATA_INGESTION_ARTIFACTS_DIR="DataIngestionArtifacts"
DATA_INGESTION_IMBALANCE_DATA_DIR="imbalanced_data.csv"
DATA_INGESTION_RAW_DATA_DIR='raw_data.csv'


#Data Transformation Constants

DATA_TRANSFORMATION_ARTIFACT_DIR = "DataTransformationArtifacts"
TRANSFORMED_FILE_NAME= "final.csv"
DATA_DIR = "data"
ID = "id"
AXIS = 1
INPLACE = True
DROP_COLUMNS= ['Unnamed: 0', 'count', 'hate_speech', 'offensive_language', 'neither']
CLASS="class"

# Model Training Constants

MODEL_TRAINER_ARTIFACTS_DIR='ModelTrainerArtifacts'
TRAINED_MODEL_DIR='trained_model'
TRAINED_MODEL_NAME='model.h5'
X_TEST_FILE_NAME='x_test.csv'
Y_TEST_FILE_NAME='y_test.csv'

X_TRAIN_FILE_NAME='x_train.csv'

RANDOM_STATE=42
EPOCH=2
BATCH_SIZE=128
VALIDATION_SPLIT=0.2

# Model Archtecture Constants
MAX_WORDS=50000
MAX_LEN=300
LOSS='binary_crossentropy'
METRICS=['accuracy']
ACTIVATION='sigmoid'


# Model Evaluations Constants

MODEL_EVALUATION_ARTIFACTS_DIR='ModelEvaluationArtifacts'
BEST_MODEL_DIR='best_model'
MODEL_EVALUATION_FILE_NAME='loss.csv'


MODEL_NAME="model.h5"
APP_HOST='0.0.0.0'
APP_PORT=8080
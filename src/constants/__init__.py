import os
from datetime import date 

# For Mongodb connection

DATABASE_NAME = "Proj1"
COLLECTION_NAME = "Proj1-Data"
MONGODB_URL_KEY = "MONGODB_URL"

PIPELINE_NAME = ''
ARTIFACT_DIR = 'artifact'

MODEL_FILE_NAME  ='model.pkl'

TARGET_COLUMN = "Response"
CURRENT_YEAR  = "date.today().year"
PREPROCESSING_OBJECT_FILE_NAME = "preprocessing.pkl"


FILE_NAME = "data.csv"
TRAIN_FILE_NAME = "train.csv"
TEST_FILE_NAME = "test.csv"
SCHEMA_FILE_PATH = os.path.join('config','schema.yaml')



"""
Data Ingestion related constant start with DATA_INGSTION VARAIBLE NAME

"""

DATA_INGESTON_COLLECTION_NAME = "Proj1-Data"

DATA_INGESTION_DIR_NAME  =  "data_ingestion"
DATA_INGESTION_FEATURE_STORE_DIR  = "feature_store"
DATA_INGESTION_INGESTED_DIR = "ingested"
DATA_INGESTION_TRAIN_TEST_SPLIT_RATION = 0.25
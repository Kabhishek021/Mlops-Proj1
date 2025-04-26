import sys 

from typing import Tuple 
import numpy as np 

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score ,f1_score, precision_score,recall_score

from src.exception import MyException
from src.logger import logging

from src.utils.main_utils import load_numpy_array_data,load_object, save_object
from src.entity.config_entity import ModelTrainerConfig
from src.entity.artifact_entity import DataTransformationArtifact, ClassificationMetricArtifact
from src.entity.estimator import MyModel

class ModelTrainer:

    def __init__(self, data_transformation_artifact :DataTransformationArtifact,
                 model_trainer_config: ModelTrainerConfig):
        
        
        self.data_transformation_artifact = data_transformation_artifact,
        self.model_trainer_config= model_trainer_config

    
    def get_model_object_and_report(self, train ,test):

        try:
            logging.info("Training RandomForestClassifier with specified parameters.")

            #spitting the train and test data into features and target variable 

            x_train,y_train,x_test,y_test = train[:,:-1] , train[:,-1] , test[:,:-1] , test[:,-1]
            logging.info("trin-tets spolit done.")

            #Initilaizer Randomforest Classifieer with specified parameters

            model = RandomForestClassifier(

                n_estimators= self.model_trainer_config._n_estimators,
                min_samples_split= self.model_trainer_config._min_samples_split,
                min_samples_leaf= self.model_trainer_config._min_samples_leaf,
                max_depth=  self.model_trainer_config._max_depth,
                criterion= self.model_trainer_config._criterion,
                random_state=self.model_trainer_config._random_state
    
            )

            #Fit the model 

            logging.info("Model training goi9ng on ...")
            model.fit(x_train ,y_train)
            logging.info('Model training done .')

            # Prediction and evluation metrics
            y_pred = model.predict(x_test)
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)

            #creating the metric artifact 

            metric_artifact = C
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression,Ridge,Lasso,ElasticNet
from sklearn.metrics import r2_score
import os
import sys
from src.logger import logging
from src.exception import CustomException
from src.utils import save_object,evaluate_model
from dataclasses import dataclass



@dataclass
class DataTrainingConfig:
    model_save_path=os.path.join('artifacts','model.pkl')



class DataTraining:
    def __init__(self):
        self.model_path=DataTrainingConfig()

    def Initiate_data_training(self,X_train,X_test,y_train,y_test):

        try:
            logging.info('Ititate data trainig')

            R_Model={"linearreagression":LinearRegression(),'ridge':Ridge(),'lasso':Lasso(),'elasticnet':ElasticNet()}

            print(R_Model)
            model_scor_report:dict=evaluate_model(X_train,y_train,X_test,y_test,R_Model)
            print(f'Model Scor Report{model_scor_report}')
            logging.info(f'Model performance{model_scor_report}')

            model_score=max(sorted(model_scor_report.values()))

            model_name=list(model_scor_report.keys())[list(model_scor_report.values()).index(model_score)]
            best_model=R_Model[model_name]

            print(f'Best Model Name {best_model}')
            logging.info(f'Best Model Name {best_model}')

            save_object(self.model_path.model_save_path,best_model)

            print("Best model pikle file saved successfully")
            logging.info("Best model pikle file saved successfully")


        except Exception as e:
            raise CustomException(e,sys)
            logging.info("Error in data training ")  









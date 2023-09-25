import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from src.logger import logging
from src.exception import CustomException
from dataclasses import dataclass
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.pipeline import  Pipeline
from sklearn.preprocessing import StandardScaler,OneHotEncoder,OrdinalEncoder
from sklearn.compose import ColumnTransformer
from src.utils import save_object


@dataclass
class DataTransformationConfig:
    preproccesor_path=os.path.join('artifacts','preprocesssor.pkl')
    # row_data_path=os.path.join('artifacts','row.csv')



class DataTransformation:
    def __init__(self):
        self.preproccessor_file_path=DataTransformationConfig()
        # self.row_data_file_path=DataTransformationConfig.row_data_path()
        




    def get_preprocessor_obj(self):

        try:

            logging.info('preprocessing start')

            categorical_cols = ['cut', 'color','clarity']
            numerical_cols = ['carat', 'depth','table', 'x', 'y', 'z']


            cut_categories = ['Fair', 'Good', 'Very Good','Premium','Ideal']
            color_categories = ['D', 'E', 'F', 'G', 'H', 'I', 'J']
            clarity_categories = ['I1','SI2','SI1','VS2','VS1','VVS2','VVS1','IF']

            logging.info('Pipeline start')

            Num_pipeline=Pipeline(
                         steps=[
                               ('imputer',SimpleImputer(strategy='median')),
                               ('scaler',StandardScaler())

                               ])
            Cat_pipeline=Pipeline(

                          steps=[

                                 ('imputer',SimpleImputer(strategy='most_frequent')),
                                 ('ordinalencodr',OrdinalEncoder(categories=[cut_categories,color_categories,clarity_categories])),
                                 ('scaler',StandardScaler())
                                ])
            

            logging.info('ColumnTransform start and returning preprocessor')
            processor=ColumnTransformer(

                                          [

                                             ('num_pipeline',Num_pipeline,numerical_cols),
                                             ('cat_pipeline',Cat_pipeline,categorical_cols)
                                          ]
                                        )
            
            logging.info("pipeline completed")


            return processor




        except Exception as e:
           
           raise CustomException(e,sys)
           logging.info('Error in transformation')



        
    def initiate_data_transform(self):

        try:

            file_path=os.path.join('notebook/data','gemstone.csv')

            preproseccor_obj=self.get_preprocessor_obj()

            df=pd.read_csv(file_path)

            
            target_colmn='price'
            drop_colmn=[target_colmn,'id']

            X=df.drop(columns=drop_colmn,axis=1)
            y=df[target_colmn]


            X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.30,random_state=40)

            X_train=pd.DataFrame(preproseccor_obj.fit_transform(X_train),columns=preproseccor_obj.get_feature_names_out())
            X_test=pd.DataFrame(preproseccor_obj.transform(X_test),columns=preproseccor_obj.get_feature_names_out())

            save_object(file_path=self.preproccessor_file_path.preproccesor_path,obj=preproseccor_obj)




            return ( X_train,X_test,y_train,y_test)





        except Exception as e:
            raise CustomException(e,sys)
            logging.info('Error in intiating data transforming')  






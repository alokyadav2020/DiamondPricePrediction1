import pandas as pd
import numpy as np
import os
import sys
from src.logger import logging
# from src.exception import CustomException
from src.components.data_ingesion import DataIngetion
from src.components.data_transformer import DataTransformation
from src.components.data_training import DataTraining


if __name__=='__main__':
    
    obj=DataIngetion()
    train_data_path,test_data_path=obj.initiate_data_ingetion()

    print(train_data_path)
    print(test_data_path)


    DataTransform_obj=DataTransformation()
    X_train,X_test,y_train,y_text=DataTransform_obj.initiate_data_transform()

    print(X_train)
    print()
    print(y_train)

    training_obj=DataTraining()
    training_obj.Initiate_data_training(X_train,X_test,y_train,y_text)










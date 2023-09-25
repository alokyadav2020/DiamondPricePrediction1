import pandas as pd
import numpy as np
from src.logger import logging
from src.exception import CustomException
import os
import sys
from src.utils import load_object
import pickle



class Predict_Pipeline:
   


    def __init__(self):
        pass


    def Predict(self,features):

        try:

            logging.info('Prediction initiated')

            Model_Path=os.path.join('artifacts','model.pkl')
            proccesr_path=os.path.join('artifacts','preprocesssor.pkl')

            processor=load_object(proccesr_path)
            # Models = pickle.load(open(Model_Path, 'rb'))
            Models=load_object(Model_Path)
            print("============================================================================")
            print()
            print(features)
            print()
            # Models=load_object(Model_Path)

            scaled_data=processor.transform(features)

            

            print("============================================================================")
            print()
            print(scaled_data)
           
            

            result=Models.predict(scaled_data)

            return result


        except Exception as e:
            raise CustomException(e,sys)
            


class InputData:
    def __init__(self,carat:float,depth:float,table:float,x:float,y:float,z:float,cut:str,color:str,clarity:str):

        self.carat=carat
        self.depth=depth
        self.table=table
        self.x=x
        self.y=y
        self.z=z
        self.cut=cut
        self.color=color
        self.clarity=clarity


    def Get_dataframe(self):
        try:

            data=[{'carat':self.carat,'depth':self.depth,'table':self.table,'x':self.x,'y':self.y,'z':self.z,'cut':self.cut,'color':self.color,'clarity':self.clarity}]
            df=pd.DataFrame(data)

            return df



        except Exception as e:
            raise CustomException(e,sys)        





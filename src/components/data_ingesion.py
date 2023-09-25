import pandas as pd
import os 
import sys
from src.logger import logging
from src.exception import CustomException
from dataclasses import dataclass
from sklearn.model_selection import train_test_split



@dataclass
class DataingetIonconfig:
    train_path=os.path.join('artifacts','train.csv')
    test_path=os.path.join('artifacts','test.csv')
    row_path=os.path.join('artifacts','row.csv')




class DataIngetion:

    def __init__(self):
        self.ingetion_config=DataingetIonconfig()


    def initiate_data_ingetion(self):

        try:
            logging.info('Data ingetion initiated')

            data_path=os.path.join('notebook/data','gemstone.csv')

            df=pd.read_csv(data_path)

            os.makedirs(os.path.dirname(self.ingetion_config.row_path),exist_ok=True)

            df.to_csv(self.ingetion_config.row_path,index=False)

            logging.info('train,test,split')

            train_set,test_set=train_test_split(df,test_size=0.30,random_state=42)

           
            train_set.to_csv(self.ingetion_config.train_path,index=False,header=True )

            test_set.to_csv(self.ingetion_config.test_path,index=False,header=True)


         
           
            logging.info('Data ingetion part complited')


            return (self.ingetion_config.train_path,self.ingetion_config.test_path)
            # print(self.ingetion_config.train_path,self.ingetion_config.test_path)
            



        except Exception as e:

            logging.info('error occurs in  data ingestiion')


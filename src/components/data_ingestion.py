import os ## to store file on path
import sys ## loggging exception
sys.path.insert(0,'/src')
#from src.logger import logging ##loggin the error
#from src.exception import Customexception ## exception
from src.logger import logging
from src.exception import Customexception

import pandas as pd
from sklearn.model_selection import train_test_split

## initialize  the data ingestion configuration
from dataclasses import dataclass

@dataclass
class DataIngestionconfig:
   train_datapath:str=os.path.join('artifacts','train.csv')
   test_datapath:str=os.path.join('artifacts','test.csv')
   raw_data:str=os.path.join('artifacts','raw.csv')

## data ingestion class
class Dataingestion:
   def __init__(self):
      self.ingestionconfig=DataIngestionconfig()
      
   def intitate_data_ingestion(self):
        logging.info("Data Ingestion start")

        try:
          df=pd. read_csv(os.path.join('notebook/data','gemstone.csv'))
          logging.info('dataset read from pandas datadf')

          os.makedirs(os.path.dirname(self.ingestionconfig.raw_data),exist_ok=True)
          pd.to_csv(self.ingestionconfig.raw_data,index=False)

          logging.info('Raw data is created')

          train_set,test_set=train_test_split(df,test_size=0.30,random_state=42)
          train_set.to_csv(self.ingestionconfig.train_datapath,index=False,header=True)
          train_set.to_csv(self.ingestionconfig.test_datapath,index=False,header=True)

          logging.info('Ingestion of data completed')

          return(
             self.ingestionconfig.train_datapath,
             self.ingestionconfig.test_datapath
          )

        except Exception as e:
            logging.info("exception ouuurec at data ingestion stage")
            raise Customexception(e,sys)
      
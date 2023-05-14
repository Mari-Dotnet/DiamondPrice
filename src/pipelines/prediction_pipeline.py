import sys
import os
import pandas as pd
import numpy as np
from src.logger import logging
from src.exception import Customexception
from src.utils import  load_object


class predict_pipeline:
    def __init__(self):
        pass

    def predict(self,feature):
        try:
            pre_processor_path=os.path.join('artifacts','preprocessor.pkl')
            model_path=os.path.join('artifacts','model.pkl')

            pre_processor=load_object(pre_processor_path)
            model=load_object(model_path)

            data_scaled=pre_processor.tranform(feature)
            pred=model.predict(data_scaled)

            return pred
        except Exception as e:
            raise Customexception(e,sys)


class CustomData:
    def __init__(self,
                 carat:float,
                 depth:float,
                 table:float,
                 x:float,
                 y:float,
                 z:float,
                 cut:str,
                 color:str,
                 clarity:str):
        
        self.carat=carat
        self.depth=depth
        self.table=table
        self.x=x
        self.y=y
        self.z=z
        self.cut = cut
        self.color = color
        self.clarity = clarity

    def get_data_as_dataframe(self):
        try:
            custom_data_input_dict = {
                'carat':[self.carat],
                'depth':[self.depth],
                'table':[self.table],
                'x':[self.x],
                'y':[self.y],
                'z':[self.z],
                'cut':[self.cut],
                'color':[self.color],
                'clarity':[self.clarity]
            }
            df = pd.DataFrame(custom_data_input_dict)
            logging.info('Dataframe Gathered')
            return df
        except Exception as e:
            logging.info('Exception Occured in prediction pipeline')
            raise Customexception(e,sys)
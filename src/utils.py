import os
import sys
import pickle
import numpy as np
import pandas as pd

from src.exception import   Customexception
from src.logger import logging
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error

def save_object(file_path,obj):
    try:
        dir_path=os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)

        with open(file_path,'wb') as file:
            pickle.dump(obj,file)
    except Exception as e:
        raise Customexception(e,sys)

def evaulatemetrics(X_train,X_test,y_train,y_test,models):
    try:

        result={}
        for i in range(len(models)):
            model=list(models.values())[i]
            model.fit(X_train,y_train)
            ##predict
            y_predict=model.predict(X_test)
            
            mae=mean_absolute_error(y_test,y_predict)
            mse=mean_squared_error(y_test,y_predict)
            r2score=r2_score(y_test,y_predict)
            result[list(models.keys())[i]]={'Mae':mae,'mse':mse,'r2score':r2score}
        return result
    except Exception as e :
        logging.info("error occured and model training")
        raise Customexception(e,sys)
    
def load_object(file_path):
    try:
        with open(file_path,'rb') as file:
            return pickle.load(file)
    except Exception as e:
        raise Customexception(e,sys)
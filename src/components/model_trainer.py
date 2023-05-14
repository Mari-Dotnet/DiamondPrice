import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression,Ridge,Lasso,ElasticNet

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from src.logger import logging
from src.exception import Customexception

from src.utils import evaulatemetrics

from src.utils import save_object
from dataclasses import dataclass
import sys,os

@dataclass
class modl_trainer_config:
    trianmodel_filepath=os.path.join('articfacts','modle.pkl')

class model_trianer_class:
   def  __init__ (self):
       self.model_triner_config=modl_trainer_config()
   def initiate_model_training(self,train,test):
        try:
           logging.info('started model trainer class')
           X_train,y_train,X_test,y_test=(
               train[:,:-1],
               train[:,-1],
               test[:,:-1],
               test[:,-1]
            )
           model={
                    'Linear':LinearRegression(),
                    'Ridge':Ridge(),
                    'Lasso':Lasso(),
                    'ElasticNet':ElasticNet(),
                    'DecisionTreeRegressor': DecisionTreeRegressor(),
                    'ensemble':RandomForestRegressor(),
                    'knn':KNeighborsRegressor()
                }
           model_report:dict=evaulatemetrics(X_train,y_train,X_test,y_test,model)

           ## to get best model score
           best_modelscore=max(sorted(model_report.values()))
           ## best model name
           best_model_name=list(model_report.keys())[
               list(model_report.values()).index(best_modelscore)
           ]

           best_model=model[best_model_name]
           print("best model name",best_model)
           logging.info("best model name : {best_model}")

           save_object(
               file_path=self.model_triner_config.trianmodel_filepath,
               obj=best_model
           )
       
        except Exception as e:
           raise Customexception(e,sys)

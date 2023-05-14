from sklearn.preprocessing import StandardScaler ## feature scalling
from sklearn.preprocessing import OrdinalEncoder ## ordinal encodin
from sklearn.impute import SimpleImputer ## hanndling missing values

##  pipeline
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import numpy as np
import pandas as pd

from src.exception import  Customexception
from src.logger import logging

import sys,os

from src.utils import save_object


from dataclasses import dataclass

@dataclass
class DataTransfromationConfig:
    preprocesser_obj_file_path=os.path.join('artifacts','preprocessor.pkl')

class get_datatranformation_transfer_object():
    def __init__(self):
        self.data_tranformation_config=DataTransfromationConfig()

    def initiate_data_transformation_obj(self):
        try:
            logging.info("Data transfermation started")

            ## categorical and numerical column
            numerical_column=['carat', 'depth', 'table', 'x', 'y', 'z']
            categorical_column=['cut', 'color', 'clarity']

            ## define custom ranking for catrgorical field values
            cut_map=['Fair','Good','Very Good','Premium','Ideal']
            clarity_map=['I1','SI2','SI1','VS2','VS1','VVS2','VVS1','IF']
            color_map=['D','E','F','G','H','I','J']
            

            logging.info('Data tranfermation pipline started')

            numpipeline=Pipeline(
                steps=[
                ('SimpleImputer',SimpleImputer(strategy='median')),
                ('scaler',StandardScaler())
                     ]
                )

            ## categorical pipeline
            catepipeline=Pipeline(
                steps=[
                    ('SimpleImputer',SimpleImputer(strategy='most_frequent')),
                    ('OrdinalEncoder',OrdinalEncoder(categories=[cut_map,color_map,clarity_map])),
                    ('scaler',StandardScaler())       
                    ]
                 )
            ## final colum tranfer
            preprocessor=ColumnTransformer([
                    ('numpipeline',numpipeline,numerical_column),
                    ('catepipeline',catepipeline,categorical_column)
                ])
            logging.info("data trasfer completed")
            return preprocessor
        except Exception as e:
            logging.info('Exception occured in Data Transfermation')
            raise Customexception(e,sys)
            
    def initiate_data_transfermation(self,train_path,test_path):
        try:
                train_df=pd.read_csv(train_path)
                test_df=pd.read_csv(test_path)

                logging.info("Trainin and testing data load completedly")

                logging.info("obtaning preprocessing")
                preprocessing_obj=self.initiate_data_transformation_obj()
                targetcolumn='price'
                dropcolumn=[targetcolumn,'id']
                

                ##dividing the dataset into independed and dependent features
                input_feature_train_df=train_df.drop(labels=dropcolumn,axis=1)
                target_feature_train_df=train_df[targetcolumn]

                ##test data 
                input_feature_test=test_df.drop(labels=dropcolumn,axis=1)
                target_feature_test=test_df[targetcolumn]

                ## Data tranfermation
                input_feature_train_array=preprocessing_obj.fit_transform(input_feature_train_df)
                input_feature_test_array=preprocessing_obj.tranform(input_feature_test)

                logging.info("Aplying train and test data")

                train_array=np.c_[input_feature_train_array,np.array(target_feature_train_df)]
                test_array=np.c_[input_feature_test_array,np.array(target_feature_test)]


                save_object(
                    file_path=self.data_tranformation_config.preprocesser_obj_file_path,
                    obj=preprocessing_obj
                      )
                return(
                     train_array,
                     test_array,
                     self.data_tranformation_config.preprocesser_obj_file_path
                )

        except Exception as e:
             raise Customexception(e,sys)
            

            
                
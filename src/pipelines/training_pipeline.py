import os
import sys
import pandas as pd
from src.logger import logging
from src.exception import Customexception

from src.components.data_ingestion import Dataingestion
from src.components.data_transformation import get_datatranformation_transfer_object

from src.components.model_trainer import model_trianer_class

if __name__=='__main__':
    obj=Dataingestion()
    train_data_path,test_data_path=obj.intitate_data_ingestion()
    print(train_data_path,test_data_path)

    data_trensfermation_obj=get_datatranformation_transfer_object()
    trian_arr,test_arr,pickle_path=data_trensfermation_obj.initiate_data_transfermation(train_data_path,test_data_path)

    model_trainer=model_trianer_class()
    model_trainer.initiate_model_training(trian_arr,test_arr)

    

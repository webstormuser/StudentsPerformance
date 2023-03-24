import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.components.data_ingestion import DataIngestionConfig,DataIngestion
from src.components.data_transformation import DataTransformation,DataTransformationConfig
from src.components.model_trainer import ModelTrainerConfig,ModelTrainer
from src.pipeline.predict_pipeline import PredictPipeline,CustomData



if __name__=="__main__":
    obj=DataIngestion()
    train_data,test_data=obj.initiate_data_ingestion()

    data_transformation=DataTransformation()
    train_arr,test_arr,_=data_transformation.initiate_data_transformation(train_data,test_data)

    modeltrainer=ModelTrainer()
    print(modeltrainer.initiate_model_trainer(train_arr,test_arr))

    predict_pipeline=PredictPipeline()
    gender='female'
    race_ethnicity='group B'
    parental_level_of_education='some college'
    lunch='standard'
    test_preparation_course='none'
    writing_score=89
    reading_score=78
    data=CustomData(gender,race_ethnicity,parental_level_of_education,lunch,test_preparation_course, 
            writing_score,reading_score)
    data_df=data.get_data_as_data_frame()
    print(data_df)
    result=predict_pipeline.predict(data_df)
    print(result[0])


        
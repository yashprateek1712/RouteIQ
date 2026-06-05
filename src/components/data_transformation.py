import sys
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer,StandardScaler

from src.exception import CustomException
from src.logger import logging
import os

from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join("artifacts","preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()
    

    def get_data_transfer_object(self):
        try:
            log_and_scale_columns = ['Order_Value_INR', 'Delivery_Time_Mins']
            scale_only_columns = ['Latitude', 'Longitude']

            log_pipeline=Pipeline(
                steps=[
                    ("log_transform",FunctionTransformer(np.log1p)),
                    ("scaler",StandardScaler())
                ]
            )           

            standard_pipeline=Pipeline(
                steps=[
                    ("scaler",StandardScaler())

                ]
            )
            logging.info(f"Applying Log + Standard scaling to: {log_and_scale_columns}")
            logging.info(f"Applying Standard scaling only to: {scale_only_columns}")   

            preprocessor=ColumnTransformer(
                [
                ("log_pipeline",log_pipeline,log_and_scale_columns),
                ("standard_pipeline",standard_pipeline,scale_only_columns)
                ],
                remainder="drop"
            )   
            return preprocessor

        except Exception as e:
            raise CustomException(e,sys)   


    def initiate_data_transformation(self,raw_data_path):
        try:
            df=pd.read_csv(raw_data_path)
            logging.info("Read master dataset completed")

            logging.info("Obtaining preprocessing object")
            preprocessing_obj=self.get_data_transfer_object()

            drop_columns = ['order_id']
            input_feature_df=df.drop(columns=drop_columns,axis=1)

            logging.info("Applying preprocessing objects on dataframe")

            data_arr=preprocessing_obj.fit_transform(input_feature_df)

            logging.info("Saved preprocessing object.")

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )
            return (
                data_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
        
        except Exception as e:
            raise CustomException(e, sys)




import os
import sys
from src.exception import CustomException
from src.logger import logging


import pandas as pd
from dataclasses import dataclass

from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig

from src.components.model_trainer import ModelTrainerConfig
from src.components.model_trainer import ModelTrainer

@dataclass
class DataIngestionConfig:
    raw_data_path:str = os.path.join("artifacts","data.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered in data ingestion process")
        try:
            df=pd.read_csv("notebook/real_delivery_data.csv")
            logging.info("Red the dataset as a DataFrame")

            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path),exist_ok=True)

            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)
            
            logging.info("Data ingestion completed")

            return self.ingestion_config.raw_data_path
  

            

        except Exception as e:
            raise CustomException(e,sys)    
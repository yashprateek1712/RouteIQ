from src.logger import logging
from src.exception import CustomException
import sys

# Import your three workers
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

if __name__ == "__main__":
    logging.info("<<< PIPELINE EXECUTION STARTED >>>")
    try:
        # STEP 1: INGESTION
        logging.info(">> Waking up Data Ingestion Worker...")
        ingestion = DataIngestion()
        raw_data_path = ingestion.initiate_data_ingestion()
        print(f"Data Ingestion Complete! Master data saved at: {raw_data_path}")

        # STEP 2: TRANSFORMATION
        logging.info(">> Waking up Data Transformation Worker...")
        transformation = DataTransformation()
        data_array, preprocessor_path = transformation.initiate_data_transformation(raw_data_path)
        print(f"Data Transformation Complete! Brain saved at: {preprocessor_path}")

        # STEP 3: MODEL TRAINING
        logging.info(">> Waking up Model Trainer Worker...")
        trainer = ModelTrainer()
        silhouette_score = trainer.initiate_model_trainer(data_array)
        
        print("===================================================")
        print("🎉 PIPELINE COMPLETED SUCCESSFULLY! 🎉")
        print(f"Optimal Warehouse Clusters found.")
        print(f"Network Separation (Silhouette Score): {silhouette_score:.4f}")
        print("Check your 'artifacts' folder for the .pkl files!")
        print("===================================================")

    except Exception as e:
        # If ANYTHING fails in any file, it gets caught right here.
        error = CustomException(e, sys)
        logging.error(error)
        print("Pipeline Failed! Check the logs folder for the exact line number.")
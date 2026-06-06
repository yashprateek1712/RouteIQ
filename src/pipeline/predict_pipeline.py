import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        # Load both brains into memory ONCE when the app starts
        preprocessor_path = 'artifacts/preprocessor.pkl'
        model_path        = 'artifacts/model.pkl'

        self.preprocessor    = load_object(file_path=preprocessor_path)
        self.model_dictionary = load_object(file_path=model_path)

        self.iso_forest = self.model_dictionary["IsolationForest"]
        self.kmeans     = self.model_dictionary["KMeans"]

    def predict(self, features):
        try:
            # Step 1: Scale the raw input using the saved preprocessor
            # All 4 features: Order_Value, Delivery_Time, Latitude, Longitude
            scaled_data = self.preprocessor.transform(features)

            # Step 2: IsoForest checks all 4 features for anomaly
            is_anomaly = self.iso_forest.predict(scaled_data)[0]

            if is_anomaly == -1:
                return "FRAUD/ANOMALY DETECTED", None

            # Step 3: KMeans assigns warehouse zone using all 4 features
            warehouse_cluster = self.kmeans.predict(scaled_data)[0]
            return "SAFE", warehouse_cluster

        except Exception as e:
            raise CustomException(e, sys)


class CustomData:
    """
    Maps HTML form inputs into a clean Pandas DataFrame
    so the prediction pipeline can process it.
    """
    def __init__(self,
                 Order_Value: float,
                 Delivery_Time_Mins: float,
                 Latitude: float,
                 Longitude: float):
        self.Order_Value        = Order_Value
        self.Delivery_Time_Mins = Delivery_Time_Mins
        self.Latitude           = Latitude
        self.Longitude          = Longitude

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "Order_Value":        [self.Order_Value],
                "Delivery_Time_Mins": [self.Delivery_Time_Mins],
                "Latitude":           [self.Latitude],
                "Longitude":          [self.Longitude],
            }
            return pd.DataFrame(custom_data_input_dict)
        except Exception as e:
            raise CustomException(e, sys)

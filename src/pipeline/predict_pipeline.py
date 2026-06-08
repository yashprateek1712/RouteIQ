import sys
import pandas as pd

from src.exception import CustomException
from src.utils import load_object


def is_within_brazil(lat, lon):
    """
    Rough geographic boundary for Brazil.
    Returns True if coordinates fall within Brazil's bounding box.
    """
    LAT_MIN, LAT_MAX = -33.75, 5.27
    LON_MIN, LON_MAX = -73.98, -34.79

    return (
        LAT_MIN <= lat <= LAT_MAX
        and LON_MIN <= lon <= LON_MAX
    )


class PredictPipeline:
    def __init__(self):
        # Load artifacts once
        preprocessor_path = 'artifacts/preprocessor.pkl'
        model_path = 'artifacts/model.pkl'

        self.preprocessor = load_object(file_path=preprocessor_path)
        self.model_dictionary = load_object(file_path=model_path)

        self.iso_forest = self.model_dictionary["IsolationForest"]
        self.kmeans = self.model_dictionary["KMeans"]

    def predict(self, features):
        try:
            # -----------------------------
            # 1. Geographic Validation
            # -----------------------------
            lat = float(features["Latitude"].iloc[0])
            lon = float(features["Longitude"].iloc[0])

            if not is_within_brazil(lat, lon):
                return (
                    "OUT OF DELIVERY ZONE - LOCATION OUTSIDE BRAZIL",
                    None
                )

            # -----------------------------
            # 2. Feature Scaling
            # -----------------------------
            scaled_data = self.preprocessor.transform(features)

            # -----------------------------
            # 3. Isolation Forest Anomaly Check
            # -----------------------------
            is_anomaly = self.iso_forest.predict(scaled_data)[0]

            if is_anomaly == -1:
                return "FRAUD/ANOMALY DETECTED", None

            # -----------------------------
            # 4. Warehouse Assignment
            # -----------------------------
            warehouse_cluster = self.kmeans.predict(scaled_data)[0]

            return "SAFE", warehouse_cluster

        except Exception as e:
            raise CustomException(e, sys)


class CustomData:
    """
    Converts user inputs into a DataFrame
    expected by the prediction pipeline.
    """

    def __init__(
        self,
        Order_Value: float,
        Delivery_Time_Mins: float,
        Latitude: float,
        Longitude: float
    ):
        self.Order_Value = Order_Value
        self.Delivery_Time_Mins = Delivery_Time_Mins
        self.Latitude = Latitude
        self.Longitude = Longitude

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "Order_Value": [self.Order_Value],
                "Delivery_Time_Mins": [self.Delivery_Time_Mins],
                "Latitude": [self.Latitude],
                "Longitude": [self.Longitude],
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)
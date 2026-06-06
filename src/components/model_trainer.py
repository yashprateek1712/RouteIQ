import os
import sys
from dataclasses import dataclass

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.metrics import silhouette_score, davies_bouldin_score

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class ModelTrainerConfig:
    trained_model_file_path:str=os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()

    def initiate_model_trainer(self, data_array):
        try:
            logging.info("Starting Unsupervised Model Training.")
            X = data_array

            # Isolation Forest
            logging.info("Training Isolation Forest...")
            iso_forest = IsolationForest(contamination=0.01, random_state=42)
            iso_forest.fit(X)
            anomaly_labels = iso_forest.predict(X)
            X_clean = X[anomaly_labels == 1]
            logging.info(f"Original dataset size: {len(X)}. Clean dataset size: {len(X_clean)}.")

            # ELBOW METHOD for finding number of clusters
        
            logging.info("Running Elbow Method to find optimal K...")
            inertias, sil_scores = [], []
            K_range = range(2, 11)

            for k in K_range:
                km_test = KMeans(n_clusters=k, init='k-means++', random_state=42)
                labels_test = km_test.fit_predict(X_clean)
                inertias.append(km_test.inertia_)
                sil_scores.append(silhouette_score(X_clean, labels_test,sample_size=5000, random_state=42))

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
            ax1.plot(list(K_range), inertias, 'bo-')
            ax1.set_title('Elbow Method')
            ax1.set_xlabel('n_clusters')
            ax1.set_ylabel('Inertia')

            ax2.plot(list(K_range), sil_scores, 'ro-')
            ax2.set_title('Silhouette Score vs K')
            ax2.set_xlabel('n_clusters')

            plt.tight_layout()
            os.makedirs('artifacts', exist_ok=True)
            plt.savefig('artifacts/elbow_plot.png')
            logging.info("Elbow plot saved to artifacts/elbow_plot.png")
            plt.close()

            #  K-MEANS CLUSTERING
            logging.info("Training K-Means Clustering on clean data...")
            kmeans = KMeans(n_clusters=5, init='k-means++', random_state=42)
            cluster_labels = kmeans.fit_predict(X_clean)

            sil = silhouette_score(X_clean, cluster_labels)
            db  = davies_bouldin_score(X_clean, cluster_labels)
            logging.info(f"Silhouette Score : {sil:.4f}  (higher is better)")
            logging.info(f"Davies-Bouldin   : {db:.4f}   (lower is better)")

            # SAVING THE BRAINS
            logging.info("Packaging models into a dictionary...")
            model_dictionary = {
                "IsolationForest": iso_forest,
                "KMeans": kmeans
            }
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=model_dictionary
            )
            logging.info("Saved model.pkl to artifacts folder.")
            return sil

        except Exception as e:
            raise CustomException(e, sys)

# RouteIQ — E-Commerce Logistics Optimizer

An end-to-end unsupervised machine learning pipeline built on **96,000+ real Brazilian e-commerce orders** from the Olist dataset. RouteIQ detects anomalous orders using Isolation Forest and assigns normal orders to one of 5 optimized delivery zones using K-Means clustering — all served through a Flask web app with an interactive map.

---

## What it does

You give it an order. It tells you two things:

1. **Is this order suspicious?** — unusual value, impossible delivery time, or an outlier location gets flagged before it enters the logistics network
2. **Which delivery zone should handle it?** — the order gets routed to one of 5 geographic hubs covering the Brazilian delivery network

---

## ML Pipeline

```
Olist Dataset (4 raw CSVs)
        ↓
Feature Engineering  →  Order Value (BRL) · Delivery Time (mins) · Lat · Lng
        ↓
Log Transform + StandardScaler  →  handles heavy right skew in financial data
        ↓
Isolation Forest (contamination=0.01)  →  flags ~963 anomalous orders
        ↓
K-Means (k=5, k-means++)  →  clusters 95,241 clean orders into delivery zones
        ↓
Flask API  →  real-time predictions via web interface
```

### Why these two models?

**Isolation Forest** — tree-based anomaly detection. Isolates outliers by randomly partitioning the feature space. Points that get isolated in fewer splits are anomalies. No assumptions about data distribution, works well on high-dimensional mixed data.

**K-Means** — tried DBSCAN first (explored eps=0.2 and eps=0.4 via K-Distance graph), but Brazilian delivery data has uneven geographic density — São Paulo alone accounts for 25% of all orders. DBSCAN collapsed everything into one giant cluster. K-Means gives us the 5 hard-boundary zones we need for warehouse assignment.

---

## Results

| Metric | Score |
|---|---|
| Silhouette Score | `run main.py to generate` |
| Davies-Bouldin Index | `run main.py to generate` |
| Anomalies detected | ~963 / 96,204 orders (~1%) |
| Training data | 96,204 orders across Brazil |

---

## Tech Stack

- **Python 3.10**
- **scikit-learn** — IsolationForest, KMeans, StandardScaler, ColumnTransformer
- **pandas / numpy** — data engineering and feature construction
- **Flask** — REST API + web interface
- **Leaflet.js** — interactive delivery location map
- **Matplotlib / Seaborn** — EDA and cluster visualization

---

## Dataset

**Olist Brazilian E-Commerce** (Kaggle) — 4 tables joined:

| Table | Used for |
|---|---|
| `olist_orders_dataset.csv` | Delivery timestamps → `Delivery_Time_Mins` |
| `olist_order_items_dataset.csv` | Price + freight → `Order_Value` |
| `olist_customers_dataset.csv` | Zip code → geolocation join |
| `olist_geolocation_dataset.csv` | Lat/Lng per zip code prefix |

Feature engineering done in `notebook/data.ipynb` — 4 raw tables joined into one clean CSV using SQL-style pandas merges.

---

## Project Structure

```
RouteIQ/
├── src/
│   ├── components/
│   │   ├── data_ingestion.py       # loads CSV into artifacts/
│   │   ├── data_transformation.py  # log transform + scaling pipeline
│   │   └── model_trainer.py        # IsoForest + KMeans + elbow plot
│   ├── pipeline/
│   │   └── predict_pipeline.py     # loads artifacts, runs prediction
│   ├── logger.py                   # timestamped log files
│   ├── exception.py                # custom traceback with file + line
│   └── utils.py                    # save/load pickle helpers
├── notebook/
│   ├── data.ipynb                  # feature engineering (4 tables → CSV)
│   └── eda.ipynb                   # EDA, DBSCAN exploration, KMeans
├── templates/
│   └── index.html                  # Flask frontend with Leaflet map
├── artifacts/                      # generated — model.pkl, preprocessor.pkl
├── logs/                           # generated — timestamped log files
├── app.py                          # Flask application
├── main.py                         # training pipeline entry point
├── setup.py                        # package setup with find_packages()
└── requirements.txt
```

---

## Setup & Run

**1. Clone and install**
```bash
git clone https://github.com/yourusername/RouteIQ.git
cd RouteIQ
pip install -r requirements.txt
```

**2. Add the Olist dataset**

Download from [Kaggle — Brazilian E-Commerce](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) and place CSVs in `notebook/data/`

**3. Generate the training data**

Run `notebook/data.ipynb` top to bottom — this joins the 4 Olist tables and saves `real_delivery_data.csv`

**4. Train the models**
```bash
python main.py
```
This runs the full pipeline and saves `artifacts/model.pkl`, `artifacts/preprocessor.pkl`, and `artifacts/elbow_plot.png`

**5. Start the web app**
```bash
python app.py
```
Open `http://localhost:5000` — select a city or click the map, enter order details, predict.

---

## How to test

Normal orders (expect **SAFE**):

| City | Order Value | Delivery Days |
|---|---|---|
| Rio de Janeiro | R$168 | 15 |
| São Paulo | R$105 | 10 |
| Porto Alegre | R$120 | 20 |

Anomalous orders (expect **FLAGGED**):

| Reason | Order Value | Delivery Days |
|---|---|---|
| Unusually high value | R$5000 | 12 |
| Impossible delivery speed | R$100 | 1 |

---

## Key design decisions

- **Log transform before scaling** — `Order_Value` skewness ~9.4, `Delivery_Time_Mins` skewness ~3.8. Raw values would dominate distance calculations and break both models.
- **IsoForest before KMeans** — clustering on clean data only. Anomalies would distort cluster centroids and produce meaningless zones.
- **Preprocessor saved as `.pkl`** — same scaling applied at training and inference. No data leakage.
- **Delivery time in minutes, not days** — internally the model uses minutes for precision. The web app accepts days and converts automatically.

---

## Author

**Prateek Srivastav** — built as an end-to-end ML project exploring unsupervised learning on real logistics data.

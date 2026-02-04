
# UrbanPulse — NYC Bike-Share Demand Predictor



![Python](https://img.shields.io/badge/Python-3.10-blue)

![DuckDB](https://img.shields.io/badge/DuckDB-OLAP-yellow)

![XGBoost](https://img.shields.io/badge/Model-XGBoost-orange)

![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red)

![GitOps](https://img.shields.io/badge/GitOps-GitHub%20Actions-green)



UrbanPulse is a small end-to-end data project that predicts hourly bike demand in NYC using Citi Bike trips and public weather data.

It’s built to be simple, reproducible and cheap to run: everything lives in GitHub, DuckDB handles analytics, and Streamlit serves the app.



---



## Problem



Bike-sharing systems constantly need to rebalance stations:



* Residential stations empty out in the morning.

* **Machine Learning Model**
  XGBoost regressor trained on time, weather, location, and historical demand features.

* **Interactive Prediction**
  User inputs (hour, temperature, rain, day, month) generate real-time station-level forecasts.

* **Scheduled Updates**
  GitHub Actions refreshes the data and retrains the model every month.

* **Basic Data Quality Checks**
  Row counts, schema, null values, and last update timestamp are visible in the app.

---

## Architecture

The system is fully local and Git-based, with no external warehouse or paid services.

* **ETL** runs in GitHub Actions
* **DuckDB** handles transformations
* **Parquet** stores the final dataset
* **Streamlit** serves the dashboard

```text
Citi Bike S3 + Open-Meteo
        ↓
   GitHub Actions
        ↓
     DuckDB ETL
        ↓
  daily_demand.parquet
        ↓
  Model Training → model.pkl
        ↓
     Streamlit App
```

---

## Code Structure

* **src/etl.py**
  Downloads data, cleans it, aggregates hourly demand, and joins weather.

* **src/train.py**
  Loads the Parquet file, engineers features, trains XGBoost, and saves the model.

* **src/features.py**
  Central feature logic (time encoding, lags, encodings).

* **src/inference.py**
  Builds feature rows for new user inputs.

* **src/plots.py**
  Chart and map utilities (Altair + PyDeck).

* **src/loader.py**
  Cached loading helpers for Streamlit.

* **src/config.py**
  Paths, feature lists, constants.

* **app.py**
  Main Streamlit application (Pipeline, EDA, Prediction tabs).

---

## Data Sources

* **Citi Bike Trip Data**
  Monthly CSV files from the public S3 bucket.

* **Open-Meteo Weather API**
  Hourly temperature, precipitation, and wind for NYC.

* **Processed Dataset**
  Hourly demand per station, merged with weather and time features, stored in:

  ```
  data/processed/daily_demand.parquet
  ```

---

## Model

* **Algorithm:** XGBoost Regressor
* **Target:** `trip_count` (hourly trips per station)

### Features

* **Cyclical time**

  * `hour_sin`, `hour_cos`
  * `day_sin`, `day_cos`
  * `month_sin`, `month_cos`

* **Binary**

  * `is_weekend`
  * `is_raining`

* **Continuous**

  * `temperature`
  * `start_lat`, `start_lng`

* **Encodings**

  * `station_avg_demand`

* **Lag features**

  * `trip_count_lag1`
  * `trip_count_lag2`
  * `trip_count_lag3`
  * `trip_count_rolling3`

Training data is ordered by time and lag features are computed per station to avoid leakage.
At inference time, lag values are approximated using precomputed historical statistics.

---

## Installation

```bash
git clone https://github.com/your-username/UrbanPulse-NYC-Predictor.git
cd UrbanPulse-NYC-Predictor
pip install -r requirements.txt
```

Run ETL:

```bash
python -m src.etl
```

Train model:

```bash
python -m src.train
```

Launch app:

```bash
streamlit run app.py
```

---

## App Sections

### Pipeline

* File size, rows, schema, null counts
* Last update timestamp

### EDA

* 3D density map
* Hourly trends (weekday vs weekend)
* Weather vs demand
* Top stations

### Prediction

* Input sliders for time and weather
* Station-level demand forecast
* Hotspot map and top 10 stations
* City-wide demand KPI

---

## Tech Stack

* Python
* DuckDB
* XGBoost
* Scikit-learn
* Streamlit
* PyDeck, Altair
* GitHub Actions

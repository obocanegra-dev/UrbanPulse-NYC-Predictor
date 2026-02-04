# UrbanPulse — NYC Bike Demand Predictor

[![Live App](https://img.shields.io/badge/Live%20App-Streamlit-blue)](https://urbanpulse-nyc-predictor.streamlit.app/)


UrbanPulse is a Streamlit application for forecasting hourly Citi Bike demand in New York City.
It combines historical trip data with public weather data and a machine learning model to estimate demand per station.

The focus of the project is not just prediction, but also automation, reproducibility, and simple deployment using open-source tools.

---

## Overview

UrbanPulse pulls monthly Citi Bike trip data and joins it with hourly weather data from Open-Meteo.
The processed dataset is stored in Parquet format and used both for model training and for an interactive dashboard.

The app allows you to:

* Inspect and validate the pipeline output
* Explore demand patterns visually
* Simulate scenarios and predict demand by station

It is meant as a practical example of a lightweight, end-to-end data product.

---

## Features

* **Automated ETL**
  Monthly pipeline that downloads raw data, joins weather, aggregates trips, and writes a compact Parquet file using DuckDB.

* **Exploratory Analysis (EDA)**
  Interactive maps and charts: station density, hourly demand curves, weather impact, and busiest stations.

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

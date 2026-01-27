# UrbanPulse: NYC Bike-Share Demand Predictor

![Python](https://img.shields.io/badge/Python-3.10-blue)
![DuckDB](https://img.shields.io/badge/DuckDB-OLAP-yellow)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red)
![GitOps](https://img.shields.io/badge/GitOps-GitHub%20Actions-green)

UrbanPulse is an end-to-end data product for predicting bike-share demand in New York City.
It combines historical Citi Bike data with weather information to support rebalancing decisions.

## Problem

Bike-sharing systems face a recurring rebalancing issue:

- Residential stations tend to empty during morning hours.
- Business districts accumulate bikes during the same period.
- The pattern reverses in the evening.
- Weather conditions significantly affect usage.

The goal of this project is to automate data ingestion and provide a simple interface to explore and predict demand.

## Architecture

The pipeline is built around GitHub Actions and DuckDB, avoiding external cloud services.

```mermaid
graph LR
    A[Citi Bike S3] -->|Extract| B[GitHub Actions]
    C[Open-Meteo API] -->|Extract| B
    B -->|"Transform (DuckDB)"| D["Daily_Demand.parquet"]
    D -->|Load| E[Streamlit App]
    E -->|Inference| F[Random Forest Model]
````

Flow:

1. GitHub Actions runs a scheduled job to download raw trip data.
2. DuckDB processes and joins weather data.
3. Aggregated data is stored as a Parquet file in the repository.
4. Streamlit loads the Parquet file and model for visualization and inference.

## Repository Structure

```text
├── .github/workflows  # ETL automation
├── data/
│   ├── raw/           # Ignored raw CSVs
│   └── processed/     # Aggregated parquet data
├── src/
│   ├── etl.py         # Data pipeline
│   └── train.py       # Model training
├── app.py             # Streamlit dashboard
├── requirements.txt
└── README.md
```

## Setup

Clone the repository:

```bash
git clone https://github.com/obocanegra-dev/UrbanPulse-NYC-Predictor.git
cd UrbanPulse-NYC-Predictor
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the ETL:

```bash
python src/etl.py
```

Train the model:

```bash
python src/train.py
```

Launch the app:

```bash
streamlit run app.py
```

## Tech Stack

* DuckDB for analytical processing.
* GitHub Actions for scheduled ETL.
* Streamlit and PyDeck for visualization.
* Scikit-learn for demand prediction.

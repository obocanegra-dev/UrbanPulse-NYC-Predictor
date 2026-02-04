import os

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")

INPUT_FILE = os.path.join(PROCESSED_DIR, "daily_demand.parquet")
MODEL_FILE = os.path.join(BASE_DIR, "model.pkl")

# External APIs
WEATHER_URL = "https://archive-api.open-meteo.com/v1/archive"
CITIBIKE_S3 = "https://s3.amazonaws.com/tripdata"

# Model Config
TARGET_COL = "trip_count"
FEATURES = [
    "hour_sin", "hour_cos",
    "day_sin", "day_cos",
    "month_sin", "month_cos",
    "is_weekend",
    "temperature",
    "is_raining",
    "station_avg_demand",
    "start_lat",
    "start_lng",
    "trip_count_lag1",
    "trip_count_lag2",
    "trip_count_lag3",
    "trip_count_rolling3",
]

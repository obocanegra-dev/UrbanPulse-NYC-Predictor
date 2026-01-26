import os
import requests
import zipfile
import duckdb
import pandas as pd
from datetime import datetime, timedelta

# Config
YEAR = 2025
MONTH = 1
MONTH_STR = f"{YEAR}{MONTH:02d}"
CITIBIKE_URL = f"https://s3.amazonaws.com/tripdata/{MONTH_STR}-citibike-tripdata.zip"

LAT = 40.71
LON = -74.01

RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"
OUTPUT_FILE = os.path.join(PROCESSED_DIR, "daily_demand.parquet")


def ensure_directories():
    os.makedirs(RAW_DIR, exist_ok=True)
    os.makedirs(PROCESSED_DIR, exist_ok=True)


def fetch_weather_data():
    print(f"Fetching weather data for {MONTH_STR}")

    start_date = f"{YEAR}-{MONTH:02d}-01"
    if MONTH == 12:
        next_month = datetime(YEAR + 1, 1, 1)
    else:
        next_month = datetime(YEAR, MONTH + 1, 1)
    end_date = (next_month - timedelta(days=1)).strftime("%Y-%m-%d")

    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": LAT,
        "longitude": LON,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": "temperature_2m,precipitation,rain,wind_speed_10m",
        "timezone": "America/New_York"
    }

    r = requests.get(url, params=params)
    hourly = r.json()["hourly"]

    return pd.DataFrame({
        "time": pd.to_datetime(hourly["time"]),
        "temperature": hourly["temperature_2m"],
        "precipitation": hourly["precipitation"],
        "wind_speed": hourly["wind_speed_10m"]
    })


def fetch_citibike_data():
    print("Downloading Citi Bike data")

    zip_path = os.path.join(RAW_DIR, "citibike_data.zip")
    r = requests.get(CITIBIKE_URL, stream=True)

    with open(zip_path, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)

    print("Extracting CSVs")
    csv_files = []
    with zipfile.ZipFile(zip_path, "r") as z:
        for name in z.namelist():
            if name.endswith(".csv") and "__MACOSX" not in name:
                z.extract(name, RAW_DIR)
                csv_files.append(os.path.join(RAW_DIR, name))

    return csv_files


def transform_and_load(df_weather):
    print("Running DuckDB transformations")

    con = duckdb.connect(database=":memory:")
    con.register("weather_data", df_weather)

    csv_glob_path = os.path.join(RAW_DIR, "*.csv")

    query = f"""
    WITH bike_clean AS (
        SELECT
            start_station_name,
            start_lat,
            start_lng,
            started_at,
            ended_at,
            date_diff('second', started_at, ended_at) AS duration_sec
        FROM read_csv_auto('{csv_glob_path}', ignore_errors=true)
    ),
    bike_agg AS (
        SELECT
            start_station_name,
            start_lat,
            start_lng,
            date_trunc('hour', CAST(started_at AS TIMESTAMP)) AS hour_timestamp,
            COUNT(*) AS trip_count
        FROM bike_clean
        WHERE duration_sec >= 60
          AND start_station_name IS NOT NULL
        GROUP BY 1,2,3,4
    )
    SELECT
        b.start_station_name,
        b.start_lat,
        b.start_lng,
        b.hour_timestamp,
        hour(b.hour_timestamp) AS hour_of_day,
        dayofweek(b.hour_timestamp) AS day_of_week,
        w.temperature,
        w.precipitation,
        w.wind_speed,
        b.trip_count
    FROM bike_agg b
    LEFT JOIN weather_data w ON b.hour_timestamp = w.time
    WHERE month(b.hour_timestamp) = {MONTH}
    """

    print(f"Saving parquet to {OUTPUT_FILE}")
    con.execute(f"COPY ({query}) TO '{OUTPUT_FILE}' (FORMAT 'parquet')")
    con.close()


def main():
    ensure_directories()
    df_weather = fetch_weather_data()
    csv_files = fetch_citibike_data()
    transform_and_load(df_weather)

    for f in csv_files:
        if os.path.exists(f):
            os.remove(f)


if __name__ == "__main__":
    main()

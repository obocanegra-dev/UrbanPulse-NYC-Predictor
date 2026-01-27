import os
import requests
import zipfile
import duckdb
import pandas as pd
from datetime import datetime, timedelta


def get_target_month():
    today = datetime.now()
    first_of_month = today.replace(day=1)
    last_month = first_of_month - timedelta(days=1)
    return last_month.year, last_month.month


YEAR, MONTH = get_target_month()
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
        next_month_start = datetime(YEAR + 1, 1, 1)
    else:
        next_month_start = datetime(YEAR, MONTH + 1, 1)

    end_date = (next_month_start - timedelta(days=1)).strftime("%Y-%m-%d")

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
    r.raise_for_status()
    hourly = r.json()["hourly"]

    return pd.DataFrame({
        "time": pd.to_datetime(hourly["time"]),
        "temperature": hourly["temperature_2m"],
        "precipitation": hourly["precipitation"],
        "wind_speed": hourly["wind_speed_10m"]
    })


def fetch_citibike_data():
    print(f"Downloading Citi Bike data: {MONTH_STR}")

    zip_path = os.path.join(RAW_DIR, "citibike_data.zip")
    r = requests.get(CITIBIKE_URL, stream=True)

    if r.status_code == 404:
        raise FileNotFoundError(f"Data for {MONTH_STR} not available")

    with open(zip_path, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)

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

    history_exists = os.path.exists(OUTPUT_FILE)
    if history_exists:
        con.execute(
            f"CREATE OR REPLACE VIEW history_data AS SELECT * FROM '{OUTPUT_FILE}'"
        )

    csv_glob_path = os.path.join(RAW_DIR, "*.csv")

    new_data_query = f"""
    WITH bike_clean AS (
        SELECT
            start_station_name,
            start_lat,
            start_lng,
            started_at,
            ended_at,
            date_diff(
              'second',
              CAST(started_at AS TIMESTAMP),
              CAST(ended_at AS TIMESTAMP)
            ) AS duration_sec
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
    LEFT JOIN weather_data w
        ON b.hour_timestamp = w.time
    WHERE month(b.hour_timestamp) = {MONTH}
    """

    con.execute(f"CREATE OR REPLACE VIEW new_data_view AS {new_data_query}")

    if history_exists:
        final_query = """
        SELECT DISTINCT * FROM (
            SELECT * FROM history_data
            UNION ALL
            SELECT * FROM new_data_view
        )
        ORDER BY hour_timestamp DESC
        """
    else:
        final_query = "SELECT * FROM new_data_view ORDER BY hour_timestamp DESC"

    print(f"Saving data to {OUTPUT_FILE}")
    con.execute(f"COPY ({final_query}) TO '{OUTPUT_FILE}' (FORMAT 'parquet')")
    con.close()


def main():
    ensure_directories()

    try:
        df_weather = fetch_weather_data()
        csv_files = fetch_citibike_data()
        transform_and_load(df_weather)

        for f in csv_files:
            if os.path.exists(f):
                os.remove(f)

        zip_path = os.path.join(RAW_DIR, "citibike_data.zip")
        if os.path.exists(zip_path):
            os.remove(zip_path)

        print("ETL complete")

    except FileNotFoundError as e:
        print(str(e))
    except Exception as e:
        print(f"Error: {e}")
        exit(1)


if __name__ == "__main__":
    main()

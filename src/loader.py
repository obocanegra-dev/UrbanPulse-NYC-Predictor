import streamlit as st
import duckdb
import joblib
import os
import pandas as pd
from src.config import INPUT_FILE, MODEL_FILE

@st.cache_data
def get_pipeline_stats():
    """Returns the exact dictionary structure expected by Tab 1."""
    with duckdb.connect() as con:
        stats_query = f"""
        WITH max_ts AS (
            SELECT MAX(hour_timestamp) AS max_hour FROM '{INPUT_FILE}'
        )
        SELECT
            COUNT(*) AS total_rows,
            ANY_VALUE(max_hour) AS last_update,
            COUNT(*) FILTER (WHERE hour_timestamp >= max_hour - INTERVAL 30 DAY) AS recent_volume
        FROM '{INPUT_FILE}', max_ts
        """
        stats = con.execute(stats_query).fetchone()

        null_check = f"""
            SELECT COUNT(*) FROM '{INPUT_FILE}'
            WHERE trip_count IS NULL OR start_station_name IS NULL OR temperature IS NULL
        """
        null_count = con.execute(null_check).fetchone()[0]
        schema_df = con.execute(f"DESCRIBE SELECT * FROM '{INPUT_FILE}'").df()
        sample_df = con.execute(f"SELECT * FROM '{INPUT_FILE}' LIMIT 50").df()

    return {
        "total_rows": stats[0],
        "last_update": stats[1],
        "recent_volume": stats[2],
        "null_count": null_count,
        "schema": schema_df,
        "sample": sample_df
    }

@st.cache_data
def get_aggregated_map_data():
    with duckdb.connect() as con:
        return con.execute(f"""
            SELECT start_station_name, AVG(start_lat) AS start_lat, AVG(start_lng) AS start_lng, SUM(trip_count) AS total_trips
            FROM '{INPUT_FILE}' GROUP BY start_station_name
        """).df()

@st.cache_data
def get_top_stations():
    with duckdb.connect() as con:
        return con.execute(f"""
            SELECT start_station_name, SUM(trip_count) as total_trips
            FROM '{INPUT_FILE}' GROUP BY start_station_name ORDER BY total_trips DESC LIMIT 10
        """).df()

@st.cache_data
def get_hourly_demand():
    with duckdb.connect() as con:
        return con.execute(f"""
            SELECT hour_of_day AS 'Hour of Day', SUM(trip_count) as 'Total Trips'
            FROM '{INPUT_FILE}' GROUP BY hour_of_day ORDER BY hour_of_day
        """).df()

@st.cache_data
def get_eda_stats():
    with duckdb.connect() as con:
        hourly_df = con.execute(f"""
            SELECT hour_of_day, CASE WHEN day_of_week IN (0, 6) THEN 'Weekend' ELSE 'Weekday' END as day_type,
            AVG(trip_count) as avg_trips FROM '{INPUT_FILE}' GROUP BY 1, 2 ORDER BY 1, 2
        """).df()

        weather_df = con.execute(f"""
            SELECT temperature, precipitation, CASE WHEN precipitation > 0.1 THEN 'Rainy' ELSE 'Clear' END as weather_cond,
            SUM(trip_count) as total_trips FROM '{INPUT_FILE}'
            GROUP BY hour_timestamp, temperature, precipitation HAVING total_trips > 10
        """).df()
    return hourly_df, weather_df

@st.cache_resource
def load_model_bundle():
    return joblib.load(MODEL_FILE) if os.path.exists(MODEL_FILE) else None

@st.cache_data
def get_unique_stations():
    with duckdb.connect() as con:
        return con.execute(f"""
            SELECT start_station_name, AVG(start_lat) AS start_lat, AVG(start_lng) AS start_lng
            FROM '{INPUT_FILE}' GROUP BY start_station_name
        """).df()

@st.cache_data
def get_historical_hourly_demand(hour):
    with duckdb.connect() as con:
        val = con.execute(f"""
            WITH hourly_totals AS (
                SELECT date_trunc('day', hour_timestamp) as date, SUM(trip_count) as total_trips
                FROM '{INPUT_FILE}' WHERE hour_of_day = {hour} GROUP BY 1
            )
            SELECT AVG(total_trips) FROM hourly_totals
        """).fetchone()[0]
    return val or 0

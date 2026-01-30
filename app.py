import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk
import altair as alt
import joblib
import os
from datetime import datetime
import duckdb

st.set_page_config(page_title="UrbanPulse - NYC Bike Demand", layout="wide")

DATA_PATH = "data/processed/daily_demand.parquet"
MODEL_PATH = "model.pkl"


# ----------------------------
# Loaders
# ----------------------------

@st.cache_data
def get_pipeline_stats():
    with duckdb.connect() as con:
        count = con.execute(f"SELECT COUNT(*) FROM '{DATA_PATH}'").fetchone()[0]
        sample = con.execute(f"SELECT * FROM '{DATA_PATH}' LIMIT 100").df()
    return count, sample


@st.cache_data
def get_aggregated_map_data():
    with duckdb.connect() as con:
        query = f"""
            SELECT start_lat, start_lng, start_station_name, SUM(trip_count) as total_trips
            FROM '{DATA_PATH}'
            GROUP BY 1, 2, 3
        """
        return con.execute(query).df()


@st.cache_data
def get_top_stations():
    with duckdb.connect() as con:
        query = f"""
            SELECT start_station_name, SUM(trip_count) as total_trips
            FROM '{DATA_PATH}'
            GROUP BY start_station_name
            ORDER BY total_trips DESC
            LIMIT 10
        """
        return con.execute(query).df()

@st.cache_data
def get_hourly_demand():
    with duckdb.connect() as con:
        query = f"""
            SELECT hour_of_day AS 'Hour of Day', SUM(trip_count) as 'Total Trips'
            FROM '{DATA_PATH}'
            GROUP BY hour_of_day
            ORDER BY hour_of_day
        """
        return con.execute(query).df()


@st.cache_resource
def load_model_bundle():
    return joblib.load(MODEL_PATH) if os.path.exists(MODEL_PATH) else None


@st.cache_data
def get_unique_stations():
    with duckdb.connect() as con:
        query = f"""
            SELECT DISTINCT start_station_name, start_lat, start_lng
            FROM '{DATA_PATH}'
        """
        return con.execute(query).df()


# ----------------------------
# Feature helpers
# ----------------------------

@st.cache_data
def prepare_hourly_features(hourly_stats):
    lag1_map, rolling3_map = {}, {}

    for h in range(24):
        prev = (h - 1) % 24

        lag1_map[h] = (
            hourly_stats[hourly_stats["hour_of_day"] == prev]
            .set_index("start_station_name")["avg_hourly_trips"]
        )

        hours = [(h - i) % 24 for i in (1, 2, 3)]
        rolling3_map[h] = (
            hourly_stats[hourly_stats["hour_of_day"].isin(hours)]
            .groupby("start_station_name")["avg_hourly_trips"]
            .mean()
        )

    return lag1_map, rolling3_map


def cyclical(val, period):
    return np.sin(2 * np.pi * val / period), np.cos(2 * np.pi * val / period)


# ----------------------------
# App
# ----------------------------

st.title("UrbanPulse - NYC Bike Demand Predictor")
st.markdown("Bike demand forecasting for Citi Bike NYC.")

bundle = load_model_bundle()

tab1, tab2, tab3 = st.tabs(["Pipeline", "EDA", "Prediction"])


# ----------------------------
# TAB 1 – Pipeline
# ----------------------------

with tab1:
    st.header("Pipeline Status")

    total_rows, sample_df = get_pipeline_stats()
    stats = os.stat(DATA_PATH)

    c1, c2, c3 = st.columns(3)
    c1.metric("File size (MB)", f"{stats.st_size / (1024 * 1024):.2f}")
    c2.metric("Rows", f"{total_rows:,}")
    c3.metric("Last update", datetime.fromtimestamp(stats.st_mtime).strftime("%Y-%m-%d %H:%M:%S"))

    st.subheader("Data Preview (First 10 rows)")
    st.dataframe(sample_df.head(10), width='stretch')


# ----------------------------
# TAB 2 – EDA
# ----------------------------

with tab2:
    st.header("Exploratory Analysis")

    map_data = get_aggregated_map_data()

    col_map, col_stats = st.columns([3, 2])

    with col_map:
        max_trips = map_data["total_trips"].max()
        map_data["norm_height"] = (map_data["total_trips"] / max_trips) * 2000

        layer = pdk.Layer(
            "ColumnLayer",
            map_data,
            get_position=["start_lng", "start_lat"],
            elevation_scale=1,
            radius=50,
            get_fill_color=[255, 165, 0, 100],
            extruded=True,
            pickable=True,
            get_elevation="norm_height",
        )

        view_state = pdk.ViewState(
            latitude=40.74, longitude=-73.99, zoom=10.5, pitch=60
        )

        st.pydeck_chart(
            pdk.Deck(
                layers=[layer],
                initial_view_state=view_state,
                tooltip={"html": "<b>Station:</b> {start_station_name}<br/><b>Trips:</b> {total_trips}"},
            )
        )

    with col_stats:
        st.subheader("Top 10 Stations")
        top_stations = get_top_stations()

        chart = (
            alt.Chart(top_stations)
            .mark_bar()
            .encode(
                x=alt.X("total_trips:Q", title="Trips"),
                y=alt.Y("start_station_name:N", sort="-x", title="Station"),
            )
        )
        st.altair_chart(chart, width='stretch')
        st.divider()

        st.subheader("Demand by Hour")
        hourly_data = get_hourly_demand()
        st.bar_chart(hourly_data.set_index("Hour of Day"))


# ----------------------------
# TAB 3 – Prediction
# ----------------------------

with tab3:
    st.header("Demand Prediction")

    if bundle is None:
        st.warning("Model not found. Train it first.")
        st.stop()

    model = bundle["model"]
    required_features = bundle["features"]
    station_stats = bundle["station_stats"]
    hourly_stats = bundle["hourly_stats"]

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        input_hour = st.slider("Hour", 0, 23, 17)
    with c2:
        input_temp = st.slider("Temperature (°C)", -10, 40, 25)
    with c3:
        is_raining_val = 1 if st.radio("Raining?", ["No", "Yes"]) == "Yes" else 0
    with c4:
        input_day = st.selectbox(
            "Day",
            options=list(range(7)),
            format_func=lambda x: ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"][x],
            index=3,
        )

    stations = get_unique_stations()
    lag1_map, rolling3_map = prepare_hourly_features(hourly_stats)

    X_pred = stations.copy()
    X_pred["hour_of_day"] = input_hour
    X_pred["day_of_week"] = input_day
    X_pred["temperature"] = input_temp
    X_pred["is_raining"] = is_raining_val
    X_pred["is_weekend"] = int(input_day in [0, 6])

    X_pred["hour_sin"], X_pred["hour_cos"] = cyclical(input_hour, 24)
    X_pred["day_sin"], X_pred["day_cos"] = cyclical(input_day, 7)

    avg_global = station_stats["station_avg_demand"].mean()
    station_avg = station_stats.set_index("start_station_name")["station_avg_demand"]

    X_pred["station_avg_demand"] = (
        X_pred["start_station_name"].map(station_avg).fillna(avg_global)
    )

    X_pred["trip_count_lag1"] = (
        X_pred["start_station_name"].map(lag1_map[input_hour]).fillna(0)
    )
    X_pred["trip_count_rolling3"] = (
        X_pred["start_station_name"].map(rolling3_map[input_hour]).fillna(0)
    )

    try:
        X_pred = X_pred[required_features]
    except KeyError as e:
        st.error(f"Missing features for model: {e}")
        st.stop()

    preds = model.predict(X_pred)
    stations["predicted_demand"] = preds.clip(min=0)
    stations["radius_norm"] = (np.sqrt(stations["predicted_demand"]) * 10)

    def get_color(val):
        if val > 15:
            return [255, 0, 0, 180]
        elif val > 5:
            return [255, 165, 0, 160]
        return [0, 128, 255, 140]

    stations["color"] = stations["predicted_demand"].apply(get_color)

    hotspots = (
        stations.sort_values("predicted_demand", ascending=False)
        .head(10)[["start_station_name", "predicted_demand"]]
    )

    col_map, col_table = st.columns([3, 1])

    with col_map:
        layer = pdk.Layer(
            "ScatterplotLayer",
            stations,
            get_position=["start_lng", "start_lat"],
            get_radius="radius_norm",
            get_fill_color="color",
            pickable=True,
        )

        view_state = pdk.ViewState(
            latitude=40.74, longitude=-73.99, zoom=11.5
        )

        st.pydeck_chart(
            pdk.Deck(
                layers=[layer],
                initial_view_state=view_state,
                tooltip={
                    "html": "<b>{start_station_name}</b><br/>Demand: {predicted_demand}"
                },
            )
        )

    with col_table:
        st.subheader("Top Predicted")
        st.dataframe(
            hotspots,
            hide_index=True,
            width="stretch",
            column_config={
                "start_station_name": st.column_config.TextColumn("Station"),
                "predicted_demand": st.column_config.NumberColumn(
                    "Predicted Demand", format="%.1f"
                ),
            },
        )

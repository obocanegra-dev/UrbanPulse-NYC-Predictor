import streamlit as st
import pandas as pd
import pydeck as pdk
import joblib
import os
from datetime import datetime
import duckdb

st.set_page_config(
    page_title="UrbanPulse - NYC Bike Demand",
    layout="wide"
)

DATA_PATH = "data/processed/daily_demand.parquet"
MODEL_PATH = "model.pkl"


@st.cache_data
def load_data():
    if not os.path.exists(DATA_PATH):
        return None
    con = duckdb.connect()
    df = con.execute(f"SELECT * FROM '{DATA_PATH}'").df()
    con.close()
    return df


@st.cache_resource
def load_model_bundle():
    if not os.path.exists(MODEL_PATH): return None
    return joblib.load(MODEL_PATH)


def get_unique_stations(df):
    return df[["start_station_name", "start_lat", "start_lng"]].drop_duplicates()


st.title("UrbanPulse - NYC Bike Demand Predictor")
st.markdown("Bike demand forecasting for Citi Bike NYC.")

df = load_data()
bundle = load_model_bundle()

if df is None:
    st.error("Processed data not found. Run ETL first.")
    st.stop()

tab1, tab2, tab3 = st.tabs(["Pipeline", "EDA", "Prediction"])


with tab1:
    st.header("Pipeline Status")

    col1, col2, col3 = st.columns(3)
    stats = os.stat(DATA_PATH)

    col1.metric("File size (MB)", f"{stats.st_size / (1024 * 1024):.2f}")
    col2.metric("Rows", f"{len(df):,}")
    col3.metric(
        "Last update",
        datetime.fromtimestamp(stats.st_mtime).strftime("%Y-%m-%d %H:%M:%S")
    )

    st.dataframe(df.head(10), use_container_width=True)


with tab2:
    st.header("Exploratory Analysis")

    col1, col2 = st.columns([2, 1])

    with col1:
        map_data = (
            df.groupby(["start_lat", "start_lng"])
            .agg({"trip_count": "sum"})
            .reset_index()
        )

        layer = pdk.Layer(
            "HexagonLayer",
            map_data,
            get_position=["start_lng", "start_lat"],
            elevation_scale=50,
            elevation_range=[0, 3000],
            extruded=True,
            pickable=True,
            get_elevation="trip_count"
        )

        view_state = pdk.ViewState(
            latitude=40.74,
            longitude=-73.99,
            zoom=11,
            pitch=50
        )

        st.pydeck_chart(
            pdk.Deck(
                layers=[layer],
                initial_view_state=view_state,
                tooltip={"html": "<b>Trips:</b> {elevationValue}"}
            )
        )

    with col2:
        top_stations = (
            df.groupby("start_station_name")["trip_count"]
            .sum()
            .nlargest(10)
            .sort_values()
        )
        st.bar_chart(top_stations, horizontal=True)

    hourly_counts = df.groupby("hour_of_day")["trip_count"].sum()
    st.bar_chart(hourly_counts)


with tab3:
    st.header("Demand Prediction")

    if bundle is None:
        st.warning("Model not found. Train it first.")
    else:
        model = bundle["model"]
        required_features = bundle["features"]

        col1, col2, col3, col4 = st.columns(4)
        with col1: input_hour = st.slider("Hour", 0, 23, 17)
        with col2: input_temp = st.slider("Temperature (°C)", -10, 40, 25)
        with col3:
            input_rain = st.radio("Raining?", ["No", "Yes"])
            is_raining_val = 1 if input_rain == "Yes" else 0
        with col4:
            input_day = st.selectbox(
                "Day",
                options=[0,1,2,3,4,5,6],
                format_func=lambda x: ["Sun","Mon","Tue","Wed","Thu","Fri","Sat"][x],
                index=3
            )

        stations = get_unique_stations(df)

        X_pred = stations.copy()
        X_pred["hour_of_day"] = input_hour
        X_pred["day_of_week"] = input_day
        X_pred["temperature"] = input_temp
        X_pred["is_raining"] = is_raining_val

        try:
            X_pred = X_pred[required_features]
        except KeyError as e:
            st.error(f"Error: The model expects features that are missing: {e}")
            st.stop()

        preds = model.predict(X_pred)
        stations["predicted_demand"] = preds.clip(min=0)

        def get_color(val):
            if val > 15:
                return [255, 0, 0, 180]
            elif val > 5:
                return [255, 165, 0, 160]
            else:
                return [0, 128, 255, 140]

        stations["color"] = stations["predicted_demand"].apply(get_color)

        layer = pdk.Layer(
            "ScatterplotLayer",
            stations,
            get_position=["start_lng", "start_lat"],
            get_radius="predicted_demand * 10",
            get_fill_color="color",
            pickable=True
        )

        view_state = pdk.ViewState(
            latitude=40.74,
            longitude=-73.99,
            zoom=11.5
        )

        st.pydeck_chart(
            pdk.Deck(
                layers=[layer],
                initial_view_state=view_state,
                tooltip={
                    "html": "<b>{start_station_name}</b><br/>Demand: {predicted_demand}"
                }
            )
        )

        hotspots = (
            stations.sort_values("predicted_demand", ascending=False)
            .head(5)
        )
        st.table(hotspots[["start_station_name", "predicted_demand"]])

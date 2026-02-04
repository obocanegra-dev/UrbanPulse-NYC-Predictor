import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk
import altair as alt
import sys
import os
from datetime import datetime
from src.config import INPUT_FILE
from src import loader, inference, plots

sys.path.append(os.path.dirname(__file__))

st.set_page_config(page_title="UrbanPulse - NYC Bike Demand", layout="wide")

# ----------------------------
# App
# ----------------------------

st.title("UrbanPulse - NYC Bike Demand Predictor")
st.markdown("Bike demand forecasting for Citi Bike NYC.")

bundle = loader.load_model_bundle()

tab1, tab2, tab3 = st.tabs(["Pipeline", "EDA", "Prediction"])


# ----------------------------
# TAB 1 – Pipeline
# ----------------------------

with tab1:
    st.header("Pipeline Health & Status")

    # Use the new loader
    stats = loader.get_pipeline_stats()

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            label="Total Rows processed",
            value=f"{stats['total_rows']:,}",
            delta=f"+{stats['recent_volume']:,} (last 30 days)"
        )

    with col2:
        last_date = stats['last_update']
        days_lag = (datetime.now() - last_date).days

        st.metric(
            label="Data Freshness",
            value=last_date.strftime('%Y-%m-%d'),
            delta=f"{days_lag} days ago",
            delta_color="inverse"
        )

    with col3:
        nulls = stats['null_count']
        st.metric(
            label="Failed Records (Nulls)",
            value=nulls,
            delta="Perfect Quality" if nulls == 0 else "Data Quality Issue",
            delta_color="normal" if nulls == 0 else "inverse"
        )

    with col4:
        file_size = os.path.getsize(INPUT_FILE) / (1024 * 1024)
        st.metric(
            label="Storage Usage (Parquet)",
            value=f"{file_size:.2f} MB",
            delta="Optimized"
        )

    st.markdown("---")

    col_schema, col_sample = st.columns([1, 2])

    with col_schema:
        st.subheader("Data Schema")
        st.caption("DuckDB auto-inferred types")
        st.dataframe(
            stats['schema'][['column_name', 'column_type']],
            width='stretch',
            hide_index=True,
            height=400
        )

    with col_sample:
        st.subheader("Latest Data Preview")
        st.caption("Sample of the last ingested rows")
        st.dataframe(
            stats['sample'],
            width='stretch',
            height=400,
            hide_index=True
        )


# ----------------------------
# TAB 2 – EDA
# ----------------------------

with tab2:
    st.header("Exploratory Analysis")

    map_data = loader.get_aggregated_map_data()

    st.pydeck_chart(plots.plot_map_density(map_data), width='stretch')

    hourly_df, weather_df = loader.get_eda_stats()

    st.subheader("Commuters vs. Tourists Pattern")
    st.caption("Average trips per station by hour of day.")

    st.altair_chart(plots.plot_hourly_trend(hourly_df), width='stretch')

    with st.expander("💡 Business Insight"):
        st.markdown("""
        * **Weekdays (Blue):** Clear peaks at **8 AM** and **6 PM**. This is the classic "Commuter" pattern (people going to/from work).
        * **Weekends (Orange):** Smooth bell curve peaking at **2 PM**. This indicates leisure/tourist usage.
        * **Action:** Rebalancing trucks should focus on business districts at 9 AM on weekdays, but focus on parks/tourist spots at noon on weekends.
        """)

    st.markdown("---")

    st.subheader("The Impact of Weather")
    st.caption("Total city-wide demand vs. Temperature & Rain.")

    st.altair_chart(plots.plot_weather_impact(weather_df), width='stretch')

    with st.expander("💡 Business Insight"):
        st.markdown("""
        * **Temperature Correlation:** There is a strong positive correlation. As temperature rises, demand increases.
        * **Rain Effect (Red Dots):** Notice how 'Rainy' points are consistently lower than 'Clear' points at the same temperature.
        * **Takeaway:** Even on a warm day (25°C), rain can cut demand by **30-50%**. The model must penalize heavy rain heavily.
        """)

    st.markdown("---")

    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Top 10 Stations")
        top_stations = loader.get_top_stations()

        chart = (
            alt.Chart(top_stations)
            .mark_bar()
            .encode(
                x=alt.X("total_trips:Q", title="Trips"),
                y=alt.Y("start_station_name:N", sort="-x", title="Station"),
            )
        )
        st.altair_chart(chart, width='stretch')

    with col_right:
        st.subheader("Demand by Hour")
        hourly_data = loader.get_hourly_demand()
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

    c1, c2, c3, c4, c5 = st.columns(5)
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

    with c5:
        input_month = st.selectbox(
            "Month",
            options=list(range(1, 13)),
            format_func=lambda x: datetime(2000, x, 1).strftime('%B'),
            index=datetime.now().month - 1
        )

    stations = loader.get_unique_stations()

    try:
        X_pred = inference.build_prediction_features(
            stations, bundle, input_hour, input_day,
            input_month, input_temp, is_raining_val
        )
    except ValueError as e:
        st.error(str(e))
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

    total_predicted_demand = stations["predicted_demand"].sum()

    avg_historical_total = loader.get_historical_hourly_demand(input_hour)

    diff = total_predicted_demand - avg_historical_total
    pct_diff = (diff / avg_historical_total) * 100 if avg_historical_total > 0 else 0

    kpi1, kpi2, kpi3 = st.columns(3)
    with kpi1:
        st.metric(
            "Total City Demand (Forecast)",
            f"{int(total_predicted_demand):,}",
            help="Sum of predicted trips across all stations"
        )
    with kpi2:
        st.metric(
            "Vs. Historical Avg",
            f"{int(avg_historical_total):,}",
            delta=f"{diff:+.0f} ({pct_diff:+.1f}%)",
            help=f"Comparison against average demand at {input_hour}:00"
        )
    with kpi3:
        if pct_diff > 25:
            st.warning("🔥 High Demand Surge expected!")
        elif pct_diff < -25:
            st.info("❄️ Lower demand than usual.")
        else:
            st.success("✅ Normal business operations.")

    st.markdown("---")

    st.pydeck_chart(plots.plot_prediction_map(stations), width='stretch')

    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Top Hotspots")
        hotspots = (
            stations.sort_values("predicted_demand", ascending=False)
            .head(10)[["start_station_name", "predicted_demand"]]
        )
        st.dataframe(
            hotspots,
            hide_index=True,
            width='stretch',
            column_config={
                "start_station_name": st.column_config.TextColumn("Station"),
                "predicted_demand": st.column_config.ProgressColumn(
                    "Demand", format="%.0f", min_value=0, max_value=max(stations["predicted_demand"])
                ),
            },
        )

    with col_right:
        st.subheader("Model Logic")
        st.caption("What factors influenced this prediction?")

        xai_chart = plots.plot_feature_importance(model, required_features)
        st.altair_chart(xai_chart, width='stretch', height=350)

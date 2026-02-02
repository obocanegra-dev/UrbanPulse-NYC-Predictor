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
        stats_query = f"""
        WITH max_ts AS (
            SELECT MAX(hour_timestamp) AS max_hour
            FROM '{DATA_PATH}'
        )
        SELECT
            COUNT(*) AS total_rows,
            ANY_VALUE(max_hour) AS last_update,
            COUNT(*) FILTER (
                WHERE hour_timestamp >= max_hour - INTERVAL 30 DAY
            ) AS recent_volume
        FROM '{DATA_PATH}', max_ts
        """


        stats = con.execute(stats_query).fetchone()
        total_rows, last_update, recent_volume = stats

        null_check_query = f"""
            SELECT COUNT(*)
            FROM '{DATA_PATH}'
            WHERE trip_count IS NULL
               OR start_station_name IS NULL
               OR temperature IS NULL
        """
        null_count = con.execute(null_check_query).fetchone()[0]

        schema_df = con.execute(f"DESCRIBE SELECT * FROM '{DATA_PATH}'").df()
        sample_df = con.execute(f"SELECT * FROM '{DATA_PATH}' LIMIT 50").df()

    return {
        "total_rows": total_rows,
        "last_update": last_update,
        "recent_volume": recent_volume,
        "null_count": null_count,
        "schema": schema_df,
        "sample": sample_df
    }


@st.cache_data
def get_aggregated_map_data():
    with duckdb.connect() as con:
        query = f"""
            SELECT
                start_station_name,
                AVG(start_lat) AS start_lat,
                AVG(start_lng) AS start_lng,
                SUM(trip_count) AS total_trips
            FROM '{DATA_PATH}'
            GROUP BY start_station_name
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


@st.cache_data
def get_eda_stats():
    with duckdb.connect() as con:
        hourly_query = f"""
            SELECT
                hour_of_day,
                CASE WHEN day_of_week IN (0, 6) THEN 'Weekend' ELSE 'Weekday' END as day_type,
                AVG(trip_count) as avg_trips
            FROM '{DATA_PATH}'
            GROUP BY 1, 2
            ORDER BY 1, 2
        """
        hourly_df = con.execute(hourly_query).df()

        weather_query = f"""
            SELECT
                temperature,
                precipitation,
                CASE WHEN precipitation > 0.1 THEN 'Rainy' ELSE 'Clear' END as weather_cond,
                SUM(trip_count) as total_trips
            FROM '{DATA_PATH}'
            GROUP BY hour_timestamp, temperature, precipitation
            HAVING total_trips > 10  -- Filtramos horas muertas (madrugada extrema) para limpiar el gráfico
        """
        weather_df = con.execute(weather_query).df()

    return hourly_df, weather_df


@st.cache_resource
def load_model_bundle():
    return joblib.load(MODEL_PATH) if os.path.exists(MODEL_PATH) else None


@st.cache_data
def get_unique_stations():
    with duckdb.connect() as con:
        query = f"""
            SELECT
                start_station_name,
                AVG(start_lat) AS start_lat,
                AVG(start_lng) AS start_lng
            FROM '{DATA_PATH}'
            GROUP BY start_station_name
        """
        return con.execute(query).df()


@st.cache_data
def get_historical_hourly_demand(hour):
    with duckdb.connect() as con:
        query = f"""
            WITH hourly_totals AS (
                SELECT date_trunc('day', hour_timestamp) as date, SUM(trip_count) as total_trips
                FROM '{DATA_PATH}'
                WHERE hour_of_day = {hour}
                GROUP BY 1
            )
            SELECT AVG(total_trips) FROM hourly_totals
        """
        return con.execute(query).fetchone()[0] or 0

def plot_feature_importance(model):
    if hasattr(model, 'feature_importances_'):
        imp = model.feature_importances_
        try:
            names = model.feature_names_in_
        except AttributeError:
            names = [f"Feat {i}" for i in range(len(imp))]

        df_imp = pd.DataFrame({'Feature': names, 'Importance': imp})
        df_imp = df_imp.sort_values('Importance', ascending=False).head(10) # Top 10

        chart = alt.Chart(df_imp).mark_bar(color='#FF4B4B').encode(
            x=alt.X('Importance', title='Weight in Decision'),
            y=alt.Y('Feature', sort='-x', title=None),
            tooltip=['Feature', 'Importance']
        ).properties(height=300)

        return chart
    else:
        return None


def get_feature_importance(model, feature_names):
    importance = model.feature_importances_
    df_imp = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importance
    }).sort_values('Importance', ascending=False).head(8)

    chart = alt.Chart(df_imp).mark_bar(color='#FF4B4B').encode(
        x=alt.X('Importance', title=None),
        y=alt.Y('Feature', sort='-x', title=None),
        tooltip=['Feature', 'Importance']
    ).properties(height=200, title="Model Drivers (Why?)")
    return chart

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
    st.header("Pipeline Health & Status")

    stats = get_pipeline_stats()

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
        file_size = os.path.getsize(DATA_PATH) / (1024 * 1024)
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

    map_data = get_aggregated_map_data()

    max_trips = map_data["total_trips"].max()
    map_data["norm_height"] = (map_data["total_trips"] / max_trips) * 5000

    layer = pdk.Layer(
        "ColumnLayer",
        map_data,
        get_position=["start_lng", "start_lat"],
        elevation_scale=1,
        radius=50,
        get_fill_color="""
            [
                255,
                165 - (norm_height / 5000) * 165,
                0,
                120
            ]
        """,
        extruded=True,
        pickable=True,
        get_elevation="norm_height",
    )

    view_state = pdk.ViewState(
        latitude=40.74, longitude=-73.99, zoom=11.2, pitch=60, bearing=120
    )

    st.pydeck_chart(
        pdk.Deck(
            layers=[layer],
            map_style=None,
            initial_view_state=view_state,
            tooltip={
                "html": "<b>Station:</b> {start_station_name}<br/><b>Trips:</b> {total_trips}"
            },
        ),
        width='stretch'
    )

    hourly_df, weather_df = get_eda_stats()

    st.subheader("Commuters vs. Tourists Pattern")
    st.caption("Average trips per station by hour of day.")

    chart_hourly = alt.Chart(hourly_df).mark_line(strokeWidth=3).encode(
        x=alt.X('hour_of_day', title='Hour (0-23)'),
        y=alt.Y('avg_trips', title='Avg Trips per Station'),
        color=alt.Color(
            'day_type',
            scale=alt.Scale(
                domain=['Weekday', 'Weekend'],
                range=['#1f77b4', '#ff7f0e']
            ),
            legend=alt.Legend(title="Day Type", orient="top-left")
        ),
        tooltip=[
            alt.Tooltip('hour_of_day', title='Hour of Day'),
            alt.Tooltip('avg_trips', title='Avg Trips per Station', format=',.1f'),
            alt.Tooltip('day_type', title='Day Type')
        ]
    ).properties(height=350)

    st.altair_chart(chart_hourly, width='stretch')

    with st.expander("💡 Business Insight"):
        st.markdown("""
        * **Weekdays (Blue):** Clear peaks at **8 AM** and **6 PM**. This is the classic "Commuter" pattern (people going to/from work).
        * **Weekends (Orange):** Smooth bell curve peaking at **2 PM**. This indicates leisure/tourist usage.
        * **Action:** Rebalancing trucks should focus on business districts at 9 AM on weekdays, but focus on parks/tourist spots at noon on weekends.
        """)

    st.markdown("---")

    st.subheader("The Impact of Weather")
    st.caption("Total city-wide demand vs. Temperature & Rain.")

    chart_weather = alt.Chart(weather_df).mark_circle(size=60).encode(
        x=alt.X('temperature', title='Temperature (°C)', scale=alt.Scale(domain=[-5, 35])),
        y=alt.Y('total_trips', title='Total Trips (City-wide)'),
        color=alt.Color(
            'weather_cond',
            scale=alt.Scale(domain=['Clear', 'Rainy'], range=['steelblue', 'red']),
            legend=alt.Legend(title="Condition")
        ),
        tooltip=[
            alt.Tooltip('temperature', title='Temperature (°C)'),
            alt.Tooltip('total_trips', title='Total Trips'),
            alt.Tooltip('weather_cond', title='Weather Condition')
        ]
    ).properties(height=400).interactive()

    st.altair_chart(chart_weather, width='stretch')

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

    with col_right:
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
    X_pred["month_sin"], X_pred["month_cos"] = cyclical(input_month, 12)

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

    total_predicted_demand = stations["predicted_demand"].sum()

    try:
        with duckdb.connect() as con:
            hist_query = f"""
                SELECT AVG(hourly_sum)
                FROM (
                    SELECT hour_timestamp, SUM(trip_count) as hourly_sum
                    FROM '{DATA_PATH}'
                    WHERE hour_of_day = {input_hour}
                    GROUP BY hour_timestamp
                )
            """
            avg_historical_total = con.execute(hist_query).fetchone()[0] or 0
    except Exception:
        avg_historical_total = 0

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

    hotspots = (
        stations.sort_values("predicted_demand", ascending=False)
        .head(10)[["start_station_name", "predicted_demand"]]
    )

    layer = pdk.Layer(
        "ScatterplotLayer",
        stations,
        get_position=["start_lng", "start_lat"],
        get_radius="radius_norm",
        get_fill_color="color",
        pickable=True,
    )

    view_state = pdk.ViewState(
        latitude=40.74, longitude=-73.99, zoom=11.2, bearing=120
    )

    st.pydeck_chart(
        pdk.Deck(
            layers=[layer],
            map_style=None,
            initial_view_state=view_state,
            tooltip={
                "html": "<b>{start_station_name}</b><br/>Demand: {predicted_demand}"
            },
        )
    )

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

        xai_chart = get_feature_importance(model, required_features)
        st.altair_chart(xai_chart, width='stretch', height=350)

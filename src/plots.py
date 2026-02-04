import altair as alt
import pydeck as pdk
import pandas as pd
import numpy as np

def plot_map_density(data):
    """3D Hex/Column Layer map for station density."""
    max_trips = data["total_trips"].max()
    data["norm_height"] = (data["total_trips"] / max_trips) * 5000

    layer = pdk.Layer(
        "ColumnLayer",
        data,
        get_position=["start_lng", "start_lat"],
        elevation_scale=1,
        radius=50,
        get_fill_color="[255, 165 - (norm_height / 5000) * 165, 0, 120]",
        extruded=True,
        pickable=True,
        get_elevation="norm_height",
    )

    view = pdk.ViewState(latitude=40.74, longitude=-73.99, zoom=11.2, pitch=60, bearing=120)
    return pdk.Deck(layers=[layer], initial_view_state=view, tooltip={"html": "<b>Station:</b> {start_station_name}<br/><b>Trips:</b> {total_trips}"})

def plot_hourly_trend(df):
    return alt.Chart(df).mark_line(strokeWidth=3).encode(
        x=alt.X('hour_of_day', title='Hour'),
        y=alt.Y('avg_trips', title='Avg Trips'),
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

def plot_weather_impact(df):
    return alt.Chart(df).mark_circle(size=60).encode(
        x=alt.X('temperature', title='Temp (°C)'),
        y=alt.Y('total_trips', title='Trips'),
        color=alt.Color(
            'weather_cond',
            scale=alt.Scale(domain=['Clear', 'Rainy'], range=['steelblue', 'red']),
            legend=alt.Legend(title="Condition", orient="top-left")
        ),
        tooltip=[
            alt.Tooltip('temperature', title='Temperature (°C)'),
            alt.Tooltip('total_trips', title='Total Trips'),
            alt.Tooltip('weather_cond', title='Weather Condition')
        ]
    ).properties(height=400).interactive()

def plot_feature_importance(model, features):
    if not hasattr(model, 'feature_importances_'): return None

    df = pd.DataFrame({'Feature': features, 'Importance': model.feature_importances_})
    df = df.sort_values('Importance', ascending=False).head(10)

    return alt.Chart(df).mark_bar(color='#FF4B4B').encode(
        x=alt.X('Importance'),
        y=alt.Y('Feature', sort='-x')
    ).properties(height=300)

def plot_prediction_map(stations):
    """Scatterplot map for predicted demand."""
    stations["radius_norm"] = np.sqrt(stations["predicted_demand"]) * 10
    stations["color"] = stations["predicted_demand"].apply(
        lambda x: [255, 0, 0, 180] if x > 15 else ([255, 165, 0, 160] if x > 5 else [0, 128, 255, 140])
    )

    layer = pdk.Layer(
        "ScatterplotLayer",
        stations,
        get_position=["start_lng", "start_lat"],
        get_radius="radius_norm",
        get_fill_color="color",
        pickable=True,
    )
    view = pdk.ViewState(latitude=40.74, longitude=-73.99, zoom=11.2, bearing=120)
    return pdk.Deck(layers=[layer], initial_view_state=view, tooltip={"html": "<b>{start_station_name}</b><br/>Pred: {predicted_demand}"})

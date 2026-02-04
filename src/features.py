import numpy as np
import pandas as pd

def cyclical_encode(val, period):
    return np.sin(2 * np.pi * val / period), np.cos(2 * np.pi * val / period)

def add_temporal_features(df):
    df = df.copy()

    # Ensure time components exist
    if 'hour_of_day' not in df.columns:
        df['hour_of_day'] = df['hour_timestamp'].dt.hour
    if 'day_of_week' not in df.columns:
        df['day_of_week'] = df['hour_timestamp'].dt.dayofweek
    if 'month' not in df.columns:
        df['month'] = df['hour_timestamp'].dt.month

    df["hour_sin"], df["hour_cos"] = cyclical_encode(df["hour_of_day"], 24)
    df["day_sin"], df["day_cos"] = cyclical_encode(df["day_of_week"], 7)
    df["month_sin"], df["month_cos"] = cyclical_encode(df["month"], 12)

    df["is_weekend"] = df["day_of_week"].isin([0, 6]).astype(int)
    return df

def add_lag_features(df):
    df = df.copy()
    grp = df.groupby("start_station_name")["trip_count"]

    df["trip_count_lag1"] = grp.shift(1)
    df["trip_count_lag2"] = grp.shift(2)
    df["trip_count_lag3"] = grp.shift(3)
    df["trip_count_rolling3"] = (
        grp.shift(1)
           .rolling(3, min_periods=1)
           .mean()
           .reset_index(0, drop=True)
    )
    return df

def prepare_inference_lags(hourly_stats):
    lag1_map, lag2_map, lag3_map, rolling3_map = {}, {}, {}, {}

    for h in range(24):
        # Logic matches training lags (1 hour ago, 2 hours ago, etc.)
        prev1 = (h - 1) % 24
        prev2 = (h - 2) % 24
        prev3 = (h - 3) % 24

        lag1_map[h] = hourly_stats[hourly_stats["hour_of_day"] == prev1].set_index("start_station_name")["avg_hourly_trips"]
        lag2_map[h] = hourly_stats[hourly_stats["hour_of_day"] == prev2].set_index("start_station_name")["avg_hourly_trips"]
        lag3_map[h] = hourly_stats[hourly_stats["hour_of_day"] == prev3].set_index("start_station_name")["avg_hourly_trips"]

        # Approximate rolling mean using last 3 hours averages
        hours_rolling = [prev1, prev2, prev3]
        rolling3_map[h] = (
            hourly_stats[hourly_stats["hour_of_day"].isin(hours_rolling)]
            .groupby("start_station_name")["avg_hourly_trips"]
            .mean()
        )

    return lag1_map, lag2_map, lag3_map, rolling3_map

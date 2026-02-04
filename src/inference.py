import pandas as pd
import numpy as np
from src.features import cyclical_encode, prepare_inference_lags

def build_prediction_features(stations, bundle, hour, day, month, temp, is_raining):
    """
    Constructs the exact DataFrame required by the model for inference.
    """
    # 1. Base Features
    X_pred = stations.copy()
    X_pred["hour_of_day"] = hour
    X_pred["day_of_week"] = day
    X_pred["temperature"] = temp
    X_pred["is_raining"] = is_raining
    X_pred["is_weekend"] = int(day in [0, 6])

    # 2. Cyclical Features
    X_pred["hour_sin"], X_pred["hour_cos"] = cyclical_encode(hour, 24)
    X_pred["day_sin"], X_pred["day_cos"] = cyclical_encode(day, 7)
    X_pred["month_sin"], X_pred["month_cos"] = cyclical_encode(month, 12)

    # 3. Station Stats (Target Encoding)
    stats = bundle["station_stats"]
    station_avg = stats.set_index("start_station_name")["station_avg_demand"]
    avg_global = stats["station_avg_demand"].mean()

    X_pred["station_avg_demand"] = (
        X_pred["start_station_name"].map(station_avg).fillna(avg_global)
    )

    # 4. Lag Features
    # We use the centralized function from src.features, then map results here
    hourly_stats = bundle["hourly_stats"]
    lag1_map, lag2_map, lag3_map, rolling3_map = prepare_inference_lags(hourly_stats)

    X_pred["trip_count_lag1"] = X_pred["start_station_name"].map(lag1_map[hour]).fillna(0)
    X_pred["trip_count_lag2"] = X_pred["start_station_name"].map(lag2_map[hour]).fillna(0)
    X_pred["trip_count_lag3"] = X_pred["start_station_name"].map(lag3_map[hour]).fillna(0)
    X_pred["trip_count_rolling3"] = X_pred["start_station_name"].map(rolling3_map[hour]).fillna(0)

    # 5. Filter features
    try:
        return X_pred[bundle["features"]]
    except KeyError as e:
        raise ValueError(f"Missing features: {e}")

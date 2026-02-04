import duckdb
import joblib
import xgboost as xgb
import pandas as pd
from sklearn.metrics import mean_squared_error
from src.config import INPUT_FILE, MODEL_FILE, FEATURES, TARGET_COL
from src.features import add_temporal_features, add_lag_features

def load_data():
    print(f"Loading data from {INPUT_FILE}")
    return duckdb.query(f"SELECT * FROM '{INPUT_FILE}'").to_df()

def train_model():
    df = load_data().sort_values("hour_timestamp")

    # 1. Station Stats (Target Encoding)
    station_popularity = (
        df.groupby("start_station_name")[TARGET_COL]
        .mean()
        .reset_index(name="station_avg_demand")
    )

    # 2. Hourly Stats (For Inference Lookups)
    hourly_stats = (
        df.groupby(["start_station_name", "hour_of_day"])[TARGET_COL]
        .mean()
        .reset_index(name="avg_hourly_trips")
    )

    # Merge stats
    df = df.merge(station_popularity, on="start_station_name", how="left")
    global_mean = df[TARGET_COL].mean()
    df["station_avg_demand"] = df["station_avg_demand"].fillna(global_mean)

    # 3. Apply Centralized Feature Engineering
    df = add_temporal_features(df)
    df["is_raining"] = (df["precipitation"] > 0.1).astype(int)

    # Sort before calculating lags
    df = df.sort_values(["start_station_name", "hour_timestamp"])
    df = add_lag_features(df)

    X = df[FEATURES]
    y = df[TARGET_COL]

    print("Training XGBoost...")
    model = xgb.XGBRegressor(
        n_estimators=350, max_depth=8, learning_rate=0.075,
        subsample=0.95, colsample_bytree=0.81, reg_alpha=3.05,
        reg_lambda=2.24, min_child_weight=6, gamma=1.8,
        objective="reg:squarederror", random_state=42, n_jobs=-1,
    )
    model.fit(X, y)
    print("Training complete.")

    artifact = {
        "model": model,
        "features": FEATURES,
        "station_stats": station_popularity,
        "hourly_stats": hourly_stats,
    }

    print(f"Saving model bundle to {MODEL_FILE}")
    joblib.dump(artifact, MODEL_FILE)

if __name__ == "__main__":
    train_model()

import duckdb
import joblib
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error

INPUT_FILE = "data/processed/daily_demand.parquet"
MODEL_FILE = "model.pkl"


# ----------------------------
# Data
# ----------------------------

def load_data():
    print(f"Loading data from {INPUT_FILE}")
    return duckdb.query(f"SELECT * FROM '{INPUT_FILE}'").to_df()


# ----------------------------
# Feature Engineering
# ----------------------------

def add_basic_features(df):
    df = df.copy()
    df = df.sort_values(["start_station_name", "hour_timestamp"])

    df["is_weekend"] = df["day_of_week"].isin([0, 6]).astype(int)

    df["hour_sin"] = np.sin(2 * np.pi * df["hour_of_day"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour_of_day"] / 24)

    df["day_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["day_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)

    df["month"] = df["hour_timestamp"].dt.month
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    df["is_raining"] = (df["precipitation"] > 0.1).astype(int)

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


# ----------------------------
# Training
# ----------------------------

def train_model():
    df = load_data().sort_values("hour_timestamp")

    station_popularity = (
        df.groupby("start_station_name")["trip_count"]
        .mean()
        .reset_index(name="station_avg_demand")
    )

    hourly_stats = (
        df.groupby(["start_station_name", "hour_of_day"])["trip_count"]
        .mean()
        .reset_index(name="avg_hourly_trips")
    )

    df = df.merge(station_popularity, on="start_station_name", how="left")

    global_mean = df["trip_count"].mean()
    df["station_avg_demand"] = df["station_avg_demand"].fillna(global_mean)

    df = add_basic_features(df)

    features = [
        "hour_sin", "hour_cos",
        "day_sin", "day_cos",
        "month_sin", "month_cos",
        "is_weekend",
        "temperature",
        "is_raining",
        "station_avg_demand",
        "start_lat",
        "start_lng",
        "trip_count_lag1",
        "trip_count_lag2",
        "trip_count_lag3",
        "trip_count_rolling3",
    ]

    target = "trip_count"

    X = df[features]
    y = df[target]

    print("Training XGBoost...")

    model = xgb.XGBRegressor(
        n_estimators=350,
        max_depth=8,
        learning_rate=0.075,
        subsample=0.95,
        colsample_bytree=0.81,
        reg_alpha=3.05,
        reg_lambda=2.24,
        min_child_weight=6,
        gamma=1.8,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=-1,
    )

    model.fit(X, y)

    print("Training complete.")

    artifact = {
        "model": model,
        "features": features,
        "station_stats": station_popularity,
        "hourly_stats": hourly_stats,
    }

    print(f"Saving model bundle to {MODEL_FILE}")
    joblib.dump(artifact, MODEL_FILE)


if __name__ == "__main__":
    train_model()

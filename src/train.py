import duckdb
import joblib
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error

INPUT_FILE = "data/processed/daily_demand.parquet"
MODEL_FILE = "model.pkl"


def load_data():
    print(f"Loading data from {INPUT_FILE}")
    return duckdb.query(f"SELECT * FROM '{INPUT_FILE}'").to_df()


def add_basic_features(df):
    df = df.copy()
    df = df.sort_values(["start_station_name", "hour_timestamp"])

    df["is_weekend"] = df["day_of_week"].isin([0, 6]).astype(int)
    df['hour_sin'] = np.sin(2 * np.pi * df['hour_of_day'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour_of_day'] / 24)
    df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    df["is_raining"] = (df["precipitation"] > 0.1).astype(int)
    df["trip_count_lag1"] = df.groupby("start_station_name")["trip_count"].shift(1)
    df["trip_count_rolling3"] = (
    df.groupby("start_station_name")["trip_count"]
      .shift(1)
      .rolling(3, min_periods=1)
      .mean()
      .reset_index(0, drop=True)
    )

    return df


def train_model():
    df = load_data()

    df = df.sort_values("hour_timestamp")

    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()

    station_popularity = train_df.groupby("start_station_name")["trip_count"].mean().reset_index()
    station_popularity.columns = ["start_station_name", "station_avg_demand"]

    hourly_stats = train_df.groupby(["start_station_name", "hour_of_day"])["trip_count"].mean().reset_index()
    hourly_stats.columns = ["start_station_name", "hour_of_day", "avg_hourly_trips"]

    train_df = train_df.merge(station_popularity, on="start_station_name", how="left")
    test_df = test_df.merge(station_popularity, on="start_station_name", how="left")

    global_mean = train_df["trip_count"].mean()
    train_df["station_avg_demand"] = train_df["station_avg_demand"].fillna(global_mean)
    test_df["station_avg_demand"] = test_df["station_avg_demand"].fillna(global_mean)

    train_df = add_basic_features(train_df)
    test_df = add_basic_features(test_df)

    features = [
        "hour_sin", "hour_cos",
        "day_sin", "day_cos",
        "is_weekend",
        "temperature",
        "is_raining",
        "station_avg_demand",
        "start_lat",
        "start_lng",
        "trip_count_lag1",
        "trip_count_rolling3",
    ]
    target = "trip_count"

    X_train = train_df[features]
    y_train = train_df[target]
    X_test = test_df[features]
    y_test = test_df[target]

    print("Training XGBoost...")
    model = xgb.XGBRegressor(
        n_estimators=294,
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
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    print(f"Model RMSE: {rmse:.2f}")

    baseline = np.repeat(y_train.mean(), len(y_test))
    baseline_rmse = np.sqrt(mean_squared_error(y_test, baseline))
    print(f"Baseline RMSE: {baseline_rmse:.2f}")

    print("Feature importance:")
    for name, score in zip(features, model.feature_importances_):
        print(f"  {name}: {score:.3f}")

    artifact = {
        "model": model,
        "features": features,
        "station_stats": station_popularity,
        "hourly_stats": hourly_stats
    }

    print(f"Saving model bundle to {MODEL_FILE}")
    joblib.dump(artifact, MODEL_FILE)


if __name__ == "__main__":
    train_model()

import pandas as pd
import duckdb
import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

INPUT_FILE = "data/processed/daily_demand.parquet"
MODEL_FILE = "model.pkl"


def load_data():
    print(f"Loading data from {INPUT_FILE}")
    return duckdb.query(f"SELECT * FROM '{INPUT_FILE}'").to_df()


def feature_engineering(df):
    df["is_raining"] = (df["precipitation"] > 0.1).astype(int)

    features = [
        "hour_of_day",
        "day_of_week",
        "temperature",
        "is_raining",
        "start_lat",
        "start_lng"
    ]

    target = "trip_count"
    return df[features], df[target]


def train_model():
    df = load_data()
    X, y = feature_engineering(df)

    print("Splitting data")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print("Training Random Forest")
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        n_jobs=-1,
        random_state=42
    )

    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    print(f"RMSE: {rmse:.2f}")

    print(f"Saving model to {MODEL_FILE}")
    joblib.dump(model, MODEL_FILE)


if __name__ == "__main__":
    train_model()

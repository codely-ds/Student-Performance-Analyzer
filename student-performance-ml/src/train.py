
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.data_loader import load_data, TARGET_COLUMN
from src.preprocessing import (
    build_preprocessor,
    clean_data,
    separate_features_target,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "models"

MODEL_PATH = MODEL_DIR / "student_model.joblib"
METRICS_PATH = MODEL_DIR / "model_metrics.json"


def evaluate_model(model, X_test, y_test):
    """Calculate regression metrics on the test set."""

    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    rmse = float(np.sqrt(mean_squared_error(y_test, predictions)))
    r2 = r2_score(y_test, predictions)

    return {
        "MAE": float(mae),
        "RMSE": rmse,
        "R2": float(r2),
    }


def train_models():
    """Train, compare, evaluate, and save regression models."""

    # 1. Load and clean the dataset.
    df = clean_data(load_data())

    # 2. Separate the features and target.
    X, y = separate_features_target(df)

    # 3. Split BEFORE fitting preprocessing.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
    )

    # 4. Define candidate regression models.
    candidates = {
        "Dummy Regressor": DummyRegressor(strategy="mean"),
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(
            n_estimators=200,
            random_state=42,
            n_jobs=-1,
            min_samples_leaf=2,
        ),
    }

    results = {}
    trained_models = {}

    # 5. Train and evaluate each complete pipeline.
    for model_name, estimator in candidates.items():
        pipeline = Pipeline(
            steps=[
                ("preprocessor", build_preprocessor(X_train)),
                ("model", estimator),
            ]
        )

        pipeline.fit(X_train, y_train)

        metrics = evaluate_model(
            pipeline,
            X_test,
            y_test,
        )

        results[model_name] = metrics
        trained_models[model_name] = pipeline

        print(f"\n{model_name}")
        print("-" * 35)
        print(f"MAE  : {metrics['MAE']:.4f}")
        print(f"RMSE : {metrics['RMSE']:.4f}")
        print(f"R²   : {metrics['R2']:.4f}")

    # 6. Choose the model with the lowest test RMSE.
    best_model_name = min(
        results,
        key=lambda name: results[name]["RMSE"],
    )

    best_model = trained_models[best_model_name]

    # 7. Create the artifact directory if needed.
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # 8. Save preprocessing and model together.
    joblib.dump(best_model, MODEL_PATH)

    # 9. Save model metrics and experiment information.
    metrics_output = {
        "target": TARGET_COLUMN,
        "selected_model": best_model_name,
        "selection_metric": "RMSE",
        "test_size": 0.20,
        "random_state": 42,
        "training_rows": int(len(X_train)),
        "testing_rows": int(len(X_test)),
        "feature_count": int(X.shape[1]),
        "results": results,
    }

    with METRICS_PATH.open("w", encoding="utf-8") as file:
        json.dump(metrics_output, file, indent=4)

    print("\n========== TRAINING SUMMARY ==========")
    print(f"Selected model: {best_model_name}")
    print(f"Saved pipeline: {MODEL_PATH}")
    print(f"Saved metrics : {METRICS_PATH}")

    return best_model, metrics_output


if __name__ == "__main__":
    train_models()

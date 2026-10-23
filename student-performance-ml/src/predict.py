
from pathlib import Path

import joblib
import pandas as pd

from src.data_loader import TARGET_COLUMN


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "student_model.joblib"


def load_model():
    """Load the trained preprocessing and model pipeline."""

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Trained model not found. Run "
            "'python -m src.train' before making predictions."
        )

    return joblib.load(MODEL_PATH)


def predict_student(student_data):
    """
    Predict an exam score from a dictionary of student features.

    Example:
        predict_student({
            "Hours_Studied": 20,
            "Attendance": 85,
            ...
        })
    """

    model = load_model()

    if not isinstance(student_data, dict):
        raise TypeError("student_data must be a dictionary.")

    if not student_data:
        raise ValueError("Student input cannot be empty.")

    if TARGET_COLUMN in student_data:
        raise ValueError(
            f"Do not provide '{TARGET_COLUMN}' as an input feature."
        )

    input_df = pd.DataFrame([student_data])

    expected_columns = list(model.feature_names_in_)
    missing_columns = [
        column for column in expected_columns
        if column not in input_df.columns
    ]
    extra_columns = [
        column for column in input_df.columns
        if column not in expected_columns
    ]

    if missing_columns or extra_columns:
        raise ValueError(
            f"Missing features: {missing_columns}. "
            f"Unexpected features: {extra_columns}."
        )

    # Preserve the exact feature order used during training.
    input_df = input_df[expected_columns]

    prediction = model.predict(input_df)[0]

    return float(prediction)


if __name__ == "__main__":
    print(
        "Prediction module is ready. "
        "Run the Streamlit application to enter student details."
    )

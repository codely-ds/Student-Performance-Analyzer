
from pathlib import Path

import pandas as pd


# Project root: parent of the src directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Dataset location
DATA_PATH = PROJECT_ROOT / "data" / "StudentPerformanceFactors.csv"

# Column we want to predict
TARGET_COLUMN = "Exam_Score"


def load_data(file_path=DATA_PATH):
    """Load the student performance CSV file."""

    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at: {file_path}\n"
            "Place StudentPerformanceFactors.csv inside the data folder."
        )

    try:
        df = pd.read_csv(file_path)
    except (pd.errors.ParserError, UnicodeDecodeError) as error:
        raise ValueError(
            f"Could not read the CSV file: {error}"
        ) from error

    if df.empty:
        raise ValueError("The dataset is empty.")

    if TARGET_COLUMN not in df.columns:
        raise ValueError(
            f"Required target column '{TARGET_COLUMN}' "
            "is missing from the dataset."
        )

    if df[TARGET_COLUMN].isna().any():
        raise ValueError(
            f"The target column '{TARGET_COLUMN}' "
            "contains missing values."
        )

    return df


def inspect_data(df):
    """Display a basic dataset quality report."""

    print("\n========== DATASET INFORMATION ==========")

    print("\nDataset shape:")
    print(df.shape)

    print("\nColumn names:")
    print(df.columns.tolist())

    print("\nData types:")
    print(df.dtypes)

    print("\nMissing values:")
    print(df.isnull().sum())

    print("\nDuplicate rows:")
    print(df.duplicated().sum())

    print("\nFirst five records:")
    print(df.head())

    print("\nTarget column statistics:")
    print(df[TARGET_COLUMN].describe())


if __name__ == "__main__":
    dataset = load_data()
    inspect_data(dataset)

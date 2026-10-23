
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.data_loader import TARGET_COLUMN


def clean_data(df):
    """
    Perform basic data cleaning without learning
    imputation values from the entire dataset.
    """

    if df.empty:
        raise ValueError("Cannot clean an empty dataset.")

    # Work on a copy so the original DataFrame stays unchanged.
    df = df.copy()

    # Remove duplicate rows.
    df = df.drop_duplicates().reset_index(drop=True)

    # Check that the target column exists.
    if TARGET_COLUMN not in df.columns:
        raise ValueError(
            f"Target column '{TARGET_COLUMN}' not found."
        )

    # The target must be numeric for regression.
    df[TARGET_COLUMN] = pd.to_numeric(
        df[TARGET_COLUMN], errors="coerce"
    )

    # Rows without a target cannot be used for supervised training.
    df = df.dropna(subset=[TARGET_COLUMN])

    # Prevent accidental training with an invalid target.
    if df.empty:
        raise ValueError("No usable rows remain after cleaning.")

    return df.reset_index(drop=True)


def separate_features_target(df):
    """
    Separate input columns (X) and target values (y).
    """

    if TARGET_COLUMN not in df.columns:
        raise ValueError(
            f"Target column '{TARGET_COLUMN}' not found."
        )

    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    return X, y


def build_preprocessor(X):
    """
    Create preprocessing steps for numerical and
    categorical features.

    Missing values are imputed when the pipeline is fitted.
    Fit this pipeline on training data only.
    """

    # Automatically detect column types.
    numeric_features = X.select_dtypes(
        include=["number"]
    ).columns.tolist()

    categorical_features = X.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()

    # Numerical preprocessing.
    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median")
            ),
            (
                "scaler",
                StandardScaler()
            ),
        ]
    )

    # Categorical preprocessing.
    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="most_frequent")
            ),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore")
            ),
        ]
    )

    # Apply the correct pipeline to each column type.
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_pipeline,
                numeric_features
            ),
            (
                "categorical",
                categorical_pipeline,
                categorical_features
            ),
        ],
        remainder="drop"
    )

    return preprocessor


if __name__ == "__main__":
    from src.data_loader import load_data

    raw_df = load_data()
    cleaned_df = clean_data(raw_df)

    X, y = separate_features_target(cleaned_df)
    preprocessor = build_preprocessor(X)

    print("\n========== PREPROCESSING REPORT ==========")

    print(f"Original rows: {len(raw_df)}")
    print(f"Rows after cleaning: {len(cleaned_df)}")
    print(f"Input features: {X.shape[1]}")
    print(f"Target column: {TARGET_COLUMN}")

    print("\nNumerical columns:")
    print(X.select_dtypes(include=["number"]).columns.tolist())

    print("\nCategorical columns:")
    print(
        X.select_dtypes(
            include=["object", "category", "bool"]
        ).columns.tolist()
    )

    print("\nMissing values before imputation:")
    print(X.isnull().sum()[X.isnull().sum() > 0])

    print("\nPreprocessing pipeline created successfully.")
    print(preprocessor)


import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.data_loader import load_data, TARGET_COLUMN
from src.preprocessing import clean_data


sns.set_theme(style="whitegrid")


def plot_exam_score_distribution(df):
    """Display the distribution of exam scores."""

    fig, ax = plt.subplots(figsize=(9, 5))

    sns.histplot(
        data=df,
        x=TARGET_COLUMN,
        bins=25,
        kde=True,
        ax=ax
    )

    ax.set_title("Distribution of Student Exam Scores")
    ax.set_xlabel("Exam Score")
    ax.set_ylabel("Number of Students")

    fig.tight_layout()
    plt.show()


def plot_study_hours_vs_score(df):
    """Explore study hours against exam scores."""

    fig, ax = plt.subplots(figsize=(9, 5))

    sns.scatterplot(
        data=df,
        x="Hours_Studied",
        y=TARGET_COLUMN,
        alpha=0.5,
        ax=ax
    )

    sns.regplot(
        data=df,
        x="Hours_Studied",
        y=TARGET_COLUMN,
        scatter=False,
        ax=ax
    )

    ax.set_title("Study Hours vs Exam Score")
    ax.set_xlabel("Hours Studied")
    ax.set_ylabel("Exam Score")

    fig.tight_layout()
    plt.show()


def plot_attendance_vs_score(df):
    """Explore attendance against exam scores."""

    fig, ax = plt.subplots(figsize=(9, 5))

    sns.scatterplot(
        data=df,
        x="Attendance",
        y=TARGET_COLUMN,
        alpha=0.5,
        ax=ax
    )

    ax.set_title("Attendance vs Exam Score")
    ax.set_xlabel("Attendance")
    ax.set_ylabel("Exam Score")

    fig.tight_layout()
    plt.show()


def plot_correlation_heatmap(df):
    """Display correlations among numerical columns."""

    numeric_df = df.select_dtypes(include="number")
    correlation = numeric_df.corr()

    fig, ax = plt.subplots(figsize=(13, 9))

    sns.heatmap(
        correlation,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        ax=ax
    )

    ax.set_title("Correlation Heatmap")
    fig.tight_layout()
    plt.show()


def plot_categorical_factor(df, column):
    """Compare exam-score distributions across categories."""

    if column not in df.columns:
        print(f"Column not found: {column}")
        return

    category_count = df[column].nunique(dropna=True)

    if category_count > 15:
        print(
            f"Skipping {column}: too many categories "
            f"({category_count})."
        )
        return

    fig, ax = plt.subplots(figsize=(10, 5))

    sns.boxplot(
        data=df,
        x=column,
        y=TARGET_COLUMN,
        ax=ax
    )

    ax.set_title(f"{TARGET_COLUMN} by {column}")
    ax.set_xlabel(column.replace("_", " "))
    ax.set_ylabel("Exam Score")

    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    plt.show()


def print_eda_summary(df):
    """Print descriptive statistics and target correlations."""

    print("\n========== EDA SUMMARY ==========")

    print("\nDataset shape:")
    print(df.shape)

    print("\nDescriptive statistics:")
    print(df.describe().round(2))

    print("\nMissing values:")
    print(df.isnull().sum()[df.isnull().sum() > 0])

    numeric_df = df.select_dtypes(include="number")

    if TARGET_COLUMN in numeric_df.columns:
        correlations = (
            numeric_df.corr()[TARGET_COLUMN]
            .drop(TARGET_COLUMN)
            .sort_values(
                key=lambda values: values.abs(),
                ascending=False
            )
        )

        print("\nNumerical correlations with Exam_Score:")
        print(correlations.round(3))


def run_eda():
    """Run the complete exploratory data analysis."""

    df = clean_data(load_data())

    print_eda_summary(df)

    plot_exam_score_distribution(df)
    plot_study_hours_vs_score(df)
    plot_attendance_vs_score(df)
    plot_correlation_heatmap(df)

    categorical_columns = [
        "Parental_Involvement",
        "Access_to_Resources",
        "Motivation_Level",
        "Teacher_Quality",
        "Family_Income",
        "School_Type",
        "Peer_Influence",
        "Tutoring_Sessions",
        "Parental_Education_Level",
        "Distance_from_Home",
    ]

    for column in categorical_columns:
        if column in df.columns:
            plot_categorical_factor(df, column)


if __name__ == "__main__":
    run_eda()

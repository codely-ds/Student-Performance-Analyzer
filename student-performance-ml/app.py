
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
import joblib

from src.data_loader import load_data, TARGET_COLUMN
from src.preprocessing import clean_data


# --------------------------------------------------
# 1. Project paths and page configuration
# --------------------------------------------------

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "student_model.joblib"
METRICS_PATH = ROOT / "models" / "model_metrics.json"

st.set_page_config(
    page_title="Student Performance ML",
    page_icon="🎓",
    layout="wide",
)


# --------------------------------------------------
# 2. Load data and model
# --------------------------------------------------

@st.cache_data
def get_dataset():
    return clean_data(load_data())


@st.cache_resource
def get_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Model not found. Run 'python -m src.train' first."
        )
    return joblib.load(MODEL_PATH)


@st.cache_data
def get_metrics():
    if not METRICS_PATH.exists():
        return None

    with METRICS_PATH.open("r", encoding="utf-8") as file:
        return json.load(file)


try:
    df = get_dataset()
    model = get_model()
    metrics_data = get_metrics()
except Exception as error:
    st.error(f"Unable to initialize the application: {error}")
    st.info(
        "Check that the CSV is in the data folder and that "
        "Part 4 training completed successfully."
    )
    st.stop()


X = df.drop(columns=[TARGET_COLUMN])
y = df[TARGET_COLUMN]


# --------------------------------------------------
# 3. Sidebar navigation
# --------------------------------------------------

st.sidebar.title("🎓 Student Performance")
st.sidebar.caption("Machine Learning Project")

page = st.sidebar.radio(
    "Navigation",
    [
        "Overview",
        "Explore Data",
        "Model Performance",
        "Predict Score",
    ],
)

st.sidebar.divider()
st.sidebar.write("**Technology**")
st.sidebar.write("Python • Pandas • Scikit-learn")
st.sidebar.write("Streamlit • Matplotlib • Seaborn")


# --------------------------------------------------
# 4. Overview page
# --------------------------------------------------

if page == "Overview":
    st.title("🎓 Student Performance Prediction")
    st.write(
        "Explore student-related factors and use a trained "
        "machine-learning model to estimate exam scores."
    )

    st.warning(
        "Predictions are estimates, not guaranteed exam results. "
        "The dataset shows associations, not necessarily causes."
    )

    col1, col2, col3 = st.columns(3)

    col1.metric("Student Records", f"{len(df):,}")
    col2.metric("Input Features", len(X.columns))
    col3.metric(
        "Average Exam Score",
        f"{y.mean():.2f}",
    )

    st.subheader("Dataset Preview")
    st.dataframe(df.head(10), use_container_width=True)

    st.subheader("Project Workflow")

    st.markdown(
        """
        1. **Data Processing** — validate and clean the CSV.
        2. **EDA** — explore score distributions and relationships.
        3. **Model Training** — compare regression algorithms.
        4. **Prediction** — estimate a score from entered features.
        """
    )


# --------------------------------------------------
# 5. Explore data page
# --------------------------------------------------

elif page == "Explore Data":
    st.title("📊 Exploratory Data Analysis")

    st.subheader("Exam Score Distribution")

    fig, ax = plt.subplots(figsize=(9, 4))
    sns.histplot(
        data=df,
        x=TARGET_COLUMN,
        kde=True,
        ax=ax,
    )
    ax.set_title("Distribution of Exam Scores")
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Study Hours vs Exam Score")

    if "Hours_Studied" in df.columns:
        fig, ax = plt.subplots(figsize=(9, 4))
        sns.scatterplot(
            data=df,
            x="Hours_Studied",
            y=TARGET_COLUMN,
            alpha=0.5,
            ax=ax,
        )
        ax.set_title("Study Hours vs Exam Score")
        st.pyplot(fig)
        plt.close(fig)
    else:
        st.info("Hours_Studied is not available in this dataset.")

    st.subheader("Attendance vs Exam Score")

    if "Attendance" in df.columns:
        fig, ax = plt.subplots(figsize=(9, 4))
        sns.scatterplot(
            data=df,
            x="Attendance",
            y=TARGET_COLUMN,
            alpha=0.5,
            ax=ax,
        )
        ax.set_title("Attendance vs Exam Score")
        st.pyplot(fig)
        plt.close(fig)
    else:
        st.info("Attendance is not available in this dataset.")

    st.subheader("Numerical Correlation Heatmap")

    numeric_df = df.select_dtypes(include="number")
    corr = numeric_df.corr()

    fig, ax = plt.subplots(figsize=(12, 7))
    sns.heatmap(
        corr,
        cmap="coolwarm",
        center=0,
        ax=ax,
    )
    ax.set_title("Numerical Feature Correlations")
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Missing Values")
    missing = df.isna().sum().sort_values(ascending=False)
    st.dataframe(
        missing[missing > 0].rename("Missing Count").to_frame(),
        use_container_width=True,
    )


# --------------------------------------------------
# 6. Model performance page
# --------------------------------------------------

elif page == "Model Performance":
    st.title("📈 Model Performance")

    if not metrics_data:
        st.warning(
            "Metrics file not found. Run 'python -m src.train' first."
        )
    else:
        st.success(
            f"Selected model: {metrics_data['selected_model']}"
        )

        st.caption(
            "Results are from the held-out test set used during "
            "the training run."
        )

        results = metrics_data.get("results", {})
        results_df = pd.DataFrame(results).T
        results_df.index.name = "Model"
        results_df = results_df.reset_index()

        st.dataframe(
            results_df.round(4),
            use_container_width=True,
            hide_index=True,
        )

        st.subheader("RMSE Comparison")

        if not results_df.empty:
            fig, ax = plt.subplots(figsize=(9, 4))
            sns.barplot(
                data=results_df,
                x="Model",
                y="RMSE",
                ax=ax,
            )
            ax.set_title("Test RMSE by Model (Lower Is Better)")
            ax.tick_params(axis="x", rotation=20)
            st.pyplot(fig)
            plt.close(fig)

        st.subheader("Selected Model Metrics")

        selected = metrics_data["selected_model"]
        selected_metrics = results.get(selected, {})

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "MAE",
            f"{selected_metrics.get('MAE', float('nan')):.3f}",
        )
        col2.metric(
            "RMSE",
            f"{selected_metrics.get('RMSE', float('nan')):.3f}",
        )
        col3.metric(
            "R²",
            f"{selected_metrics.get('R2', float('nan')):.3f}",
        )


# --------------------------------------------------
# 7. Prediction page
# --------------------------------------------------

elif page == "Predict Score":
    st.title("🔮 Predict Student Exam Score")

    st.write(
        "Enter the student's details below. "
        "The saved model will estimate an exam score."
    )

    # Build the form using the training dataset's input columns.
    input_data = {}

    with st.form("student_prediction_form"):
        st.subheader("Student Details")

        columns = st.columns(2)

        for index, column in enumerate(X.columns):
            with columns[index % 2]:
                series = X[column]

                if pd.api.types.is_numeric_dtype(series):
                    values = pd.to_numeric(
                        series, errors="coerce"
                    ).dropna()

                    if values.empty:
                        input_data[column] = st.number_input(
                            column.replace("_", " "),
                            value=0.0,
                        )
                        continue

                    minimum = float(values.min())
                    maximum = float(values.max())
                    median = float(values.median())

                    if pd.api.types.is_integer_dtype(series):
                        low = int(minimum)
                        high = int(maximum)
                        default = int(round(median))

                        # Handle constant-valued columns.
                        if low == high:
                            input_data[column] = st.number_input(
                                column.replace("_", " "),
                                value=default,
                                disabled=True,
                            )
                        else:
                            input_data[column] = st.number_input(
                                column.replace("_", " "),
                                min_value=low,
                                max_value=high,
                                value=max(low, min(high, default)),
                                step=1,
                            )
                    else:
                        if minimum == maximum:
                            input_data[column] = st.number_input(
                                column.replace("_", " "),
                                value=median,
                                disabled=True,
                            )
                        else:
                            input_data[column] = st.number_input(
                                column.replace("_", " "),
                                min_value=minimum,
                                max_value=maximum,
                                value=median,
                            )

                else:
                    options = sorted(
                        series.dropna().astype(str).unique().tolist()
                    )

                    if not options:
                        input_data[column] = st.text_input(
                            column.replace("_", " "),
                        )
                    else:
                        input_data[column] = st.selectbox(
                            column.replace("_", " "),
                            options=options,
                        )

        submitted = st.form_submit_button(
            "Predict Exam Score",
            use_container_width=True,
        )

    if submitted:
        try:
            input_df = pd.DataFrame([input_data])
            input_df = input_df[list(model.feature_names_in_)]

            prediction = float(model.predict(input_df)[0])

            st.subheader("Prediction Result")

            col1, col2 = st.columns(2)

            col1.metric(
                "Predicted Exam Score",
                f"{prediction:.2f}",
            )
            col2.metric(
                "Dataset Average",
                f"{y.mean():.2f}",
            )

            st.caption(
                "This is a model estimate based on patterns in the "
                "training dataset. It is not a guaranteed score."
            )

        except Exception as error:
            st.error(f"Prediction failed: {error}")

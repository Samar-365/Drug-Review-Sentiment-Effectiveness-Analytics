import logging
import os
import time
from io import BytesIO

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)

from ml_pipeline.models import (
    load_pipeline,
    predict_single_review,
)
from ml_pipeline.utils import setup_logging


setup_logging("streamlit_app.log")

st.set_page_config(
    page_title="Drug Review Sentiment Analysis",
    page_icon="💊",
    layout="wide",
)

MODEL_PATH = "models/logistic_pipeline.joblib"


@st.cache_resource(show_spinner=False)
def load_cached_pipeline(path):
    return load_pipeline(path)


@st.cache_data(show_spinner=False)
def load_csv(file_bytes):
    return pd.read_csv(
        BytesIO(file_bytes)
    )


@st.cache_data(show_spinner=False)
def clean_dataset(df):
    df = df.copy()

    required_columns = [
        "drugName",
        "condition",
        "review",
        "rating",
    ]

    for column in required_columns:
        if column not in df.columns:
            raise ValueError(
                f"Required column '{column}' is missing."
            )

    df["review"] = (
        df["review"]
        .fillna("")
        .astype(str)
    )

    df["drugName"] = (
        df["drugName"]
        .fillna("Unknown")
        .astype(str)
    )

    df["condition"] = (
        df["condition"]
        .fillna("")
        .astype(str)
    )

    invalid_condition = (
        df["condition"].str.contains(
            r"</?span|users found this comment",
            case=False,
            regex=True,
            na=False,
        )
        |
        (
            df["condition"]
            .str.strip()
            == ""
        )
    )

    df = df.loc[
        ~invalid_condition
    ].copy()

    numeric_rating = pd.to_numeric(
        df["rating"],
        errors="coerce",
    )

    df = df.loc[
        numeric_rating.notna()
    ].copy()

    df["rating"] = pd.to_numeric(
        df["rating"],
        errors="coerce",
    )

    df["sentiment"] = (
        df["rating"] > 5
    ).astype(int)

    return df


def evaluate_pipeline(
    df,
    pipeline,
    threshold,
):
    model = pipeline["model"]
    vectorizer = pipeline["vectorizer"]

    X_test = vectorizer.transform(
        df["review"].tolist()
    )

    y_test = (
        df["sentiment"]
        .values
    )

    if hasattr(
        model,
        "predict_proba",
    ):
        y_proba = (
            model
            .predict_proba(X_test)[:, 1]
        )

        y_pred = (
            y_proba >= threshold
        ).astype(int)

        if len(
            np.unique(y_test)
        ) > 1:
            roc_auc = roc_auc_score(
                y_test,
                y_proba,
            )
        else:
            roc_auc = None

    else:
        y_pred = model.predict(
            X_test
        )

        y_proba = None
        roc_auc = None

    accuracy = accuracy_score(
        y_test,
        y_pred,
    )

    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0,
    )

    return {
        "y_test": y_test,
        "y_pred": y_pred,
        "y_proba": y_proba,
        "accuracy": accuracy,
        "f1": f1,
        "roc_auc": roc_auc,
    }


st.title(
    "💊 Drug Review Sentiment Analysis Dashboard"
)

if not os.path.exists(
    MODEL_PATH
):
    st.error(
        f"Saved model was not found at: "
        f"{MODEL_PATH}"
    )

    st.code(
        "python scripts\\train_and_save.py "
        "--model logistic"
    )

    st.stop()


load_start = time.perf_counter()

try:
    pipeline = load_cached_pipeline(
        MODEL_PATH
    )

except Exception as error:
    logging.exception(
        "Failed to load saved model pipeline."
    )

    st.error(
        f"Could not load the saved model: "
        f"{error}"
    )

    st.stop()


load_time = (
    time.perf_counter()
    - load_start
)

logging.info(
    "Cached model pipeline loaded/accessed "
    "in %.4f seconds",
    load_time,
)


with st.sidebar:
    st.header(
        "Upload Data"
    )

    train_file = st.file_uploader(
        "Training Data CSV",
        type=["csv"],
        key="train_file",
    )

    test_file = st.file_uploader(
        "Test Data CSV",
        type=["csv"],
        key="test_file",
    )

    threshold = st.slider(
        "Decision Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.50,
        step=0.01,
    )


if (
    train_file is not None
    and test_file is not None
):
    try:
        train_df = load_csv(
            train_file.getvalue()
        )

        test_df = load_csv(
            test_file.getvalue()
        )

        train_df = clean_dataset(
            train_df
        )

        test_df = clean_dataset(
            test_df
        )

    except Exception as error:
        st.error(
            "Could not process uploaded "
            f"datasets: {error}"
        )

        st.stop()

    conditions = sorted(
        set(
            train_df[
                "condition"
            ]
            .dropna()
            .unique()
        )
        |
        set(
            test_df[
                "condition"
            ]
            .dropna()
            .unique()
        )
    )

    condition = st.sidebar.selectbox(
        "Filter by Condition",
        ["All"] + conditions,
    )

    if condition == "All":
        filtered_train = (
            train_df.copy()
        )

        filtered_test = (
            test_df.copy()
        )

    else:
        filtered_train = train_df[
            train_df[
                "condition"
            ]
            == condition
        ].copy()

        filtered_test = test_df[
            test_df[
                "condition"
            ]
            == condition
        ].copy()

    metric1, metric2, metric3 = (
        st.columns(3)
    )

    metric1.metric(
        "Training Reviews",
        f"{len(filtered_train):,}",
    )

    metric2.metric(
        "Test Reviews",
        f"{len(filtered_test):,}",
    )

    if not filtered_train.empty:
        top_drug_series = (
            filtered_train[
                "drugName"
            ]
            .replace(
                "",
                np.nan,
            )
            .dropna()
            .value_counts()
        )

        if not top_drug_series.empty:
            top_drug = (
                top_drug_series
                .index[0]
            )
        else:
            top_drug = "-"

    else:
        top_drug = "-"

    metric3.metric(
        "Top Drug (Train)",
        top_drug,
    )

    st.divider()

    st.subheader(
        "Saved Model Performance"
    )

    if filtered_test.empty:
        st.warning(
            "No test reviews are available "
            "for the selected condition."
        )

    else:
        evaluation_start = (
            time.perf_counter()
        )

        try:
            evaluation = (
                evaluate_pipeline(
                    filtered_test,
                    pipeline,
                    threshold,
                )
            )

        except Exception as error:
            logging.exception(
                "Model evaluation failed."
            )

            st.error(
                "Model evaluation failed: "
                f"{error}"
            )

            st.stop()

        evaluation_time = (
            time.perf_counter()
            - evaluation_start
        )

        result1, result2, result3 = (
            st.columns(3)
        )

        result1.metric(
            "Accuracy",
            f"{evaluation['accuracy']:.3f}",
        )

        result2.metric(
            "F1 Score",
            f"{evaluation['f1']:.3f}",
        )

        if (
            evaluation["roc_auc"]
            is not None
        ):
            result3.metric(
                "ROC-AUC",
                f"{evaluation['roc_auc']:.3f}",
            )

        else:
            result3.metric(
                "ROC-AUC",
                "N/A",
            )

        st.caption(
            "Evaluation completed in "
            f"{evaluation_time:.3f} seconds."
        )

        st.divider()

        st.subheader(
            "Confusion Matrix"
        )

        cm = confusion_matrix(
            evaluation["y_test"],
            evaluation["y_pred"],
            labels=[0, 1],
        )

        cm_df = pd.DataFrame(
            cm,
            index=[
                "Actual Negative",
                "Actual Positive",
            ],
            columns=[
                "Predicted Negative",
                "Predicted Positive",
            ],
        )

        st.dataframe(
            cm_df,
            width="stretch",
        )

        with st.expander(
            "Classification Report"
        ):
            report = (
                classification_report(
                    evaluation[
                        "y_test"
                    ],
                    evaluation[
                        "y_pred"
                    ],
                    labels=[0, 1],
                    output_dict=True,
                    zero_division=0,
                )
            )

            report_df = (
                pd.DataFrame(
                    report
                )
                .transpose()
            )

            st.dataframe(
                report_df,
                width="stretch",
            )

        st.divider()

        st.subheader(
            "Dataset Insights"
        )

        insight1, insight2 = (
            st.columns(2)
        )

        with insight1:
            st.markdown(
                "#### Sentiment Distribution"
            )

            sentiment_counts = (
                filtered_test[
                    "sentiment"
                ]
                .map(
                    {
                        0: "Negative",
                        1: "Positive",
                    }
                )
                .value_counts()
            )

            st.bar_chart(
                sentiment_counts
            )

        with insight2:
            st.markdown(
                "#### Top Drugs"
            )

            top_drugs = (
                filtered_train[
                    "drugName"
                ]
                .value_counts()
                .head(10)
            )

            st.bar_chart(
                top_drugs
            )

        st.divider()

        st.subheader(
            "Prediction Download"
        )

        prediction_df = (
            filtered_test.copy()
        )

        prediction_df[
            "predicted_sentiment"
        ] = np.where(
            evaluation[
                "y_pred"
            ]
            == 1,
            "Positive",
            "Negative",
        )

        if (
            evaluation["y_proba"]
            is not None
        ):
            prediction_df[
                "positive_probability"
            ] = evaluation[
                "y_proba"
            ]

        prediction_csv = (
            prediction_df.to_csv(
                index=False
            )
        )

        st.download_button(
            "Download Predictions CSV",
            data=prediction_csv,
            file_name=(
                "drug_sentiment_predictions.csv"
            ),
            mime="text/csv",
        )

else:
    st.info(
        "Upload both training and test CSV "
        "files to view dataset analytics."
    )


st.divider()

st.subheader(
    "Single Review Prediction"
)

review_text = st.text_area(
    "Enter a drug review",
    placeholder=(
        "Example: This medicine worked "
        "very well and improved my condition."
    ),
    height=120,
)

if st.button(
    "Predict Sentiment",
    type="primary",
):
    try:
        prediction_start = (
            time.perf_counter()
        )

        result = (
            predict_single_review(
                review_text,
                pipeline,
                threshold,
            )
        )

        prediction_time = (
            time.perf_counter()
            - prediction_start
        )

        if (
            result["sentiment"]
            == "Positive"
        ):
            st.success(
                "Sentiment: "
                f"{result['sentiment']}"
            )

        else:
            st.error(
                "Sentiment: "
                f"{result['sentiment']}"
            )

        if (
            result["confidence"]
            is not None
        ):
            st.metric(
                "Confidence",
                (
                    f"{result['confidence'] * 100:.2f}%"
                ),
            )

            probability_df = (
                pd.DataFrame(
                    {
                        "Sentiment": [
                            "Negative",
                            "Positive",
                        ],
                        "Probability": [
                            result[
                                "probabilities"
                            ][
                                "negative"
                            ],
                            result[
                                "probabilities"
                            ][
                                "positive"
                            ],
                        ],
                    }
                )
            )

            st.dataframe(
                probability_df,
                width="stretch",
                hide_index=True,
            )

        st.caption(
            "Prediction completed in "
            f"{prediction_time:.4f} seconds."
        )

    except (
        ValueError,
        TypeError,
    ) as error:
        st.warning(
            str(error)
        )

    except Exception as error:
        logging.exception(
            "Single-review prediction failed."
        )

        st.error(
            "Prediction failed: "
            f"{error}"
        )


with st.sidebar:
    st.divider()

    st.caption(
        "Cached model access: "
        f"{load_time:.4f} sec"
    )
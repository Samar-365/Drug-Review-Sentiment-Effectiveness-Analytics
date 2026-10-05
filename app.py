import logging
import os
import time

import numpy as np
import pandas as pd
import streamlit as st

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)

from ml_pipeline.base import TextPreprocessor
from ml_pipeline.models import (
    BaseSentimentModel,
    load_pipeline,
    predict_single_review,
)
from ml_pipeline.utils import (
    get_model,
    setup_logging,
)


# ============================================================
# CONFIGURATION
# ============================================================

setup_logging("streamlit_app.log")

st.set_page_config(
    page_title="Drug Review Sentiment & Effectiveness Analytics",
        layout="wide",
)


# ============================================================
# PERSISTED MODEL PATHS
# ============================================================

MODEL_PATHS = {
    "Logistic Regression": "models/logistic_pipeline.joblib",
    "Random Forest": "models/random_forest_pipeline.joblib",
    "SVM": "models/svm_pipeline.joblib",
    "Naive Bayes": "models/naive_bayes_pipeline.joblib",
    "Gradient Boosting": "models/gbt_pipeline.joblib",
}


# ============================================================
# PERSISTED MODEL LOADING
# ============================================================

@st.cache_resource(show_spinner=False)
def load_cached_pipeline(path):
    return load_pipeline(path)


loaded_pipelines = {}
model_load_times = {}

for model_name, model_path in MODEL_PATHS.items():

    if os.path.exists(model_path):

        try:
            load_start = time.perf_counter()

            loaded_pipelines[model_name] = (
                load_cached_pipeline(
                    model_path
                )
            )

            model_load_times[model_name] = (
                time.perf_counter()
                - load_start
            )

            logging.info(
                "Persisted %s pipeline loaded/accessed "
                "in %.4f seconds",
                model_name,
                model_load_times[model_name],
            )

        except Exception:
            logging.exception(
                "Failed to load persisted pipeline "
                "for %s",
                model_name,
            )

    else:
        logging.warning(
            "Persisted pipeline not found for %s: %s",
            model_name,
            model_path,
        )


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 4px;
    }

    .subtitle {
        color: #94a3b8;
        font-size: 1rem;
        margin-bottom: 20px;
    }

    .demo-banner {
        padding: 14px 18px;
        border-radius: 10px;
        background: rgba(59, 130, 246, 0.15);
        border: 1px solid rgba(59, 130, 246, 0.4);
        color: #e2e8f0;
        margin-bottom: 20px;
        line-height: 1.5;
        font-size: 0.95rem;
    }

    .demo-banner strong {
        color: #60a5fa;
        font-size: 1.05rem;
    }

    .analyzer-card {
        padding: 18px 20px;
        border-radius: 12px;
        border: 1px solid rgba(148, 163, 184, 0.25);
        background: rgba(148, 163, 184, 0.08);
        min-height: 120px;
    }

    .card-title {
        font-size: 1.1rem;
        font-weight: 600;
        color: #f1f5f9;
    }

    .card-text {
        color: #94a3b8;
        margin-top: 8px;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# PAGE HEADER
# ============================================================

st.markdown(
    '<div class="main-title">'
    'Drug Review Sentiment & Effectiveness Analytics'
    '</div>',
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="subtitle">
        Analyze drug reviews, compare sentiment models,
        and explore review-based insights.
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header(
        "Dashboard Settings"
    )

    st.subheader(
        "Upload Data"
    )

    train_file = st.file_uploader(
        "Training Data CSV",
        type=["csv"],
        help="Upload the training dataset.",
    )

    test_file = st.file_uploader(
        "Test Data CSV",
        type=["csv"],
        help="Upload the test dataset.",
    )

    st.markdown("---")

    threshold = st.slider(
        "Decision Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.01,
        help=(
            "Threshold used for binary "
            "sentiment prediction."
        ),
    )

    st.markdown("---")

    st.subheader(
        "Persisted Models"
    )

    for model_name in MODEL_PATHS:

        if model_name in loaded_pipelines:

            st.success(
                f"{model_name} loaded"
            )

            st.caption(
                "Cached access: "
                f"{model_load_times[model_name]:.4f} sec"
            )

        else:

            st.warning(
                f"{model_name} not available"
            )


# ============================================================
# DATASET LOADING
# ============================================================

sample_csv_path = os.path.join(
    "data",
    "sample",
    "sample_drug_reviews.csv",
)

df_train = None
df_test = None
demo_mode = False


# ------------------------------------------------------------
# CUSTOM DATASET
# ------------------------------------------------------------

if (
    train_file is not None
    and test_file is not None
):

    try:
        df_train = pd.read_csv(
            train_file
        )

        df_test = pd.read_csv(
            test_file
        )

    except Exception as exc:
        logging.exception(
            "Unable to read uploaded datasets."
        )

        st.error(
            "Unable to read uploaded CSV files: "
            f"{exc}"
        )


# ------------------------------------------------------------
# DEMO MODE
# ------------------------------------------------------------

elif os.path.exists(
    sample_csv_path
):

    try:
        demo_mode = True

        sample_df = pd.read_csv(
            sample_csv_path
        )

        if len(sample_df) >= 2:

            split_index = int(
                len(sample_df) * 0.8
            )

            split_index = max(
                1,
                min(
                    split_index,
                    len(sample_df) - 1,
                ),
            )

            df_train = (
                sample_df.iloc[
                    :split_index
                ]
                .copy()
            )

            df_test = (
                sample_df.iloc[
                    split_index:
                ]
                .copy()
            )

    except Exception as exc:
        logging.exception(
            "Unable to load sample dataset."
        )

        st.error(
            "Unable to load sample dataset: "
            f"{exc}"
        )


# ============================================================
# DEMO MODE BANNER
# ============================================================

if demo_mode:

    st.markdown(
        """
        <div class="demo-banner">
            <strong>Demo Mode Active</strong><br>
            The dashboard is currently using the bundled
            sample dataset.
            Upload your own Training and Test CSV files
            from the sidebar to analyze custom data.
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# TWO-TAB LAYOUT
# ============================================================

tab1, tab2 = st.tabs(
    [
        "Dataset Analytics & Model Leaderboard",
        "Live Review Analyzer",
    ]
)


# ============================================================
# TAB 1
# DATASET ANALYTICS & MODEL LEADERBOARD
# ============================================================

with tab1:

    st.header(
        "Dataset Analytics & Model Leaderboard"
    )

    # --------------------------------------------------------
    # NO DATASET
    # --------------------------------------------------------

    if (
        df_train is None
        or df_test is None
    ):

        st.info(
            "No dataset loaded yet."
        )

        st.markdown(
            """
            ### Get Started

            Upload both files from the sidebar:

            - Training Data CSV
            - Test Data CSV

            You can also use the bundled sample dataset
            if it is available.
            """
        )

        st.markdown("---")

        st.subheader(
            "Dashboard Features"
        )

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown(
                """
                ### Dataset Analytics

                View review counts, drug statistics,
                sentiment distribution, trends,
                and model performance.
                """
            )

        with col2:
            st.markdown(
                """
                ### Model Leaderboard

                Compare sentiment models using
                Accuracy, ROC-AUC, and F1-score.
                """
            )

        with col3:
            st.markdown(
                """
                ### Live Review Analyzer

                Enter a review, select a persisted model,
                and analyze the review.
                """
            )

    # --------------------------------------------------------
    # DATASET AVAILABLE
    # --------------------------------------------------------

    else:

        required_columns = [
            "review",
            "rating",
        ]

        missing_train = [
            column
            for column in required_columns
            if column not in df_train.columns
        ]

        missing_test = [
            column
            for column in required_columns
            if column not in df_test.columns
        ]

        if missing_train:

            st.error(
                "Training CSV is missing: "
                + ", ".join(
                    missing_train
                )
            )

        elif missing_test:

            st.error(
                "Test CSV is missing: "
                + ", ".join(
                    missing_test
                )
            )

        else:

            # ------------------------------------------------
            # CONDITION FILTER
            # ------------------------------------------------

            condition = "All"

            if (
                "condition" in df_train.columns
                and "condition" in df_test.columns
            ):

                conditions = sorted(
                    set(
                        df_train[
                            "condition"
                        ]
                        .dropna()
                        .astype(str)
                        .unique()
                    )
                    |
                    set(
                        df_test[
                            "condition"
                        ]
                        .dropna()
                        .astype(str)
                        .unique()
                    )
                )

                condition = (
                    st.sidebar.selectbox(
                        "Filter by Condition",
                        ["All"] + conditions,
                    )
                )

                if condition != "All":

                    df_train = (
                        df_train[
                            df_train[
                                "condition"
                            ]
                            .astype(str)
                            == condition
                        ]
                        .copy()
                    )

                    df_test = (
                        df_test[
                            df_test[
                                "condition"
                            ]
                            .astype(str)
                            == condition
                        ]
                        .copy()
                    )

            # ------------------------------------------------
            # DATASET METRICS
            # ------------------------------------------------

            st.subheader(
                "Dataset Overview"
            )

            col1, col2, col3, col4 = (
                st.columns(4)
            )

            col1.metric(
                "Training Reviews",
                f"{len(df_train):,}",
            )

            col2.metric(
                "Test Reviews",
                f"{len(df_test):,}",
            )

            if (
                "drugName"
                in df_train.columns
            ):

                drug_counts = (
                    df_train[
                        "drugName"
                    ]
                    .dropna()
                    .astype(str)
                    .value_counts()
                )

                top_drug = (
                    drug_counts.index[0]
                    if not drug_counts.empty
                    else "N/A"
                )

            else:
                top_drug = "N/A"

            col3.metric(
                "Top Drug",
                top_drug,
            )

            col4.metric(
                "Condition",
                condition,
            )

            st.markdown("---")

            # ------------------------------------------------
            # MODEL LEADERBOARD
            # ------------------------------------------------

            st.subheader(
                "Model Performance Leaderboard"
            )

            st.caption(
                "Model comparison is optional because "
                "training multiple models may take "
                "several minutes on large datasets."
            )

            run_evaluation = st.button(
                "Run Model Comparison",
                key="run_model_comparison",
            )

            include_hf = st.checkbox(
                "Include Hugging Face Transformer (slow)",
                value=False,
            )

            if not run_evaluation:

                st.info(
                    "Click 'Run Model Comparison' "
                    "to train and compare the models."
                )

            else:

                try:

                    preprocessor = (
                        TextPreprocessor()
                    )

                    X_train, X_test = (
                        preprocessor.fit_transform(
                            df_train[
                                "review"
                            ]
                            .fillna(""),
                            df_test[
                                "review"
                            ]
                            .fillna(""),
                        )
                    )

                    y_train = (
                        (
                            pd.to_numeric(
                                df_train[
                                    "rating"
                                ],
                                errors="coerce",
                            )
                            > 5
                        )
                        .astype(int)
                        .values
                    )

                    y_test = (
                        (
                            pd.to_numeric(
                                df_test[
                                    "rating"
                                ],
                                errors="coerce",
                            )
                            > 5
                        )
                        .astype(int)
                        .values
                    )

                    model_names = [
                        "gbt",
                        "logistic",
                        "naive_bayes",
                        "random_forest",
                        "svm",
                    ]

                    if include_hf:

                        model_names.append(
                            "hf_transformer"
                        )

                    results = []
                    predictions = {}

                    with st.spinner(
                        "Evaluating sentiment models..."
                    ):

                        for name in model_names:

                            try:

                                # ----------------------------
                                # HUGGING FACE
                                # ----------------------------

                                if (
                                    name
                                    == "hf_transformer"
                                ):

                                    try:
                                        from ml_pipeline.hf_sentiment import (
                                            HFSentimentModel,
                                        )

                                    except ImportError as error:

                                        st.warning(
                                            "Hugging Face model "
                                            "is unavailable: "
                                            f"{error}"
                                        )

                                        continue

                                    sentiment_model = (
                                        HFSentimentModel()
                                    )

                                    y_pred = (
                                        sentiment_model.predict(
                                            df_test[
                                                "review"
                                            ]
                                            .fillna("")
                                        )
                                    )

                                    y_proba = None

                                # ----------------------------
                                # CLASSICAL MODELS
                                # ----------------------------

                                else:

                                    model = (
                                        get_model(
                                            name
                                        )
                                    )

                                    sentiment_model = (
                                        BaseSentimentModel(
                                            model
                                        )
                                    )

                                    sample_weight = None

                                    if name == "gbt":

                                        class_counts = (
                                            np.bincount(
                                                y_train
                                            )
                                        )

                                        if (
                                            len(
                                                class_counts
                                            )
                                            == 2
                                        ):

                                            total = (
                                                len(
                                                    y_train
                                                )
                                            )

                                            class_weights = {
                                                0: (
                                                    total
                                                    /
                                                    (
                                                        2
                                                        * class_counts[
                                                            0
                                                        ]
                                                    )
                                                )
                                                if (
                                                    class_counts[
                                                        0
                                                    ]
                                                    > 0
                                                )
                                                else 1.0,

                                                1: (
                                                    total
                                                    /
                                                    (
                                                        2
                                                        * class_counts[
                                                            1
                                                        ]
                                                    )
                                                )
                                                if (
                                                    class_counts[
                                                        1
                                                    ]
                                                    > 0
                                                )
                                                else 1.0,
                                            }

                                            sample_weight = (
                                                np.array(
                                                    [
                                                        class_weights.get(
                                                            y,
                                                            1.0,
                                                        )
                                                        for y
                                                        in y_train
                                                    ]
                                                )
                                            )

                                    sentiment_model.train(
                                        X_train,
                                        y_train,
                                        sample_weight=(
                                            sample_weight
                                        ),
                                    )

                                    (
                                        y_pred,
                                        y_proba,
                                    ) = (
                                        sentiment_model.evaluate(
                                            X_test,
                                            y_test,
                                            threshold=(
                                                threshold
                                            ),
                                        )
                                    )

                                # ----------------------------
                                # METRICS
                                # ----------------------------

                                accuracy = (
                                    accuracy_score(
                                        y_test,
                                        y_pred,
                                    )
                                )

                                if (
                                    y_proba is not None
                                    and len(
                                        np.unique(
                                            y_test
                                        )
                                    )
                                    > 1
                                ):

                                    roc_auc = (
                                        roc_auc_score(
                                            y_test,
                                            y_proba,
                                        )
                                    )

                                else:
                                    roc_auc = None

                                f1 = f1_score(
                                    y_test,
                                    y_pred,
                                    zero_division=0,
                                )

                                results.append(
                                    {
                                        "Model": name.upper(),
                                        "Accuracy": accuracy,
                                        "ROC-AUC": roc_auc,
                                        "F1-score": f1,
                                    }
                                )

                                predictions[
                                    name
                                ] = (
                                    y_pred,
                                    y_proba,
                                )

                            except Exception:

                                logging.exception(
                                    "%s failed",
                                    name,
                                )

                                st.warning(
                                    f"{name.upper()} "
                                    "could not be evaluated."
                                )

                    # ----------------------------------------
                    # RESULTS
                    # ----------------------------------------

                    if results:

                        results_df = (
                            pd.DataFrame(
                                results
                            )
                        )

                        display_df = (
                            results_df.copy()
                        )

                        display_df[
                            "Accuracy"
                        ] = (
                            display_df[
                                "Accuracy"
                            ]
                            .map(
                                lambda x:
                                f"{x:.2%}"
                            )
                        )

                        display_df[
                            "ROC-AUC"
                        ] = (
                            display_df[
                                "ROC-AUC"
                            ]
                            .map(
                                lambda x:
                                "N/A"
                                if pd.isna(x)
                                else f"{x:.3f}"
                            )
                        )

                        display_df[
                            "F1-score"
                        ] = (
                            display_df[
                                "F1-score"
                            ]
                            .map(
                                lambda x:
                                f"{x:.3f}"
                            )
                        )

                        st.dataframe(
                            display_df,
                            width="stretch",
                            hide_index=True,
                        )

                        best_row = (
                            results_df
                            .sort_values(
                                "F1-score",
                                ascending=False,
                            )
                            .iloc[0]
                        )

                        st.success(
                            "Best Performing Model: "
                            f"**{best_row['Model']}**"
                        )

                        # ------------------------------------
                        # INSIGHTS
                        # ------------------------------------

                        st.markdown("---")

                        st.subheader(
                            "Dataset Insights"
                        )

                        (
                            insight1,
                            insight2,
                        ) = st.columns(2)

                        with insight1:

                            st.markdown(
                                "**Sentiment Distribution**"
                            )

                            distribution = (
                                pd.Series(
                                    y_test
                                )
                                .value_counts()
                                .sort_index()
                            )

                            distribution.index = [
                                (
                                    "Negative"
                                    if x == 0
                                    else "Positive"
                                )
                                for x
                                in distribution.index
                            ]

                            st.bar_chart(
                                distribution
                            )

                        with insight2:

                            st.markdown(
                                "**Top Drugs by Review Volume**"
                            )

                            if (
                                "drugName"
                                in df_train.columns
                            ):

                                top_drugs = (
                                    df_train[
                                        "drugName"
                                    ]
                                    .dropna()
                                    .astype(str)
                                    .value_counts()
                                    .head(10)
                                )

                                st.dataframe(
                                    top_drugs.rename(
                                        "Review Count"
                                    ),
                                    width="stretch",
                                )

                            else:

                                st.info(
                                    "drugName column "
                                    "is not available."
                                )

                        # ------------------------------------
                        # CONFUSION MATRIX
                        # ------------------------------------

                        st.markdown("---")

                        best_model_key = (
                            best_row[
                                "Model"
                            ]
                            .lower()
                        )

                        best_y_pred = (
                            predictions[
                                best_model_key
                            ][0]
                        )

                        st.subheader(
                            "Confusion Matrix"
                        )

                        cm = confusion_matrix(
                            y_test,
                            best_y_pred,
                            labels=[
                                0,
                                1,
                            ],
                        )

                        cm_df = (
                            pd.DataFrame(
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
                        )

                        st.dataframe(
                            cm_df,
                            width="stretch",
                        )

                        # ------------------------------------
                        # TOP RATED DRUGS
                        # ------------------------------------

                        if (
                            "drugName"
                            in df_test.columns
                            and "rating"
                            in df_test.columns
                        ):

                            st.markdown("---")

                            st.subheader(
                                "Top-Rated Drugs"
                            )

                            rating_data = (
                                df_test.copy()
                            )

                            rating_data[
                                "rating"
                            ] = (
                                pd.to_numeric(
                                    rating_data[
                                        "rating"
                                    ],
                                    errors="coerce",
                                )
                            )

                            top_rated = (
                                rating_data
                                .dropna(
                                    subset=[
                                        "drugName",
                                        "rating",
                                    ]
                                )
                                .groupby(
                                    "drugName"
                                )[
                                    "rating"
                                ]
                                .mean()
                                .sort_values(
                                    ascending=False
                                )
                                .head(10)
                            )

                            st.bar_chart(
                                top_rated
                            )

                    else:

                        st.warning(
                            "No models were "
                            "successfully evaluated."
                        )

                except Exception as exc:

                    logging.exception(
                        "Unable to evaluate models."
                    )

                    st.error(
                        "Unable to evaluate models: "
                        f"{exc}"
                    )


# ============================================================
# TAB 2
# LIVE REVIEW ANALYZER
# ============================================================

with tab2:

    st.header(
        "Live Review Analyzer"
    )

    st.caption(
        "Enter a drug review and select a persisted "
        "sentiment model for interactive analysis."
    )

    st.markdown("---")

    # --------------------------------------------------------
    # REVIEW INPUT
    # --------------------------------------------------------

    review_text = st.text_area(
        "Enter Drug Review",
        placeholder=(
            "Example: This medicine worked very well for me "
            "and I experienced significant improvement."
        ),
        height=180,
        help=(
            "Enter the drug review "
            "you want to analyze."
        ),
    )

    # --------------------------------------------------------
    # MODEL SELECTION
    # --------------------------------------------------------

    selected_model = st.selectbox(
        "Select Sentiment Model",
        list(
            MODEL_PATHS.keys()
        ),
        help=(
            "Select a persisted classical "
            "sentiment model."
        ),
    )

    # --------------------------------------------------------
    # MODEL AVAILABILITY
    # --------------------------------------------------------

    if (
        selected_model
        in loaded_pipelines
    ):

        st.caption(
            f"{selected_model} persisted pipeline is ready."
        )

    else:

        st.warning(
            f"{selected_model} has not been persisted yet. "
            "Train and save it before using live inference."
        )

    # --------------------------------------------------------
    # ANALYZE BUTTON
    # --------------------------------------------------------

    analyze_clicked = st.button(
        "Analyze Sentiment",
        type="primary",
        width="stretch",
    )

    # --------------------------------------------------------
    # LIVE ANALYSIS
    # --------------------------------------------------------

    if analyze_clicked:

        if not review_text.strip():

            st.warning(
                "Please enter a review "
                "before analyzing."
            )

        elif (
            selected_model
            not in loaded_pipelines
        ):

            model_key_map = {
                "Logistic Regression": "logistic",
                "Random Forest": "random_forest",
                "SVM": "svm",
                "Naive Bayes": "naive_bayes",
                "Gradient Boosting": "gbt",
            }

            model_key = (
                model_key_map[
                    selected_model
                ]
            )

            st.error(
                f"{selected_model} persisted "
                "pipeline is not available."
            )

            st.code(
                "python scripts\\train_and_save.py "
                f"--model {model_key}"
            )

        else:

            try:

                selected_pipeline = (
                    loaded_pipelines[
                        selected_model
                    ]
                )

                prediction_start = (
                    time.perf_counter()
                )

                result = (
                    predict_single_review(
                        review_text,
                        selected_pipeline,
                        threshold,
                    )
                )

                prediction_time = (
                    time.perf_counter()
                    - prediction_start
                )

                st.markdown(
                    "### Analysis Result"
                )

                (
                    result1,
                    result2,
                    result3,
                ) = st.columns(3)

                # --------------------------------------------
                # SENTIMENT
                # --------------------------------------------

                with result1:

                    sentiment_val = result["sentiment"]
                    if sentiment_val == "Positive":
                        st.success(
                            "Sentiment\n\n"
                            f"### {sentiment_val}"
                        )
                    elif sentiment_val == "Neutral":
                        st.warning(
                            "Sentiment\n\n"
                            f"### {sentiment_val}"
                        )
                    else:
                        st.error(
                            "Sentiment\n\n"
                            f"### {sentiment_val}"
                        )

                # --------------------------------------------
                # CONFIDENCE
                # --------------------------------------------

                with result2:

                    confidence = (
                        result.get(
                            "confidence"
                        )
                    )

                    if (
                        confidence
                        is not None
                    ):

                        st.metric(
                            "Confidence",
                            (
                                f"{confidence * 100:.2f}%"
                            ),
                        )

                    else:

                        st.metric(
                            "Confidence",
                            "N/A",
                        )

                # --------------------------------------------
                # SELECTED MODEL
                # --------------------------------------------

                with result3:

                    st.metric(
                        "Selected Model",
                        selected_model,
                    )

                # --------------------------------------------
                # PROBABILITIES
                # --------------------------------------------

                probabilities = (
                    result.get(
                        "probabilities"
                    )
                )

                if probabilities:

                    st.markdown(
                        "### Class Probabilities"
                    )

                    probability_df = (
                        pd.DataFrame(
                            {
                                "Sentiment": list(probabilities.keys()),
                                "Probability": [float(p) for p in probabilities.values()],
                            }
                        )
                    )

                    (
                        probability_col1,
                        probability_col2,
                    ) = st.columns(2)

                    with probability_col1:

                        st.dataframe(
                            probability_df,
                            use_container_width=True,
                            hide_index=True,
                        )

                    with probability_col2:

                        st.bar_chart(
                            probability_df
                            .set_index(
                                "Sentiment"
                            )[
                                "Probability"
                            ]
                        )

                else:

                    st.info(
                        "This model does not provide "
                        "class probabilities. "
                        "The predicted class is shown above."
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
                    "Single-review "
                    "prediction failed."
                )

                st.error(
                    "Prediction failed: "
                    f"{error}"
                )

    else:

        # ----------------------------------------------------
        # INITIAL CARDS
        # ----------------------------------------------------

        st.markdown(
            "### Analysis Preview"
        )

        (
            preview1,
            preview2,
            preview3,
        ) = st.columns(3)

        with preview1:

            st.markdown(
                """
                <div class="analyzer-card">
                    <div class="card-title">
                        Sentiment
                    </div>
                    <div class="card-text">
                        Sentiment result
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with preview2:

            st.markdown(
                """
                <div class="analyzer-card">
                    <div class="card-title">
                        Confidence
                    </div>
                    <div class="card-text">
                        Confidence score
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with preview3:

            st.markdown(
                """
                <div class="analyzer-card">
                    <div class="card-title">
                        Class Probabilities
                    </div>
                    <div class="card-text">
                        Prediction probabilities
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

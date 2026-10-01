import os
import logging

import streamlit as st
import pandas as pd
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    f1_score,
    confusion_matrix,
)

from ml_pipeline.base import TextPreprocessor
from ml_pipeline.models import BaseSentimentModel
from ml_pipeline.utils import get_model, setup_logging
from ml_pipeline.hf_sentiment import HFSentimentModel


# ============================================================
# CONFIGURATION
# ============================================================

setup_logging("streamlit_app.log")

st.set_page_config(
    page_title="Drug Review Sentiment & Effectiveness Analytics",
    page_icon="💊",
    layout="wide",
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
        color: #6b7280;
        font-size: 1rem;
        margin-bottom: 20px;
    }

    .demo-banner {
        padding: 15px 18px;
        border-radius: 10px;
        background-color: #eff6ff;
        border: 1px solid #93c5fd;
        margin-bottom: 20px;
    }

    .analyzer-card {
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #e5e7eb;
        background-color: #f8fafc;
        min-height: 120px;
    }

    .card-title {
        font-size: 1.1rem;
        font-weight: 600;
    }

    .card-text {
        color: #6b7280;
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
    '<div class="main-title">💊 Drug Review Sentiment & Effectiveness Analytics</div>',
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

    st.header("⚙️ Dashboard Settings")

    st.subheader("Upload Data")

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
        help="Threshold used for binary sentiment prediction.",
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
# Custom Dataset
# ------------------------------------------------------------

if train_file is not None and test_file is not None:

    try:

        df_train = pd.read_csv(train_file)
        df_test = pd.read_csv(test_file)

    except Exception as exc:

        st.error(
            f"Unable to read uploaded CSV files: {exc}"
        )


# ------------------------------------------------------------
# Demo Mode
# ------------------------------------------------------------

elif os.path.exists(sample_csv_path):

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

            df_train = sample_df.iloc[
                :split_index
            ].copy()

            df_test = sample_df.iloc[
                split_index:
            ].copy()

    except Exception as exc:

        st.error(
            f"Unable to load sample dataset: {exc}"
        )


# ============================================================
# DEMO MODE BANNER
# ============================================================

if demo_mode:

    st.markdown(
        """
        <div class="demo-banner">
            ℹ️ <strong>Demo Mode Active</strong><br>
            The dashboard is currently using the bundled sample dataset.
            Upload your own Training and Test CSV files from the sidebar
            to analyze custom data.
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# TWO-TAB LAYOUT
# ============================================================

tab1, tab2 = st.tabs(
    [
        "📊 Dataset Analytics & Model Leaderboard",
        "🔬 Live Review Analyzer",
    ]
)


# ============================================================
# TAB 1
# DATASET ANALYTICS & MODEL LEADERBOARD
# ============================================================

with tab1:

    st.header(
        "📊 Dataset Analytics & Model Leaderboard"
    )

    # --------------------------------------------------------
    # No Dataset
    # --------------------------------------------------------

    if df_train is None or df_test is None:

        st.info(
            "📂 No dataset loaded yet."
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
            "🚀 Dashboard Features"
        )

        col1, col2, col3 = st.columns(3)

        with col1:

            st.markdown(
                """
                ### 📊 Dataset Analytics

                View review counts, drug statistics,
                sentiment distribution, trends,
                and model performance.
                """
            )

        with col2:

            st.markdown(
                """
                ### 🤖 Model Leaderboard

                Compare sentiment models using
                Accuracy, ROC-AUC, and F1-score.
                """
            )

        with col3:

            st.markdown(
                """
                ### 🔬 Live Review Analyzer

                Enter a review, select a model,
                and analyze the review.
                """
            )

    # --------------------------------------------------------
    # Dataset Available
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
                + ", ".join(missing_train)
            )

        elif missing_test:

            st.error(
                "Test CSV is missing: "
                + ", ".join(missing_test)
            )

        else:

            # ------------------------------------------------
            # Condition Filter
            # ------------------------------------------------

            condition = "All"

            if (
                "condition" in df_train.columns
                and "condition" in df_test.columns
            ):

                conditions = sorted(
                    set(
                        df_train["condition"]
                        .dropna()
                        .astype(str)
                        .unique()
                    )
                    |
                    set(
                        df_test["condition"]
                        .dropna()
                        .astype(str)
                        .unique()
                    )
                )

                condition = st.sidebar.selectbox(
                    "Filter by Condition",
                    ["All"] + conditions,
                )

                if condition != "All":

                    df_train = df_train[
                        df_train["condition"]
                        .astype(str)
                        == condition
                    ].copy()

                    df_test = df_test[
                        df_test["condition"]
                        .astype(str)
                        == condition
                    ].copy()

            # ------------------------------------------------
            # Dataset Metrics
            # ------------------------------------------------

            st.subheader(
                "📌 Dataset Overview"
            )

            col1, col2, col3, col4 = st.columns(4)

            col1.metric(
                "Training Reviews",
                f"{len(df_train):,}",
            )

            col2.metric(
                "Test Reviews",
                f"{len(df_test):,}",
            )

            if "drugName" in df_train.columns:

                drug_counts = (
                    df_train["drugName"]
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
            # Model Leaderboard
            # ------------------------------------------------

            st.subheader(
                "🤖 Model Performance Leaderboard"
            )

            try:

                preprocessor = TextPreprocessor()

                X_train, X_test = (
                    preprocessor.fit_transform(
                        df_train["review"].fillna(""),
                        df_test["review"].fillna(""),
                    )
                )

                y_train = (
                    pd.to_numeric(
                        df_train["rating"],
                        errors="coerce",
                    ) > 5
                ).astype(int).values

                y_test = (
                    pd.to_numeric(
                        df_test["rating"],
                        errors="coerce",
                    ) > 5
                ).astype(int).values

                model_names = [
                    "gbt",
                    "logistic",
                    "naive_bayes",
                    "random_forest",
                    "svm",
                    "hf_transformer",
                ]

                results = []
                predictions = {}

                with st.spinner(
                    "Evaluating sentiment models..."
                ):

                    for name in model_names:

                        try:

                            if name == "hf_transformer":

                                sentiment_model = (
                                    HFSentimentModel()
                                )

                                y_pred = (
                                    sentiment_model.predict(
                                        df_test["review"]
                                        .fillna("")
                                    )
                                )

                                y_proba = None

                            else:

                                model = get_model(name)

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

                                    if len(
                                        class_counts
                                    ) == 2:

                                        total = len(
                                            y_train
                                        )

                                        class_weights = {
                                            0: total / (
                                                2
                                                * class_counts[0]
                                            )
                                            if class_counts[0] > 0
                                            else 1.0,

                                            1: total / (
                                                2
                                                * class_counts[1]
                                            )
                                            if class_counts[1] > 0
                                            else 1.0,
                                        }

                                        sample_weight = np.array(
                                            [
                                                class_weights.get(
                                                    y,
                                                    1.0,
                                                )
                                                for y
                                                in y_train
                                            ]
                                        )

                                sentiment_model.train(
                                    X_train,
                                    y_train,
                                    sample_weight=sample_weight,
                                )

                                y_pred, y_proba = (
                                    sentiment_model.evaluate(
                                        X_test,
                                        y_test,
                                        threshold=threshold,
                                    )
                                )

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
                                ) > 1
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

                            predictions[name] = (
                                y_pred,
                                y_proba,
                            )

                        except Exception as exc:

                            logging.exception(
                                f"{name} failed"
                            )

                            st.warning(
                                f"{name.upper()} could not be evaluated."
                            )

                if results:

                    results_df = pd.DataFrame(
                        results
                    )

                    display_df = results_df.copy()

                    display_df["Accuracy"] = (
                        display_df["Accuracy"]
                        .map(
                            lambda x: f"{x:.2%}"
                        )
                    )

                    display_df["ROC-AUC"] = (
                        display_df["ROC-AUC"]
                        .map(
                            lambda x:
                            "N/A"
                            if pd.isna(x)
                            else f"{x:.3f}"
                        )
                    )

                    display_df["F1-score"] = (
                        display_df["F1-score"]
                        .map(
                            lambda x: f"{x:.3f}"
                        )
                    )

                    st.dataframe(
                        display_df,
                        use_container_width=True,
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
                        f"🏆 Best Performing Model: "
                        f"**{best_row['Model']}**"
                    )

                    # ----------------------------------------
                    # Insights
                    # ----------------------------------------

                    st.markdown("---")

                    st.subheader(
                        "📈 Dataset Insights"
                    )

                    insight1, insight2 = st.columns(2)

                    with insight1:

                        st.markdown(
                            "**Sentiment Distribution**"
                        )

                        distribution = (
                            pd.Series(y_test)
                            .value_counts()
                            .sort_index()
                        )

                        distribution.index = [
                            "Negative"
                            if x == 0
                            else "Positive"
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

                        if "drugName" in df_train.columns:

                            top_drugs = (
                                df_train["drugName"]
                                .dropna()
                                .astype(str)
                                .value_counts()
                                .head(10)
                            )

                            st.dataframe(
                                top_drugs.rename(
                                    "Review Count"
                                ),
                                use_container_width=True,
                            )

                        else:

                            st.info(
                                "drugName column is not available."
                            )

                    # ----------------------------------------
                    # Confusion Matrix
                    # ----------------------------------------

                    st.markdown("---")

                    best_model_key = (
                        best_row["Model"].lower()
                    )

                    best_y_pred = predictions[
                        best_model_key
                    ][0]

                    st.subheader(
                        "🔲 Confusion Matrix"
                    )

                    cm = confusion_matrix(
                        y_test,
                        best_y_pred,
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
                        use_container_width=True,
                    )

                    # ----------------------------------------
                    # Top Rated Drugs
                    # ----------------------------------------

                    if (
                        "drugName" in df_test.columns
                        and "rating" in df_test.columns
                    ):

                        st.markdown("---")

                        st.subheader(
                            "⭐ Top-Rated Drugs"
                        )

                        rating_data = df_test.copy()

                        rating_data["rating"] = (
                            pd.to_numeric(
                                rating_data["rating"],
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
                            )["rating"]
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
                        "No models were successfully evaluated."
                    )

            except Exception as exc:

                st.error(
                    f"Unable to evaluate models: {exc}"
                )


# ============================================================
# TAB 2
# LIVE REVIEW ANALYZER
# ============================================================

with tab2:

    st.header(
        "🔬 Live Review Analyzer"
    )

    st.caption(
        "Enter a drug review and select a sentiment model "
        "for interactive analysis."
    )

    st.markdown("---")

    # --------------------------------------------------------
    # Review Input
    # --------------------------------------------------------

    review_text = st.text_area(
        "📝 Enter Drug Review",
        placeholder=(
            "Example: This medicine worked very well for me "
            "and I experienced significant improvement."
        ),
        height=180,
        help="Enter the drug review you want to analyze.",
    )

    # --------------------------------------------------------
    # Model Selection
    # --------------------------------------------------------

    selected_model = st.selectbox(
        "🤖 Select Sentiment Model",
        [
            "Logistic Regression",
            "Random Forest",
            "SVM",
            "Naive Bayes",
            "Gradient Boosting",
            "Hugging Face Transformer",
        ],
        help="Select the sentiment model.",
    )

    # --------------------------------------------------------
    # Analyze Button
    # --------------------------------------------------------

    analyze_clicked = st.button(
        "🔍 Analyze Sentiment",
        type="primary",
        use_container_width=True,
    )

    # --------------------------------------------------------
    # Review Validation
    # --------------------------------------------------------

    if analyze_clicked:

        if not review_text.strip():

            st.warning(
                "⚠️ Please enter a review before analyzing."
            )

        else:

            st.success(
                "Review submitted successfully."
            )

            st.markdown(
                "### 📊 Analysis Result"
            )

            result1, result2, result3 = st.columns(3)

            with result1:

                st.markdown(
                    """
                    <div class="analyzer-card">
                        <div class="card-title">
                            😊 Sentiment
                        </div>
                        <div class="card-text">
                            Sentiment result
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            with result2:

                st.markdown(
                    """
                    <div class="analyzer-card">
                        <div class="card-title">
                            🎯 Confidence
                        </div>
                        <div class="card-text">
                            Confidence score
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            with result3:

                st.markdown(
                    f"""
                    <div class="analyzer-card">
                        <div class="card-title">
                            🤖 Selected Model
                        </div>
                        <div class="card-text">
                            {selected_model}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    else:

        # ----------------------------------------------------
        # Initial Cards
        # ----------------------------------------------------

        st.markdown(
            "### 📋 Analysis Preview"
        )

        preview1, preview2, preview3 = st.columns(3)

        with preview1:

            st.markdown(
                """
                <div class="analyzer-card">
                    <div class="card-title">
                        😊 Sentiment
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
                        🎯 Confidence
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
                        📊 Class Probabilities
                    </div>
                    <div class="card-text">
                        Prediction probabilities
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
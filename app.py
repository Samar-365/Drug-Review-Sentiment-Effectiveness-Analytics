# -*- coding: utf-8 -*-
"""
Drug Review Sentiment & Clinical Effectiveness Analytics Dashboard.
Supports 3-Class Sentiment Classification, Dataset Analytics, Model Leaderboard,
and Real-Time Interactive Review Inference.
"""

import os
import time
import logging
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from ml_pipeline.base import SentimentDataLoader, TextPreprocessor, map_sentiment_3class, clean_review_text
from ml_pipeline.models import BaseSentimentModel, save_pipeline, load_pipeline, predict_single_review, SENTIMENT_LABELS, SENTIMENT_COLORS
from ml_pipeline.utils import get_model, setup_logging
from config import DATA_TRAIN_PATH, DATA_TEST_PATH, SAMPLE_DATA_PATH, MODEL_SAVE_PATH

# Configure Logging & Page
setup_logging("streamlit_app.log")
st.set_page_config(
    page_title="Drug Review Sentiment Analytics",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .reportview-container { background-color: #0e1117; }
    .metric-card {
        background-color: #1e293b;
        border-radius: 8px;
        padding: 16px;
        border: 1px solid #334155;
    }
    .badge-demo {
        background-color: #0284c7;
        color: white;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .badge-live {
        background-color: #059669;
        color: white;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.85rem;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Cached Resource & Data Loaders
# ---------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_cached_data(train_upload=None, test_upload=None):
    """Loads dataset from upload or local default paths with Demo Mode fallback."""
    if train_upload and test_upload:
        df_train = pd.read_csv(train_upload)
        df_test = pd.read_csv(test_upload)
        mode = "User Uploaded Full Data"
    elif os.path.exists(DATA_TRAIN_PATH) and os.path.exists(DATA_TEST_PATH):
        df_train = pd.read_csv(DATA_TRAIN_PATH)
        df_test = pd.read_csv(DATA_TEST_PATH)
        mode = "Full Benchmark Datasets"
    elif os.path.exists(SAMPLE_DATA_PATH):
        df_sample = pd.read_csv(SAMPLE_DATA_PATH)
        # 80/20 split for demo mode
        split_idx = int(len(df_sample) * 0.8)
        df_train = df_sample.iloc[:split_idx].copy()
        df_test = df_sample.iloc[split_idx:].copy()
        mode = "Demo Mode (Curated Sample)"
    else:
        return None, None, "No Data Available"

    for df in [df_train, df_test]:
        if 'rating' in df.columns and 'sentiment' not in df.columns:
            df['sentiment'] = df['rating'].apply(map_sentiment_3class)
        if 'review' in df.columns:
            df['review'] = df['review'].fillna('').apply(clean_review_text)
        if 'condition' in df.columns:
            df['condition'] = df['condition'].fillna('Unknown').astype(str).str.replace(r"<.*?>", "", regex=True).str.strip()

    return df_train, df_test, mode

@st.cache_resource(show_spinner=False)
def get_cached_pipeline():
    """Loads pre-trained sentiment pipeline or creates a fast default pipeline."""
    model_path = os.path.join(MODEL_SAVE_PATH, "sentiment_pipeline.joblib")
    if os.path.exists(model_path):
        return load_pipeline(model_path)
    
    # Fallback fit on sample dataset if pipeline artifact not yet saved
    if os.path.exists(SAMPLE_DATA_PATH):
        df = pd.read_csv(SAMPLE_DATA_PATH)
        df['clean_review'] = df['review'].apply(clean_review_text)
        df['sentiment'] = df['rating'].apply(map_sentiment_3class)
        
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        
        vec = TfidfVectorizer(max_features=2500, stop_words="english", ngram_range=(1, 2))
        X = vec.fit_transform(df['clean_review'])
        clf = LogisticRegression(max_iter=500, class_weight="balanced", random_state=42)
        clf.fit(X, df['sentiment'])
        
        os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
        save_pipeline(vec, clf, model_path)
        return {"vectorizer": vec, "model": clf, "labels": SENTIMENT_LABELS, "num_classes": 3}
    return None

# ---------------------------------------------------------
# Sidebar Configuration
# ---------------------------------------------------------
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/pill.png", width=64)
    st.title("Settings & Data")
    
    st.markdown("### 📂 Data Source")
    train_file = st.file_uploader("Upload Training CSV", type="csv")
    test_file = st.file_uploader("Upload Test CSV", type="csv")
    
    st.markdown("---")
    st.markdown("### ⚙️ Evaluation Tuning")
    model_select = st.multiselect(
        "Select Models to Benchmark",
        ["logistic", "naive_bayes", "random_forest", "gbt"],
        default=["logistic", "naive_bayes", "random_forest", "gbt"]
    )

# ---------------------------------------------------------
# Main App Header & Mode Status
# ---------------------------------------------------------
df_train, df_test, current_mode = load_cached_data(train_file, test_file)

col_title, col_badge = st.columns([4, 1])
with col_title:
    st.title("💊 Drug Review Sentiment & Effectiveness Analytics")
    st.markdown("*3-Class Patient Sentiment Intelligence, Clinical Imbalance Audits & Real-Time Inference*")

with col_badge:
    st.markdown("<br>", unsafe_allow_html=True)
    if "Demo" in current_mode:
        st.markdown(f'<span class="badge-demo">🚀 {current_mode}</span>', unsafe_allow_html=True)
    else:
        st.markdown(f'<span class="badge-live">⚡ {current_mode}</span>', unsafe_allow_html=True)

st.markdown("---")

# ---------------------------------------------------------
# Navigation Tabs
# ---------------------------------------------------------
tab_analytics, tab_live = st.tabs([
    "📊 Dataset Analytics & Benchmark Leaderboard",
    "🔬 Live Interactive Review Analyzer"
])

# =========================================================
# TAB 1: DATASET ANALYTICS & BENCHMARK LEADERBOARD
# =========================================================
with tab_analytics:
    if df_train is None or df_train.empty:
        st.warning("⚠️ No dataset found. Please upload data or ensure sample data exists in `data/sample/`.")
    else:
        # Top-level Metric Cards
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Training Records", f"{len(df_train):,}")
        c2.metric("Test Records", f"{len(df_test):,}")
        c3.metric("Unique Conditions", f"{df_train['condition'].nunique():,}")
        c4.metric("Unique Drugs", f"{df_train['drugName'].nunique():,}")

        st.markdown("### 🔍 Condition Filter")
        all_conditions = ["All"] + sorted([str(c) for c in df_train['condition'].unique() if pd.notna(c)])
        selected_condition = st.selectbox("Filter dataset visualizations by clinical condition:", all_conditions)
        
        filtered_train = df_train if selected_condition == "All" else df_train[df_train['condition'] == selected_condition]
        filtered_test = df_test if selected_condition == "All" else df_test[df_test['condition'] == selected_condition]

        # Analytics Visualizations
        col_chart1, col_chart2 = st.columns(2)
        with col_chart1:
            st.subheader("Sentiment Class Distribution")
            sentiment_counts = filtered_train['sentiment'].value_counts().sort_index().rename(SENTIMENT_LABELS)
            st.bar_chart(sentiment_counts)

        with col_chart2:
            st.subheader("Top Conditions / Medications")
            top_drugs = filtered_train['drugName'].value_counts().head(8)
            st.bar_chart(top_drugs)

        st.markdown("---")
        st.subheader("🏆 Multi-Model 3-Class Benchmark Leaderboard")

        if st.button("▶ Run Full Benchmark Evaluation", key="run_benchmark_btn"):
            with st.spinner("Training and evaluating selected models across 3 classes..."):
                start_time = time.time()
                preprocessor = TextPreprocessor(max_features=2500)
                X_tr, X_te = preprocessor.fit_transform(filtered_train['review'], filtered_test['review'])
                y_tr = filtered_train['sentiment'].values
                y_te = filtered_test['sentiment'].values

                benchmark_results = []
                for m_name in model_select:
                    raw_model = get_model(m_name)
                    model_wrapper = BaseSentimentModel(raw_model, num_classes=3)
                    model_wrapper.train(X_tr, y_tr)
                    metrics = model_wrapper.evaluate(X_te, y_te)
                    
                    benchmark_results.append({
                        "Model": m_name.replace("_", " ").title(),
                        "Accuracy": f"{metrics['accuracy']*100:.2f}%",
                        "Macro F1": round(metrics['macro_f1'], 4),
                        "Weighted F1": round(metrics['weighted_f1'], 4),
                        "Macro Precision": round(metrics['macro_precision'], 4),
                        "Macro Recall": round(metrics['macro_recall'], 4)
                    })

                bench_df = pd.DataFrame(benchmark_results).sort_values("Macro F1", ascending=False)
                st.dataframe(bench_df, use_container_width=True)
                st.success(f" Benchmark completed in {time.time() - start_time:.2f} seconds!")

# =========================================================
# TAB 2: LIVE INTERACTIVE REVIEW ANALYZER
# =========================================================
with tab_live:
    st.subheader("🔬 Real-Time Clinical Sentiment & Nuance Prediction")
    st.markdown("Enter patient review text below or click a sample preset to test clinical edge-case classifications.")

    preset_samples = {
        "Custom Input": "",
        "Positive (High Efficacy)": "This medication has truly given me my life back! Within four weeks all depressive symptoms and fatigue lifted completely.",
        "Neutral / Mixed (Moderate with Side Effect)": "Cured my intense migraines within 30 minutes, but gave me noticeable nausea and mild dizziness all afternoon.",
        "Negative (Severe Adverse Reaction)": "Ended up in the ER with severe hives, swelling, and extreme chest tightness after taking only one dose.",
        "Delayed Onset (Nuanced Positive)": "Did not feel any improvement for the first three weeks, then suddenly my anxiety symptoms improved dramatically."
    }

    selected_preset = st.selectbox("💡 Load Sample Patient Review Preset:", list(preset_samples.keys()))
    default_text = preset_samples[selected_preset]

    review_input = st.text_area(
        "Patient Review Text:",
        value=default_text,
        height=140,
        placeholder="Type or paste a patient review here (e.g. 'Helped with my symptoms, but caused mild headaches...')"
    )

    if st.button("🚀 Analyze Sentiment & Probabilities", type="primary"):
        if not review_input.strip():
            st.warning("Please enter some review text before analyzing.")
        else:
            pipeline = get_cached_pipeline()
            if pipeline is None:
                st.error("No sentiment pipeline artifact found. Please run the benchmark first.")
            else:
                inference_res = predict_single_review(review_input, pipeline)
                
                # Results Card Layout
                res_col1, res_col2 = st.columns([1, 2])
                with res_col1:
                    st.markdown("### 🏷️ Predicted Sentiment")
                    pred_label = inference_res["sentiment_label"]
                    confidence = inference_res["confidence"]
                    
                    if pred_label == "Positive":
                        st.success(f"### 😊 {pred_label}")
                    elif pred_label == "Neutral":
                        st.warning(f"### 😐 {pred_label}")
                    else:
                        st.error(f"### 😡 {pred_label}")
                    
                    st.metric("Model Confidence", f"{confidence * 100:.1f}%")

                with res_col2:
                    st.markdown("### 📊 Class Probability Breakdown")
                    proba_series = pd.Series(inference_res["probabilities"])
                    st.bar_chart(proba_series)

                with st.expander("🔍 Cleaned Tokens & Technical Details"):
                    st.json({
                        "original_input": inference_res["text"],
                        "normalized_input": inference_res["cleaned_text"],
                        "predicted_class_id": inference_res["predicted_class"],
                        "probability_vector": inference_res["probabilities"]
                    })

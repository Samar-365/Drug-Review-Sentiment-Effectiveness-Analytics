"""
End-to-end pipeline integration tests.

Validates the full lifecycle:
    load CSV  →  preprocess  →  train  →  save artifact
    →  load artifact  →  predict on raw text

No external data files are required — all data comes from
the synthetic fixtures in conftest.py.
"""

import os
import pytest
import numpy as np

from ml_pipeline.base import SentimentDataLoader, TextPreprocessor
from ml_pipeline.models import (
    BaseSentimentModel,
    save_pipeline,
    load_pipeline,
    predict_single_review,
)
from ml_pipeline.utils import get_model


# ============================================================
# Helpers
# ============================================================

def _run_full_pipeline(train_csv, test_csv, model_name="logistic", max_features=200):
    """
    Execute the complete pipeline end-to-end and return all artifacts.
    Used by multiple tests to avoid repetition.
    """
    # 1. Load
    loader = SentimentDataLoader(train_csv, test_csv)
    df_train, df_test = loader.load()

    # 2. Preprocess
    prep = TextPreprocessor(max_features=max_features)
    X_train, X_test = prep.fit_transform(df_train["review"], df_test["review"])

    y_train = df_train["sentiment"].values
    y_test = df_test["sentiment"].values

    # 3. Train
    model = get_model(model_name)
    sm = BaseSentimentModel(model)
    sm.train(X_train, y_train)

    return sm, prep.vectorizer, X_train, X_test, y_train, y_test


# ============================================================
# Stage 1 — Data Loading
# ============================================================

class TestIntegrationDataLoading:

    def test_loader_reads_train_csv(self, mock_train_csv, mock_test_csv):
        loader = SentimentDataLoader(mock_train_csv, mock_test_csv)
        df_train, _ = loader.load()
        assert len(df_train) == 40

    def test_loader_reads_test_csv(self, mock_train_csv, mock_test_csv):
        loader = SentimentDataLoader(mock_train_csv, mock_test_csv)
        _, df_test = loader.load()
        assert len(df_test) == 10

    def test_sentiment_labels_are_3class(self, mock_train_csv, mock_test_csv):
        loader = SentimentDataLoader(mock_train_csv, mock_test_csv)
        df_train, df_test = loader.load()
        all_labels = set(df_train["sentiment"]) | set(df_test["sentiment"])
        assert all_labels.issubset({0, 1, 2})


# ============================================================
# Stage 2 — Preprocessing
# ============================================================

class TestIntegrationPreprocessing:

    def test_feature_matrices_have_correct_row_counts(
        self, mock_train_csv, mock_test_csv
    ):
        sm, vectorizer, X_train, X_test, _, _ = _run_full_pipeline(
            mock_train_csv, mock_test_csv
        )
        assert X_train.shape[0] == 40
        assert X_test.shape[0] == 10

    def test_train_and_test_have_same_feature_dimension(
        self, mock_train_csv, mock_test_csv
    ):
        sm, vectorizer, X_train, X_test, _, _ = _run_full_pipeline(
            mock_train_csv, mock_test_csv
        )
        assert X_train.shape[1] == X_test.shape[1]


# ============================================================
# Stage 3 — Training
# ============================================================

class TestIntegrationTraining:

    def test_model_is_fitted_after_train(self, mock_train_csv, mock_test_csv):
        sm, _, _, _, _, _ = _run_full_pipeline(mock_train_csv, mock_test_csv)
        assert hasattr(sm.model, "classes_")

    def test_predictions_cover_valid_label_set(self, mock_train_csv, mock_test_csv):
        sm, _, _, X_test, _, _ = _run_full_pipeline(mock_train_csv, mock_test_csv)
        preds = sm.model.predict(X_test)
        assert set(preds).issubset({0, 1, 2})

    @pytest.mark.parametrize("model_name", ["logistic", "naive_bayes", "random_forest"])
    def test_multiple_models_train_successfully(
        self, mock_train_csv, mock_test_csv, model_name
    ):
        sm, _, _, X_test, _, _ = _run_full_pipeline(
            mock_train_csv, mock_test_csv, model_name=model_name
        )
        preds = sm.model.predict(X_test)
        assert len(preds) == 10


# ============================================================
# Stage 4 — Save Artifact
# ============================================================

class TestIntegrationSaveArtifact:

    def test_save_pipeline_creates_joblib_file(
        self, mock_train_csv, mock_test_csv, tmp_model_dir
    ):
        sm, vectorizer, _, _, _, _ = _run_full_pipeline(mock_train_csv, mock_test_csv)
        path = os.path.join(tmp_model_dir, "integration_pipeline.joblib")
        save_pipeline(sm.model, vectorizer, path)
        assert os.path.exists(path)

    def test_saved_file_is_nonzero_size(
        self, mock_train_csv, mock_test_csv, tmp_model_dir
    ):
        sm, vectorizer, _, _, _, _ = _run_full_pipeline(mock_train_csv, mock_test_csv)
        path = os.path.join(tmp_model_dir, "size_check_pipeline.joblib")
        save_pipeline(sm.model, vectorizer, path)
        assert os.path.getsize(path) > 0


# ============================================================
# Stage 5 — Load Artifact
# ============================================================

class TestIntegrationLoadArtifact:

    def test_load_pipeline_returns_model_and_vectorizer(
        self, mock_train_csv, mock_test_csv, tmp_model_dir
    ):
        sm, vectorizer, _, _, _, _ = _run_full_pipeline(mock_train_csv, mock_test_csv)
        path = os.path.join(tmp_model_dir, "load_test_pipeline.joblib")
        save_pipeline(sm.model, vectorizer, path)

        pipeline = load_pipeline(path)
        assert "model" in pipeline
        assert "vectorizer" in pipeline
        assert pipeline["model"] is not None
        assert pipeline["vectorizer"] is not None

    def test_loaded_model_predictions_match_original(
        self, mock_train_csv, mock_test_csv, tmp_model_dir
    ):
        sm, vectorizer, _, X_test, _, _ = _run_full_pipeline(
            mock_train_csv, mock_test_csv
        )
        path = os.path.join(tmp_model_dir, "repro_test_pipeline.joblib")
        save_pipeline(sm.model, vectorizer, path)

        pipeline = load_pipeline(path)
        preds_original = sm.model.predict(X_test)
        preds_loaded = pipeline["model"].predict(X_test)
        np.testing.assert_array_equal(preds_original, preds_loaded)


# ============================================================
# Stage 6 — End-to-End Predict on Raw Text
# ============================================================

class TestIntegrationPredictRawText:

    def test_positive_review_returns_valid_result(
        self, mock_train_csv, mock_test_csv, tmp_model_dir
    ):
        sm, vectorizer, _, _, _, _ = _run_full_pipeline(mock_train_csv, mock_test_csv)
        path = os.path.join(tmp_model_dir, "e2e_pipeline.joblib")
        save_pipeline(sm.model, vectorizer, path)
        pipeline = load_pipeline(path)

        result = predict_single_review(
            "This medication worked perfectly and helped my condition.",
            pipeline,
        )
        assert result["sentiment"] in {"Positive", "Negative", "Neutral"}
        assert "confidence" in result
        assert "probabilities" in result

    def test_negative_review_returns_valid_result(
        self, mock_train_csv, mock_test_csv, tmp_model_dir
    ):
        sm, vectorizer, _, _, _, _ = _run_full_pipeline(mock_train_csv, mock_test_csv)
        path = os.path.join(tmp_model_dir, "e2e_neg_pipeline.joblib")
        save_pipeline(sm.model, vectorizer, path)
        pipeline = load_pipeline(path)

        result = predict_single_review(
            "Terrible side effects, did not help at all.",
            pipeline,
        )
        assert result["sentiment"] in {"Positive", "Negative", "Neutral"}

    def test_prediction_confidence_is_valid_probability(
        self, mock_train_csv, mock_test_csv, tmp_model_dir
    ):
        sm, vectorizer, _, _, _, _ = _run_full_pipeline(mock_train_csv, mock_test_csv)
        path = os.path.join(tmp_model_dir, "e2e_conf_pipeline.joblib")
        save_pipeline(sm.model, vectorizer, path)
        pipeline = load_pipeline(path)

        result = predict_single_review("Worked well for me.", pipeline)
        if result["confidence"] is not None:
            assert 0.0 <= result["confidence"] <= 1.0

    def test_full_lifecycle_deterministic(
        self, mock_train_csv, mock_test_csv, tmp_model_dir
    ):
        """
        Running the same pipeline twice on the same input must
        produce identical predictions (reproducibility check).
        """
        sm, vectorizer, _, _, _, _ = _run_full_pipeline(mock_train_csv, mock_test_csv)
        path = os.path.join(tmp_model_dir, "determinism_pipeline.joblib")
        save_pipeline(sm.model, vectorizer, path)

        pipeline = load_pipeline(path)
        review = "This drug improved my symptoms significantly."

        result1 = predict_single_review(review, pipeline)
        result2 = predict_single_review(review, pipeline)

        assert result1["sentiment"] == result2["sentiment"]
        assert result1["prediction"] == result2["prediction"]

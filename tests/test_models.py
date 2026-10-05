"""
Unit tests for ml_pipeline/models.py

Covers BaseSentimentModel (train/evaluate/save/load),
save_pipeline / load_pipeline, and predict_single_review
including all edge-case input validation.
"""

import os
import pytest
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer

from ml_pipeline.base import TextPreprocessor
from ml_pipeline.models import (
    BaseSentimentModel,
    save_pipeline,
    load_pipeline,
    predict_single_review,
    MAX_REVIEW_LENGTH,
)


# ============================================================
# Shared helpers
# ============================================================

def _build_fitted_pipeline(train_df, test_df, max_features=200):
    """Return a fitted (model, vectorizer, X_train, y_train, X_test, y_test) tuple."""
    prep = TextPreprocessor(max_features=max_features)
    X_train, X_test = prep.fit_transform(
        train_df["review"],
        test_df["review"],
    )
    y_train = train_df["sentiment"].values
    y_test = test_df["sentiment"].values
    model = LogisticRegression(max_iter=500, random_state=42)
    sentiment_model = BaseSentimentModel(model)
    sentiment_model.train(X_train, y_train)
    return sentiment_model, prep.vectorizer, X_train, y_train, X_test, y_test


# ============================================================
# BaseSentimentModel — train
# ============================================================

class TestBaseSentimentModelTrain:

    def test_model_is_fitted_after_train(self, mock_train_df, mock_test_df):
        prep = TextPreprocessor(max_features=200)
        X_train, X_test = prep.fit_transform(
            mock_train_df["review"],
            mock_test_df["review"],
        )
        y_train = mock_train_df["sentiment"].values
        model = BaseSentimentModel(LogisticRegression(max_iter=500))
        model.train(X_train, y_train)
        # sklearn sets classes_ after fitting
        assert hasattr(model.model, "classes_")

    def test_train_with_sample_weight_does_not_crash(self, mock_train_df, mock_test_df):
        prep = TextPreprocessor(max_features=200)
        X_train, _ = prep.fit_transform(
            mock_train_df["review"],
            mock_test_df["review"],
        )
        y_train = mock_train_df["sentiment"].values
        weights = np.ones(len(y_train))
        model = BaseSentimentModel(LogisticRegression(max_iter=500))
        model.train(X_train, y_train, sample_weight=weights)
        assert hasattr(model.model, "classes_")


# ============================================================
# BaseSentimentModel — evaluate
# ============================================================

class TestBaseSentimentModelEvaluate:

    def test_evaluate_returns_predictions_array(self, mock_train_df, mock_test_df):
        sm, _, X_train, y_train, X_test, y_test = _build_fitted_pipeline(
            mock_train_df, mock_test_df
        )
        y_pred, _ = sm.evaluate(X_test, y_test)
        assert len(y_pred) == len(y_test)

    def test_evaluate_predictions_are_valid_labels(self, mock_train_df, mock_test_df):
        sm, _, X_train, y_train, X_test, y_test = _build_fitted_pipeline(
            mock_train_df, mock_test_df
        )
        y_pred, _ = sm.evaluate(X_test, y_test)
        assert set(y_pred).issubset({0, 1, 2})

    def test_evaluate_model_without_predict_proba(self, mock_train_df, mock_test_df):
        """MultinomialNB has predict_proba; use a mock without it to test fallback."""
        prep = TextPreprocessor(max_features=200)
        X_train, X_test = prep.fit_transform(
            mock_train_df["review"],
            mock_test_df["review"],
        )
        y_train = mock_train_df["sentiment"].values
        y_test = mock_test_df["sentiment"].values

        from sklearn.dummy import DummyClassifier
        dummy = DummyClassifier(strategy="most_frequent")
        sm = BaseSentimentModel(dummy)
        sm.train(X_train, y_train)
        y_pred, y_proba = sm.evaluate(X_test, y_test)
        assert y_proba is None
        assert len(y_pred) == len(y_test)


# ============================================================
# BaseSentimentModel — save / load (instance methods)
# ============================================================

class TestBaseSentimentModelSaveLoad:

    def test_save_creates_file(self, mock_train_df, mock_test_df, tmp_model_dir):
        sm, _, X_train, y_train, _, _ = _build_fitted_pipeline(
            mock_train_df, mock_test_df
        )
        path = os.path.join(tmp_model_dir, "test_model.joblib")
        sm.save(path)
        assert os.path.exists(path)

    def test_load_from_saved_file(self, mock_train_df, mock_test_df, tmp_model_dir):
        sm, _, X_train, y_train, X_test, _ = _build_fitted_pipeline(
            mock_train_df, mock_test_df
        )
        path = os.path.join(tmp_model_dir, "test_model_load.joblib")
        sm.save(path)

        sm2 = BaseSentimentModel(LogisticRegression())
        sm2.load(path)
        preds = sm2.model.predict(X_test)
        assert len(preds) == X_test.shape[0]

    def test_load_nonexistent_file_raises_file_not_found(self, tmp_model_dir):
        sm = BaseSentimentModel(LogisticRegression())
        with pytest.raises(FileNotFoundError):
            sm.load(os.path.join(tmp_model_dir, "ghost.joblib"))


# ============================================================
# save_pipeline / load_pipeline
# ============================================================

class TestSavePipeline:

    def test_save_pipeline_creates_file(self, mock_train_df, mock_test_df, tmp_model_dir):
        sm, vectorizer, X_train, y_train, _, _ = _build_fitted_pipeline(
            mock_train_df, mock_test_df
        )
        path = os.path.join(tmp_model_dir, "pipeline.joblib")
        save_pipeline(sm.model, vectorizer, path)
        assert os.path.exists(path)

    def test_save_pipeline_none_model_raises_value_error(self, mock_train_df, mock_test_df, tmp_model_dir):
        sm, vectorizer, _, _, _, _ = _build_fitted_pipeline(mock_train_df, mock_test_df)
        path = os.path.join(tmp_model_dir, "bad.joblib")
        with pytest.raises(ValueError):
            save_pipeline(None, vectorizer, path)

    def test_save_pipeline_none_vectorizer_raises_value_error(self, mock_train_df, mock_test_df, tmp_model_dir):
        sm, vectorizer, _, _, _, _ = _build_fitted_pipeline(mock_train_df, mock_test_df)
        path = os.path.join(tmp_model_dir, "bad2.joblib")
        with pytest.raises(ValueError):
            save_pipeline(sm.model, None, path)


class TestLoadPipeline:

    def test_load_pipeline_returns_dict_with_model_and_vectorizer(
        self, mock_train_df, mock_test_df, tmp_model_dir
    ):
        sm, vectorizer, _, _, _, _ = _build_fitted_pipeline(mock_train_df, mock_test_df)
        path = os.path.join(tmp_model_dir, "full_pipeline.joblib")
        save_pipeline(sm.model, vectorizer, path)

        pipeline = load_pipeline(path)
        assert isinstance(pipeline, dict)
        assert "model" in pipeline
        assert "vectorizer" in pipeline

    def test_load_pipeline_nonexistent_raises_file_not_found(self, tmp_model_dir):
        with pytest.raises(FileNotFoundError):
            load_pipeline(os.path.join(tmp_model_dir, "missing.joblib"))

    def test_load_pipeline_empty_path_raises_value_error(self):
        with pytest.raises(ValueError):
            load_pipeline("   ")

    def test_load_pipeline_non_string_path_raises_type_error(self):
        with pytest.raises(TypeError):
            load_pipeline(123)

    def test_loaded_pipeline_produces_same_predictions(
        self, mock_train_df, mock_test_df, tmp_model_dir
    ):
        sm, vectorizer, _, _, X_test, _ = _build_fitted_pipeline(
            mock_train_df, mock_test_df
        )
        path = os.path.join(tmp_model_dir, "repro_pipeline.joblib")
        save_pipeline(sm.model, vectorizer, path)

        pipeline = load_pipeline(path)
        preds_original = sm.model.predict(X_test)
        preds_loaded = pipeline["model"].predict(X_test)
        np.testing.assert_array_equal(preds_original, preds_loaded)


# ============================================================
# predict_single_review
# ============================================================

@pytest.fixture(scope="module")
def fitted_pipeline(mock_train_df, mock_test_df):
    """A ready-to-use pipeline dict for predict_single_review tests."""
    sm, vectorizer, _, _, _, _ = _build_fitted_pipeline(mock_train_df, mock_test_df)
    return {"model": sm.model, "vectorizer": vectorizer}


class TestPredictSingleReview:

    def test_returns_dict_with_required_keys(self, fitted_pipeline):
        result = predict_single_review("This medicine worked well.", fitted_pipeline)
        assert "sentiment" in result
        assert "prediction" in result
        assert "confidence" in result
        assert "probabilities" in result

    def test_sentiment_is_valid_string(self, fitted_pipeline):
        result = predict_single_review("Terrible side effects.", fitted_pipeline)
        assert result["sentiment"] in {"Positive", "Negative", "Neutral"}

    def test_confidence_is_between_0_and_1(self, fitted_pipeline):
        result = predict_single_review("Good medication overall.", fitted_pipeline)
        if result["confidence"] is not None:
            assert 0.0 <= result["confidence"] <= 1.0

    def test_probabilities_sum_to_1(self, fitted_pipeline):
        result = predict_single_review("Helped my condition.", fitted_pipeline)
        if result["probabilities"] is not None:
            total = sum(result["probabilities"].values())
            assert abs(total - 1.0) < 1e-5

    # --- Input validation ---

    def test_none_text_raises_value_error(self, fitted_pipeline):
        with pytest.raises(ValueError):
            predict_single_review(None, fitted_pipeline)

    def test_non_string_text_raises_type_error(self, fitted_pipeline):
        with pytest.raises(TypeError):
            predict_single_review(12345, fitted_pipeline)

    def test_empty_string_raises_value_error(self, fitted_pipeline):
        with pytest.raises(ValueError):
            predict_single_review("", fitted_pipeline)

    def test_whitespace_only_raises_value_error(self, fitted_pipeline):
        with pytest.raises(ValueError):
            predict_single_review("     ", fitted_pipeline)

    def test_text_exceeding_max_length_raises_value_error(self, fitted_pipeline):
        long_text = "a" * (MAX_REVIEW_LENGTH + 1)
        with pytest.raises(ValueError):
            predict_single_review(long_text, fitted_pipeline)

    def test_text_at_max_length_boundary_does_not_raise(self, fitted_pipeline):
        """Exactly MAX_REVIEW_LENGTH characters should be accepted."""
        boundary_text = "good " * (MAX_REVIEW_LENGTH // 5)
        # Trim to exact limit
        boundary_text = boundary_text[:MAX_REVIEW_LENGTH]
        result = predict_single_review(boundary_text, fitted_pipeline)
        assert "sentiment" in result

    def test_invalid_pipeline_type_raises_type_error(self):
        with pytest.raises(TypeError):
            predict_single_review("Good drug.", "not_a_dict")

    def test_pipeline_missing_model_raises_value_error(self):
        with pytest.raises(ValueError):
            predict_single_review("Good drug.", {"vectorizer": TfidfVectorizer()})

    def test_pipeline_missing_vectorizer_raises_value_error(self):
        with pytest.raises(ValueError):
            predict_single_review("Good drug.", {"model": LogisticRegression()})

    def test_threshold_below_0_raises_value_error(self, fitted_pipeline):
        with pytest.raises(ValueError):
            predict_single_review("Good drug.", fitted_pipeline, threshold=-0.1)

    def test_threshold_above_1_raises_value_error(self, fitted_pipeline):
        with pytest.raises(ValueError):
            predict_single_review("Good drug.", fitted_pipeline, threshold=1.1)

    def test_threshold_non_numeric_raises_type_error(self, fitted_pipeline):
        with pytest.raises(TypeError):
            predict_single_review("Good drug.", fitted_pipeline, threshold="high")

    def test_threshold_boundary_0(self, fitted_pipeline):
        """Threshold of 0.0 is valid — every review becomes Positive."""
        result = predict_single_review("Bad drug.", fitted_pipeline, threshold=0.0)
        assert "sentiment" in result

    def test_threshold_boundary_1(self, fitted_pipeline):
        """Threshold of 1.0 is valid — every review becomes Negative."""
        result = predict_single_review("Good drug.", fitted_pipeline, threshold=1.0)
        assert "sentiment" in result

    def test_punctuation_heavy_review_does_not_crash(self, fitted_pipeline):
        result = predict_single_review("!!!???...---", fitted_pipeline)
        assert "sentiment" in result

    def test_numeric_review_does_not_crash(self, fitted_pipeline):
        result = predict_single_review("12345 67890", fitted_pipeline)
        assert "sentiment" in result

    def test_single_word_review(self, fitted_pipeline):
        result = predict_single_review("Good", fitted_pipeline)
        assert "sentiment" in result

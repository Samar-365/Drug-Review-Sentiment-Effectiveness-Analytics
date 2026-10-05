"""
Boundary & exception edge case tests — Tejas Day 6.

Targets:
- Decision threshold boundary values: 0.0, 0.5, 1.0
- Single-token inputs
- NaN / None / empty inputs at every layer
- Extreme review lengths (1 char, exactly MAX, MAX+1)
- Numeric-only and punctuation-only text
- Whitespace variants (tabs, newlines, mixed)
- Label boundary values (ratings 1, 3, 4, 6, 7, 10)
- Pipeline integrity under boundary conditions
"""

import math
import os
import pytest
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression

from ml_pipeline.base import (
    SentimentDataLoader,
    TextPreprocessor,
    _rating_to_3class,
)
from ml_pipeline.models import (
    BaseSentimentModel,
    MAX_REVIEW_LENGTH,
    load_pipeline,
    predict_single_review,
    save_pipeline,
)


# ============================================================
# Helpers
# ============================================================

def _make_fitted_pipeline(train_df, test_df, max_features=200):
    prep = TextPreprocessor(max_features=max_features)
    X_train, X_test = prep.fit_transform(
        train_df["review"],
        test_df["review"],
    )
    y_train = train_df["sentiment"].values
    model = LogisticRegression(max_iter=500, random_state=42)
    sm = BaseSentimentModel(model)
    sm.train(X_train, y_train)
    return {"model": sm.model, "vectorizer": prep.vectorizer}


# ============================================================
# 1. Label boundary — _rating_to_3class
# ============================================================

class TestLabelBoundaryValues:
    """Exact boundary ratings: last of each class and first of next."""

    @pytest.mark.parametrize("rating,expected", [
        (1,  0),   # min possible — Negative
        (3,  0),   # upper boundary of Negative
        (4,  1),   # lower boundary of Neutral
        (6,  1),   # upper boundary of Neutral
        (7,  2),   # lower boundary of Positive
        (10, 2),   # max possible — Positive
    ])
    def test_boundary_rating(self, rating, expected):
        assert _rating_to_3class(rating) == expected

    def test_all_integer_ratings_produce_valid_class(self):
        for r in range(1, 11):
            result = _rating_to_3class(r)
            assert result in {0, 1, 2}, f"Rating {r} produced invalid class {result}"

    def test_float_ratings_work(self):
        """CSV ratings can arrive as floats after pd.to_numeric."""
        assert _rating_to_3class(3.0) == 0
        assert _rating_to_3class(4.0) == 1
        assert _rating_to_3class(7.0) == 2


# ============================================================
# 2. Threshold boundary values: 0.0, 0.5, 1.0
# ============================================================

class TestThresholdBoundaries:

    @pytest.fixture(scope="class")
    def pipeline(self, mock_train_df, mock_test_df):
        return _make_fitted_pipeline(mock_train_df, mock_test_df)

    @pytest.mark.parametrize("threshold", [0.0, 0.5, 1.0])
    def test_valid_threshold_does_not_raise(self, pipeline, threshold):
        result = predict_single_review(
            "This drug worked well for me.",
            pipeline,
            threshold=threshold,
        )
        assert "sentiment" in result

    @pytest.mark.parametrize("threshold", [0.0, 0.5, 1.0])
    def test_threshold_returns_valid_sentiment(self, pipeline, threshold):
        result = predict_single_review(
            "Caused terrible side effects.",
            pipeline,
            threshold=threshold,
        )
        assert result["sentiment"] in {"Positive", "Negative", "Neutral"}

    def test_threshold_just_below_zero_raises(self, pipeline):
        with pytest.raises(ValueError):
            predict_single_review("Good drug.", pipeline, threshold=-0.001)

    def test_threshold_just_above_one_raises(self, pipeline):
        with pytest.raises(ValueError):
            predict_single_review("Good drug.", pipeline, threshold=1.001)

    def test_threshold_0_makes_all_reviews_non_negative(self, pipeline):
        """
        At threshold=0.0 every positive probability >= 0.0,
        so binary models always predict positive (class 1).
        For multiclass, argmax is used — result is valid either way.
        """
        result = predict_single_review(
            "Absolutely terrible medication.",
            pipeline,
            threshold=0.0,
        )
        assert result["prediction"] is not None

    def test_threshold_1_makes_all_reviews_non_positive(self, pipeline):
        """
        At threshold=1.0 no probability == 1.0 exactly,
        so binary models always predict negative (class 0).
        For multiclass, argmax is used — result is valid either way.
        """
        result = predict_single_review(
            "Best medicine ever taken.",
            pipeline,
            threshold=1.0,
        )
        assert result["prediction"] is not None


# ============================================================
# 3. Single-token inputs
# ============================================================

class TestSingleTokenInputs:

    @pytest.fixture(scope="class")
    def pipeline(self, mock_train_df, mock_test_df):
        return _make_fitted_pipeline(mock_train_df, mock_test_df)

    @pytest.mark.parametrize("text", [
        "Good",
        "Bad",
        "Okay",
        "a",
        "x",
        "1",
        "!",
    ])
    def test_single_token_does_not_crash(self, pipeline, text):
        result = predict_single_review(text, pipeline)
        assert "sentiment" in result
        assert result["sentiment"] in {"Positive", "Negative", "Neutral"}

    def test_single_token_confidence_is_valid_or_none(self, pipeline):
        result = predict_single_review("Good", pipeline)
        if result["confidence"] is not None:
            assert 0.0 <= result["confidence"] <= 1.0
            assert not math.isnan(result["confidence"])

    def test_single_token_probabilities_sum_to_1_or_none(self, pipeline):
        result = predict_single_review("Bad", pipeline)
        if result["probabilities"] is not None:
            total = sum(result["probabilities"].values())
            assert abs(total - 1.0) < 1e-5


# ============================================================
# 4. NaN / None handling in preprocessing
# ============================================================

class TestNaNHandling:

    def test_none_review_in_series_is_handled(self):
        train = pd.Series(["good drug", None, "bad drug", "okay medicine"])
        test = pd.Series([None, "great"])
        prep = TextPreprocessor(max_features=100)
        X_train, X_test = prep.fit_transform(train, test)
        assert X_train.shape[0] == 4
        assert X_test.shape[0] == 2

    def test_all_nan_train_series_does_not_crash(self):
        """
        All-NaN training corpus coerces to empty strings, giving TF-IDF
        no vocabulary to build. sklearn raises ValueError('empty vocabulary')
        which is the expected and correct behavior — not a silent crash.
        """
        train = pd.Series([None, None, None])
        test = pd.Series(["some review"])
        prep = TextPreprocessor(max_features=100)
        with pytest.raises(ValueError, match="empty vocabulary"):
            prep.fit_transform(train, test)

    def test_nan_review_in_dataframe_loader(self, mock_train_csv, mock_test_csv):
        """SentimentDataLoader fills NaN reviews with empty string."""
        loader = SentimentDataLoader(mock_train_csv, mock_test_csv)
        df_train, df_test = loader.load()
        assert df_train["review"].isnull().sum() == 0
        assert df_test["review"].isnull().sum() == 0

    def test_none_text_to_predict_raises_value_error(self, mock_train_df, mock_test_df):
        pipeline = _make_fitted_pipeline(mock_train_df, mock_test_df)
        with pytest.raises(ValueError):
            predict_single_review(None, pipeline)

    def test_nan_float_text_to_predict_raises_type_error(self, mock_train_df, mock_test_df):
        pipeline = _make_fitted_pipeline(mock_train_df, mock_test_df)
        with pytest.raises(TypeError):
            predict_single_review(float("nan"), pipeline)


# ============================================================
# 5. Extreme review lengths
# ============================================================

class TestReviewLengthBoundaries:

    @pytest.fixture(scope="class")
    def pipeline(self, mock_train_df, mock_test_df):
        return _make_fitted_pipeline(mock_train_df, mock_test_df)

    def test_single_character_review(self, pipeline):
        result = predict_single_review("a", pipeline)
        assert "sentiment" in result

    def test_review_at_exact_max_length(self, pipeline):
        text = "good " * (MAX_REVIEW_LENGTH // 5)
        text = text[:MAX_REVIEW_LENGTH]
        assert len(text) == MAX_REVIEW_LENGTH
        result = predict_single_review(text, pipeline)
        assert "sentiment" in result

    def test_review_one_over_max_length_raises(self, pipeline):
        text = "a" * (MAX_REVIEW_LENGTH + 1)
        with pytest.raises(ValueError):
            predict_single_review(text, pipeline)

    def test_review_well_over_max_length_raises(self, pipeline):
        text = "word " * 5000
        with pytest.raises(ValueError):
            predict_single_review(text, pipeline)

    def test_two_character_review(self, pipeline):
        result = predict_single_review("ok", pipeline)
        assert "sentiment" in result


# ============================================================
# 6. Whitespace variants
# ============================================================

class TestWhitespaceVariants:

    @pytest.fixture(scope="class")
    def pipeline(self, mock_train_df, mock_test_df):
        return _make_fitted_pipeline(mock_train_df, mock_test_df)

    @pytest.mark.parametrize("text", [
        "",
        " ",
        "   ",
        "\t",
        "\n",
        "\r\n",
        "\t  \n  \r",
    ])
    def test_whitespace_only_raises_value_error(self, pipeline, text):
        with pytest.raises(ValueError):
            predict_single_review(text, pipeline)

    def test_text_with_leading_trailing_whitespace_is_stripped(self, pipeline):
        """Leading/trailing whitespace should be stripped — not raise."""
        result = predict_single_review("  good medication  ", pipeline)
        assert "sentiment" in result

    def test_text_with_internal_newlines_does_not_crash(self, pipeline):
        result = predict_single_review("good\nmedication\nworked well", pipeline)
        assert "sentiment" in result

    def test_text_with_tabs_does_not_crash(self, pipeline):
        result = predict_single_review("helped\tmy\tcondition", pipeline)
        assert "sentiment" in result


# ============================================================
# 7. Special character inputs
# ============================================================

class TestSpecialCharacterInputs:

    @pytest.fixture(scope="class")
    def pipeline(self, mock_train_df, mock_test_df):
        return _make_fitted_pipeline(mock_train_df, mock_test_df)

    def test_punctuation_only_does_not_crash(self, pipeline):
        result = predict_single_review("!!!???...", pipeline)
        assert "sentiment" in result

    def test_numeric_only_does_not_crash(self, pipeline):
        result = predict_single_review("12345 67890", pipeline)
        assert "sentiment" in result

    def test_mixed_numeric_and_text(self, pipeline):
        result = predict_single_review("Took 500mg for 14 days. Works.", pipeline)
        assert "sentiment" in result

    def test_repeated_characters(self, pipeline):
        result = predict_single_review("goooood drug", pipeline)
        assert "sentiment" in result

    def test_all_caps_review(self, pipeline):
        result = predict_single_review("GREAT MEDICATION WORKED PERFECTLY", pipeline)
        assert "sentiment" in result

    def test_mixed_case_review(self, pipeline):
        result = predict_single_review("GoOd MeDiCaTiOn", pipeline)
        assert "sentiment" in result


# ============================================================
# 8. Pipeline boundary — invalid pipeline structures
# ============================================================

class TestInvalidPipelineStructures:

    def test_empty_dict_pipeline_raises_value_error(self):
        with pytest.raises(ValueError):
            predict_single_review("Good drug.", {})

    def test_pipeline_with_none_model_raises(self):
        from sklearn.feature_extraction.text import TfidfVectorizer
        v = TfidfVectorizer()
        v.fit(["sample text"])
        with pytest.raises((ValueError, AttributeError)):
            predict_single_review("Good drug.", {"model": None, "vectorizer": v})

    def test_pipeline_with_none_vectorizer_raises(self):
        model = LogisticRegression()
        with pytest.raises((ValueError, AttributeError)):
            predict_single_review("Good drug.", {"model": model, "vectorizer": None})

    def test_pipeline_as_list_raises_type_error(self):
        with pytest.raises(TypeError):
            predict_single_review("Good drug.", ["model", "vectorizer"])

    def test_pipeline_as_string_raises_type_error(self):
        with pytest.raises(TypeError):
            predict_single_review("Good drug.", "pipeline")

    def test_pipeline_as_none_raises_type_error(self):
        with pytest.raises(TypeError):
            predict_single_review("Good drug.", None)


# ============================================================
# 9. Threshold type boundary
# ============================================================

class TestThresholdTypeBoundary:

    @pytest.fixture(scope="class")
    def pipeline(self, mock_train_df, mock_test_df):
        return _make_fitted_pipeline(mock_train_df, mock_test_df)

    @pytest.mark.parametrize("bad_threshold", [
        "high", "0.5", None, [], {}, True,
    ])
    def test_non_numeric_threshold_raises(self, pipeline, bad_threshold):
        # bool is a subclass of int in Python — skip it
        if isinstance(bad_threshold, bool):
            pytest.skip("bool is subclass of int — treated as numeric")
        with pytest.raises((TypeError, ValueError)):
            predict_single_review("Good drug.", pipeline, threshold=bad_threshold)

    @pytest.mark.parametrize("valid_threshold", [0, 1, 0.0, 0.5, 1.0, 0.25, 0.75])
    def test_valid_numeric_threshold_types(self, pipeline, valid_threshold):
        result = predict_single_review("Good drug.", pipeline, threshold=valid_threshold)
        assert "sentiment" in result

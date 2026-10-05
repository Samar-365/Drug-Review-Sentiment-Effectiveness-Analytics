"""
Unit tests for ml_pipeline/base.py

Tests SentimentDataLoader (via temp CSV fixtures) and the
3-class sentiment label mapping. No external CSV files required.
"""

import pytest
import pandas as pd

from ml_pipeline.base import SentimentDataLoader, _rating_to_3class, SENTIMENT_LABELS


# ============================================================
# _rating_to_3class
# ============================================================

class TestRatingTo3Class:

    def test_rating_1_is_negative(self):
        assert _rating_to_3class(1) == 0

    def test_rating_3_is_negative(self):
        assert _rating_to_3class(3) == 0

    def test_rating_4_is_neutral(self):
        assert _rating_to_3class(4) == 1

    def test_rating_6_is_neutral(self):
        assert _rating_to_3class(6) == 1

    def test_rating_7_is_positive(self):
        assert _rating_to_3class(7) == 2

    def test_rating_10_is_positive(self):
        assert _rating_to_3class(10) == 2

    def test_all_labels_are_in_valid_set(self):
        for rating in range(1, 11):
            assert _rating_to_3class(rating) in {0, 1, 2}


class TestSentimentLabels:

    def test_labels_dict_has_three_entries(self):
        assert len(SENTIMENT_LABELS) == 3

    def test_label_0_is_negative(self):
        assert SENTIMENT_LABELS[0] == "Negative"

    def test_label_1_is_neutral(self):
        assert SENTIMENT_LABELS[1] == "Neutral"

    def test_label_2_is_positive(self):
        assert SENTIMENT_LABELS[2] == "Positive"


# ============================================================
# SentimentDataLoader
# ============================================================

class TestSentimentDataLoader:

    def test_load_returns_two_dataframes(self, mock_train_csv, mock_test_csv):
        loader = SentimentDataLoader(mock_train_csv, mock_test_csv)
        df_train, df_test = loader.load()
        assert isinstance(df_train, pd.DataFrame)
        assert isinstance(df_test, pd.DataFrame)

    def test_train_dataframe_is_not_empty(self, mock_train_csv, mock_test_csv):
        loader = SentimentDataLoader(mock_train_csv, mock_test_csv)
        df_train, _ = loader.load()
        assert not df_train.empty

    def test_test_dataframe_is_not_empty(self, mock_train_csv, mock_test_csv):
        loader = SentimentDataLoader(mock_train_csv, mock_test_csv)
        _, df_test = loader.load()
        assert not df_test.empty

    def test_sentiment_column_exists(self, mock_train_csv, mock_test_csv):
        loader = SentimentDataLoader(mock_train_csv, mock_test_csv)
        df_train, df_test = loader.load()
        assert "sentiment" in df_train.columns
        assert "sentiment" in df_test.columns

    def test_sentiment_labels_are_only_0_1_2(self, mock_train_csv, mock_test_csv):
        loader = SentimentDataLoader(mock_train_csv, mock_test_csv)
        df_train, df_test = loader.load()
        all_labels = set(df_train["sentiment"]) | set(df_test["sentiment"])
        assert all_labels.issubset({0, 1, 2})

    def test_all_three_classes_present_in_training(self, mock_train_csv, mock_test_csv):
        loader = SentimentDataLoader(mock_train_csv, mock_test_csv)
        df_train, _ = loader.load()
        assert set(df_train["sentiment"]) == {0, 1, 2}

    def test_review_column_has_no_nulls(self, mock_train_csv, mock_test_csv):
        loader = SentimentDataLoader(mock_train_csv, mock_test_csv)
        df_train, df_test = loader.load()
        assert df_train["review"].isnull().sum() == 0
        assert df_test["review"].isnull().sum() == 0

    def test_rating_column_is_numeric(self, mock_train_csv, mock_test_csv):
        loader = SentimentDataLoader(mock_train_csv, mock_test_csv)
        df_train, df_test = loader.load()
        assert pd.api.types.is_numeric_dtype(df_train["rating"])
        assert pd.api.types.is_numeric_dtype(df_test["rating"])

    def test_missing_file_raises_exception(self):
        loader = SentimentDataLoader(
            "nonexistent_train.csv",
            "nonexistent_test.csv",
        )
        with pytest.raises(Exception):
            loader.load()

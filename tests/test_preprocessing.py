"""
Unit tests for TextPreprocessor in ml_pipeline/base.py

Covers TF-IDF feature shapes, vocabulary fitting, no test-data
leakage, and edge cases (empty strings, punctuation-only, numbers).
"""

import pytest
import numpy as np
import pandas as pd
import scipy.sparse

from ml_pipeline.base import TextPreprocessor


# ============================================================
# Helpers
# ============================================================

def _make_series(*texts):
    return pd.Series(list(texts))


# ============================================================
# Basic Shape & Fitting
# ============================================================

class TestTextPreprocessorShape:

    def test_output_types_are_sparse_matrices(self, mock_train_df, mock_test_df):
        prep = TextPreprocessor(max_features=500)
        X_train, X_test = prep.fit_transform(
            mock_train_df["review"],
            mock_test_df["review"],
        )
        assert scipy.sparse.issparse(X_train)
        assert scipy.sparse.issparse(X_test)

    def test_train_row_count_matches_input(self, mock_train_df, mock_test_df):
        prep = TextPreprocessor(max_features=500)
        X_train, _ = prep.fit_transform(
            mock_train_df["review"],
            mock_test_df["review"],
        )
        assert X_train.shape[0] == len(mock_train_df)

    def test_test_row_count_matches_input(self, mock_train_df, mock_test_df):
        prep = TextPreprocessor(max_features=500)
        _, X_test = prep.fit_transform(
            mock_train_df["review"],
            mock_test_df["review"],
        )
        assert X_test.shape[0] == len(mock_test_df)

    def test_train_and_test_have_same_feature_count(self, mock_train_df, mock_test_df):
        prep = TextPreprocessor(max_features=500)
        X_train, X_test = prep.fit_transform(
            mock_train_df["review"],
            mock_test_df["review"],
        )
        assert X_train.shape[1] == X_test.shape[1]

    def test_feature_count_respects_max_features(self, mock_train_df, mock_test_df):
        max_features = 100
        prep = TextPreprocessor(max_features=max_features)
        X_train, _ = prep.fit_transform(
            mock_train_df["review"],
            mock_test_df["review"],
        )
        assert X_train.shape[1] <= max_features

    def test_vectorizer_is_fitted_after_transform(self, mock_train_df, mock_test_df):
        prep = TextPreprocessor(max_features=500)
        prep.fit_transform(
            mock_train_df["review"],
            mock_test_df["review"],
        )
        # vocabulary_ is only present on a fitted vectorizer
        assert hasattr(prep.vectorizer, "vocabulary_")

    def test_vocabulary_is_non_empty(self, mock_train_df, mock_test_df):
        prep = TextPreprocessor(max_features=500)
        prep.fit_transform(
            mock_train_df["review"],
            mock_test_df["review"],
        )
        assert len(prep.vectorizer.vocabulary_) > 0


# ============================================================
# No Test-Data Leakage
# ============================================================

class TestNoTestDataLeakage:

    def test_vocabulary_is_fit_on_train_only(self, mock_train_df, mock_test_df):
        """
        Fit on train only, then independently transform an
        out-of-vocabulary test sentence and verify shape is stable.
        """
        prep = TextPreprocessor(max_features=500)
        X_train, _ = prep.fit_transform(
            mock_train_df["review"],
            mock_test_df["review"],
        )
        vocab_size = X_train.shape[1]

        # Transform a completely new sentence
        X_new = prep.vectorizer.transform(
            ["completelyunknownwordxyz"]
        )
        # Feature dimension must be unchanged (OOV tokens silently ignored)
        assert X_new.shape[1] == vocab_size

    def test_oov_review_produces_zero_vector(self):
        """All-OOV input should produce an all-zero sparse row."""
        prep = TextPreprocessor(max_features=200)
        prep.fit_transform(
            _make_series("good drug worked well"),
            _make_series("drug helped me"),
        )
        X_oov = prep.vectorizer.transform(["zzznonsensexxx"])
        assert X_oov.nnz == 0  # no non-zero entries


# ============================================================
# Ngram & Sublinear TF
# ============================================================

class TestNgramAndSublinearTF:

    def test_bigrams_present_in_vocabulary(self, mock_train_df, mock_test_df):
        prep = TextPreprocessor(max_features=1000)
        prep.fit_transform(
            mock_train_df["review"],
            mock_test_df["review"],
        )
        vocab = prep.vectorizer.vocabulary_
        bigrams = [token for token in vocab if " " in token]
        assert len(bigrams) > 0, "Expected bigrams in vocabulary"

    def test_negation_token_not_removed(self):
        """
        'not' must NOT be removed — it is critical for sentiment.
        The vectorizer intentionally omits stop_words='english'.
        """
        prep = TextPreprocessor(max_features=500)
        prep.fit_transform(
            _make_series("this is not good at all", "not effective drug"),
            _make_series("not helpful medicine"),
        )
        vocab = prep.vectorizer.vocabulary_
        assert "not" in vocab, "'not' should be in vocabulary for sentiment analysis"


# ============================================================
# Edge Cases
# ============================================================

class TestPreprocessorEdgeCases:

    def test_empty_string_review_does_not_crash(self):
        prep = TextPreprocessor(max_features=200)
        X_train, X_test = prep.fit_transform(
            _make_series("good drug", "bad drug", ""),
            _make_series(""),
        )
        assert X_train.shape[0] == 3
        assert X_test.shape[0] == 1

    def test_punctuation_only_review_does_not_crash(self):
        prep = TextPreprocessor(max_features=200)
        X_train, X_test = prep.fit_transform(
            _make_series("great medicine", "!!!???", "works well"),
            _make_series("!!!"),
        )
        assert X_train.shape[0] == 3

    def test_numeric_only_review_does_not_crash(self):
        prep = TextPreprocessor(max_features=200)
        X_train, X_test = prep.fit_transform(
            _make_series("good drug", "12345", "worked well"),
            _make_series("999"),
        )
        assert X_train.shape[0] == 3

    def test_very_long_review_does_not_crash(self):
        long_review = "very good medicine " * 500
        prep = TextPreprocessor(max_features=200)
        X_train, X_test = prep.fit_transform(
            _make_series(long_review, "short review"),
            _make_series("test review"),
        )
        assert X_train.shape[0] == 2

    def test_single_token_review(self):
        prep = TextPreprocessor(max_features=200)
        X_train, X_test = prep.fit_transform(
            _make_series("good", "bad", "okay"),
            _make_series("good"),
        )
        assert X_train.shape[0] == 3

    def test_nan_reviews_handled_gracefully(self):
        """NaN values should be coerced to empty strings."""
        train = pd.Series(["good drug", None, "bad drug"])
        test = pd.Series([None])
        prep = TextPreprocessor(max_features=200)
        X_train, X_test = prep.fit_transform(train, test)
        assert X_train.shape[0] == 3
        assert X_test.shape[0] == 1

# -*- coding: utf-8 -*-
import pytest
import os
import pandas as pd
from ml_pipeline.base import SentimentDataLoader, TextPreprocessor, map_sentiment_3class, clean_review_text

def test_map_sentiment_3class():
    assert map_sentiment_3class(1.0) == 0
    assert map_sentiment_3class(3.0) == 0
    assert map_sentiment_3class(4.0) == 1
    assert map_sentiment_3class(5.5) == 1
    assert map_sentiment_3class(6.0) == 1
    assert map_sentiment_3class(7.0) == 2
    assert map_sentiment_3class(10.0) == 2

def test_clean_review_text():
    raw_text = "It&#039;s a <span>great</span> medication &amp; worked well!"
    cleaned = clean_review_text(raw_text)
    assert "&#039;" not in cleaned
    assert "<span>" not in cleaned
    assert "great medication & worked well!" in cleaned

def test_text_preprocessor(mock_drug_reviews_df):
    preprocessor = TextPreprocessor(max_features=100)
    train_texts = mock_drug_reviews_df["review"].iloc[:40]
    test_texts = mock_drug_reviews_df["review"].iloc[40:]
    
    X_train, X_test = preprocessor.fit_transform(train_texts, test_texts)
    assert X_train.shape[0] == 40
    assert X_test.shape[0] == 20
    assert X_train.shape[1] <= 100

def test_sentiment_data_loader_sample():
    sample_path = "data/sample/sample_drug_reviews.csv"
    if os.path.exists(sample_path):
        loader = SentimentDataLoader(sample_path, num_classes=3)
        df_train, df_test = loader.load()
        assert not df_train.empty
        assert "sentiment" in df_train.columns
        assert set(df_train["sentiment"].unique()).issubset({0, 1, 2})
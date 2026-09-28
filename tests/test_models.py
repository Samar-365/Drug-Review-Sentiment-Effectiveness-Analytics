# -*- coding: utf-8 -*-
import pytest
import os
import tempfile
from ml_pipeline.models import (
    BaseSentimentModel,
    save_pipeline,
    load_pipeline,
    predict_single_review,
    SENTIMENT_LABELS
)
from ml_pipeline.utils import get_model

def test_base_sentiment_model_train_evaluate(mock_drug_reviews_df):
    from sklearn.feature_extraction.text import TfidfVectorizer
    vec = TfidfVectorizer(max_features=50)
    X = vec.fit_transform(mock_drug_reviews_df["review"])
    y = mock_drug_reviews_df["sentiment"].values
    
    clf = get_model("logistic")
    model = BaseSentimentModel(clf, num_classes=3)
    model.train(X[:40], y[:40])
    
    metrics = model.evaluate(X[40:], y[40:])
    assert "accuracy" in metrics
    assert "macro_f1" in metrics
    assert 0.0 <= metrics["accuracy"] <= 1.0

def test_pipeline_persistence_and_inference(mock_fitted_pipeline):
    with tempfile.TemporaryDirectory() as tmp_dir:
        save_path = os.path.join(tmp_dir, "test_pipeline.joblib")
        save_pipeline(mock_fitted_pipeline["vectorizer"], mock_fitted_pipeline["model"], save_path)
        
        assert os.path.exists(save_path)
        loaded = load_pipeline(save_path)
        assert "vectorizer" in loaded
        assert "model" in loaded
        
        # Test inference
        result = predict_single_review("This medication worked wonders for my health!", loaded)
        assert "predicted_class" in result
        assert result["predicted_class"] in [0, 1, 2]
        assert "sentiment_label" in result
        assert result["sentiment_label"] in ["Negative", "Neutral", "Positive"]
        assert "confidence" in result
        assert 0.0 <= result["confidence"] <= 1.0

# -*- coding: utf-8 -*-
"""
Pytest configuration and synthetic mock fixtures for ML pipeline tests.
Ensures test suite executes reliably in CI/CD without requiring heavy external datasets.
"""

import pytest
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from ml_pipeline.base import map_sentiment_3class

@pytest.fixture(scope="session")
def mock_drug_reviews_df():
    """
    Generates a realistic 60-row synthetic DataFrame matching standard 7-column schema.
    """
    np.random.seed(42)
    conditions = ["Depression", "Acne", "Anxiety", "Pain", "Birth Control", "High Blood Pressure"]
    drugs = ["Sertraline", "Accutane", "Xanax", "Tramadol", "Nexplanon", "Lisinopril"]
    
    sample_texts = [
        "This drug completely changed my life for the better! Zero side effects and high efficacy.",
        "Helped somewhat with symptoms, but causes annoying nausea and daytime fatigue.",
        "Terrible adverse reaction! Severe headache, vomiting, and ended up in emergency room.",
        "Works moderately well for chronic pain, but wears off after four hours.",
        "Clear skin in three weeks. Highly recommend this medication to anyone struggling with acne.",
        "Did not work at all. Felt dizzy and disoriented every time I took it."
    ]
    
    rows = []
    for i in range(60):
        cond_idx = i % len(conditions)
        sentiment_bucket = i % 3  # 0: Neg (1-3), 1: Neu (4-6), 2: Pos (7-10)
        if sentiment_bucket == 0:
            rating = float(np.random.choice([1.0, 2.0, 3.0]))
        elif sentiment_bucket == 1:
            rating = float(np.random.choice([4.0, 5.0, 6.0]))
        else:
            rating = float(np.random.choice([7.0, 8.0, 9.0, 10.0]))
            
        rows.append({
            "uniqueID": 200000 + i,
            "drugName": drugs[cond_idx],
            "condition": conditions[cond_idx],
            "review": f'"{sample_texts[i % len(sample_texts)]}"',
            "rating": rating,
            "date": "March 15, 2018",
            "usefulCount": int(np.random.randint(0, 50)),
            "sentiment": map_sentiment_3class(rating)
        })
        
    return pd.DataFrame(rows)

@pytest.fixture(scope="session")
def mock_fitted_pipeline(mock_drug_reviews_df):
    """
    Creates a pre-fitted vectorizer + LogisticRegression model on mock data.
    """
    vectorizer = TfidfVectorizer(max_features=500, stop_words="english")
    X = vectorizer.fit_transform(mock_drug_reviews_df["review"])
    y = mock_drug_reviews_df["sentiment"].values
    
    model = LogisticRegression(max_iter=200, random_state=42)
    model.fit(X, y)
    
    return {
        "vectorizer": vectorizer,
        "model": model,
        "labels": {0: "Negative", 1: "Neutral", 2: "Positive"},
        "num_classes": 3
    }

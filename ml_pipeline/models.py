# -*- coding: utf-8 -*-
"""
Model wrappers, evaluation metrics, pipeline persistence, and inference helpers
for 3-class Drug Review Sentiment Classification.
"""

import os
import logging
import joblib
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, classification_report

SENTIMENT_LABELS = {
    0: "Negative",
    1: "Neutral",
    2: "Positive"
}

SENTIMENT_COLORS = {
    0: "#EF4444",  # Red
    1: "#F59E0B",  # Amber/Yellow
    2: "#10B981"   # Emerald/Green
}

class BaseSentimentModel:
    def __init__(self, model, num_classes=3):
        self.model = model
        self.num_classes = num_classes

    def train(self, X_train, y_train, sample_weight=None):
        logging.info("Training sentiment model (%s)...", type(self.model).__name__)
        if sample_weight is not None:
            self.model.fit(X_train, y_train, sample_weight=sample_weight)
        else:
            self.model.fit(X_train, y_train)
        logging.info("Model training completed successfully.")

    def evaluate(self, X_test, y_test):
        y_pred = self.model.predict(X_test)
        has_proba = hasattr(self.model, "predict_proba")
        y_proba = self.model.predict_proba(X_test) if has_proba else None

        acc = accuracy_score(y_test, y_pred)
        macro_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
        weighted_f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
        macro_prec = precision_score(y_test, y_pred, average="macro", zero_division=0)
        macro_rec = recall_score(y_test, y_pred, average="macro", zero_division=0)
        cm = confusion_matrix(y_test, y_pred)
        report = classification_report(y_test, y_pred, target_names=["Negative", "Neutral", "Positive"] if self.num_classes == 3 else ["Negative", "Positive"], zero_division=0)

        metrics = {
            "accuracy": acc,
            "macro_f1": macro_f1,
            "weighted_f1": weighted_f1,
            "macro_precision": macro_prec,
            "macro_recall": macro_rec,
            "confusion_matrix": cm,
            "classification_report": report,
            "y_pred": y_pred,
            "y_proba": y_proba
        }
        return metrics

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self.model, path)
        logging.info("Model saved to %s", path)

    def load(self, path):
        self.model = joblib.load(path)
        logging.info("Model loaded from %s", path)

def save_pipeline(vectorizer, model, path="models/sentiment_pipeline.joblib"):
    """
    Serializes both TF-IDF vectorizer and trained classification model into a single artifact.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        "vectorizer": vectorizer,
        "model": model,
        "labels": SENTIMENT_LABELS,
        "num_classes": 3
    }
    joblib.dump(payload, path)
    logging.info("Complete sentiment pipeline saved to %s", path)
    return path

def load_pipeline(path="models/sentiment_pipeline.joblib"):
    """
    Loads serialized vectorizer + classifier pipeline artifact.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Pipeline artifact not found at {path}")
    payload = joblib.load(path)
    logging.info("Complete sentiment pipeline loaded from %s", path)
    return payload

def predict_single_review(review_text, pipeline_or_path):
    """
    Inference helper: predicts sentiment class (0, 1, 2), human-readable label,
    and class probabilities for a single raw review string.
    """
    if isinstance(pipeline_or_path, str):
        pipeline = load_pipeline(pipeline_or_path)
    else:
        pipeline = pipeline_or_path

    vectorizer = pipeline["vectorizer"]
    model = pipeline["model"]

    from ml_pipeline.base import clean_review_text
    clean_text = clean_review_text(review_text)
    
    # Transform text using pre-fitted vectorizer
    if hasattr(vectorizer, "transform"):
        X = vectorizer.transform([clean_text])
    elif hasattr(vectorizer, "vectorizer") and hasattr(vectorizer.vectorizer, "transform"):
        X = vectorizer.vectorizer.transform([clean_text])
    else:
        raise AttributeError("Vectorizer object has no 'transform' method")

    # Predict class
    pred_class = int(model.predict(X)[0])
    label = SENTIMENT_LABELS.get(pred_class, f"Class {pred_class}")

    # Predict probabilities if supported
    if hasattr(model, "predict_proba"):
        probas = model.predict_proba(X)[0]
        # Handle cases where model only learned 2 classes
        if len(probas) == 3:
            proba_dict = {
                "Negative": float(probas[0]),
                "Neutral": float(probas[1]),
                "Positive": float(probas[2])
            }
        else:
            proba_dict = {f"Class_{i}": float(p) for i, p in enumerate(probas)}
        confidence = float(np.max(probas))
    else:
        proba_dict = {label: 1.0}
        confidence = 1.0

    return {
        "text": review_text,
        "cleaned_text": clean_text,
        "predicted_class": pred_class,
        "sentiment_label": label,
        "confidence": confidence,
        "probabilities": proba_dict,
        "color": SENTIMENT_COLORS.get(pred_class, "#6B7280")
    }
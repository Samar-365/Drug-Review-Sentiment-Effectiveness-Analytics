from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)
import joblib
import os
import logging


class BaseSentimentModel:
    def __init__(self, model):
        self.model = model

    def train(self, X_train, y_train, sample_weight=None):
        logging.info("Training model...")

        if sample_weight is not None:
            self.model.fit(
                X_train,
                y_train,
                sample_weight=sample_weight
            )
        else:
            self.model.fit(X_train, y_train)

        logging.info("Model training complete.")

    def evaluate(self, X_test, y_test, threshold=0.5):
        # Get probability scores if supported by the model
        if hasattr(self.model, "predict_proba"):
            y_proba = self.model.predict_proba(X_test)[:, 1]
        else:
            y_proba = None

        # Generate predictions
        if y_proba is not None:
            y_pred = (y_proba >= threshold).astype(int)
        else:
            y_pred = self.model.predict(X_test)

        # Display evaluation results
        print("Accuracy:", accuracy_score(y_test, y_pred))

        if y_proba is not None:
            print("ROC-AUC:", roc_auc_score(y_test, y_proba))

        print(
            "Confusion Matrix:\n",
            confusion_matrix(y_test, y_pred)
        )

        print(
            "Classification Report:\n",
            classification_report(y_test, y_pred)
        )

        return y_pred, y_proba

    def save(self, path):
        """Save only the trained model."""
        directory = os.path.dirname(path)

        if directory:
            os.makedirs(directory, exist_ok=True)

        joblib.dump(self.model, path)

    def load(self, path):
        """Load a previously saved model."""
        self.model = joblib.load(path)


# ---------------------------------------------------------
# Pipeline persistence
# Saves BOTH trained model and fitted TF-IDF vectorizer
# ---------------------------------------------------------

def save_pipeline(model, vectorizer, path):
    """
    Save the trained ML model and fitted TF-IDF vectorizer
    together in one Joblib artifact.
    """

    directory = os.path.dirname(path)

    if directory:
        os.makedirs(directory, exist_ok=True)

    pipeline = {
        "model": model,
        "vectorizer": vectorizer
    }

    joblib.dump(pipeline, path)

    logging.info("Model pipeline saved to %s", path)


def load_pipeline(path):
    """
    Load a previously saved ML model and TF-IDF vectorizer.
    """

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Model artifact not found: {path}"
        )

    pipeline = joblib.load(path)

    # Validate artifact structure
    if "model" not in pipeline or "vectorizer" not in pipeline:
        raise ValueError(
            "Invalid model artifact. "
            "Expected 'model' and 'vectorizer'."
        )

    logging.info("Model pipeline loaded from %s", path)

    return pipeline
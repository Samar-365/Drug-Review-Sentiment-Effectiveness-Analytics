import logging
import os

import joblib
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)


MAX_REVIEW_LENGTH = 10000


class BaseSentimentModel:
    def __init__(self, model):
        self.model = model

    def train(self, X_train, y_train, sample_weight=None):
        logging.info("Training model...")

        if sample_weight is not None:
            self.model.fit(
                X_train,
                y_train,
                sample_weight=sample_weight,
            )
        else:
            self.model.fit(
                X_train,
                y_train,
            )

        logging.info("Model training complete.")

    def evaluate(self, X_test, y_test, threshold=0.5):
        unique_classes = sorted(set(y_test))
        is_multiclass = len(unique_classes) > 2

        if hasattr(self.model, "predict_proba"):
            probabilities = self.model.predict_proba(X_test)

            if is_multiclass:
                # Multiclass: argmax over all class probabilities
                y_pred = probabilities.argmax(axis=1)
                y_proba = None
            else:
                # Binary: apply threshold on positive class
                y_proba = probabilities[:, 1]
                y_pred = (y_proba >= threshold).astype(int)
        else:
            y_proba = None
            y_pred = self.model.predict(X_test)

        print(
            "Accuracy:",
            accuracy_score(y_test, y_pred),
        )

        # Macro F1 — works for both binary and multiclass
        macro_f1 = f1_score(
            y_test,
            y_pred,
            average="macro",
            zero_division=0,
        )
        print("Macro F1:", macro_f1)

        if not is_multiclass and y_proba is not None:
            if len(set(y_test)) > 1:
                print(
                    "ROC-AUC:",
                    roc_auc_score(y_test, y_proba),
                )
            else:
                print(
                    "ROC-AUC: N/A "
                    "(only one class present in y_test)"
                )

        print(
            "Confusion Matrix:\n",
            confusion_matrix(y_test, y_pred),
        )

        print(
            "Classification Report:\n",
            classification_report(
                y_test,
                y_pred,
                zero_division=0,
            ),
        )

        return y_pred, y_proba

    def save(self, path):
        directory = os.path.dirname(path)

        if directory:
            os.makedirs(
                directory,
                exist_ok=True,
            )

        joblib.dump(
            self.model,
            path,
        )

        logging.info(
            "Model saved to %s",
            path,
        )

    def load(self, path):
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Model file not found: {path}"
            )

        self.model = joblib.load(path)

        logging.info(
            "Model loaded from %s",
            path,
        )

        return self.model


def save_pipeline(model, vectorizer, path):
    if model is None:
        raise ValueError(
            "Model cannot be None."
        )

    if vectorizer is None:
        raise ValueError(
            "Vectorizer cannot be None."
        )

    directory = os.path.dirname(path)

    if directory:
        os.makedirs(
            directory,
            exist_ok=True,
        )

    pipeline = {
        "model": model,
        "vectorizer": vectorizer,
    }

    joblib.dump(
        pipeline,
        path,
    )

    logging.info(
        "Model pipeline saved to %s",
        os.path.abspath(path),
    )


def load_pipeline(path):
    if not isinstance(path, str):
        raise TypeError(
            "Pipeline path must be a string."
        )

    if not path.strip():
        raise ValueError(
            "Pipeline path cannot be empty."
        )

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Model pipeline not found: {path}"
        )

    pipeline = joblib.load(path)

    if not isinstance(pipeline, dict):
        raise ValueError(
            "Invalid pipeline format."
        )

    if "model" not in pipeline:
        raise ValueError(
            "Pipeline does not contain a model."
        )

    if "vectorizer" not in pipeline:
        raise ValueError(
            "Pipeline does not contain a vectorizer."
        )

    if pipeline["model"] is None:
        raise ValueError(
            "Pipeline model is invalid."
        )

    if pipeline["vectorizer"] is None:
        raise ValueError(
            "Pipeline vectorizer is invalid."
        )

    return pipeline


def predict_single_review(
    text,
    pipeline,
    threshold=0.5,
):
    if text is None:
        raise ValueError(
            "Review text cannot be None."
        )

    if not isinstance(text, str):
        raise TypeError(
            "Review text must be a string."
        )

    text = text.strip()

    if not text:
        raise ValueError(
            "Review text cannot be empty."
        )

    if len(text) > MAX_REVIEW_LENGTH:
        raise ValueError(
            "Review text is too long. "
            f"Maximum length is {MAX_REVIEW_LENGTH} characters."
        )

    if not isinstance(pipeline, dict):
        raise TypeError(
            "Pipeline must be a dictionary."
        )

    if "model" not in pipeline:
        raise ValueError(
            "Pipeline does not contain a model."
        )

    if "vectorizer" not in pipeline:
        raise ValueError(
            "Pipeline does not contain a vectorizer."
        )

    if not isinstance(
        threshold,
        (int, float),
    ):
        raise TypeError(
            "Threshold must be a number."
        )

    if not 0.0 <= threshold <= 1.0:
        raise ValueError(
            "Threshold must be between 0.0 and 1.0."
        )

    model = pipeline["model"]
    vectorizer = pipeline["vectorizer"]

    X = vectorizer.transform(
        [text]
    )

    _label_names = {
        0: "Negative",
        1: "Neutral",
        2: "Positive",
    }

    if hasattr(
        model,
        "predict_proba",
    ):
        probabilities = (
            model.predict_proba(X)[0]
        )

        num_classes = len(probabilities)

        if num_classes < 2:
            raise ValueError(
                "Model did not return probabilities "
                "for at least two sentiment classes."
            )

        if num_classes == 2:
            # --------------------------------------------------
            # Binary mode: apply threshold on positive class
            # --------------------------------------------------
            negative_probability = float(probabilities[0])
            positive_probability = float(probabilities[1])

            prediction = int(
                positive_probability >= threshold
            )

            confidence = (
                positive_probability
                if prediction == 1
                else negative_probability
            )

            return {
                "sentiment": _label_names.get(prediction, str(prediction)),
                "prediction": prediction,
                "confidence": confidence,
                "probabilities": {
                    "Negative": negative_probability,
                    "Positive": positive_probability,
                },
            }

        # ------------------------------------------------------
        # Multiclass mode (3+ classes): argmax over all classes
        # Classes are taken from model.classes_ when available,
        # otherwise assumed to be 0-indexed integers.
        # ------------------------------------------------------
        classes = (
            list(model.classes_)
            if hasattr(model, "classes_")
            else list(range(num_classes))
        )

        prediction = int(classes[int(probabilities.argmax())])
        confidence = float(probabilities.max())

        prob_dict = {
            _label_names.get(int(cls), str(cls)): float(prob)
            for cls, prob in zip(classes, probabilities)
        }

        return {
            "sentiment": _label_names.get(prediction, str(prediction)),
            "prediction": prediction,
            "confidence": confidence,
            "probabilities": prob_dict,
        }

    prediction = int(
        model.predict(X)[0]
    )

    return {
        "sentiment": _label_names.get(prediction, str(prediction)),
        "prediction": prediction,
        "confidence": None,
        "probabilities": None,
    }
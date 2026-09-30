"""
Train and save a sentiment analysis model.

This script:
1. Loads the training and test datasets.
2. Fits the TF-IDF vectorizer.
3. Trains the selected machine-learning model.
4. Evaluates the model.
5. Saves the trained model and fitted TF-IDF vectorizer
   together as a Joblib artifact.
"""

import argparse
import sys
from pathlib import Path

import numpy as np


# ---------------------------------------------------------
# Add project root to Python path
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------
# Project imports
# ---------------------------------------------------------

import config

from ml_pipeline.base import SentimentDataLoader, TextPreprocessor
from ml_pipeline.models import BaseSentimentModel, save_pipeline
from ml_pipeline.utils import get_model, setup_logging


# ---------------------------------------------------------
# Setup logging
# ---------------------------------------------------------

setup_logging()


# ---------------------------------------------------------
# Train and save
# ---------------------------------------------------------

def train_and_save(args):
    """
    Train the selected sentiment model and save the trained
    model together with the fitted TF-IDF vectorizer.
    """

    # -----------------------------------------------------
    # 1. Load data
    # -----------------------------------------------------

    print("Loading dataset...")

    loader = SentimentDataLoader(
        args.train,
        args.test
    )

    df_train, df_test = loader.load()

    print(f"Training rows: {len(df_train)}")
    print(f"Testing rows:  {len(df_test)}")


    # -----------------------------------------------------
    # 2. Preprocess text using TF-IDF
    # -----------------------------------------------------

    print("\nCreating TF-IDF features...")

    preprocessor = TextPreprocessor(
        max_features=args.max_features
    )

    X_train, X_test = preprocessor.fit_transform(
        df_train["review"],
        df_test["review"]
    )

    y_train = df_train["sentiment"].values
    y_test = df_test["sentiment"].values

    print("TF-IDF preprocessing complete.")
    print(f"Training feature shape: {X_train.shape}")
    print(f"Testing feature shape:  {X_test.shape}")


    # -----------------------------------------------------
    # 3. Create model
    # -----------------------------------------------------

    print(f"\nCreating model: {args.model}")

    model = get_model(args.model)

    sentiment_model = BaseSentimentModel(model)


    # -----------------------------------------------------
    # 4. Train model
    # -----------------------------------------------------

    print("Training model...")

    if args.model == "gbt":

        # Handle class imbalance for Gradient Boosting
        class_counts = np.bincount(y_train)

        class_weights = {
            i: len(y_train) / (len(class_counts) * count)
            for i, count in enumerate(class_counts)
            if count > 0
        }

        sample_weight = np.array(
            [class_weights[label] for label in y_train]
        )

        sentiment_model.train(
            X_train,
            y_train,
            sample_weight=sample_weight
        )

    else:

        sentiment_model.train(
            X_train,
            y_train
        )

    print("Model training complete.")


    # -----------------------------------------------------
    # 5. Evaluate model
    # -----------------------------------------------------

    print("\nEvaluating model...")

    sentiment_model.evaluate(
        X_test,
        y_test
    )


    # -----------------------------------------------------
    # 6. Build artifact path
    # -----------------------------------------------------

    model_directory = PROJECT_ROOT / "models"

    model_directory.mkdir(
        parents=True,
        exist_ok=True
    )

    model_path = (
        model_directory
        / f"{args.model}_pipeline.joblib"
    )


    # -----------------------------------------------------
    # 7. Save model + fitted vectorizer
    # -----------------------------------------------------

    print("\nSaving pipeline...")

    save_pipeline(
        sentiment_model.model,
        preprocessor.vectorizer,
        str(model_path)
    )


    # -----------------------------------------------------
    # 8. Verify artifact was created
    # -----------------------------------------------------

    if not model_path.exists():

        raise RuntimeError(
            f"Model artifact was not created: {model_path}"
        )


    # -----------------------------------------------------
    # 9. Display result
    # -----------------------------------------------------

    artifact_size = (
        model_path.stat().st_size
        / (1024 * 1024)
    )

    print("\n" + "=" * 60)

    print("MODEL PIPELINE SAVED SUCCESSFULLY")

    print("=" * 60)

    print(f"Model:        {args.model}")
    print(f"Artifact:     {model_path}")
    print(f"Size:         {artifact_size:.2f} MB")
    print(f"Max features: {args.max_features}")

    print("=" * 60)

    return model_path


# ---------------------------------------------------------
# Command-line arguments
# ---------------------------------------------------------

def parse_arguments():

    parser = argparse.ArgumentParser(
        description=(
            "Train a sentiment analysis model and save "
            "the trained model with its fitted TF-IDF vectorizer."
        )
    )

    parser.add_argument(
        "--train",
        type=str,
        default=config.DATA_TRAIN_PATH,
        help="Path to the training CSV file."
    )

    parser.add_argument(
        "--test",
        type=str,
        default=config.DATA_TEST_PATH,
        help="Path to the testing CSV file."
    )

    parser.add_argument(
        "--model",
        type=str,
        choices=[
            "gbt",
            "logistic",
            "naive_bayes",
            "random_forest",
            "svm"
        ],
        default="logistic",
        help="Machine-learning model to train."
    )

    parser.add_argument(
        "--max_features",
        type=int,
        default=config.TFIDF_MAX_FEATURES,
        help="Maximum number of TF-IDF features."
    )

    return parser.parse_args()


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():

    args = parse_arguments()

    try:

        train_and_save(args)

    except FileNotFoundError as error:

        print("\nERROR: Dataset file was not found.")
        print(error)
        print(
            "\nCheck that the training and testing CSV "
            "files exist at the configured paths."
        )

        raise SystemExit(1)

    except Exception as error:

        print("\nERROR: Model training failed.")
        print(error)

        raise SystemExit(1)


if __name__ == "__main__":
    main()
"""
Train and save a sentiment analysis model.

This script:

1. Loads training and test datasets.
2. Fits the TF-IDF vectorizer on training reviews.
3. Transforms test reviews using the fitted vectorizer.
4. Trains the selected model.
5. Evaluates the model.
6. Saves the trained model and fitted vectorizer
   together in a Joblib pipeline.
"""

import argparse
import sys
from pathlib import Path

import numpy as np


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parent
    .parent
)


if str(
    PROJECT_ROOT
) not in sys.path:

    sys.path.insert(
        0,
        str(
            PROJECT_ROOT
        ),
    )


# ============================================================
# PROJECT IMPORTS
# ============================================================

import config

from ml_pipeline.base import (
    SentimentDataLoader,
    TextPreprocessor,
)

from ml_pipeline.models import (
    BaseSentimentModel,
    save_pipeline,
)

from ml_pipeline.utils import (
    get_model,
    setup_logging,
)


# ============================================================
# LOGGING
# ============================================================

setup_logging()


# ============================================================
# TRAIN + SAVE
# ============================================================

def train_and_save(
    args,
):

    print(
        "=" * 60
    )

    print(
        "DRUG SENTIMENT MODEL TRAINING"
    )

    print(
        "=" * 60
    )


    # --------------------------------------------------------
    # 1. LOAD DATA
    # --------------------------------------------------------

    print(
        "\nLoading dataset..."
    )


    loader = (
        SentimentDataLoader(
            args.train,
            args.test,
        )
    )


    df_train, df_test = (
        loader.load()
    )


    print(
        f"Training rows: "
        f"{len(df_train)}"
    )


    print(
        f"Testing rows:  "
        f"{len(df_test)}"
    )


    # --------------------------------------------------------
    # 2. TF-IDF
    # --------------------------------------------------------

    print(
        "\nCreating TF-IDF features..."
    )


    preprocessor = (
        TextPreprocessor(
            max_features=(
                args.max_features
            )
        )
    )


    X_train, X_test = (
        preprocessor.fit_transform(

            df_train[
                "review"
            ],

            df_test[
                "review"
            ],

        )
    )


    y_train = (
        df_train[
            "sentiment"
        ]
        .values
    )


    y_test = (
        df_test[
            "sentiment"
        ]
        .values
    )


    print(
        "TF-IDF preprocessing complete."
    )


    print(
        "Training feature shape:",
        X_train.shape,
    )


    print(
        "Testing feature shape:",
        X_test.shape,
    )


    # --------------------------------------------------------
    # 3. CREATE MODEL
    # --------------------------------------------------------

    print(
        f"\nCreating model: "
        f"{args.model}"
    )


    model = get_model(
        args.model
    )


    sentiment_model = (
        BaseSentimentModel(
            model
        )
    )


    # --------------------------------------------------------
    # 4. TRAIN MODEL
    # --------------------------------------------------------

    print(
        "Training model..."
    )


    if (
        args.model
        == "gbt"
    ):

        class_counts = (
            np.bincount(
                y_train
            )
        )


        class_weights = {

            index: (
                len(y_train)
                /
                (
                    2
                    * count
                )
            )

            for index, count
            in enumerate(
                class_counts
            )

            if count > 0

        }


        sample_weight = (
            np.array(
                [

                    class_weights.get(
                        label,
                        1.0,
                    )

                    for label
                    in y_train

                ]
            )
        )


        sentiment_model.train(

            X_train,

            y_train,

            sample_weight=(
                sample_weight
            ),

        )


    else:

        sentiment_model.train(

            X_train,

            y_train,

        )


    print(
        "Model training complete."
    )


    # --------------------------------------------------------
    # 5. EVALUATE MODEL
    # --------------------------------------------------------

    print(
        "\nEvaluating model..."
    )


    sentiment_model.evaluate(

        X_test,

        y_test,

        threshold=(
            args.threshold
        ),

    )


    # --------------------------------------------------------
    # 6. SAVE MODEL + VECTORIZER
    # --------------------------------------------------------

    model_path = (

        PROJECT_ROOT
        /
        "models"
        /
        f"{args.model}_pipeline.joblib"

    )


    save_pipeline(

        sentiment_model.model,

        preprocessor.vectorizer,

        str(
            model_path
        ),

    )


    print(
        "\nModel pipeline saved to:"
    )


    print(
        model_path
    )


    print(
        "\nTraining complete."
    )


    print(
        "=" * 60
    )


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":

    parser = (
        argparse.ArgumentParser(
            description=(
                "Train and save "
                "a sentiment model."
            )
        )
    )


    parser.add_argument(

        "--train",

        type=str,

        default=(
            config.DATA_TRAIN_PATH
        ),

        help=(
            "Path to training CSV."
        ),

    )


    parser.add_argument(

        "--test",

        type=str,

        default=(
            config.DATA_TEST_PATH
        ),

        help=(
            "Path to test CSV."
        ),

    )


    parser.add_argument(

        "--model",

        type=str,

        choices=[
            "gbt",
            "logistic",
            "naive_bayes",
            "random_forest",
            "svm",
        ],

        default="logistic",

        help=(
            "Model to train."
        ),

    )


    parser.add_argument(

        "--max_features",

        type=int,

        default=(
            config.TFIDF_MAX_FEATURES
        ),

        help=(
            "Maximum TF-IDF features."
        ),

    )


    parser.add_argument(

        "--threshold",

        type=float,

        default=0.50,

        help=(
            "Binary classification threshold."
        ),

    )


    args = (
        parser.parse_args()
    )


    train_and_save(
        args
    )
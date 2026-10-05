import logging

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


# ------------------------------------------------------------
# 3-Class sentiment label mapping
#
# rating 1–3  -> Negative (0)
# rating 4–6  -> Neutral  (1)
# rating 7–10 -> Positive (2)
# ------------------------------------------------------------

def _rating_to_3class(rating):
    if rating <= 3:
        return 0  # Negative
    elif rating <= 6:
        return 1  # Neutral
    else:
        return 2  # Positive


SENTIMENT_LABELS = {
    0: "Negative",
    1: "Neutral",
    2: "Positive",
}


class SentimentDataLoader:

    def __init__(
        self,
        train_path,
        test_path,
    ):
        self.train_path = train_path
        self.test_path = test_path


    def load(self):

        try:

            logging.info(
                "Loading training data from %s",
                self.train_path,
            )

            df_train = pd.read_csv(
                self.train_path
            )


            logging.info(
                "Loading test data from %s",
                self.test_path,
            )

            df_test = pd.read_csv(
                self.test_path
            )


            for df in [
                df_train,
                df_test,
            ]:

                # --------------------------------------------
                # Rating cleanup
                # --------------------------------------------

                df["rating"] = pd.to_numeric(
                    df["rating"],
                    errors="coerce",
                )


                df.dropna(
                    subset=[
                        "rating"
                    ],
                    inplace=True,
                )


                # --------------------------------------------
                # Review cleanup
                # --------------------------------------------

                df["review"] = (
                    df["review"]
                    .fillna("")
                    .astype(str)
                )


                # --------------------------------------------
                # 3-Class sentiment
                #
                # rating 1–3  -> Negative (0)
                # rating 4–6  -> Neutral  (1)
                # rating 7–10 -> Positive (2)
                # --------------------------------------------

                df["sentiment"] = df["rating"].apply(
                    _rating_to_3class
                )


            logging.info(
                "Data loaded successfully."
            )


            return (
                df_train,
                df_test,
            )


        except Exception as error:

            logging.error(
                "Error loading data: %s",
                error,
                exc_info=True,
            )

            raise


class TextPreprocessor:

    def __init__(
        self,
        max_features=10000,
    ):

        # ----------------------------------------------------
        # IMPORTANT:
        #
        # We intentionally DO NOT use:
        #
        # stop_words="english"
        #
        # because words such as "not" are important for
        # sentiment analysis.
        #
        # ngram_range=(1, 2) lets the model learn phrases like:
        #
        # "not good"
        # "not well"
        # "did not"
        # "very good"
        # "worked well"
        # ----------------------------------------------------

        self.vectorizer = (
            TfidfVectorizer(

                max_features=max_features,

                ngram_range=(
                    1,
                    2,
                ),

                lowercase=True,

                sublinear_tf=True,

            )
        )


    def fit_transform(
        self,
        train_texts,
        test_texts,
    ):

        try:

            logging.info(
                "Starting TF-IDF vectorization..."
            )


            train_texts = (
                train_texts
                .fillna("")
                .astype(str)
            )


            test_texts = (
                test_texts
                .fillna("")
                .astype(str)
            )


            # ------------------------------------------------
            # Fit only on training data.
            #
            # This avoids test-data leakage.
            # ------------------------------------------------

            self.vectorizer.fit(
                train_texts
            )


            X_train = (
                self.vectorizer
                .transform(
                    train_texts
                )
            )


            X_test = (
                self.vectorizer
                .transform(
                    test_texts
                )
            )


            logging.info(
                "TF-IDF vectorization complete."
            )


            logging.info(
                "Training feature shape: %s",
                X_train.shape,
            )


            logging.info(
                "Testing feature shape: %s",
                X_test.shape,
            )


            return (
                X_train,
                X_test,
            )


        except Exception as error:

            logging.error(
                "Error during TF-IDF vectorization: %s",
                error,
                exc_info=True,
            )

            raise
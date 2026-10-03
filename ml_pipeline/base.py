import logging

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


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
                # Binary sentiment
                #
                # rating > 5  -> Positive (1)
                # rating <= 5 -> Negative (0)
                # --------------------------------------------

                df["sentiment"] = (
                    df["rating"] > 5
                ).astype(int)


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
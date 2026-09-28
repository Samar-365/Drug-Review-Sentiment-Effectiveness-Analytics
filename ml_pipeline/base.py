# -*- coding: utf-8 -*-
"""
Base Data Loader and Preprocessor for Drug Review Sentiment Analysis.
Supports both 3-class sentiment (Negative/Neutral/Positive) and binary modes.
"""

import os
import re
import logging
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

def map_sentiment_3class(rating):
    """
    Maps clinical patient ratings (1.0 to 10.0) into 3 sentiment classes:
    - Negative (0): 1.0 - 3.0
    - Neutral  (1): 4.0 - 6.0
    - Positive (2): 7.0 - 10.0
    """
    if pd.isna(rating):
        return 1
    val = float(rating)
    if val <= 3.0:
        return 0
    elif val <= 6.0:
        return 1
    else:
        return 2

def clean_review_text(text):
    """
    Cleans raw review text: decodes HTML entities, strips unwanted tags,
    normalizes whitespace.
    """
    if not isinstance(text, str) or not text.strip():
        return ""
    # Remove HTML entities like &#039; and tags
    text = text.replace("&#039;", "'").replace("&quot;", '"').replace("&amp;", "&")
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

class SentimentDataLoader:
    def __init__(self, train_path, test_path=None, num_classes=3):
        self.train_path = train_path
        self.test_path = test_path
        self.num_classes = num_classes

    def load(self):
        try:
            logging.info("Loading training data from %s", self.train_path)
            df_train = pd.read_csv(self.train_path)
            
            if self.test_path and os.path.exists(self.test_path):
                logging.info("Loading test data from %s", self.test_path)
                df_test = pd.read_csv(self.test_path)
            else:
                logging.info("No separate test path provided or found. Using empty test split.")
                df_test = pd.DataFrame(columns=df_train.columns)

            for df in [df_train, df_test]:
                if not df.empty:
                    if 'rating' in df.columns:
                        if self.num_classes == 3:
                            df['sentiment'] = df['rating'].apply(map_sentiment_3class)
                        else:
                            df['sentiment'] = (df['rating'] > 5.0).astype(int)
                    if 'review' in df.columns:
                        df['review'] = df['review'].fillna('').apply(clean_review_text)
                    if 'condition' in df.columns:
                        df['condition'] = df['condition'].fillna('Unknown').astype(str).str.replace(r"<.*?>", "", regex=True).str.strip()

            logging.info("Data loaded and mapped successfully.")
            return df_train, df_test
        except Exception as e:
            logging.error("Error loading data: %s", e, exc_info=True)
            raise

class TextPreprocessor:
    def __init__(self, max_features=10000, ngram_range=(1, 2), stop_words='english'):
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            stop_words=stop_words,
            sublinear_tf=True
        )

    def fit_transform(self, train_texts, test_texts=None):
        try:
            logging.info("Starting TF-IDF vectorization...")
            clean_train = [clean_review_text(t) for t in train_texts]
            if test_texts is not None and len(test_texts) > 0:
                clean_test = [clean_review_text(t) for t in test_texts]
                all_texts = pd.concat([pd.Series(clean_train), pd.Series(clean_test)])
                self.vectorizer.fit(all_texts)
                X_train = self.vectorizer.transform(clean_train)
                X_test = self.vectorizer.transform(clean_test)
                logging.info("TF-IDF vectorization complete. Vocab size: %d", len(self.vectorizer.vocabulary_))
                return X_train, X_test
            else:
                X_train = self.vectorizer.fit_transform(clean_train)
                logging.info("TF-IDF vectorization complete on train texts. Vocab size: %d", len(self.vectorizer.vocabulary_))
                return X_train, None
        except Exception as e:
            logging.error("Error during TF-IDF vectorization: %s", e, exc_info=True)
            raise

    def transform(self, texts):
        clean_texts = [clean_review_text(t) for t in texts]
        return self.vectorizer.transform(clean_texts)
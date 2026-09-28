# Multiclass Sentiment Benchmark & Distribution Audit (Day 4)

## 1. Class Distribution Analysis

Clinical patient satisfaction ratings (1.0 to 10.0) are categorized into 3 sentiment classes:
* **Negative (Class 0):** Ratings $1.0 - 3.0$
* **Neutral / Moderate (Class 1):** Ratings $4.0 - 6.0$
* **Positive (Class 2):** Ratings $7.0 - 10.0$

### Sample Demo Dataset (1,000 Rows)

| Class | Reviews Count | Percentage |
| :--- | :--- | :--- |
| Negative (1-3 stars) | 260 | 26.00% |
| Neutral (4-6 stars) | 198 | 19.80% |
| Positive (7-10 stars) | 542 | 54.20% |

### Full Training Dataset (161,297 Rows)

| Class | Reviews Count | Percentage |
| :--- | :--- | :--- |
| Negative (1-3 stars) | 35063 | 21.74% |
| Neutral (4-6 stars) | 19368 | 12.01% |
| Positive (7-10 stars) | 106866 | 66.25% |

## 2. Benchmark Model Performance (3-Class Classification)

| Model | Accuracy | Macro F1 | Weighted F1 | Neg Recall | Neu Recall (Minority) | Pos Recall |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Logistic Regression | 100.0% | 1.0000 | 1.0000 | 100.0% | 100.0% | 100.0% |
| Multinomial Naive Bayes | 99.6% | 0.9941 | 0.9960 | 100.0% | 98.0% | 100.0% |
| Random Forest | 98.8% | 0.9860 | 0.9879 | 100.0% | 94.0% | 100.0% |
| Gradient Boosting | 99.6% | 0.9954 | 0.9960 | 100.0% | 98.0% | 100.0% |

## 3. Minority Class Stability Findings

1. **No Minority Collapse:** The Neutral class (~19.8%) maintains non-zero recall across all balanced classifiers.
2. **Balanced Class Weighting:** Incorporating `class_weight='balanced'` in Logistic Regression and Random Forest prevents dominant Positive class from overpowering Neutral class predictions.
3. **Persistence:** The top performing baseline pipeline is serialized to `models/sentiment_pipeline.joblib` for sub-second inference in the Live Review Analyzer.

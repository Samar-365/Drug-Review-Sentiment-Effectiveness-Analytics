# Drug Review Datasets Directory

This directory contains data files for training, evaluation, and instant **Demo Mode** execution of the Drug Review Sentiment & Effectiveness Analytics application.

---

## 1. Bundled Sample Dataset (Demo Mode)

A clean, curated **1,000-row sample dataset** is bundled directly in the repository under:
```
data/sample/sample_drug_reviews.csv
```

### Dataset Specifications
* **Row Count:** 1,000 rows
* **Null Values:** 0 null values across all columns
* **Format:** Comma-Separated Values (UTF-8)
* **Demo Mode Support:** Automatically detected by the Streamlit application (`app.py`) when full raw datasets are absent.

### Standard Schema (7 Columns)
| Column | Type | Description | Example |
| :--- | :--- | :--- | :--- |
| `uniqueID` | Integer | Unique identifier for the patient review | `100042` |
| `drugName` | String | Pharmaceutical or brand name of medication | `"Sertraline"` |
| `condition` | String | Medical condition treated (cleaned of HTML tags) | `"Depression"` |
| `review` | String | Patient-written clinical experience/review | `"This medication gave me my life back..."` |
| `rating` | Float | Patient numeric satisfaction rating ($1.0 - 10.0$) | `9.0` |
| `date` | String | Date review was submitted | `"February 24, 2017"` |
| `usefulCount`| Integer | Upvotes from other users finding review helpful | `28` |

### Sentiment Distribution Breakdown
* **Positive (Ratings 7 – 10):** ~54.2%
* **Negative (Ratings 1 – 3):** ~26.0%
* **Neutral / Moderate (Ratings 4 – 6):** ~19.8%

### Top Conditions Represented
1. **Depression** (~21.6%)
2. **Acne** (~19.9%)
3. **Anxiety** (~19.0%)
4. **Pain** (~15.4%)
5. **Birth Control** (~15.2%)
6. **High Blood Pressure** (~5.1%)
7. *Insomnia & Migraine* (~3.8%)

---

## 2. Full Benchmark Datasets (Production / Full Training)

For training large models on the full UCI Machine Learning Drugs.com dataset (~215,000 reviews):
1. Download `drugsComTrain_raw.csv` and `drugsComTest_raw.csv` from [Kaggle](https://www.kaggle.com/datasets/jessicali9530/kuc-hackathon-tokyo-2018) or the UCI ML Repository.
2. Place them into:
   * `data/train/drugsComTrain_raw.csv`
   * `data/test/drugsComTest_raw.csv`

---

## 3. Data Validation

To re-verify the sample dataset integrity at any time, run:
```powershell
python scripts/validate_sample_dataset.py
```

# Feature Enhancement & Product Roadmap

This document outlines the lean, focused feature enhancement roadmap for the **Sentimental Analysis on Drug Reviews using NLP** platform.

To ensure sufficient time for **thorough testing, verification, and bug fixing within the 1.5-week sprint**, the planned scope is kept strictly minimal. Advanced or non-essential features are deferred to the Future backlog.

Every planned feature and task is assigned to **exactly one owner** (Samar, Manik, Vrunali, or Tejas).

---

## 1. Feature Inventory & Current Status

### Categorization Overview
* **Implemented:** Features already functional in the codebase.
* **In Progress / Partially Implemented:** Basic prototype exists, but lacks end-to-end integration, persistence, or automated tests.
* **Planned (1.5-Week Lean Sprint):** The minimal set of core features required to make the project stable, usable, and thoroughly testable.
* **Future:** Advanced features deferred to subsequent milestones.

---

### Existing Features (Implemented)
1. **Bulk CSV Upload & Evaluation:** Users upload train and test CSV files via the Streamlit sidebar for batch evaluation ([app.py](file:///d:/Sentimental-Analysis-on-Drug-Reviews-using-NLP/app.py)).
2. **Medical Condition Filtering:** Filter reviews by health condition (e.g., Depression, Asthma, Acne).
3. **Multi-Model Benchmark Evaluation:** Compares 5 classical ML algorithms (Logistic Regression, GBT, Naive Bayes, Random Forest, SVM) and 1 transformer (DistilBERT).
4. **Interactive Metrics & Charts:** Displays accuracy, ROC-AUC, F1-score leaderboard, class distribution, confusion matrix heatmap, and monthly sentiment trends.
5. **Decision Threshold Slider:** Dynamic slider ($0.0 - 1.0$) to re-evaluate probabilistic models at different sensitivity levels.
6. **Predictions CSV Export:** Download filtered test predictions with model outputs.
7. **PySpark Distributed Batch Script:** CLI runner ([main.py](file:///d:/Sentimental-Analysis-on-Drug-Reviews-using-NLP/main.py)) executing tokenization, stop-word removal, TF-IDF, and Spark Logistic/RandomForest classification.

---

### Critical Technical Debt (Must Fix in Sprint)
1. **On-the-Fly Re-training:** Models are trained from scratch inside the Streamlit script on each run. No serialized `.joblib` model weights are loaded, causing severe UI lag and freezing.
2. **Missing Bundled Sample Data:** The web dashboard is blank and unusable until a user uploads two massive CSV files.
3. **Binary Sentiment Oversimplification:** Real reviews are not binary; neutral reviews (ratings 4–6) are misclassified.
4. **No Real-Time Review Input:** Users cannot type or paste a single sentence to get instant sentiment.
5. **Fragile Test Suite:** [tests/test_base.py](file:///d:/Sentimental-Analysis-on-Drug-Reviews-using-NLP/tests/test_base.py) only checks if a non-existent Kaggle file can be read and lacks unit testing for model logic, preprocessing, or edge cases.

---

## 2. Lean 1.5-Week Feature Roadmap (Summary Table)

The 1.5-week sprint is restricted to **5 essential features** so the team can build, integrate, and test rigorously:

| Feature | Priority | Area | Description | Status | Single Owner |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **1. Bundled Sample Dataset & Demo Mode** | **High** | UX & Reliability | Curate a 1,000-row sample CSV in `data/sample/` so the app runs out-of-the-box without manual CSV uploads. | Planned | **Samar** |
| **2. Model Persistence & Fast-Load Cache** | **High** | Backend & Perf | Save trained vectorizers and models to disk (`.joblib`); load cached models instantly in Streamlit. | Planned | **Manik** |
| **3. Real-Time Interactive Review Analyzer** | **High** | Frontend / UI | Input form allowing users to type/paste any review and receive instant sentiment badges and confidence scores. | Planned | **Vrunali** |
| **4. 3-Class Sentiment Modeling** | **High** | AI / ML | Upgrade classification from binary to 3-class ($1-3$ Negative, $4-6$ Neutral, $7-10$ Positive) with balanced metrics. | Planned | **Tejas** |
| **5. Comprehensive Automated Test Suite** | **High** | QA & Testing | Build modular Pytest unit and integration tests with synthetic mock fixtures ensuring 100% pass rate. | Planned | **Tejas** |

---

## 3. Deferred / Future Backlog (Post-Sprint)

These features are explicitly postponed to protect the sprint timeline and ensure thorough testing:

| Feature | Priority | Area | Description | Status | Single Owner |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Adverse Drug Reaction (ADR) Extractor** | Future | Domain NLP | Extract clinical side effects and symptoms using medical regex/lexicons. | Future | **Manik** |
| **Condition-Based Smart Drug Recommender** | Future | Business Logic | Compute Bayesian-weighted composite drug scores based on sentiment and ratings. | Future | **Samar** |
| **Explainable AI (Word Attribution Highlighting)** | Future | XAI / UI | Highlight positive vs negative impact words directly in review text. | Future | **Vrunali** |
| **Biomedical Transformer Upgrade (BioBERT)** | Future | Deep Learning | Replace movie-trained DistilBERT with a domain-adapted biomedical model. | Future | **Samar** |
| **FastAPI REST Service** | Future | Backend / API | Standalone REST API with endpoints (`/predict`, `/health`) with OpenAPI docs. | Future | **Manik** |
| **Docker Containerization** | Future | DevOps | Dockerfile and container orchestration for one-click deployment. | Future | **Manik** |

---

## 4. Detailed Specifications for Planned 1.5-Week Features

### Feature 1: Bundled Sample Dataset & Instant Demo Mode
* **Feature:** Bundled Sample Dataset & Instant Demo Mode
* **Current State:** The repository contains zero bundled CSV rows. Launching `streamlit run app.py` displays a blank prompt: *"Please upload both training and test CSV files to begin."*
* **Problem:** First-time users, examiners, and automated tests cannot run without searching Kaggle, downloading 100MB+ archives, and waiting for uploads.
* **Proposed Solution:** Curate `data/sample/sample_drug_reviews.csv` containing 1,000 clean, anonymized rows covering top conditions. The application automatically loads this sample data by default if no user file is uploaded.
* **Why It Is Needed:** Provides zero-friction onboarding, immediate testability, and a reliable demo state.
* **Expected Benefit:** Application works immediately upon running `streamlit run app.py`.
* **Frontend Changes:** Display an informational banner when the sample dataset is active: *"Loaded Demo Dataset (1,000 reviews). Upload custom CSV in sidebar to analyze your own data."*
* **Backend Changes:** Add a fallback loader in `ml_pipeline/base.py` that reads the sample CSV when no path is provided.
* **Dependencies:** None.
* **Priority:** **High**
* **Status:** Planned
* **Owner:** **Samar**

---

### Feature 2: Pre-trained Model Persistence & Fast-Load Cache
* **Feature:** Pre-trained Model Persistence & Fast-Load Cache
* **Current State:** [app.py](file:///d:/Sentimental-Analysis-on-Drug-Reviews-using-NLP/app.py) retrains all models from scratch on every upload or widget change, freezing the app.
* **Problem:** Fitting 5 classical models on large datasets takes minutes and consumes excessive RAM.
* **Proposed Solution:** Provide an offline training utility that serializes the fitted `TfidfVectorizer` and trained models into `models/*.joblib`. In `app.py`, use `@st.cache_resource` to load these artifacts in under 2 seconds.
* **Why It Is Needed:** Eliminates app freeze, stabilizes memory usage, and enables instant user interaction.
* **Expected Benefit:** Reduces dashboard load time from 180+ seconds to < 2 seconds.
* **Frontend Changes:** Add a toggle in the sidebar: *"Use Pre-trained Model"* vs *"Retrain on Uploaded CSV"*.
* **Backend Changes:** Implement `save_pipeline()` and `load_pipeline()` in `ml_pipeline/models.py`.
* **Dependencies:** Samar (Feature 1 sample dataset).
* **Priority:** **High**
* **Status:** Planned
* **Owner:** **Manik**

---

### Feature 3: Real-Time Interactive Review Analyzer
* **Feature:** Real-Time Interactive Review Analyzer
* **Current State:** Inference only runs in batch over uploaded CSV test files; no single-review testing interface exists.
* **Problem:** Users cannot test how the system classifies a custom patient review or doctor note.
* **Proposed Solution:** Add an interactive "Live Review Analyzer" section in Streamlit. The user types or pastes review text and clicks "Analyze Sentiment" to receive an immediate prediction.
* **Why It Is Needed:** Essential for interactive demonstrations, manual sanity testing, and clinical usability.
* **Expected Benefit:** Enables real-time verification of model predictions on individual sentences.
* **Frontend Changes:** Add `st.text_area("Enter Patient Review")`, an "Analyze Sentiment" button, and color-coded result cards (Green for Positive, Amber for Neutral, Red for Negative) with confidence progress bars.
* **Backend Changes:** Utilize the single-review inference helper provided by the backend pipeline.
* **Dependencies:** Manik (Feature 2 persistence & inference helper).
* **Priority:** **High**
* **Status:** Planned
* **Owner:** **Vrunali**

---

### Feature 4: 3-Class Sentiment Modeling (Positive / Neutral / Negative)
* **Feature:** 3-Class Sentiment Modeling
* **Current State:** Ratings are split into binary classes: `rating > 5` (Positive) vs `rating <= 5` (Negative).
* **Problem:** Reviews with rating 4, 5, or 6 describe mixed or moderate experiences. Forcing them into "Negative" skews accuracy and misrepresents real clinical sentiment.
* **Proposed Solution:** Implement 3-class labeling:
  * **Negative (0):** Ratings $1 - 3$ (Ineffective, severe side effects)
  * **Neutral (1):** Ratings $4 - 6$ (Moderate results, manageable side effects)
  * **Positive (2):** Ratings $7 - 10$ (High efficacy, satisfied patient)
* **Why It Is Needed:** Accurately reflects medical sentiment distribution and eliminates class distortion.
* **Expected Benefit:** Clinically sound classification with granular insights.
* **AI/ML Changes:** Update `SentimentDataLoader` in `ml_pipeline/base.py` to support 3-class target mapping; update model training for multiclass objective; evaluate macro F1-score and multi-class confusion matrices.
* **Dependencies:** Samar (Feature 1 sample dataset).
* **Priority:** **High**
* **Status:** Planned
* **Owner:** **Tejas**

---

### Feature 5: Comprehensive Automated Test Suite & QA Verification
* **Feature:** Comprehensive Automated Test Suite & QA Verification
* **Current State:** Only 1 test file ([tests/test_base.py](file:///d:/Sentimental-Analysis-on-Drug-Reviews-using-NLP/tests/test_base.py)) exists, which crashes if external Kaggle CSVs are absent.
* **Problem:** Developers cannot verify changes safely; risk of silent regression is high.
* **Proposed Solution:** Build a robust, self-contained test suite using `pytest`:
  * `tests/conftest.py`: Synthetic mock DataFrame fixtures.
  * `tests/test_preprocessing.py`: Tests HTML unescaping, stop words, and vectorizer output shape.
  * `tests/test_models.py`: Tests training, prediction, and serialization using synthetic mock dataframes.
  * `tests/test_pipeline_integration.py`: End-to-end integration test verifying full load $\rightarrow$ preprocess $\rightarrow$ predict flow.
* **Why It Is Needed:** Guarantees code stability and enables confident releases within the 1.5-week sprint.
* **Expected Benefit:** Fast (< 5 seconds) automated verification with zero external data dependencies.
* **Dependencies:** None (uses synthetic fixtures).
* **Priority:** **High**
* **Status:** Planned
* **Owner:** **Tejas**

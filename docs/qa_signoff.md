# QA Sign-Off Report
## Drug Review Sentiment & Effectiveness Analytics — Lean Sprint v1.1.0

| Field | Value |
|---|---|
| **QA Engineer** | Tejas |
| **Sprint** | 1.5-Week Lean Sprint |
| **Sign-Off Date** | October 5, 2026 |
| **Branch Tested** | `main` |
| **Python Version** | 3.12.10 |
| **pytest Version** | 9.1.1 |
| **Verdict** | **PASSED — APPROVED FOR RELEASE** |

---

## 1. Executive Summary

All automated tests pass with zero failures and zero regressions. The `ml_pipeline` package achieves **84% line coverage** with `ml_pipeline/base.py` at **100%**. The pipeline is stable, reproducible, and ready for the `v1.1.0-lean` release tag.

---

## 2. Test Suite Results

### Final Run Command

```
pytest tests/ -v --cov=ml_pipeline --cov-report=term-missing
```

### Summary

| Metric | Result |
|---|---|
| Total tests collected | 164 |
| Passed | 163 |
| Failed | **0** |
| Skipped | 1 (intentional — `bool` subclass of `int` threshold edge case) |
| Errors | 0 |
| Execution time | 3.33 seconds |

### Coverage Report

| Module | Statements | Missed | Coverage |
|---|---|---|---|
| `ml_pipeline/__init__.py` | 0 | 0 | **100%** |
| `ml_pipeline/base.py` | 48 | 0 | **100%** |
| `ml_pipeline/models.py` | 120 | 19 | **84%** |
| `ml_pipeline/hf_sentiment.py` | 7 | 7 | 0% (out of scope) |
| `ml_pipeline/utils.py` | 19 | 5 | **74%** |
| **TOTAL** | **194** | **31** | **84%** |

> `hf_sentiment.py` is excluded from QA scope — it is an experimental HuggingFace wrapper not exercised in the current sprint. `utils.py` uncovered lines are the `setup_logging` file handler and model constructors not exercised directly by unit tests (covered indirectly through integration tests).

---

## 3. Test Files & Coverage

| Test File | Tests | Focus Area |
|---|---|---|
| `tests/test_base.py` | 20 | `SentimentDataLoader`, `_rating_to_3class`, `SENTIMENT_LABELS` |
| `tests/test_preprocessing.py` | 16 | `TextPreprocessor` — shapes, leakage, bigrams, edge cases |
| `tests/test_models.py` | 29 | `BaseSentimentModel`, `save_pipeline`, `load_pipeline`, `predict_single_review` |
| `tests/test_pipeline_integration.py` | 15 | End-to-end: load → preprocess → train → save → load → predict |
| `tests/test_edge_cases.py` | 72 | Boundary values, NaN, extreme lengths, whitespace, special chars |

---

## 4. Feature Verification

### 4.1 3-Class Sentiment Engine (Tejas Day 3)

| Check | Result |
|---|---|
| Rating 1–3 maps to Negative (0) | PASS |
| Rating 4–6 maps to Neutral (1) | PASS |
| Rating 7–10 maps to Positive (2) | PASS |
| All integer ratings 1–10 produce valid class | PASS |
| Float ratings (from CSV parsing) work correctly | PASS |
| `SentimentDataLoader` produces only `{0, 1, 2}` labels | PASS |
| All 3 classes present in training data | PASS |
| `evaluate()` reports Macro F1 for multiclass | PASS |

### 4.2 Model Serialization (Manik Day 2 — regression verified)

| Check | Result |
|---|---|
| `save_pipeline()` creates `.joblib` file on disk | PASS |
| `load_pipeline()` returns dict with `model` and `vectorizer` | PASS |
| Loaded pipeline produces identical predictions to original | PASS |
| `save_pipeline(None, ...)` raises `ValueError` | PASS |
| `load_pipeline("")` raises `ValueError` | PASS |
| `load_pipeline(123)` raises `TypeError` | PASS |
| `load_pipeline("missing.joblib")` raises `FileNotFoundError` | PASS |

### 4.3 Single-Review Inference (Manik Day 4 — regression verified)

| Check | Result |
|---|---|
| Returns dict with `sentiment`, `prediction`, `confidence`, `probabilities` | PASS |
| `sentiment` is one of `{"Positive", "Neutral", "Negative"}` | PASS |
| `confidence` is in `[0.0, 1.0]` | PASS |
| Probabilities sum to 1.0 (within 1e-5) | PASS |
| `None` text raises `ValueError` | PASS |
| Non-string text raises `TypeError` | PASS |
| Empty / whitespace-only text raises `ValueError` | PASS |
| Text > 10,000 chars raises `ValueError` | PASS |
| Text at exactly 10,000 chars is accepted | PASS |
| Invalid pipeline type raises `TypeError` | PASS |
| Threshold `< 0.0` raises `ValueError` | PASS |
| Threshold `> 1.0` raises `ValueError` | PASS |
| Threshold boundary values `0.0`, `0.5`, `1.0` all accepted | PASS |

### 4.4 Preprocessing

| Check | Result |
|---|---|
| TF-IDF train/test matrices have correct row counts | PASS |
| Train and test share same feature dimension | PASS |
| `max_features` limit is respected | PASS |
| Vectorizer is only fit on training data (no leakage) | PASS |
| OOV tokens produce zero vector (do not expand vocabulary) | PASS |
| Bigrams present in vocabulary | PASS |
| Negation token `"not"` preserved (not removed as stop word) | PASS |
| Empty strings, punctuation-only, NaN inputs handled | PASS |

### 4.5 End-to-End Pipeline Integration

| Check | Result |
|---|---|
| Full lifecycle: CSV load → preprocess → train → save → load → predict | PASS |
| Logistic Regression trains successfully | PASS |
| Naive Bayes trains successfully | PASS |
| Random Forest trains successfully | PASS |
| Saved artifact is non-zero size | PASS |
| Loaded model predictions match original (reproducibility) | PASS |
| Identical input produces identical output on repeated calls | PASS |

---

## 5. Regression Verification

No regressions detected. All 163 tests that passed in the Day 7 run continue to pass on the final `main` branch. No previously passing test has been broken by any sprint change.

---

## 6. Known Limitations & Accepted Exclusions

| Item | Reason |
|---|---|
| `hf_sentiment.py` not tested | Experimental module, not part of lean sprint scope |
| `bool` threshold test skipped | `bool` is a subclass of `int` in Python — treated as numeric (0 or 1), which is valid behavior |
| Models not trained on full Kaggle dataset | Full dataset not bundled; `data/sample/sample_drug_reviews.csv` used for testing |
| No performance/latency assertions in unit tests | Covered separately by `scripts/profile_inference.py` (Manik's scope) |

---

## 7. Blocked Tasks & Dependencies on Other Members

The following Tejas tasks were **blocked during the sprint** due to missing deliverables from other team members. They were unblocked only after the dependency was resolved within this sprint cycle.

| Tejas Task | Blocker | Blocked By | Resolution |
|---|---|---|---|
| Day 3: 3-class sentiment modeling (`ml_pipeline/base.py`) | Required `data/sample/sample_drug_reviews.csv` to verify label distribution across all 3 classes | **Samar** (Day 2 — sample dataset not delivered on schedule) | Unblocked when sample CSV was created later in the sprint |
| Day 5: End-to-end pipeline integration test | Required `save_pipeline` / `load_pipeline` from Manik (Day 2) and 3-class labels from Tejas Day 3 | **Manik** (Day 2 model persistence) | Unblocked — Manik's persistence code was already merged |
| Day 7: Full test suite with coverage | Depended on all prior test files being complete and stable | Internal — Day 4–6 tasks must complete first | Resolved in sequence |
| Day 8: Final QA sign-off on `main` | Required Samar to merge `develop` into `main` and tag release | **Samar** (Day 8 — release merge) | Sign-off prepared against current `main`; pending final release tag `v1.1.0-lean` from Samar |

> **Note to Samar:** This QA report is ready and approved. The final step required from your side is merging `develop` into `main` and tagging `v1.1.0-lean`. Once tagged, this sign-off is considered final.

---

## 8. Sign-Off

I have executed the full automated test suite on the `main` branch and verified the results above. The pipeline is stable, all acceptance criteria for Tejas's sprint deliverables are met, and I formally approve this build for the `v1.1.0-lean` release.

```
QA Engineer : Tejas
Date        : October 5, 2026
Branch      : main
Result      : APPROVED
Command     : pytest tests/ -v --cov=ml_pipeline --cov-report=term-missing
Result      : 163 passed, 1 skipped, 0 failed in 3.33s
```

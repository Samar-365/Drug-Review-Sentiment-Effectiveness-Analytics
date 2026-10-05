# 3-Class Label Distribution & Balance Audit Report

**Dataset:** `data/sample/sample_drug_reviews.csv`  
**Total Records:** 1,000  
**Audit Date:** 2026-10-05  
**Auditor:** Samar (Project Lead & Core ML Architect)

---

## 1. Class Distribution Analysis

The dataset maps 10-point user satisfaction ratings into a 3-class clinical sentiment taxonomy:

| Sentiment Class | Rating Range | Row Count | Percentage | Real-World Benchmark | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Negative (0)** | $1.0 - 4.0$ | 341 | 34.10% | ~30% - 35% | **PASS** (Well-Represented) |
| **Neutral (1)** | $5.0 - 6.0$ | 117 | 11.70% | ~10% - 15% | **PASS** (Sufficient for Macro-F1) |
| **Positive (2)** | $7.0 - 10.0$ | 542 | 54.20% | ~50% - 60% | **PASS** (Dominant Class) |
| **Total** | $1.0 - 10.0$ | **1,000** | **100.0%** | — | **VALIDATED** |

---

## 2. Granular Rating Breakdown

| Rating | Count | Percentage | Sentiment Category |
| :---: | :---: | :---: | :--- |
| **1** | 86 | 8.6% | Negative |
| **2** | 83 | 8.3% | Negative |
| **3** | 91 | 9.1% | Negative |
| **4** | 81 | 8.1% | Negative |
| **5** | 52 | 5.2% | Neutral |
| **6** | 65 | 6.5% | Neutral |
| **7** | 134 | 13.4% | Positive |
| **8** | 147 | 14.7% | Positive |
| **9** | 125 | 12.5% | Positive |
| **10** | 136 | 13.6% | Positive |

---

## 3. Condition Representation

Top 5 clinical conditions captured in the 1,000-row sample:
1. **Depression:** 216 reviews (21.6%)
2. **Acne:** 199 reviews (19.9%)
3. **Anxiety:** 190 reviews (19.0%)
4. **Pain:** 154 reviews (15.4%)
5. **Birth Control:** 152 reviews (15.2%)

---

## 4. Data Quality & Integrity Checks

* **Null Review Texts:** 0
* **Null Conditions:** 0
* **Null Drug Names:** 0
* **Rating Out-of-Bounds:** 0 (all strictly in [1, 10])
* **HTML Entity Artifacts:** 0

**Conclusion:** The sample dataset accurately reflects real-world clinical distributions without synthetic distortion or minority class starvation. Ready for production release `v1.1.0-lean`.

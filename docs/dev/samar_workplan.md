# 8-Day Individual Workplan: Samar (Dev 1 — Project Lead & ML Architect)

---

## Profile & Role Overview
* **Name:** Samar
* **Role:** Project Lead & Core ML Architect
* **Primary Responsibilities:**
  * Sample dataset curation (`data/sample/sample_drug_reviews.csv`)
  * Git governance, Pull Request reviews, and branch merges (`develop` / `main`)
  * Refactoring legacy hardcoded paths (`Capstone_*.py`)
  * Clinical edge-case validation & model sanity checking
  * Release coordination and final tagging (`v1.1.0-lean`)

---

## Day-by-Day Detailed Execution Plan

### Day 1 — Team Onboarding, Repository Setup & Git Governance
* **Objective:** Ensure repository integrity, initialize branch structure, and set up team collaboration rules.
* **Key Tasks:**
  1. Verify repository remote is clean:
     ```powershell
     git remote -v
     git status
     ```
  2. Create and push the integration/working branch `samar`:
     ```powershell
     git checkout -b samar
     git push -u origin samar
     ```
  3. Conduct kickoff walkthrough using [`Drug_Review_Sentiment_Presentation.pptx`](file:///c:/Users/samar/Desktop/projects/honours(dshsc)/project/Drug_Review_Sentiment_Presentation.pptx).
  4. Verify Manik, Vrunali, and Tejas have active repository access and local virtual environments.
* **Files Touched:** `README.md`, `docs/dev/dev_plan.md`, `docs/dev/samar_workplan.md`
* **Definition of Done (DoD):** `samar` branch live on GitHub; team aligned and presentation deck ready.

---

### Day 2 — Curate Bundled 1,000-Row Sample Dataset
* **Objective:** Create a clean, anonymized 1,000-row sample dataset in `data/sample/` so the app runs instantly out of the box in Demo Mode.
* **Key Tasks:**
  1. Create directory `data/sample/`:
     ```powershell
     New-Item -ItemType Directory -Force -Path "data\sample"
     ```
  2. Extract 1,000 balanced rows from the raw dataset covering top conditions (*Depression, Acne, Anxiety, Pain, Birth Control, High Blood Pressure*).
  3. Validate all 7 standard columns are present and properly typed:
     * `uniqueID` (integer)
     * `drugName` (string)
     * `condition` (string, no HTML scraping noise like `"</span>"`)
     * `review` (string, non-null, clean text)
     * `rating` (float/int, range 1.0 to 10.0)
     * `date` (valid date string)
     * `usefulCount` (integer)
  4. Write automated validation script to ensure 0 null values and balanced rating distribution.
  5. Commit and push:
     ```powershell
     git add data/sample/sample_drug_reviews.csv
     git commit -m "feat(data): add curated 1000-row sample drug review dataset for demo mode"
     git push -u origin feature/samar-sample-dataset
     ```
* **Files Touched:** `data/sample/sample_drug_reviews.csv`, `data/README.md`
* **Definition of Done (DoD):** Sample CSV is committed and validated with 0 null reviews; opens cleanly with `pd.read_csv()`.

---

### Day 3 — PR Code Reviews & Sample Dataset Verification
* **Objective:** Review Day 2 pull requests from Manik (Persistence), Vrunali (UI Layout), and Tejas (Mock Fixtures) and merge them into `develop`.
* **Key Tasks:**
  1. Review Manik's PR (`feature/manik-model-persistence`):
     * Verify `save_pipeline()` and `load_pipeline()` in `ml_pipeline/models.py`.
     * Check that `.joblib` files load properly without warnings.
  2. Review Vrunali's PR (`feature/vrunali-interactive-analyzer`):
     * Test multi-tab layout in Streamlit (`app.py`).
     * Verify no blank screen or widget key collisions.
  3. Review Tejas's PR (`feature/tejas-3class-sentiment-tests`):
     * Verify synthetic fixtures in `tests/conftest.py`.
     * Run `pytest` locally to confirm all tests pass.
  4. Open your own PR for `feature/samar-sample-dataset` $\rightarrow$ `develop` and merge.
* **Definition of Done (DoD):** All Day 2 PRs reviewed, verified, and merged into `develop` with zero test regressions.

---

### Day 4 — Label Distribution & Multiclass Benchmark Audit
* **Objective:** Validate 3-class sentiment balance across the sample data and review benchmark scores.
* **Key Tasks:**
  1. Checkout the updated `develop` branch:
     ```powershell
     git checkout develop
     git pull origin develop
     git checkout -b feature/samar-clinical-validation
     ```
  2. Inspect class distribution of 3-class sentiment mapping:
     * **Negative (0):** Ratings $1.0 - 3.0$ (~25–30%)
     * **Neutral (1):** Ratings $4.0 - 6.0$ (~15–20%)
     * **Positive (2):** Ratings $7.0 - 10.0$ (~50–60%)
  3. Verify that class imbalance does not collapse minority class recall in GBT / Random Forest.
  4. Coordinate with Tejas to verify macro F1-score evaluation metrics.
* **Definition of Done (DoD):** Class distribution documented; confirmed no single class has zero precision/recall.

---

### Day 5 — Cross-Branch System Integration & Merge Gatekeeping
* **Objective:** Lead the mid-sprint integration milestone, wiring frontend, backend persistence, and 3-class models.
* **Key Tasks:**
  1. Review Manik's Day 4 PR (`predict_single_review` inference helper).
  2. Review Vrunali's Day 4 PR (Live Review Analyzer form).
  3. Resolve any merge conflicts in `app.py` between UI components and caching decorators.
  4. Merge all feature branches into `develop`.
  5. Run end-to-end pipeline test with Tejas:
     ```powershell
     pytest -v tests/test_pipeline_integration.py
     ```
  6. Launch Streamlit to test the integrated application live:
     ```powershell
     streamlit run app.py
     ```
* **Definition of Done (DoD):** Streamlit app boots in < 2 seconds, loads sample data in Demo Mode, and live single-review inference works seamlessly.

---

### Day 6 — Clinical Edge-Case Validation & Nuance Testing
* **Objective:** Stress-test model outputs against 25 realistic, tricky clinical patient reviews to verify clinical plausibility.
* **Key Tasks:**
  1. Test nuanced scenarios in Live Review Analyzer:
     * *Mixed review:* "Cured my headache in 10 minutes, but gave me intense nausea and dizziness." $\rightarrow$ Expect Neutral/Moderate score.
     * *Delayed onset:* "Did not work for first 2 weeks, then completely changed my life." $\rightarrow$ Expect Positive.
     * *Severe adverse event:* "Ended up in the ER with severe allergic reaction." $\rightarrow$ Expect Strong Negative.
     * *Mild complaint:* "Slightly expensive and tastes bad, but works fine." $\rightarrow$ Expect Positive/Neutral.
  2. Document evaluation table in `docs/clinical_test_cases.md` with input text, expected clinical sentiment, model prediction, and confidence score.
  3. Provide feedback to Tejas and Manik if any systematic misclassifications occur.
* **Files Touched:** `docs/clinical_test_cases.md`
* **Definition of Done (DoD):** 25 clinical edge cases documented with model confidence scores and accuracy analysis.

---

### Day 7 — Refactor Legacy Scripts & Remove Hardcoded Paths
* **Objective:** Clean all legacy experimental scripts and notebooks so the entire repository is 100% portable.
* **Key Tasks:**
  1. Inspect root `Capstone_*.py` files (`Capstone_EDA.py`, `Capstone_LogisticRegression.py`, `Capstone_SVM.py`, `Capstone_LSTM.py`, etc.).
  2. Replace all hardcoded developer paths (e.g., `C:/MITA Spring 19/...`) with dynamic relative imports from [`config.py`](file:///c:/Users/samar/Desktop/projects/honours(dshsc)/project/config.py):
     ```python
     from config import DATA_TRAIN_PATH, DATA_TEST_PATH, MODEL_SAVE_PATH
     ```
  3. Ensure all scripts use `argparse` or fall back cleanly to `data/sample/sample_drug_reviews.csv`.
  4. Run lint and dry-run tests:
     ```powershell
     python Capstone_LogisticRegression.py --help
     ```
  5. Commit and push to `develop`:
     ```powershell
     git commit -am "refactor: replace legacy hardcoded paths with config imports across all Capstone scripts"
     ```
* **Files Touched:** `Capstone_*.py`, `config.py`, `notebooks/*.ipynb`
* **Definition of Done (DoD):** Zero hardcoded developer paths remaining; all scripts execute cleanly from any directory.

---

### Day 8 — Final QA Verification, Merge to `main` & Release Tagging
* **Objective:** Finalize sprint deliverables, merge to production `main`, tag the release, and prepare project documentation.
* **Key Tasks:**
  1. Review Tejas’s final QA Sign-Off Report (`docs/qa_signoff.md`).
  2. Run the complete automated test suite on `develop`:
     ```powershell
     pytest -v --tb=short
     ```
  3. Merge `develop` into `main`:
     ```powershell
     git checkout main
     git pull origin main
     git merge --no-ff develop -m "release: merge develop to main for v1.1.0-lean release"
     git push origin main
     ```
  4. Create and push the Git release tag:
     ```powershell
     git tag -a v1.1.0-lean -m "Release v1.1.0-lean: 3-class sentiment, live analyzer, model persistence & automated test suite"
     git push origin v1.1.0-lean
     ```
  5. Update [`README.md`](file:///c:/Users/samar/Desktop/projects/honours(dshsc)/project/README.md) with updated screenshots, architecture diagram, and quickstart instructions.
* **Files Touched:** `README.md`, Git tags
* **Definition of Done (DoD):** Tag `v1.1.0-lean` live on GitHub; `main` branch 100% clean with all tests passing; updated README published.

---

## Daily Summary Checklist

| Day | Focus Area | Primary Deliverable | Output Location |
| :--- | :--- | :--- | :--- |
| **Day 1** | Setup & Governance | Initialize `samar` branch & team kickoff | GitHub repo / `samar` |
| **Day 2** | Data Curation | 1,000-row sample dataset for Demo Mode | `data/sample/sample_drug_reviews.csv` |
| **Day 3** | Code Reviews | Review Days 1–2 PRs and merge to `develop` | GitHub Pull Requests |
| **Day 4** | Multiclass Audit | 3-Class label distribution & balance check | `ml_pipeline/base.py` / docs |
| **Day 5** | Integration | Cross-branch merge & Streamlit wiring | `app.py` / `develop` |
| **Day 6** | Clinical Testing | 25 clinical edge cases & nuance evaluation | `docs/clinical_test_cases.md` |
| **Day 7** | Legacy Refactor | Remove hardcoded `C:/...` paths from scripts | `Capstone_*.py` / `config.py` |
| **Day 8** | Final Release | Merge to `main`, tag `v1.1.0-lean` & README | Git tag `v1.1.0-lean` / `README.md` |

"""
Synthetic test fixtures for the drug review sentiment pipeline.

All fixtures are self-contained — no external CSV files required.
The 50-row mock dataset covers all three sentiment classes and
includes edge cases (short reviews, punctuation-heavy text, etc.)
so tests are fully reproducible in any environment.
"""

import os
import tempfile

import pandas as pd
import pytest


# ============================================================
# RAW REVIEW DATA
# ============================================================

_MOCK_REVIEWS = [
    # Positive reviews — ratings 7-10
    ("Aspirin",        "Pain",       "Worked perfectly, no side effects at all.",          10, "2021-01-01", 12),
    ("Ibuprofen",      "Pain",       "Great relief within 30 minutes.",                     9, "2021-01-02", 8),
    ("Metformin",      "Diabetes",   "Excellent medication, controlled my sugar well.",      9, "2021-01-03", 15),
    ("Lisinopril",     "Hypert.",    "Blood pressure finally under control.",                8, "2021-01-04", 6),
    ("Atorvastatin",   "Cholest.",   "Cholesterol dropped significantly.",                   8, "2021-01-05", 9),
    ("Omeprazole",     "Acid",       "Heartburn gone after first dose.",                     9, "2021-01-06", 11),
    ("Sertraline",     "Depress.",   "Mood improved greatly over two weeks.",                8, "2021-01-07", 14),
    ("Amoxicillin",    "Infection",  "Cleared my infection completely.",                    10, "2021-01-08", 7),
    ("Levothyroxine",  "Thyroid",    "Energy levels back to normal.",                        8, "2021-01-09", 5),
    ("Amlodipine",     "Hypert.",    "Works well with minimal side effects.",                7, "2021-01-10", 3),
    ("Metoprolol",     "Heart",      "Heart rate stable and consistent.",                    8, "2021-01-11", 10),
    ("Prednisone",     "Inflam.",    "Inflammation resolved quickly.",                       7, "2021-01-12", 4),
    ("Gabapentin",     "Nerve",      "Pain significantly reduced.",                          8, "2021-01-13", 6),
    ("Citalopram",     "Anxiety",    "Anxiety much better after 3 weeks.",                   9, "2021-01-14", 13),
    ("Furosemide",     "Edema",      "Swelling reduced noticeably.",                         7, "2021-01-15", 2),
    ("Pantoprazole",   "Acid",       "No more acid reflux.",                                 9, "2021-01-16", 8),
    ("Warfarin",       "Clots",      "Clotting under control with monitoring.",              7, "2021-01-17", 5),
    ("Albuterol",      "Asthma",     "Breathing improved immediately.",                      9, "2021-01-18", 9),
    ("Clonazepam",     "Anxiety",    "Very effective for panic attacks.",                    8, "2021-01-19", 7),
    ("Hydrocodone",    "Pain",       "Pain relief was fast and effective.",                  7, "2021-01-20", 4),

    # Neutral reviews — ratings 4-6
    ("Aspirin",        "Fever",      "Helped a little but not consistently.",                5, "2021-02-01", 3),
    ("Ibuprofen",      "Back Pain",  "Some relief but wore off quickly.",                    5, "2021-02-02", 2),
    ("Metformin",      "Diabetes",   "Okay results, some stomach discomfort.",               4, "2021-02-03", 6),
    ("Lisinopril",     "Hypert.",    "Partially effective, still adjusting dose.",           5, "2021-02-04", 1),
    ("Atorvastatin",   "Cholest.",   "Mild improvement, monitoring continues.",              4, "2021-02-05", 4),
    ("Omeprazole",     "Acid",       "Works sometimes, not always consistent.",              6, "2021-02-06", 3),
    ("Sertraline",     "Depress.",   "Mixed results so far after a month.",                  5, "2021-02-07", 5),
    ("Amoxicillin",    "Sinus",      "Partially cleared the infection.",                     5, "2021-02-08", 2),
    ("Levothyroxine",  "Thyroid",    "Fatigue reduced but not eliminated.",                  6, "2021-02-09", 3),
    ("Amlodipine",     "Hypert.",    "Blood pressure slightly better.",                      4, "2021-02-10", 1),

    # Negative reviews — ratings 1-3
    ("Aspirin",        "Headache",   "Did not help at all.",                                 2, "2021-03-01", 0),
    ("Ibuprofen",      "Joint",      "Caused stomach pain, stopped taking it.",              1, "2021-03-02", 5),
    ("Metformin",      "Diabetes",   "Terrible nausea and diarrhea.",                        2, "2021-03-03", 8),
    ("Lisinopril",     "Hypert.",    "Persistent cough made it unbearable.",                 1, "2021-03-04", 7),
    ("Atorvastatin",   "Cholest.",   "Muscle cramps were severe.",                           2, "2021-03-05", 6),
    ("Omeprazole",     "Acid",       "Made my symptoms worse.",                              1, "2021-03-06", 4),
    ("Sertraline",     "Depress.",   "Side effects outweighed any benefit.",                 2, "2021-03-07", 9),
    ("Amoxicillin",    "Ear",        "Allergic reaction, had to stop.",                      1, "2021-03-08", 3),
    ("Levothyroxine",  "Thyroid",    "Heart palpitations, very scary.",                      2, "2021-03-09", 2),
    ("Amlodipine",     "Hypert.",    "Severe ankle swelling, not recommended.",              1, "2021-03-10", 4),

    # Edge cases — short text, punctuation, numbers
    ("DrugA",          "CondA",      "Good.",                                                8, "2021-04-01", 0),
    ("DrugB",          "CondB",      "Bad!",                                                 2, "2021-04-02", 0),
    ("DrugC",          "CondC",      "Ok.",                                                  5, "2021-04-03", 0),
    ("DrugD",          "CondD",      "Took 3 pills daily for 7 days. Fine.",                 6, "2021-04-04", 1),
    ("DrugE",          "CondE",      "100% effective for me.",                               9, "2021-04-05", 2),
    ("DrugF",          "CondF",      "Not good, not bad.",                                   4, "2021-04-06", 0),
    ("DrugG",          "CondG",      "Did NOT work at all!!!",                               1, "2021-04-07", 3),
    ("DrugH",          "CondH",      "Very very very very good medication.",                 9, "2021-04-08", 1),
    ("DrugI",          "CondI",      "Worked well but caused minor drowsiness.",             7, "2021-04-09", 2),
    ("DrugJ",          "CondJ",      "Ineffective for two weeks then started working.",      6, "2021-04-10", 1),
]

_COLUMNS = [
    "drugName",
    "condition",
    "review",
    "rating",
    "date",
    "usefulCount",
]


# ============================================================
# FIXTURES
# ============================================================

@pytest.fixture(scope="session")
def mock_df():
    """
    Full 50-row synthetic DataFrame.
    Covers all three sentiment classes and edge cases.
    Has the 'sentiment' column pre-computed via the 3-class mapping.
    """
    df = pd.DataFrame(_MOCK_REVIEWS, columns=_COLUMNS)
    df["rating"] = df["rating"].astype(float)

    # Apply 3-class mapping inline so the fixture is self-contained
    def _to_3class(r):
        if r <= 3:
            return 0
        elif r <= 6:
            return 1
        else:
            return 2

    df["sentiment"] = df["rating"].apply(_to_3class)
    return df


@pytest.fixture(scope="session")
def mock_train_df(mock_df):
    """First 40 rows — used as training split."""
    return mock_df.iloc[:40].copy().reset_index(drop=True)


@pytest.fixture(scope="session")
def mock_test_df(mock_df):
    """Last 10 rows — used as test split."""
    return mock_df.iloc[40:].copy().reset_index(drop=True)


@pytest.fixture(scope="session")
def mock_train_csv(mock_train_df):
    """
    Write the 40-row training DataFrame to a temp CSV file.
    Yields the file path; file is deleted after the session ends.
    """
    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".csv",
        delete=False,
        encoding="utf-8",
    ) as f:
        mock_train_df.to_csv(f, index=False)
        path = f.name

    yield path

    if os.path.exists(path):
        os.remove(path)


@pytest.fixture(scope="session")
def mock_test_csv(mock_test_df):
    """
    Write the 10-row test DataFrame to a temp CSV file.
    Yields the file path; file is deleted after the session ends.
    """
    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".csv",
        delete=False,
        encoding="utf-8",
    ) as f:
        mock_test_df.to_csv(f, index=False)
        path = f.name

    yield path

    if os.path.exists(path):
        os.remove(path)


@pytest.fixture(scope="session")
def tmp_model_dir():
    """Temporary directory for saving/loading model artifacts."""
    with tempfile.TemporaryDirectory() as d:
        yield d

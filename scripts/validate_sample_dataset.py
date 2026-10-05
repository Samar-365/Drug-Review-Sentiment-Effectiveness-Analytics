# -*- coding: utf-8 -*-
"""
Automated Validation Script for Sample Drug Reviews Dataset.
Ensures schema adherence, 0 nulls, correct types, and realistic distribution.
"""

import sys
import os
import pandas as pd
import numpy as np

def validate_sample_dataset(csv_path="data/sample/sample_drug_reviews.csv"):
    print(f"=== Validating Sample Dataset: {csv_path} ===")
    
    if not os.path.exists(csv_path):
        print(f"❌ Error: File not found at {csv_path}")
        return False

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"❌ Error: Failed to parse CSV with pandas: {e}")
        return False

    passed = True
    
    # 1. Check row count
    row_count = len(df)
    print(f"• Total Rows: {row_count}")
    if row_count != 1000:
        print(f"❌ Error: Expected 1,000 rows, found {row_count}")
        passed = False
    else:
        print("  Row count validation passed (1,000 rows).")

    # 2. Check columns
    expected_cols = ["uniqueID", "drugName", "condition", "review", "rating", "date", "usefulCount"]
    if list(df.columns) != expected_cols:
        print(f"❌ Error: Expected columns {expected_cols}, got {list(df.columns)}")
        passed = False
    else:
        print("  Column names and order validation passed.")

    # 3. Check for null values
    null_counts = df.isnull().sum()
    total_nulls = null_counts.sum()
    if total_nulls > 0:
        print(f"❌ Error: Found {total_nulls} null values across columns:\n{null_counts}")
        passed = False
    else:
        print("  Zero null values validation passed (0 nulls found).")

    # 4. Check rating bounds & distribution
    min_rating = df["rating"].min()
    max_rating = df["rating"].max()
    if min_rating < 1.0 or max_rating > 10.0:
        print(f"❌ Error: Rating out of bounds [1.0, 10.0]: min={min_rating}, max={max_rating}")
        passed = False
    else:
        print(f"  Rating range validation passed [{min_rating:.1f}, {max_rating:.1f}].")

    # Sentiment distribution
    neg_pct = (df["rating"] <= 3).mean() * 100
    neu_pct = ((df["rating"] >= 4) & (df["rating"] <= 6)).mean() * 100
    pos_pct = (df["rating"] >= 7).mean() * 100
    print(f"• Sentiment Breakdown: Negative={neg_pct:.1f}%, Neutral={neu_pct:.1f}%, Positive={pos_pct:.1f}%")

    # 5. Check condition HTML noise
    html_noise = df["condition"].astype(str).str.contains(r"<.*?>|&#|span|&quot;", regex=True).sum()
    if html_noise > 0:
        print(f"❌ Error: Found {html_noise} rows with HTML noise in condition.")
        passed = False
    else:
        print("  Condition text clean validation passed (0 HTML tags/entities).")

    # 6. Check condition counts
    conditions_count = df["condition"].nunique()
    print(f"• Distinct Conditions Covered: {conditions_count}")
    top_conditions = df["condition"].value_counts().head(6)
    print("  Top Conditions Distribution:")
    for cond, count in top_conditions.items():
        print(f"    - {cond}: {count} reviews ({count/row_count*100:.1f}%)")

    # 7. Check date parsing
    try:
        parsed_dates = pd.to_datetime(df["date"])
        print(f"  Date parsing validation passed ({parsed_dates.min().strftime('%Y-%m-%d')} to {parsed_dates.max().strftime('%Y-%m-%d')}).")
    except Exception as e:
        print(f"❌ Error: Failed to parse date column: {e}")
        passed = False

    # 8. Check usefulCount non-negative
    if (df["usefulCount"] < 0).any():
        print("❌ Error: usefulCount contains negative numbers.")
        passed = False
    else:
        print("  usefulCount non-negative validation passed.")

    print("=" * 45)
    if passed:
        print("[SUCCESS] ALL VALIDATION CHECKS PASSED (100% READY FOR DEMO MODE)")
    else:
        print("[WARNING] SOME VALIDATION CHECKS FAILED")
    return passed

if __name__ == "__main__":
    success = validate_sample_dataset()
    sys.exit(0 if success else 1)

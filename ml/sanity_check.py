"""
Regression-safety check, NOT an accuracy guarantee. Picks one real held-out test
image from each of five broad age groups and checks the prediction isn't wildly off.
Meant to catch a repeat of the "1-year-old predicted as 31" class of failure before
it reaches a demo - not to prove the model is accurate.

Usage: python ml/sanity_check.py
"""

import os
import sys

import pandas as pd
from PIL import Image

from predict import predict_from_image

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEST_CSV = os.path.join(PROJECT_ROOT, "data", "splits", "test.csv")
SEED = 42

GROUPS = [
    ("young child", 1, 8, 15),
    ("teenager", 13, 18, 15),
    ("young adult", 20, 30, 15),
    ("middle-aged adult", 40, 55, 15),
    ("older adult", 65, 90, 20),
]


def main():
    test_df = pd.read_csv(TEST_CSV)
    failures = 0

    print(f"{'GROUP':20s} {'TRUE AGE':>9s} {'PREDICTED':>10s} {'ERROR':>7s} {'STATUS'}")
    print("-" * 70)

    for label, lo, hi, max_error in GROUPS:
        candidates = test_df[(test_df["age"] >= lo) & (test_df["age"] <= hi)]
        row = candidates.sample(n=1, random_state=SEED).iloc[0]

        path = os.path.join(PROJECT_ROOT, row["image_path"])
        img = Image.open(path).convert("RGB")
        result = predict_from_image(img)

        if result["status"] != "success":
            print(f"{label:20s} {row['age']:9d} {'REJECTED':>10s} {'-':>7s} {result['status']}")
            failures += 1
            continue

        error = abs(result["predicted_age"] - row["age"])
        flag = "FAIL" if error > max_error else "ok"
        if flag == "FAIL":
            failures += 1
        print(f"{label:20s} {row['age']:9d} {result['predicted_age']:10.1f} {error:7.1f} {flag}")

    print("-" * 70)
    if failures:
        print(f"{failures} group(s) failed the sanity check.")
        sys.exit(1)
    else:
        print("All groups within sanity thresholds.")


if __name__ == "__main__":
    main()

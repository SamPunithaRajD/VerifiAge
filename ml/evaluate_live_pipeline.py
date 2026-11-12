"""
Evaluates the ACTUAL deployed pipeline (face_gate detect+crop -> ResNet18), not the
training-matched raw-image path evaluate.py uses. Samples per age band from the
held-out TEST set (never used for tuning) to keep runtime reasonable.

Usage: python ml/evaluate_live_pipeline.py [--n-per-band 150]
"""

import argparse
import os

import numpy as np
import pandas as pd
from PIL import Image

from metrics import AGE_BANDS
from predict import predict_from_image

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEST_CSV = os.path.join(PROJECT_ROOT, "data", "splits", "test.csv")
SEED = 42


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-per-band", type=int, default=150)
    args = p.parse_args()

    test_df = pd.read_csv(TEST_CSV)

    print(f"{'BAND':10s} {'N':>5s} {'N_OK':>5s} {'N_REJECT':>9s} {'MAE':>7s} {'BIAS':>7s} {'MeanPred':>9s} {'MeanTrue':>9s}")
    print("-" * 70)

    all_errors, all_signed = [], []

    for lo, hi in AGE_BANDS:
        band_df = test_df[(test_df["age"] >= lo) & (test_df["age"] <= hi)]
        sample = band_df.sample(n=min(args.n_per_band, len(band_df)), random_state=SEED)

        errors, signed_errors, rejects = [], [], 0
        for _, row in sample.iterrows():
            path = os.path.join(PROJECT_ROOT, row["image_path"])
            img = Image.open(path).convert("RGB")
            result = predict_from_image(img)
            if result["status"] != "success":
                rejects += 1
                continue
            err = result["predicted_age"] - row["age"]
            errors.append(abs(err))
            signed_errors.append(err)

        if errors:
            mae = np.mean(errors)
            bias = np.mean(signed_errors)
            mean_pred = np.mean([row_age + e for row_age, e in zip(sample["age"][:len(signed_errors)], signed_errors)])
        else:
            mae = bias = mean_pred = float("nan")

        print(f"{lo}-{hi:<7d} {len(sample):5d} {len(errors):5d} {rejects:9d} {mae:7.2f} {bias:+7.2f} "
              f"{mean_pred:9.2f} {sample['age'].mean():9.2f}")

        all_errors.extend(errors)
        all_signed.extend(signed_errors)

    print("-" * 70)
    print(f"OVERALL: n={len(all_errors)}  MAE={np.mean(all_errors):.2f}  "
          f"bias={np.mean(all_signed):+.2f}  CS(5)={np.mean(np.array(all_errors) <= 5) * 100:.1f}%")


if __name__ == "__main__":
    main()

"""
Bias/calibration analysis, done properly:
  1. Run inference on the VALIDATION set -> fit a linear calibration (true ~ a*pred + b)
     on validation predictions only.
  2. Apply that calibration to the already-computed TEST set predictions
     (results/test_predictions.csv, produced by evaluate.py) to see whether it
     actually improves held-out MAE.
  3. Only save/use the calibration if it genuinely improves test MAE - never hardcode
     a fixed offset.

Usage: python ml/evaluate.py   (first, to produce results/test_predictions.csv)
       python ml/calibrate.py
"""

import json
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from dataset import AgeDataset
from metrics import compute_metrics
from model import create_model
from preprocessing import eval_transform

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHECKPOINT_PATH = os.path.join(PROJECT_ROOT, "checkpoints", "best_resnet18.pth")
VAL_CSV = os.path.join(PROJECT_ROOT, "data", "splits", "val.csv")
TEST_PREDICTIONS_CSV = os.path.join(PROJECT_ROOT, "results", "test_predictions.csv")
CALIBRATION_PATH = os.path.join(PROJECT_ROOT, "results", "calibration.json")
CONFIDENCE_TABLE_PATH = os.path.join(PROJECT_ROOT, "results", "confidence_by_band.json")

CONFIDENCE_BAND_WIDTH = 10  # years, bucketed by PREDICTED age (true age is unknown at inference time)
MIN_BAND_SAMPLES = 20       # bands with fewer test samples than this fall back to the overall rate


def build_confidence_table(pred, true, tolerance=3):
    """For each band of PREDICTED age, what fraction of test predictions in that band were
    within `tolerance` years of the true age? This is a genuine empirical reliability rate
    from held-out test data - not a per-image calibrated probability, and it's labeled as
    such in the UI. Bands with too few samples fall back to the overall rate."""
    abs_err = np.abs(pred - true)
    overall_rate = float(np.mean(abs_err <= tolerance) * 100)

    bands = {}
    max_age = int(np.ceil(pred.max() / CONFIDENCE_BAND_WIDTH) * CONFIDENCE_BAND_WIDTH)
    for lo in range(0, max_age, CONFIDENCE_BAND_WIDTH):
        hi = lo + CONFIDENCE_BAND_WIDTH
        mask = (pred >= lo) & (pred < hi)
        n = int(mask.sum())
        if n >= MIN_BAND_SAMPLES:
            rate = float(np.mean(abs_err[mask] <= tolerance) * 100)
        else:
            rate = overall_rate
        bands[f"{lo}-{hi}"] = {"confidence_pct": round(rate, 1), "n": n}

    return {
        "tolerance_years": tolerance,
        "overall_rate_pct": round(overall_rate, 1),
        "methodology": (
            f"For predictions with predicted age in a given 10-year band, this is the "
            f"empirical percentage of TEST-SET predictions in that band whose age was within "
            f"+/-{tolerance} years of the true age. It is a held-out reliability rate, not a "
            f"per-image calibrated probability. Bands with fewer than {MIN_BAND_SAMPLES} test "
            f"samples fall back to the overall rate."
        ),
        "bands": bands,
    }


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def run_inference_on_val():
    device = get_device()
    val_df = pd.read_csv(VAL_CSV)
    print(f"Validation images: {len(val_df)}")

    ds = AgeDataset(val_df, transform=eval_transform)
    loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=0)

    model = create_model(pretrained=False).to(device)
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    preds, targets = [], []
    with torch.no_grad():
        for images, ages in loader:
            images = images.to(device)
            p = model(images).squeeze(1)
            preds.extend(p.cpu().numpy())
            targets.extend(ages.numpy())

    return np.array(preds), np.array(targets)


def main():
    print("Running inference on validation set to fit calibration...")
    val_pred, val_true = run_inference_on_val()

    # Fit true ~= a * pred + b using validation data only
    a, b = np.polyfit(val_pred, val_true, deg=1)
    print(f"\nFitted calibration on validation set: calibrated_age = {a:.4f} * pred + {b:.4f}")

    val_metrics_raw = compute_metrics(val_true, val_pred)
    val_calibrated = a * val_pred + b
    val_metrics_cal = compute_metrics(val_true, val_calibrated)
    print(f"Validation MAE  raw: {val_metrics_raw['mae']:.3f}  ->  calibrated: {val_metrics_cal['mae']:.3f}")

    if not os.path.exists(TEST_PREDICTIONS_CSV):
        print(f"\n{TEST_PREDICTIONS_CSV} not found - run ml/evaluate.py first to check test-set impact.")
        return

    test_df = pd.read_csv(TEST_PREDICTIONS_CSV)
    test_pred = test_df["predicted_age"].to_numpy()
    test_true = test_df["actual_age"].to_numpy()

    test_metrics_raw = compute_metrics(test_true, test_pred)
    test_calibrated = a * test_pred + b
    test_metrics_cal = compute_metrics(test_true, test_calibrated)

    print("\n" + "=" * 60)
    print("TEST SET: RAW vs CALIBRATED (held-out, not used for fitting)")
    print("=" * 60)
    print(f"MAE   raw: {test_metrics_raw['mae']:.3f}   calibrated: {test_metrics_cal['mae']:.3f}")
    print(f"RMSE  raw: {test_metrics_raw['rmse']:.3f}   calibrated: {test_metrics_cal['rmse']:.3f}")
    print(f"Bias  raw: {test_metrics_raw['mean_signed_error']:+.3f}   "
          f"calibrated: {test_metrics_cal['mean_signed_error']:+.3f}")
    print(f"CS(5) raw: {test_metrics_raw['cs5']:.1f}%   calibrated: {test_metrics_cal['cs5']:.1f}%")

    improves = test_metrics_cal["mae"] < test_metrics_raw["mae"]
    print(f"\nCalibration {'IMPROVES' if improves else 'does NOT improve'} held-out test MAE "
          f"({test_metrics_raw['mae']:.3f} -> {test_metrics_cal['mae']:.3f}).")

    result = {
        "a": float(a),
        "b": float(b),
        "fitted_on": "validation_set",
        "validated_on": "test_set",
        "val_mae_raw": val_metrics_raw["mae"],
        "val_mae_calibrated": val_metrics_cal["mae"],
        "test_mae_raw": test_metrics_raw["mae"],
        "test_mae_calibrated": test_metrics_cal["mae"],
        "apply_calibration": bool(improves),
    }

    with open(CALIBRATION_PATH, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved {CALIBRATION_PATH} (apply_calibration={improves})")

    # Build the confidence-by-band table from whichever prediction set we're actually deploying
    deployed_pred = test_calibrated if improves else test_pred
    confidence_table = build_confidence_table(deployed_pred, test_true, tolerance=3)
    with open(CONFIDENCE_TABLE_PATH, "w") as f:
        json.dump(confidence_table, f, indent=2)
    print(f"Saved {CONFIDENCE_TABLE_PATH}")


if __name__ == "__main__":
    main()

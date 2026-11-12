"""
Final evaluation on the held-out TEST set. Run only after model selection is complete.

Usage: python ml/evaluate.py
"""

import json
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from dataset import AgeDataset
from metrics import compute_metrics, mae_by_age, mae_by_band
from model import create_model
from preprocessing import eval_transform

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHECKPOINT_PATH = os.path.join(PROJECT_ROOT, "checkpoints", "best_resnet18.pth")
TEST_CSV = os.path.join(PROJECT_ROOT, "data", "splits", "test.csv")
OUTPUT_PATH = os.path.join(PROJECT_ROOT, "results", "metrics.json")


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main():
    device = get_device()
    print("Using device:", device)

    if not os.path.exists(CHECKPOINT_PATH):
        raise FileNotFoundError(f"No checkpoint found at {CHECKPOINT_PATH}. Train the model first.")

    test_df = pd.read_csv(TEST_CSV)
    print(f"Test images: {len(test_df)}")

    test_ds = AgeDataset(test_df, transform=eval_transform)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=0)

    model = create_model(pretrained=False).to(device)
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"Loaded checkpoint from epoch {checkpoint.get('epoch')}")

    all_preds, all_targets = [], []
    with torch.no_grad():
        for images, ages in test_loader:
            images = images.to(device)
            preds = model(images).squeeze(1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(ages.numpy())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    overall = compute_metrics(all_targets, all_preds)
    by_age = mae_by_age(all_targets, all_preds)
    by_band = mae_by_band(all_targets, all_preds)

    print("\n" + "=" * 60)
    print("TEST SET RESULTS")
    print("=" * 60)
    print(f"MAE:  {overall['mae']:.4f} years")
    print(f"RMSE: {overall['rmse']:.4f} years")
    print(f"R2:   {overall['r2']:.4f}")
    print(f"CS(2): {overall['cs2']:.1f}%   CS(3): {overall['cs3']:.1f}%   CS(5): {overall['cs5']:.1f}%")
    print(f"Mean signed error: {overall['mean_signed_error']:+.3f} years "
          f"({'over' if overall['mean_signed_error'] > 0 else 'under'}-predicts on average)")

    print("\nMAE and bias by age band:")
    for band, stats in by_band.items():
        print(f"  {band:8s} n={stats['n']:5d}  MAE={stats['mae']:6.2f}  "
              f"bias={stats['mean_signed_error']:+6.2f}")

    results = {
        "checkpoint_epoch": checkpoint.get("epoch"),
        "n_test_samples": len(test_df),
        "overall": overall,
        "mae_by_age_band": by_band,
        "mae_by_age": by_age,
    }

    os.makedirs("results", exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {OUTPUT_PATH}")

    pd.DataFrame({
        "actual_age": all_targets,
        "predicted_age": all_preds,
        "absolute_error": np.abs(all_preds - all_targets),
    }).to_csv("results/test_predictions.csv", index=False)
    print("Saved results/test_predictions.csv")


if __name__ == "__main__":
    main()

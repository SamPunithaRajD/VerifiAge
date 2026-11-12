"""
Validation-set MAE by age band, for model-selection/experiment comparison (never test
data - test is reserved for the final evaluation only). Uses the training-matched
direct-image path (same as train.py/evaluate.py), not the face_gate live pipeline, so
different training experiments are compared on equal footing.

Usage: python ml/evaluate_val_bands.py [--checkpoint checkpoints/best_resnet18.pth]
"""

import argparse
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
VAL_CSV = os.path.join(PROJECT_ROOT, "data", "splits", "val.csv")

BANDS = [(1, 10), (11, 20), (21, 40), (41, 60), (61, 80), (81, 100)]


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default=os.path.join(PROJECT_ROOT, "checkpoints", "best_resnet18.pth"))
    args = p.parse_args()

    device = get_device()
    val_df = pd.read_csv(VAL_CSV)
    print(f"Validation images: {len(val_df)}")
    print(f"Checkpoint: {args.checkpoint}")

    ds = AgeDataset(val_df, transform=eval_transform)
    loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=0)

    model = create_model(pretrained=False).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    preds, targets = [], []
    with torch.no_grad():
        for images, ages in loader:
            images = images.to(device)
            p_ = model(images).squeeze(1)
            preds.extend(p_.cpu().numpy())
            targets.extend(ages.numpy())

    preds = np.array(preds)
    targets = np.array(targets)

    overall = compute_metrics(targets, preds)
    print("\n" + "=" * 60)
    print("OVERALL VALIDATION METRICS")
    print("=" * 60)
    print(f"MAE:  {overall['mae']:.4f}")
    print(f"RMSE: {overall['rmse']:.4f}")
    print(f"R2:   {overall['r2']:.4f}")
    print(f"CS(5): {overall['cs5']:.1f}%")
    print(f"Bias: {overall['mean_signed_error']:+.3f}")

    print("\n" + "=" * 60)
    print("VALIDATION MAE BY REQUESTED AGE BAND")
    print("=" * 60)
    for lo, hi in BANDS:
        mask = (targets >= lo) & (targets <= hi)
        n = mask.sum()
        if n == 0:
            continue
        band_errors = np.abs(preds[mask] - targets[mask])
        band_bias = (preds[mask] - targets[mask]).mean()
        print(f"  {lo:3d}-{hi:<3d}  n={n:5d}  MAE={band_errors.mean():6.2f}  bias={band_bias:+6.2f}")

    # young-age (0-20) specific summary, since that's the priority
    young_mask = targets <= 20
    young_mae = np.abs(preds[young_mask] - targets[young_mask]).mean()
    print(f"\n0-20 MAE (priority band): {young_mae:.3f}  (n={young_mask.sum()})")


if __name__ == "__main__":
    main()

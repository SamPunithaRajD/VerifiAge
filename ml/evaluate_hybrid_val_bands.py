"""
Validation MAE by age band for the EXPERIMENTAL hybrid model (regression + coarse
age-group classification head). Mirrors evaluate_val_bands.py so results are directly
comparable to the production regression-only model - the hybrid model's final age
comes from its regression head only (the classification head is training-time only).

Usage: python ml/evaluate_hybrid_val_bands.py --checkpoint checkpoints/experiment_hybrid.pth
"""

import argparse
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from model_hybrid import create_hybrid_model
from preprocessing import eval_transform
from train_hybrid import HybridAgeDataset

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VAL_CSV = os.path.join(PROJECT_ROOT, "data", "splits", "val.csv")
BANDS = [(1, 10), (11, 20), (21, 40), (41, 60), (61, 80), (81, 100)]


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default=os.path.join(PROJECT_ROOT, "checkpoints", "experiment_hybrid.pth"))
    args = p.parse_args()

    device = get_device()
    val_df = pd.read_csv(VAL_CSV)
    val_df["image_path"] = val_df["image_path"].apply(lambda p: os.path.join(PROJECT_ROOT, p))
    print(f"Validation images: {len(val_df)}")
    print(f"Checkpoint: {args.checkpoint}")

    ds = HybridAgeDataset(val_df, transform=eval_transform)
    loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=0)

    model = create_hybrid_model(pretrained=False).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    preds, targets = [], []
    with torch.no_grad():
        for images, ages, groups in loader:
            images = images.to(device)
            age_pred, _ = model(images)
            preds.extend(age_pred.squeeze(1).cpu().numpy())
            targets.extend(ages.numpy())

    preds = np.array(preds)
    targets = np.array(targets)

    errors = np.abs(preds - targets)
    signed = preds - targets
    print(f"\nOverall MAE: {errors.mean():.4f}  bias: {signed.mean():+.3f}")

    print("\nMAE BY AGE BAND")
    for lo, hi in BANDS:
        mask = (targets >= lo) & (targets <= hi)
        n = mask.sum()
        if n == 0:
            continue
        print(f"  {lo:3d}-{hi:<3d}  n={n:5d}  MAE={errors[mask].mean():6.2f}  bias={signed[mask].mean():+6.2f}")

    young_mask = targets <= 20
    print(f"\n0-20 MAE (priority band): {errors[young_mask].mean():.3f}  (n={young_mask.sum()})")


if __name__ == "__main__":
    main()

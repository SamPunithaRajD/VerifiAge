"""Trivial baseline: predict the mean training age for every sample. The trained model must beat this."""

import numpy as np
import pandas as pd


def compute_baseline_mae(train_csv="data/splits/train.csv", val_csv="data/splits/val.csv"):
    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)

    mean_age = train_df["age"].mean()
    predictions = np.full(len(val_df), mean_age)

    mae = np.mean(np.abs(predictions - val_df["age"].to_numpy()))

    print(f"Mean training age: {mean_age:.2f}")
    print(f"Baseline (mean predictor) validation MAE: {mae:.4f} years")

    return mean_age, mae


if __name__ == "__main__":
    compute_baseline_mae()

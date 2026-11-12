"""
Train ResNet18 for age regression.

Usage:
  python ml/train.py --smoke                    # tiny subset, 1 epoch, sanity check
  python ml/train.py --epochs 3 --frac 0.1       # pilot run on 10% of train data
  python ml/train.py --epochs 30                 # full run (early stopping on val MAE)
  python ml/train.py --weighted-sampler          # A/B: age-balanced sampling
"""

import argparse
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler

from dataset import AgeDataset
from metrics import compute_metrics
from model import create_model
from preprocessing import eval_transform, train_transform

SEED = 42
CHECKPOINT_PATH = "checkpoints/best_resnet18.pth"
HISTORY_PATH = "results/training_history.csv"


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true", help="tiny subset, 1 epoch, sanity check")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--backbone-lr", type=float, default=1e-5)
    p.add_argument("--head-lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--patience", type=int, default=4, help="early-stopping patience (epochs without val MAE improvement)")
    p.add_argument("--frac", type=float, default=1.0, help="fraction of train.csv to use (for pilot runs)")
    p.add_argument("--val-n", type=int, default=None, help="subsample validation set to this many images (for time-boxed runs)")
    p.add_argument("--weighted-sampler", action="store_true", help="use age-balanced WeightedRandomSampler instead of random shuffle")
    p.add_argument("--sampler-strength", type=float, default=1.0, help="0.0=random, 1.0=full inverse-sqrt balancing (only used with --weighted-sampler)")
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--output", default=None, help="checkpoint output path (default: checkpoints/best_resnet18.pth) - "
                                                    "use a different path for experiments so the production checkpoint isn't overwritten")
    return p.parse_args()


def run_epoch(model, loader, criterion, optimizer, device, scaler, train_mode):
    model.train() if train_mode else model.eval()

    total_loss = 0.0
    all_preds, all_targets = [], []

    context = torch.enable_grad() if train_mode else torch.no_grad()
    with context:
        for images, ages in loader:
            images = images.to(device)
            ages = ages.to(device, dtype=torch.float32)

            if train_mode:
                optimizer.zero_grad()

            use_amp = scaler is not None
            with torch.autocast(device_type=device.type, enabled=use_amp):
                preds = model(images).squeeze(1)
                loss = criterion(preds, ages)

            if train_mode:
                if use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

            total_loss += loss.item() * images.size(0)
            all_preds.extend(preds.detach().cpu().numpy())
            all_targets.extend(ages.detach().cpu().numpy())

    avg_loss = total_loss / len(all_targets)
    metrics = compute_metrics(all_targets, all_preds)
    return avg_loss, metrics


def build_weighted_sampler(train_df, strength=1.0, clip=5.0):
    """Inverse-sqrt-frequency weighting by age, clipped to avoid rare ages dominating.
    strength blends between uniform random (0.0) and full balancing (1.0) - a milder
    strength keeps most of the benefit for rare ages while costing less on the
    dominant band, since full-strength balancing overcorrects (measured: it fixed
    young/old-age MAE but regressed the dominant 21-40 band by a similar margin)."""
    age_counts = train_df["age"].value_counts()
    balanced_weights = train_df["age"].map(lambda age: 1 / np.sqrt(age_counts[age]))
    balanced_weights = balanced_weights / balanced_weights.mean()

    blended = strength * balanced_weights + (1 - strength) * 1.0
    blended = blended.clip(upper=clip)

    return WeightedRandomSampler(weights=torch.tensor(blended.to_numpy(), dtype=torch.double),
                                  num_samples=len(blended), replacement=True)


def main():
    args = parse_args()
    device = get_device()
    print("Using device:", device)

    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    train_df = pd.read_csv("data/splits/train.csv")
    val_df = pd.read_csv("data/splits/val.csv")

    if args.smoke:
        train_df = train_df.sample(n=min(200, len(train_df)), random_state=SEED).reset_index(drop=True)
        val_df = val_df.sample(n=min(50, len(val_df)), random_state=SEED).reset_index(drop=True)
        args.epochs = 1
    else:
        if args.frac < 1.0:
            train_df = train_df.sample(frac=args.frac, random_state=SEED).reset_index(drop=True)
        if args.val_n is not None and args.val_n < len(val_df):
            val_df = val_df.sample(n=args.val_n, random_state=SEED).reset_index(drop=True)

    checkpoint_path = args.output or CHECKPOINT_PATH
    history_path = (os.path.splitext(checkpoint_path)[0] + "_history.csv") if args.output else HISTORY_PATH
    print(f"Training images: {len(train_df)}")
    print(f"Validation images: {len(val_df)}")
    print(f"Sampling: {'weighted (age-balanced, strength=' + str(args.sampler_strength) + ')' if args.weighted_sampler else 'random'}")
    print(f"Checkpoint output: {checkpoint_path}")

    train_ds = AgeDataset(train_df, transform=train_transform)
    val_ds = AgeDataset(val_df, transform=eval_transform)

    if args.weighted_sampler:
        sampler = build_weighted_sampler(train_df, strength=args.sampler_strength)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler,
                                   num_workers=args.num_workers, pin_memory=(device.type == "cuda"))
    else:
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                                   num_workers=args.num_workers, pin_memory=(device.type == "cuda"))

    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
    )

    model = create_model().to(device)
    criterion = nn.SmoothL1Loss()

    backbone_params = [p for name, p in model.named_parameters() if not name.startswith("fc.")]
    head_params = model.fc.parameters()
    optimizer = torch.optim.AdamW([
        {"params": backbone_params, "lr": args.backbone_lr},
        {"params": head_params, "lr": args.head_lr},
    ], weight_decay=args.weight_decay)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)
    scaler = torch.cuda.amp.GradScaler() if device.type == "cuda" else None

    best_val_mae = float("inf")
    epochs_without_improvement = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        train_loss, train_metrics = run_epoch(model, train_loader, criterion, optimizer, device, scaler, train_mode=True)
        val_loss, val_metrics = run_epoch(model, val_loader, criterion, optimizer, device, scaler, train_mode=False)
        scheduler.step(val_metrics["mae"])

        elapsed = time.time() - t0

        print(f"\nEpoch {epoch}/{args.epochs} ({elapsed:.1f}s)")
        print(f"  train_loss={train_loss:.4f} train_mae={train_metrics['mae']:.4f}")
        print(f"  val_loss={val_loss:.4f} val_mae={val_metrics['mae']:.4f} "
              f"val_rmse={val_metrics['rmse']:.4f} val_r2={val_metrics['r2']:.4f} val_cs5={val_metrics['cs5']:.1f}%")

        history.append({
            "epoch": epoch, "train_loss": train_loss, "train_mae": train_metrics["mae"],
            "val_loss": val_loss, "val_mae": val_metrics["mae"], "val_rmse": val_metrics["rmse"],
            "val_r2": val_metrics["r2"], "val_cs5": val_metrics["cs5"], "elapsed_sec": elapsed,
        })

        if val_metrics["mae"] < best_val_mae:
            best_val_mae = val_metrics["mae"]
            epochs_without_improvement = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_mae": val_metrics["mae"],
                "val_rmse": val_metrics["rmse"],
            }, checkpoint_path)
            print(f"  -> saved new best model (val_mae={best_val_mae:.4f})")
        else:
            epochs_without_improvement += 1
            print(f"  no improvement ({epochs_without_improvement}/{args.patience})")
            if epochs_without_improvement >= args.patience:
                print("  early stopping triggered")
                break

    pd.DataFrame(history).to_csv(history_path, index=False)
    print(f"\nTraining history saved to {history_path}")
    print(f"Best validation MAE: {best_val_mae:.4f}")
    print(f"Best model saved to {checkpoint_path}")


if __name__ == "__main__":
    main()

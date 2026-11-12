"""
EXPERIMENTAL: trains the hybrid ResNet18 (regression + coarse age-group classification
head) and compares it against the production regression-only model on VALIDATION MAE
only (never test data - test is reserved for final evaluation).

Does not touch train.py, model.py, or the production checkpoint. Saves to its own
checkpoint path.

Usage:
  python ml/train_hybrid.py --epochs 2 --frac 0.2   # pilot, matches prior sampling-strategy pilots
"""

import argparse
import os
import time

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from PIL import Image

from metrics import compute_metrics
from model_hybrid import age_to_group, create_hybrid_model, NUM_GROUPS
from preprocessing import eval_transform, train_transform

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEED = 42
GROUP_LOSS_WEIGHT = 1.0  # SmoothL1 (age, in years) and CrossEntropy (4 classes) start at
                          # a comparable order of magnitude, so 1.0 is a reasonable starting
                          # balance - not separately tuned given this is a pilot-scale check.


class HybridAgeDataset(Dataset):
    """Same image loading as dataset.AgeDataset, plus a coarse age-group label."""
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        image_path = self.dataframe.loc[index, "image_path"]
        age = self.dataframe.loc[index, "age"]

        image = Image.open(image_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)

        age_t = torch.tensor(age, dtype=torch.float32)
        group_t = torch.tensor(age_to_group(age), dtype=torch.long)
        return image, age_t, group_t


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--backbone-lr", type=float, default=1e-5)
    p.add_argument("--head-lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--frac", type=float, default=1.0)
    p.add_argument("--val-n", type=int, default=None)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--output", default=os.path.join(PROJECT_ROOT, "checkpoints", "experiment_hybrid.pth"))
    return p.parse_args()


def run_epoch(model, loader, age_criterion, group_criterion, optimizer, device, train_mode):
    model.train() if train_mode else model.eval()

    total_loss = 0.0
    all_preds, all_targets = [], []
    correct_groups, n_samples = 0, 0

    context = torch.enable_grad() if train_mode else torch.no_grad()
    with context:
        for images, ages, groups in loader:
            images = images.to(device)
            ages = ages.to(device, dtype=torch.float32)
            groups = groups.to(device)

            if train_mode:
                optimizer.zero_grad()

            age_pred, group_logits = model(images)
            age_pred = age_pred.squeeze(1)

            age_loss = age_criterion(age_pred, ages)
            group_loss = group_criterion(group_logits, groups)
            loss = age_loss + GROUP_LOSS_WEIGHT * group_loss

            if train_mode:
                loss.backward()
                optimizer.step()

            batch_size = images.size(0)
            total_loss += loss.item() * batch_size
            all_preds.extend(age_pred.detach().cpu().numpy())
            all_targets.extend(ages.detach().cpu().numpy())
            correct_groups += (group_logits.argmax(dim=1) == groups).sum().item()
            n_samples += batch_size

    avg_loss = total_loss / n_samples
    metrics = compute_metrics(all_targets, all_preds)
    group_acc = correct_groups / n_samples
    return avg_loss, metrics, group_acc


def main():
    args = parse_args()
    device = get_device()
    print("Using device:", device)
    print("EXPERIMENTAL hybrid model - not touching production checkpoint or model.py")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    train_df = pd.read_csv(os.path.join(PROJECT_ROOT, "data", "splits", "train.csv"))
    val_df = pd.read_csv(os.path.join(PROJECT_ROOT, "data", "splits", "val.csv"))
    # CSV paths are relative to PROJECT_ROOT
    train_df["image_path"] = train_df["image_path"].apply(lambda p: os.path.join(PROJECT_ROOT, p))
    val_df["image_path"] = val_df["image_path"].apply(lambda p: os.path.join(PROJECT_ROOT, p))

    if args.frac < 1.0:
        train_df = train_df.sample(frac=args.frac, random_state=SEED).reset_index(drop=True)
    if args.val_n is not None and args.val_n < len(val_df):
        val_df = val_df.sample(n=args.val_n, random_state=SEED).reset_index(drop=True)

    print(f"Training images: {len(train_df)}")
    print(f"Validation images: {len(val_df)}")
    print(f"Checkpoint output: {args.output}")

    train_ds = HybridAgeDataset(train_df, transform=train_transform)
    val_ds = HybridAgeDataset(val_df, transform=eval_transform)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = create_hybrid_model().to(device)
    age_criterion = nn.SmoothL1Loss()
    group_criterion = nn.CrossEntropyLoss()

    backbone_params = model.backbone.parameters()
    head_params = list(model.age_head.parameters()) + list(model.group_head.parameters())
    optimizer = torch.optim.AdamW([
        {"params": backbone_params, "lr": args.backbone_lr},
        {"params": head_params, "lr": args.head_lr},
    ], weight_decay=args.weight_decay)

    best_val_mae = float("inf")
    history = []

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        train_loss, train_metrics, train_group_acc = run_epoch(
            model, train_loader, age_criterion, group_criterion, optimizer, device, train_mode=True)
        val_loss, val_metrics, val_group_acc = run_epoch(
            model, val_loader, age_criterion, group_criterion, optimizer, device, train_mode=False)

        elapsed = time.time() - t0
        print(f"\nEpoch {epoch}/{args.epochs} ({elapsed:.1f}s)")
        print(f"  train_loss={train_loss:.4f} train_mae={train_metrics['mae']:.4f} train_group_acc={train_group_acc:.3f}")
        print(f"  val_loss={val_loss:.4f} val_mae={val_metrics['mae']:.4f} val_r2={val_metrics['r2']:.4f} "
              f"val_cs5={val_metrics['cs5']:.1f}% val_group_acc={val_group_acc:.3f}")

        history.append({
            "epoch": epoch, "train_loss": train_loss, "train_mae": train_metrics["mae"],
            "val_loss": val_loss, "val_mae": val_metrics["mae"], "val_group_acc": val_group_acc,
            "elapsed_sec": elapsed,
        })

        if val_metrics["mae"] < best_val_mae:
            best_val_mae = val_metrics["mae"]
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict(),
                        "val_mae": val_metrics["mae"]}, args.output)
            print(f"  -> saved new best model (val_mae={best_val_mae:.4f})")

    history_path = os.path.splitext(args.output)[0] + "_history.csv"
    pd.DataFrame(history).to_csv(history_path, index=False)
    print(f"\nBest validation MAE: {best_val_mae:.4f}")
    print(f"Model saved to {args.output}")


if __name__ == "__main__":
    main()

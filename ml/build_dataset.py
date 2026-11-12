"""
One-time (cached) dataset prep pipeline:
  scan raw images -> corrupt check -> exact-hash dedup -> conflicting-label
  removal -> age-stratified 70/15/15 split.

Safe to re-run: skips work whose output already exists on disk.
"""

import hashlib
import os

import numpy as np
import pandas as pd
from PIL import Image

SEED = 42
RAW_DIR = "data/raw"
CLEAN_METADATA_PATH = "data/clean/clean_metadata.csv"
SPLITS_DIR = "data/splits"

SUBSETS = [
    # (folder relative to RAW_DIR, source label, age folder width)
    ("age_prediction_up/age_prediction", "full_1_100"),
    ("20-50/20-50", "partial_20_50"),
]


def scan_raw_images():
    records = []
    for subset_dir, source_label in SUBSETS:
        base = os.path.join(RAW_DIR, subset_dir)
        for split in ("train", "test"):
            split_dir = os.path.join(base, split)
            if not os.path.isdir(split_dir):
                continue
            for age_folder in sorted(os.listdir(split_dir)):
                age_path = os.path.join(split_dir, age_folder)
                if not os.path.isdir(age_path):
                    continue
                try:
                    age = int(age_folder)
                except ValueError:
                    continue
                for fname in os.listdir(age_path):
                    if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
                        continue
                    records.append({
                        "image_path": os.path.join(age_path, fname).replace("\\", "/"),
                        "age": age,
                        "source_subset": source_label,
                        "orig_split": split,
                    })
    return pd.DataFrame(records)


def check_and_hash(df):
    """Verify each image opens correctly and compute a content hash + dims."""
    hashes = []
    widths = []
    heights = []
    corrupt_idx = []

    for i, path in enumerate(df["image_path"]):
        if i % 20000 == 0:
            print(f"  hashing {i}/{len(df)}")
        try:
            with open(path, "rb") as f:
                data = f.read()
            h = hashlib.md5(data).hexdigest()
            with Image.open(path) as img:
                img.verify()
            with Image.open(path) as img:
                w, h_px = img.size
        except Exception:
            corrupt_idx.append(i)
            hashes.append(None)
            widths.append(None)
            heights.append(None)
            continue

        hashes.append(h)
        widths.append(w)
        heights.append(h_px)

    df = df.copy()
    df["hash"] = hashes
    df["width"] = widths
    df["height"] = heights

    print(f"Corrupt/unreadable images: {len(corrupt_idx)}")
    df = df.drop(index=corrupt_idx).reset_index(drop=True)
    return df


def dedup(df):
    """Exact duplicate handling: same hash + same age -> keep one. Same hash + different age -> drop group."""
    keep_rows = []
    n_same_age_dupes = 0
    n_conflicting_groups = 0
    n_conflicting_images = 0

    for h, group in df.groupby("hash"):
        if len(group) == 1:
            keep_rows.append(group.iloc[0])
            continue

        unique_ages = group["age"].unique()
        if len(unique_ages) == 1:
            # exact same image repeated with same label -> keep one representative
            rep = group.sort_values("image_path").iloc[0]
            keep_rows.append(rep)
            n_same_age_dupes += len(group) - 1
        else:
            # same image, conflicting age labels -> drop entire group, don't guess
            n_conflicting_groups += 1
            n_conflicting_images += len(group)

    clean_df = pd.DataFrame(keep_rows).reset_index(drop=True)

    print(f"Duplicate images collapsed (same label): {n_same_age_dupes}")
    print(f"Conflicting-label duplicate groups removed: {n_conflicting_groups} "
          f"({n_conflicting_images} images dropped)")

    return clean_df


def stratified_split(df, seed=SEED):
    """Age-stratified 70/15/15 split. Small age groups (<3 images) go entirely to train."""
    train_parts, val_parts, test_parts = [], [], []

    for age, age_df in df.groupby("age"):
        age_df = age_df.sample(frac=1, random_state=seed)
        n = len(age_df)

        if n < 3:
            train_parts.append(age_df)
            continue

        n_val = max(1, round(n * 0.15))
        n_test = max(1, round(n * 0.15))
        n_val = min(n_val, n - 2)
        n_test = min(n_test, n - n_val - 1)

        val_parts.append(age_df.iloc[:n_val])
        test_parts.append(age_df.iloc[n_val:n_val + n_test])
        train_parts.append(age_df.iloc[n_val + n_test:])

    train_df = pd.concat(train_parts, ignore_index=True).sample(frac=1, random_state=seed).reset_index(drop=True)
    val_df = pd.concat(val_parts, ignore_index=True).sample(frac=1, random_state=seed).reset_index(drop=True)
    test_df = pd.concat(test_parts, ignore_index=True).sample(frac=1, random_state=seed).reset_index(drop=True)

    return train_df, val_df, test_df


def main():
    os.makedirs("data/clean", exist_ok=True)
    os.makedirs(SPLITS_DIR, exist_ok=True)

    if os.path.exists(CLEAN_METADATA_PATH):
        print(f"Found existing {CLEAN_METADATA_PATH}, loading instead of rescanning.")
        clean_df = pd.read_csv(CLEAN_METADATA_PATH)
    else:
        print("Scanning raw images...")
        raw_df = scan_raw_images()
        print(f"Total raw images found: {len(raw_df)}")
        print("\nPer-subset counts:")
        print(raw_df.groupby(["source_subset", "orig_split"]).size())

        print("\nVerifying + hashing images (this reads every file once)...")
        hashed_df = check_and_hash(raw_df)

        print("\nDeduplicating...")
        clean_df = dedup(hashed_df)

        clean_df.to_csv(CLEAN_METADATA_PATH, index=False)
        print(f"\nSaved {CLEAN_METADATA_PATH} ({len(clean_df)} rows)")

    train_path = os.path.join(SPLITS_DIR, "train.csv")
    val_path = os.path.join(SPLITS_DIR, "val.csv")
    test_path = os.path.join(SPLITS_DIR, "test.csv")

    if os.path.exists(train_path) and os.path.exists(val_path) and os.path.exists(test_path):
        print("\nSplits already exist, skipping split generation.")
        train_df = pd.read_csv(train_path)
        val_df = pd.read_csv(val_path)
        test_df = pd.read_csv(test_path)
    else:
        print("\nBuilding age-stratified 70/15/15 splits...")
        train_df, val_df, test_df = stratified_split(clean_df)

        train_df.to_csv(train_path, index=False)
        val_df.to_csv(val_path, index=False)
        test_df.to_csv(test_path, index=False)
        print(f"Saved {train_path} ({len(train_df)})")
        print(f"Saved {val_path} ({len(val_df)})")
        print(f"Saved {test_path} ({len(test_df)})")

    # ---- Leakage / integrity checks ----
    print("\n" + "=" * 60)
    print("SPLIT INTEGRITY CHECKS")
    print("=" * 60)

    train_hashes = set(train_df["hash"])
    val_hashes = set(val_df["hash"])
    test_hashes = set(test_df["hash"])

    print("Train & Val  overlap:", len(train_hashes & val_hashes))
    print("Train & Test overlap:", len(train_hashes & test_hashes))
    print("Val & Test   overlap:", len(val_hashes & test_hashes))

    print("\n" + "=" * 60)
    print("FINAL DATASET SUMMARY")
    print("=" * 60)
    print(f"Clean total: {len(clean_df)}")
    print(f"Train: {len(train_df)}  Val: {len(val_df)}  Test: {len(test_df)}")
    print(f"Age range: {clean_df['age'].min()} - {clean_df['age'].max()}")

    if "width" in clean_df.columns:
        dims = clean_df.groupby(["width", "height"]).size().sort_values(ascending=False)
        print("\nTop image dimensions:")
        print(dims.head(10))

    print("\nAge distribution (clean set):")
    print(clean_df["age"].value_counts().sort_index().to_string())


if __name__ == "__main__":
    main()

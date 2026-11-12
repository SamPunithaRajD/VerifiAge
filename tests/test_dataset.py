import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ml"))

from dataset import AgeDataset
from preprocessing import IMAGE_SIZE, eval_transform


def test_dataset_loading_and_one_preprocessing_pass():
    train_csv = os.path.join(os.path.dirname(__file__), "..", "data", "splits", "train.csv")
    if not os.path.exists(train_csv):
        return  # dataset not built yet in this environment

    df = pd.read_csv(train_csv).head(2)
    ds = AgeDataset(df, transform=eval_transform)

    assert len(ds) == len(df)

    image, age = ds[0]
    assert image.shape == (3, IMAGE_SIZE, IMAGE_SIZE)
    assert age.dtype.is_floating_point

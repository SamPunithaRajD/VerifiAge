import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ml"))

from model import create_model


def test_model_forward_pass():
    model = create_model(pretrained=False)
    model.eval()

    x = torch.randn(2, 3, 128, 128)
    with torch.no_grad():
        out = model(x)

    assert out.shape == (2, 1)

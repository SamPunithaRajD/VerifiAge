"""
EXPERIMENTAL - not the production model. ResNet18 with two heads sharing one backbone:
  - continuous age regression (same as the production model)
  - coarse age-group classification: 1-12, 13-19, 20-59, 60-100

The classification head is an auxiliary training signal only (multi-task learning) -
final age predictions still come from the regression head, so this stays comparable
to the production model via validation MAE. Kept in its own file so the official
model.py (used by predict.py/backend) is never touched by this experiment.
"""

import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18

AGE_GROUP_BINS = [(1, 12), (13, 19), (20, 59), (60, 100)]
NUM_GROUPS = len(AGE_GROUP_BINS)


def age_to_group(age):
    for i, (lo, hi) in enumerate(AGE_GROUP_BINS):
        if lo <= age <= hi:
            return i
    return NUM_GROUPS - 1  # clamp anything outside the known range to the last bin


class HybridResNet18(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        backbone = resnet18(weights=weights)
        in_features = backbone.fc.in_features
        backbone.fc = nn.Identity()  # strip the original single head, we add our own below
        self.backbone = backbone
        self.age_head = nn.Linear(in_features, 1)
        self.group_head = nn.Linear(in_features, NUM_GROUPS)

    def forward(self, x):
        features = self.backbone(x)
        age = self.age_head(features)
        group_logits = self.group_head(features)
        return age, group_logits


def create_hybrid_model(pretrained=True):
    return HybridResNet18(pretrained=pretrained)


if __name__ == "__main__":
    import torch

    model = create_hybrid_model()
    x = torch.randn(2, 3, 128, 128)
    age, group_logits = model(x)
    print("age shape:", age.shape, "group_logits shape:", group_logits.shape)

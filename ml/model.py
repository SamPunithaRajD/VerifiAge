"""Official VerifiAge model: ResNet18 pretrained on ImageNet, single regression head for age."""

import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18


def create_model(pretrained=True):
    weights = ResNet18_Weights.DEFAULT if pretrained else None
    model = resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, 1)
    return model


if __name__ == "__main__":
    import torch

    model = create_model()
    x = torch.randn(2, 3, 128, 128)
    out = model(x)
    print("output shape:", out.shape)

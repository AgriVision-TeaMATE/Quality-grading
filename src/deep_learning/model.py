"""
CNN Model Architectures (Approach B)
=======================================
Provides:
    - SimpleCNN: small custom CNN baseline (trained from scratch)
    - get_pretrained_backbone: transfer learning with ResNet18 / EfficientNet-B0 /
      MobileNetV3-Small, fine-tuned for tea grade classification.
"""
import torch
import torch.nn as nn
from torchvision import models


class SimpleCNN(nn.Module):
    """A small custom CNN baseline — useful to gauge whether transfer learning
    is actually necessary, or whether a lightweight model trained from scratch
    is sufficient for this particle classification task."""

    def __init__(self, num_classes, input_size=224):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


def get_pretrained_backbone(name, num_classes, pretrained=True):
    """Load a torchvision backbone and replace its head for num_classes outputs."""
    if name == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        model = models.resnet18(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif name == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        model = models.efficientnet_b0(weights=weights)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    elif name == "mobilenet_v3_small":
        weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        model = models.mobilenet_v3_small(weights=weights)
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
    else:
        raise ValueError(f"Unknown backbone: {name}")
    return model


def build_model(cfg_dl, num_classes):
    backbone = cfg_dl.get("backbone", "simple_cnn")
    if backbone == "simple_cnn":
        return SimpleCNN(num_classes, input_size=cfg_dl["image_size"][0])
    return get_pretrained_backbone(backbone, num_classes, pretrained=cfg_dl.get("pretrained", True))

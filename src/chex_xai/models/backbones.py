# src/chex_xai/models/backbones.py

from typing import Tuple

import torch.nn as nn
import torchvision.models as tv


def build_backbone(name: str, pretrained: bool) -> Tuple[nn.Module, int]:
    """
    Build a torchvision backbone and return (backbone, num_features).
    The returned backbone should output a feature tensor we can flatten.
    """
    name = name.lower()

    if name == "densenet121":
        m = tv.densenet121(
            weights=tv.DenseNet121_Weights.DEFAULT if pretrained else None
        )
        num_feat = m.classifier.in_features
        # Replace classifier with identity to get features
        m.classifier = nn.Identity()
        return m, num_feat

    if name == "resnet50":
        m = tv.resnet50(weights=tv.ResNet50_Weights.DEFAULT if pretrained else None)
        num_feat = m.fc.in_features
        m.fc = nn.Identity()
        return m, num_feat

    if name == "efficientnet_b0":
        m = tv.efficientnet_b0(
            weights=tv.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        )
        num_feat = m.classifier[1].in_features
        m.classifier = nn.Identity()
        return m, num_feat

    raise ValueError(f"Unknown backbone: {name}")

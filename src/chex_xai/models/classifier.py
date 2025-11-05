# src/chex_xai/models/classifier.py

import torch
import torch.nn as nn

from .backbones import build_backbone


class MultiLabelClassifier(nn.Module):
    """
    Wraps a backbone and adds a classification head for multi-label prediction.
    """

    def __init__(
        self, name: str, num_classes: int, pretrained: bool = True, dropout: float = 0.2
    ):
        super().__init__()
        self.backbone, in_features = build_backbone(name, pretrained)
        head = []
        if dropout and dropout > 0:
            head.append(nn.Dropout(p=dropout))
        head.append(nn.Linear(in_features, num_classes))
        self.head = nn.Sequential(*head)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # For DenseNet/ResNet/EfficientNet we already output a 2D feature vector
        feats = self.backbone(x)
        # Some backbones may output feature maps; ensure it's flattened if needed
        if feats.ndim > 2:
            feats = torch.flatten(feats, 1)
        logits = self.head(feats)
        return logits

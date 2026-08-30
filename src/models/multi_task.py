"""Multi-Task классификатор тракторов (модель + состояние) на ConvNeXt-Tiny."""

from __future__ import annotations

import torch
from torch import nn
from torchvision.models import ConvNeXt_Tiny_Weights, convnext_tiny

from src.config.classes import NUM_MODEL_CLASSES, NUM_STATE_CLASSES

CONVNEXT_TINY_FEATURES: int = 768


class MultiTaskTractorClassifier(nn.Module):
    """Multi-Task классификатор: голова модели трактора + голова состояния."""

    def __init__(
        self,
        num_model_classes: int = NUM_MODEL_CLASSES,
        num_state_classes: int = NUM_STATE_CLASSES,
    ) -> None:
        """Инициализировать модель.

        Args:
            num_model_classes: Число классов модели трактора.
            num_state_classes: Число классов состояния (clean/dirty).
        """
        super().__init__()
        self.backbone = self._create_backbone()
        self.model_head = nn.Linear(CONVNEXT_TINY_FEATURES, num_model_classes)
        self.state_head = nn.Linear(CONVNEXT_TINY_FEATURES, num_state_classes)

    def _create_backbone(self) -> nn.Module:
        """Создать ConvNeXt-Tiny backbone без оригинальной головы.

        Returns:
            Backbone с ``classifier = Identity``.
        """
        weights = ConvNeXt_Tiny_Weights.DEFAULT
        backbone = convnext_tiny(weights=weights)
        backbone.classifier = nn.Identity()
        return backbone

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Прямой проход.

        Args:
            x: Батч изображений ``(B, 3, H, W)``.

        Returns:
            Кортеж ``(model_logits, state_logits)``.
        """
        features = self.backbone(x)
        features = features.flatten(1)
        return self.model_head(features), self.state_head(features)

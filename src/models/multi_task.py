"""
Multi-Task модель для классификации тракторов.

Один backbone (ConvNeXt-Tiny) с двумя головами:
- Голова 1: модель трактора (5 классов)
- Голова 2: состояние грязный/чистый (2 класса)
"""

from __future__ import annotations

import logging

import torch
from torch import nn
from torchvision.models import ConvNeXt_Tiny_Weights, convnext_tiny

logger = logging.getLogger(__name__)

MODEL_CLASSES: list[str] = [
    "chtz_b10m",
    "johndeere",
    "kirovets_k744",
    "mtz_1221",
    "mtz_82",
]

STATE_CLASSES: list[str] = ["clean", "dirty"]

NUM_MODEL_CLASSES = len(MODEL_CLASSES)
NUM_STATE_CLASSES = len(STATE_CLASSES)
CONVNEXT_TINY_FEATURES = 768
FEATURE_FLATTEN_START_DIM = 1


class MultiTaskTractorClassifier(nn.Module):
    """Multi-Task классификатор тракторов.

    Args:
        num_model_classes: Количество классов моделей тракторов.
        num_state_classes: Количество классов состояния (clean/dirty).
    """

    def __init__(
        self,
        num_model_classes: int = NUM_MODEL_CLASSES,
        num_state_classes: int = NUM_STATE_CLASSES,
    ) -> None:
        super().__init__()

        self.backbone = self._create_backbone()
        self._freeze_backbone()

        self.model_head = nn.Linear(CONVNEXT_TINY_FEATURES, num_model_classes)
        self.state_head = nn.Linear(CONVNEXT_TINY_FEATURES, num_state_classes)

        self.num_model_classes = num_model_classes
        self.num_state_classes = num_state_classes

        logger.info(
            "MultiTaskTractorClassifier initialized: "
            "model_classes=%d, state_classes=%d",
            num_model_classes,
            num_state_classes,
        )

    def _create_backbone(self) -> nn.Module:
        """Создаёт предобученный ConvNeXt-Tiny без классификатора.

        Returns:
            ConvNeXt-Tiny с Identity вместо classification head.

        Raises:
            RuntimeError: Если не удалось загрузить предобученные веса.
        """
        try:
            backbone = convnext_tiny(weights=ConvNeXt_Tiny_Weights.DEFAULT)
            backbone.classifier = nn.Identity()
            return backbone
        except Exception as exc:
            msg = "Не удалось создать ConvNeXt-Tiny backbone с предобученными весами"
            logger.error(msg)
            raise RuntimeError(msg) from exc

    def _freeze_backbone(self) -> None:
        """Замораживает все параметры backbone."""
        for param in self.backbone.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Выполняет прямой проход через обе головы классификации.

        Args:
            x: Батч изображений формы (B, C, H, W).

        Returns:
            Кортеж (model_logits, state_logits) форм (B, num_model_classes)
            и (B, num_state_classes) соответственно.
        """
        features = self.backbone(x)
        features = features.flatten(FEATURE_FLATTEN_START_DIM)
        return self.model_head(features), self.state_head(features)

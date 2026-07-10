"""
Классификатор моделей тракторов на базе предобученной CNN.

Использует transfer learning с частичной разморозкой ConvNeXt-Tiny.
"""

from __future__ import annotations

import logging

import torch
from torch import nn
from torchvision.models import ConvNeXt_Tiny_Weights, convnext_tiny

logger = logging.getLogger(__name__)

CLASS_NAMES: list[str] = [
    "chtz_b10m",
    "johndeere",
    "kirovets_k744",
    "mtz_1221",
    "mtz_82",
]

NUM_CLASSES = len(CLASS_NAMES)
CLASSIFIER_HEAD_INDEX = 2
NUM_UNFROZEN_STAGES = 0


class TractorClassifier(nn.Module):
    """Классификатор тракторов с переносом обучения.

    Args:
        num_classes: Количество классов моделей тракторов.
    """

    def __init__(self, num_classes: int = NUM_CLASSES) -> None:
        super().__init__()

        self.backbone = self._create_backbone()
        self._freeze_backbone()
        self._replace_classifier(num_classes)

        self.num_classes = num_classes

        logger.info(
            "TractorClassifier initialized: num_classes=%d, unfrozen_stages=%d",
            num_classes,
            NUM_UNFROZEN_STAGES,
        )

    def _create_backbone(self) -> nn.Module:
        """Создаёт предобученный ConvNeXt-Tiny.

        Returns:
            Модуль ConvNeXt-Tiny с весами ImageNet.

        Raises:
            RuntimeError: Если не удалось загрузить предобученные веса.
        """
        try:
            return convnext_tiny(weights=ConvNeXt_Tiny_Weights.DEFAULT)
        except Exception as exc:
            msg = "Не удалось создать ConvNeXt-Tiny backbone с предобученными весами"
            logger.error(msg)
            raise RuntimeError(msg) from exc

    def _freeze_backbone(self) -> None:
        """Замораживает все слои, затем размораживает последние N стадий."""
        for param in self.backbone.parameters():
            param.requires_grad = False

        for param in self.backbone.features[-NUM_UNFROZEN_STAGES:].parameters():
            param.requires_grad = True

    def _replace_classifier(self, num_classes: int) -> None:
        """Заменяет последний слой классификатора на целевой.

        Args:
            num_classes: Количество выходных классов.

        Raises:
            IndexError: Если структура classifier backbone не соответствует ожидаемой.
        """
        try:
            num_features = self.backbone.classifier[CLASSIFIER_HEAD_INDEX].in_features
        except (IndexError, AttributeError) as exc:
            msg = f"Не удалось получить in_features из classifier[{CLASSIFIER_HEAD_INDEX}]"
            logger.error(msg)
            raise IndexError(msg) from exc

        self.backbone.classifier[CLASSIFIER_HEAD_INDEX] = nn.Linear(
            num_features, num_classes
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Выполняет прямой проход: изображение → логиты классов.

        Args:
            x: Батч изображений формы (B, C, H, W).

        Returns:
            Логиты формы (B, num_classes).
        """
        return self.backbone(x)

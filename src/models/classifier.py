"""Single-Task классификатор моделей тракторов на базе ConvNeXt-Tiny.

Число классов и их имена берутся ИСКЛЮЧИТЕЛЬНО из :mod:`src.config.classes`.
Локальные ``CLASS_NAMES`` / ``NUM_CLASSES = 5`` удалены — они были источником
рассинхрона (5 классов до слияния ``mtz_1221 -> mtz_82``). Теперь единственный
источник правды — ``MODEL_CLASSES`` (4 класса).
"""

from __future__ import annotations

import torch
from torch import nn
from torchvision.models import ConvNeXt_Tiny_Weights, convnext_tiny

from src.config.classes import NUM_MODEL_CLASSES

# Индекс слоя Linear внутри classifier-Sequential ConvNeXt-Tiny
# (classifier = [LayerNorm2d, Flatten, Linear]).
CLASSIFIER_HEAD_INDEX: int = 2
# Число размороженных стадий backbone: 0 — полностью заморожен (single-task
# базовая линия). Partial fine-tuning реализован в multi_task_train.
NUM_UNFROZEN_STAGES: int = 0
CONVNEXT_TINY_FEATURES: int = 768


class TractorClassifier(nn.Module):
    """Классификатор тракторов с переносом обучения на ConvNeXt-Tiny."""

    def __init__(self, num_classes: int = NUM_MODEL_CLASSES) -> None:
        """Инициализировать классификатор.

        Args:
            num_classes: Число классов модели трактора. По умолчанию берётся из
                :data:`src.config.classes.NUM_MODEL_CLASSES` (4 после слияния).
        """
        super().__init__()
        self.num_classes = num_classes
        self.backbone = self._create_backbone()
        self._freeze_backbone()
        self._replace_classifier(num_classes)

    def _create_backbone(self) -> nn.Module:
        """Создать предобученный ConvNeXt-Tiny backbone.

        Returns:
            Backbone с оригинальной ImageNet-головой (заменяется далее).
        """
        weights = ConvNeXt_Tiny_Weights.DEFAULT
        return convnext_tiny(weights=weights)

    def _freeze_backbone(self) -> None:
        """Заморозить все параметры backbone (feature-extraction режим)."""
        for param in self.backbone.parameters():
            param.requires_grad_(False)

    def _replace_classifier(self, num_classes: int) -> None:
        """Заменить голову классификатора под нужное число классов.

        Заменяется только слой Linear (индекс :data:`CLASSIFIER_HEAD_INDEX`),
        LayerNorm2d и Flatten сохраняются. Новый слой обучаем.

        Args:
            num_classes: Число выходных классов.
        """
        self.backbone.classifier[CLASSIFIER_HEAD_INDEX] = nn.Linear(
            CONVNEXT_TINY_FEATURES, num_classes
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Прямой проход.

        Args:
            x: Батч изображений ``(B, 3, H, W)``.

        Returns:
            Логиты классов ``(B, num_classes)``.
        """
        logits: torch.Tensor = self.backbone(x)
        return logits

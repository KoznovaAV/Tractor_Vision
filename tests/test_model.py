"""Тесты single-task классификатора TractorClassifier.
Классы и их число берутся из :mod:`src.config.classes` — локальный
``CLASS_NAMES`` удалён вместе с хардкодом пяти классов.
"""

from __future__ import annotations

import torch

from src.config.classes import MODEL_CLASSES, NUM_MODEL_CLASSES
from src.models.classifier import CLASSIFIER_HEAD_INDEX, CONVNEXT_TINY_FEATURES, TractorClassifier


def test_num_classes_from_config() -> None:
    """Число классов модели равно len(MODEL_CLASSES) из конфига."""
    model = TractorClassifier()
    assert model.num_classes == NUM_MODEL_CLASSES == len(MODEL_CLASSES)


def test_forward_output_shape() -> None:
    """Логиты имеют форму (B, NUM_MODEL_CLASSES)."""
    model = TractorClassifier()
    logits = model(torch.randn(2, 3, 64, 64))
    assert logits.shape == (2, NUM_MODEL_CLASSES)


def test_classifier_head_replaced() -> None:
    """Голова заменена на Linear с нужным числом выходов и обучема."""
    model = TractorClassifier()
    head = model.backbone.classifier[CLASSIFIER_HEAD_INDEX]
    assert isinstance(head, torch.nn.Linear)
    assert head.in_features == CONVNEXT_TINY_FEATURES
    assert head.out_features == NUM_MODEL_CLASSES
    assert head.weight.requires_grad and head.bias.requires_grad


def test_backbone_frozen_except_head() -> None:
    """Все параметры backbone заморожены, кроме заменённой головы."""
    model = TractorClassifier()
    head_ids = {id(p) for p in model.backbone.classifier[CLASSIFIER_HEAD_INDEX].parameters()}
    for param in model.backbone.parameters():
        if id(param) in head_ids:
            assert param.requires_grad
        else:
            assert not param.requires_grad

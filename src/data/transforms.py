"""Albumentations-аугментации для тренировки и валидации."""

from __future__ import annotations

import albumentations as A
from albumentations.pytorch import ToTensorV2

IMAGENET_MEAN: tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: tuple[float, float, float] = (0.229, 0.224, 0.225)


def get_train_transforms(image_size: int = 384) -> A.Compose:
    """Аугментации для обучения.

    Args:
        image_size: Целевой размер стороны квадрата.

    Returns:
        Композиция трансформаций Albumentations.
    """
    return A.Compose(
        [
            A.Resize(image_size, image_size),
            A.HorizontalFlip(p=0.5),
            A.Affine(scale=(0.9, 1.1), rotate=(-15, 15), p=0.5),
            A.RandomBrightnessContrast(0.2, 0.2, p=0.5),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ]
    )


def get_val_transforms(image_size: int = 384) -> A.Compose:
    """Аугментации для валидации/инференса.

    Args:
        image_size: Целевой размер стороны квадрата.

    Returns:
        Композиция трансформаций Albumentations.
    """
    return A.Compose(
        [
            A.Resize(image_size, image_size),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ]
    )

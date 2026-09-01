"""Albumentations-аугментации для тренировки и валидации.

Размер по умолчанию берётся из ``config.yaml`` (``image_size``) — единый для
train, eval и инференса.
"""

from __future__ import annotations

import albumentations as A
from albumentations.pytorch import ToTensorV2

from src.config.config_loader import load_config

IMAGENET_MEAN: tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: tuple[float, float, float] = (0.229, 0.224, 0.225)

DEFAULT_IMAGE_SIZE: int = load_config().image_size


def get_train_transforms(image_size: int = DEFAULT_IMAGE_SIZE) -> A.Compose:
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
            # Расширенный jitter яркости/контраста/насыщенности: чистые тракторы
            # на солнце и в тени, глянцевые блики — не должны читаться как dirty.
            A.RandomBrightnessContrast(0.3, 0.3, p=0.6),
            A.HueSaturationValue(hue_shift_limit=5, sat_shift_limit=25, val_shift_limit=10, p=0.3),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ]
    )


def get_val_transforms(image_size: int = DEFAULT_IMAGE_SIZE) -> A.Compose:
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

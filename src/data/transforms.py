"""
Аугментации изображений для обучения модели.

Использует Albumentations для создания вариаций данных.
"""

from albumentations import (
    Affine,
    Blur,
    Compose,
    GaussNoise,
    HorizontalFlip,
    MedianBlur,
    MotionBlur,
    Normalize,
    OneOf,
    RandomBrightnessContrast,
    Resize,
)
from albumentations.pytorch import ToTensorV2

DEFAULT_IMAGE_SIZE = 224

# Стандартные значения нормализации ImageNet
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Вероятности и параметры аугментаций (train)
HORIZONTAL_FLIP_PROB = 0.5
AFFINE_TRANSLATE_RANGE = (-0.1, 0.1)
AFFINE_SCALE_RANGE = (0.9, 1.1)
AFFINE_ROTATE_RANGE = (-15, 15)
AFFINE_PROB = 0.5
BRIGHTNESS_LIMIT = 0.2
CONTRAST_LIMIT = 0.2
BRIGHTNESS_CONTRAST_PROB = 0.5
BLUR_ONEOF_PROB = 0.3
MOTION_BLUR_PROB = 0.2
MEDIAN_BLUR_PROB = 0.2
BLUR_PROB = 0.2
BLUR_LIMIT = 3
GAUSS_NOISE_STD_RANGE = (0.01, 0.1)
GAUSS_NOISE_PROB = 0.3


def get_train_transforms(image_size: int = DEFAULT_IMAGE_SIZE) -> Compose:
    """Возвращает пайплайн аугментаций для train-набора.

    Включает: горизонтальное отражение, аффинные преобразования,
    изменение яркости/контраста, размытие и гауссов шум.

    Args:
        image_size: Целевой размер стороны изображения (квадрат).

    Returns:
        Скомпонованный пайплайн Albumentations для обучения.
    """
    return Compose(
        [
            Resize(image_size, image_size),
            HorizontalFlip(p=HORIZONTAL_FLIP_PROB),
            Affine(
                translate_percent={
                    "x": AFFINE_TRANSLATE_RANGE,
                    "y": AFFINE_TRANSLATE_RANGE,
                },
                scale=AFFINE_SCALE_RANGE,
                rotate=AFFINE_ROTATE_RANGE,
                p=AFFINE_PROB,
            ),
            RandomBrightnessContrast(
                brightness_limit=BRIGHTNESS_LIMIT,
                contrast_limit=CONTRAST_LIMIT,
                p=BRIGHTNESS_CONTRAST_PROB,
            ),
            OneOf(
                [
                    MotionBlur(p=MOTION_BLUR_PROB),
                    MedianBlur(blur_limit=BLUR_LIMIT, p=MEDIAN_BLUR_PROB),
                    Blur(blur_limit=BLUR_LIMIT, p=BLUR_PROB),
                ],
                p=BLUR_ONEOF_PROB,
            ),
            GaussNoise(std_range=GAUSS_NOISE_STD_RANGE, p=GAUSS_NOISE_PROB),
            Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ]
    )


def get_val_transforms(image_size: int = DEFAULT_IMAGE_SIZE) -> Compose:
    """Возвращает пайплайн преобразований для val/test-набора.

    Применяет только resize и нормализацию без случайных аугментаций.

    Args:
        image_size: Целевой размер стороны изображения (квадрат).

    Returns:
        Скомпонованный пайплайн Albumentations для валидации/теста.
    """
    return Compose(
        [
            Resize(image_size, image_size),
            Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ]
    )

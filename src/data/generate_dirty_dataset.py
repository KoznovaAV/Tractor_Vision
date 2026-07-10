"""
Генерация синтетического датасета "грязный/чистый".
Структура: {split}/{model_class}/{clean|dirty}/image.jpg
"""

import argparse
import logging
import random
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

DIRTY_TRANSFORMS = A.Compose(
    [
        # Сильный шум
        A.GaussNoise(std_range=(0.1, 0.3), p=0.9),
        # Сильное размытие
        A.OneOf(
            [
                A.MotionBlur(blur_limit=(10, 25), p=0.5),
                A.MedianBlur(blur_limit=(7, 15), p=0.5),
            ],
            p=0.9,
        ),
        # Сильное затемнение
        A.RandomBrightnessContrast(
            brightness_limit=(-0.6, 0.0), contrast_limit=(-0.5, 0.0), p=0.9
        ),
        # Сильное изменение цвета
        A.HueSaturationValue(
            hue_shift_limit=(-20, 20),
            sat_shift_limit=(-50, 0),
            val_shift_limit=(-50, 0),
            p=0.9,
        ),
        # Добавляем "грязные пятна"
        A.CoarseDropout(
            max_holes=8, max_height=50, max_width=50, min_holes=3, fill_value=0, p=0.7
        ),
    ]
)

CLEAN_TRANSFORMS = A.Compose(
    [
        A.Resize(384, 384),
    ]
)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def apply_transforms(image: np.ndarray, transforms) -> np.ndarray:
    """Применяет Albumentations трансформации."""
    return transforms(image=image)["image"]


def save_image(image: np.ndarray, path: Path) -> None:
    """Сохраняет изображение в BGR формате."""
    cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))


def process_image(
    img_path: Path,
    model_class: str,
    output_dir: Path,
    split: str,
    dirty_ratio: float = 1.0,
) -> tuple[int, int]:
    """Создаёт чистую и грязную версии изображения."""
    image = cv2.imread(str(img_path))
    if image is None:
        return 0, 0

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Пути для сохранения
    clean_dir = output_dir / split / model_class / "clean"
    dirty_dir = output_dir / split / model_class / "dirty"
    clean_dir.mkdir(parents=True, exist_ok=True)
    dirty_dir.mkdir(parents=True, exist_ok=True)

    # Чистая версия
    clean_img = apply_transforms(image, CLEAN_TRANSFORMS)
    save_image(clean_img, clean_dir / f"{img_path.stem}_clean.jpg")

    # Грязные версии
    num_dirty = int(dirty_ratio) + (1 if random.random() < (dirty_ratio % 1) else 0)
    for i in range(num_dirty):
        dirty_img = apply_transforms(image, DIRTY_TRANSFORMS)
        save_image(dirty_img, dirty_dir / f"{img_path.stem}_dirty_{i}.jpg")

    return 1, num_dirty


def generate_dataset(
    source_dir: Path,
    output_dir: Path,
    dirty_ratio: float = 1.0,
    val_split: float = 0.2,
) -> None:
    """Генерирует Multi-Task датасет."""
    # Собираем все изображения с их моделями
    all_images = []
    for split in ["train", "val", "test"]:
        split_dir = source_dir / split
        if not split_dir.exists():
            continue

        for model_dir in split_dir.iterdir():
            if not model_dir.is_dir():
                continue

            for img_path in model_dir.iterdir():
                if img_path.suffix.lower() in IMAGE_EXTENSIONS:
                    all_images.append((img_path, model_dir.name))

    logger.info("Найдено %d изображений", len(all_images))

    # Перемешиваем и разделяем
    random.shuffle(all_images)
    val_size = int(len(all_images) * val_split)
    val_images = all_images[:val_size]
    train_images = all_images[val_size:]

    # Обрабатываем train
    logger.info("Генерация train набора...")
    total_clean, total_dirty = 0, 0
    for img_path, model_class in tqdm(train_images):
        clean_count, dirty_count = process_image(
            img_path, model_class, output_dir, "train", dirty_ratio
        )
        total_clean += clean_count
        total_dirty += dirty_count

    logger.info("Train: clean=%d, dirty=%d", total_clean, total_dirty)

    # Обрабатываем val
    logger.info("Генерация val набора...")
    val_clean, val_dirty = 0, 0
    for img_path, model_class in tqdm(val_images):
        clean_count, dirty_count = process_image(
            img_path, model_class, output_dir, "val", dirty_ratio
        )
        val_clean += clean_count
        val_dirty += dirty_count

    logger.info("Val: clean=%d, dirty=%d", val_clean, val_dirty)
    logger.info("Готово! Датасет сохранён в %s", output_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Генерация Multi-Task датасета")
    parser.add_argument("--source_dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--output_dir", type=Path, default=Path("data/dirty_clean"))
    parser.add_argument("--dirty_ratio", type=float, default=1.0)
    parser.add_argument("--val_split", type=float, default=0.2)
    return parser.parse_args()


def main():
    args = parse_args()

    logger.info("Генерация Multi-Task датасета")
    logger.info("Источник: %s", args.source_dir)
    logger.info("Выход: %s", args.output_dir)
    logger.info("Dirty ratio: %.1f", args.dirty_ratio)
    logger.info("=" * 60)

    generate_dataset(
        source_dir=args.source_dir,
        output_dir=args.output_dir,
        dirty_ratio=args.dirty_ratio,
        val_split=args.val_split,
    )


if __name__ == "__main__":
    main()

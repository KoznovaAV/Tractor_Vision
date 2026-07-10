"""
DataLoader для загрузки батчей изображений тракторов.
"""

from __future__ import annotations

import logging
from pathlib import Path

from torch.utils.data import DataLoader

from src.data.dataset import TractorDataset
from src.data.transforms import DEFAULT_IMAGE_SIZE, get_train_transforms, get_val_transforms

logger = logging.getLogger(__name__)

DEFAULT_BATCH_SIZE = 16
DEFAULT_NUM_WORKERS = 2
TRAIN_SPLIT = "train"


def get_dataloader(
    data_dir: Path,
    split: str,
    batch_size: int = DEFAULT_BATCH_SIZE,
    image_size: int = DEFAULT_IMAGE_SIZE,
    num_workers: int = DEFAULT_NUM_WORKERS,
    multi_task: bool = False,
) -> DataLoader:
    """Создаёт DataLoader для заданного сплита.

    Args:
        data_dir: Корневая директория датасета (содержит train/val/test).
        split: Имя сплита (например, ``train``, ``val``, ``test``).
        batch_size: Размер батча.
        image_size: Целевой размер изображения для трансформаций.
        num_workers: Количество worker-процессов для загрузки данных.
        multi_task: Если True, загружает метки состояния (clean/dirty).

    Returns:
        Настроенный PyTorch DataLoader.

    Raises:
        FileNotFoundError: Если директория сплита не существует.
    """
    split_dir = data_dir / split

    if not split_dir.exists():
        raise FileNotFoundError(f"Сплит '{split}' не найден: {split_dir}")

    is_train = split == TRAIN_SPLIT
    transform = (
        get_train_transforms(image_size) if is_train else get_val_transforms(image_size)
    )

    dataset = TractorDataset(
        root_dir=split_dir,
        transform=transform,
        multi_task=multi_task,
    )

    logger.info(
        "DataLoader created: split=%s, batch_size=%d, samples=%d, multi_task=%s",
        split,
        batch_size,
        len(dataset),
        multi_task,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=is_train,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=is_train,
    )

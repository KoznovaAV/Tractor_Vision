"""Фабрика DataLoader для train/val/test сплитов."""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.data.dataset import TractorDataset
from src.data.transforms import get_train_transforms, get_val_transforms

DEFAULT_BATCH_SIZE: int = 16
DEFAULT_NUM_WORKERS: int = 2
DEFAULT_IMAGE_SIZE: int = 384


def get_dataloader(
    data_dir: Path,
    split: str,
    batch_size: int = DEFAULT_BATCH_SIZE,
    image_size: int = DEFAULT_IMAGE_SIZE,
    num_workers: int = DEFAULT_NUM_WORKERS,
    multi_task: bool = False,
) -> DataLoader:
    """Создать DataLoader для указанного сплита.

    Args:
        data_dir: Корень дерева датасета.
        split: Имя сплита (``train``/``val``/``test``).
        batch_size: Размер батча.
        image_size: Размер изображения.
        num_workers: Число воркеров загрузки.
        multi_task: Multi-task режим датасета.

    Returns:
        Настроенный DataLoader.
    """
    is_train = split == "train"
    transform = get_train_transforms(image_size) if is_train else get_val_transforms(image_size)
    dataset = TractorDataset(
        root_dir=Path(data_dir) / split,
        transform=transform,
        multi_task=multi_task,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=is_train,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

"""
PyTorch Dataset для Multi-Task классификации тракторов.

Структура: {split}/{model_class}/{clean|dirty}/image.jpg
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
from PIL import Image, UnidentifiedImageError
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS: frozenset[str] = frozenset({".jpg", ".jpeg", ".png", ".webp"})
STATE_CLASSES: list[str] = ["clean", "dirty"]
STATE_TO_IDX: dict[str, int] = {state: idx for idx, state in enumerate(STATE_CLASSES)}

Sample = tuple[Path, int, Optional[int]]


class TractorDataset(Dataset):
    """Multi-Task датасет: модель трактора + состояние (clean/dirty).

    Args:
        root_dir: Корневая директория сплита (содержит папки классов моделей).
        transform: Опциональная функция трансформации (Albumentations Compose).
        multi_task: Если True, загружает метки состояния из подпапок clean/dirty.
    """

    def __init__(
        self,
        root_dir: Path,
        transform: Optional[Any] = None,
        multi_task: bool = False,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.multi_task = multi_task

        if not self.root_dir.exists():
            raise FileNotFoundError(f"Директория датасета не найдена: {self.root_dir}")
        if not self.root_dir.is_dir():
            raise NotADirectoryError(
                f"Путь датасета должен быть директорией: {self.root_dir}"
            )

        self.model_classes: list[str] = sorted(
            d.name for d in self.root_dir.iterdir() if d.is_dir()
        )
        if not self.model_classes:
            logger.warning("В %s не найдено директорий классов моделей", self.root_dir)

        self.model_to_idx: dict[str, int] = {
            cls: idx for idx, cls in enumerate(self.model_classes)
        }
        self.samples: list[Sample] = self._load_samples()

        logger.info(
            "TractorDataset: %d samples, %d model classes, multi_task=%s",
            len(self.samples),
            len(self.model_classes),
            self.multi_task,
        )

    @staticmethod
    def _is_image_file(path: Path) -> bool:
        """Проверяет, является ли файл изображением.

        Args:
            path: Путь к файлу.

        Returns:
            True, если расширение файла входит в IMAGE_EXTENSIONS.
        """
        return path.suffix.lower() in IMAGE_EXTENSIONS

    def _collect_images(self, directory: Path) -> list[Path]:
        """Собирает все изображения из директории.

        Args:
            directory: Директория с изображениями.

        Returns:
            Список путей к файлам изображений.

        Raises:
            NotADirectoryError: Если directory не является директорией.
        """
        if not directory.is_dir():
            raise NotADirectoryError(
                f"Ожидалась директория с изображениями: {directory}"
            )
        return [f for f in directory.iterdir() if self._is_image_file(f)]

    def _load_samples(self) -> list[Sample]:
        """Собирает (путь, модель, состояние) для каждого изображения.

        Returns:
            Список кортежей (путь_к_изображению, индекс_модели, индекс_состояния).
            Для single-task индекс_состояния равен None.
        """
        samples: list[Sample] = []

        for model_class in self.model_classes:
            model_dir = self.root_dir / model_class
            model_idx = self.model_to_idx[model_class]

            if self.multi_task:
                for state_class in STATE_CLASSES:
                    state_dir = model_dir / state_class
                    if not state_dir.exists():
                        continue

                    state_idx = STATE_TO_IDX[state_class]
                    for img_path in self._collect_images(state_dir):
                        samples.append((img_path, model_idx, state_idx))
            else:
                for img_path in self._collect_images(model_dir):
                    samples.append((img_path, model_idx, None))

        return samples

    def __len__(self) -> int:
        """Возвращает количество образцов в датасете."""
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | np.ndarray]:
        """Загружает и возвращает один образец по индексу.

        Args:
            idx: Индекс образца.

        Returns:
            Словарь с ключами ``image``, ``model_label`` и опционально
            ``state_label``.

        Raises:
            RuntimeError: Если изображение не удалось прочитать.
        """
        img_path, model_idx, state_idx = self.samples[idx]

        try:
            image = Image.open(img_path).convert("RGB")
        except (OSError, UnidentifiedImageError) as exc:
            msg = f"Не удалось загрузить изображение: {img_path}"
            logger.error(msg)
            raise RuntimeError(msg) from exc

        image_array = np.array(image)

        if self.transform is not None:
            image_array = self.transform(image=image_array)["image"]

        result: dict[str, torch.Tensor | np.ndarray] = {
            "image": image_array,
            "model_label": torch.tensor(model_idx, dtype=torch.long),
        }

        if self.multi_task and state_idx is not None:
            result["state_label"] = torch.tensor(state_idx, dtype=torch.long)

        return result

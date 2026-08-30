"""PyTorch Dataset для Multi-Task классификации тракторов."""

from __future__ import annotations

from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from src.config.classes import MODEL_CLASSES, STATE_CLASSES, class_to_idx, state_to_idx

IMAGE_EXTENSIONS: frozenset[str] = frozenset({".jpg", ".jpeg", ".png", ".webp", ".bmp"})


class Sample(NamedTuple):
    """Один образец датасета."""

    path: Path
    model_label: int
    state_label: int


class TractorDataset(Dataset):
    """Multi-Task датасет: класс модели трактора + состояние (clean/dirty).

    Классы модели итерируются ЯВНО из :data:`MODEL_CLASSES`, а не через glob
    произвольных поддиректорий сплита. Это гарантирует, что служебные папки
    (например, ``to_review``) не попадут в датасет как отдельный класс.
    """

    def __init__(
        self,
        root_dir: Path,
        transform: Any | None = None,
        multi_task: bool = False,
    ) -> None:
        """Инициализировать датасет.

        Args:
            root_dir: Корень сплита (например, ``data/dirty_clean/train``).
            transform: Albumentations-трансформация (принимает ``image=``).
            multi_task: Если ``True`` — ожидается уровень ``clean/dirty`` и
                возвращается ``state_label``.
        """
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.multi_task = multi_task
        self.samples = self._load_samples()

    @staticmethod
    def _is_image_file(path: Path) -> bool:
        """Проверить, является ли путь файлом-изображением.

        Args:
            path: Проверяемый путь.

        Returns:
            ``True`` для поддерживаемых расширений изображений.
        """
        return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS

    @classmethod
    def _collect_images(cls, directory: Path) -> list[Path]:
        """Собрать все изображения из директории (рекурсивно).

        Args:
            directory: Директория обхода.

        Returns:
            Отсортированный список путей изображений.
        """
        if not directory.is_dir():
            return []
        return sorted(p for p in directory.rglob("*") if cls._is_image_file(p))

    def _load_samples(self) -> list[Sample]:
        """Построить список образцов, итерируя классы явно из MODEL_CLASSES.

        Returns:
            Список :class:`Sample`.
        """
        samples: list[Sample] = []
        for model_class in MODEL_CLASSES:
            class_dir = self.root_dir / model_class
            if not class_dir.is_dir():
                continue
            model_label = class_to_idx(model_class)

            if self.multi_task:
                for state_class in STATE_CLASSES:
                    state_dir = class_dir / state_class
                    state_label = state_to_idx(state_class)
                    for image_path in self._collect_images(state_dir):
                        samples.append(Sample(image_path, model_label, state_label))
            else:
                # Single-task: состояние не используется, метка-заглушка 0.
                for image_path in self._collect_images(class_dir):
                    samples.append(Sample(image_path, model_label, 0))
        return samples

    def __len__(self) -> int:
        """Вернуть число образцов."""
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | np.ndarray]:
        """Вернуть образец по индексу.

        Args:
            idx: Индекс образца.

        Returns:
            Словарь с ключами ``image``, ``model_label`` и (для multi-task)
            ``state_label``.
        """
        sample = self.samples[idx]
        image = np.array(Image.open(sample.path).convert("RGB"))

        if self.transform is not None:
            image = self.transform(image=image)["image"]

        item: dict[str, torch.Tensor | np.ndarray] = {
            "image": image,
            "model_label": torch.tensor(sample.model_label, dtype=torch.long),
        }
        if self.multi_task:
            item["state_label"] = torch.tensor(sample.state_label, dtype=torch.long)
        return item

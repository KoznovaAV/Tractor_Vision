"""Тесты для data pipeline (dataloader, transforms)."""

from pathlib import Path

import pytest

from src.data.dataloader import get_dataloader
from src.data.transforms import get_train_transforms, get_val_transforms


class TestTransforms:
    """Тесты для трансформаций."""

    def test_train_transforms(self):
        """Тест train трансформаций."""
        transform = get_train_transforms(224)
        assert transform is not None

    def test_val_transforms(self):
        """Тест val трансформаций."""
        transform = get_val_transforms(224)
        assert transform is not None

    def test_transform_output_shape(self):
        """Тест формы выхода трансформаций."""
        import numpy as np

        from src.data.transforms import get_val_transforms

        transform = get_val_transforms(224)
        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

        result = transform(image=image)["image"]
        # PyTorch формат: (C, H, W) = (3, 224, 224)
        assert result.shape == (3, 224, 224)


class TestDataLoader:
    """Тесты для DataLoader."""

    def test_get_dataloader(self):
        """Тест создания DataLoader с реальными данными."""
        data_dir = Path("data/processed")
        if not data_dir.exists():
            pytest.skip("Папка data/processed не существует")

        loader = get_dataloader(
            data_dir=data_dir,
            split="train",
            batch_size=2,
            image_size=224,
            num_workers=0,
        )

        assert loader is not None

        batch = next(iter(loader))
        assert "image" in batch
        assert "model_label" in batch
        assert batch["image"].shape[0] == 2

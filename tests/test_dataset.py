"""Тесты для Dataset и DataLoader."""

from PIL import Image

from src.config.classes import MODEL_CLASSES, STATE_CLASSES
from src.data.dataset import TractorDataset


class TestTractorDataset:
    """Тесты для TractorDataset."""

    def test_dataset_initialization(self, tmp_path):
        """Тест инициализации датасета."""
        train_dir = tmp_path / "train" / MODEL_CLASSES[0]
        train_dir.mkdir(parents=True)

        img = Image.new("RGB", (100, 100), color="red")
        img.save(train_dir / "test.jpg")

        dataset = TractorDataset(root_dir=tmp_path / "train")

        assert len(dataset) == 1
        assert dataset.samples[0].model_label == 0

    def test_dataset_getitem(self, tmp_path):
        """Тест получения элемента из датасета."""
        train_dir = tmp_path / "train" / MODEL_CLASSES[0]
        train_dir.mkdir(parents=True)

        img = Image.new("RGB", (100, 100), color="red")
        img.save(train_dir / "test.jpg")

        dataset = TractorDataset(root_dir=tmp_path / "train")
        sample = dataset[0]

        assert "image" in sample
        assert "model_label" in sample
        assert sample["model_label"] == 0

    def test_multi_task_dataset(self, tmp_path):
        """Тест Multi-Task датасета."""
        state_dir = tmp_path / "train" / MODEL_CLASSES[0] / STATE_CLASSES[0]
        state_dir.mkdir(parents=True)

        img = Image.new("RGB", (100, 100), color="red")
        img.save(state_dir / "test_clean.jpg")

        dataset = TractorDataset(root_dir=tmp_path / "train", multi_task=True)
        sample = dataset[0]

        assert "image" in sample
        assert "model_label" in sample
        assert "state_label" in sample
        assert sample["state_label"] == 0

    def test_invalid_image_extensions(self, tmp_path):
        """Тест игнорирования файлов с неправильными расширениями."""
        train_dir = tmp_path / "train" / MODEL_CLASSES[0]
        train_dir.mkdir(parents=True)

        (train_dir / "test.txt").write_text("not an image")

        dataset = TractorDataset(root_dir=tmp_path / "train")
        assert len(dataset) == 0

    def test_service_dirs_are_not_classes(self, tmp_path):
        """Служебные папки (to_review) не должны становиться классами."""
        review_dir = tmp_path / "train" / "to_review"
        review_dir.mkdir(parents=True)

        img = Image.new("RGB", (100, 100), color="red")
        img.save(review_dir / "x.jpg")

        dataset = TractorDataset(root_dir=tmp_path / "train")

        assert len(dataset) == 0

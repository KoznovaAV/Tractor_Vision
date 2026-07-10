"""Тесты для Dataset и DataLoader."""

from PIL import Image

from src.data.dataset import TractorDataset


class TestTractorDataset:
    """Тесты для TractorDataset."""

    def test_dataset_initialization(self, tmp_path):
        """Тест инициализации датасета."""
        train_dir = tmp_path / "train" / "class_a"
        train_dir.mkdir(parents=True)

        img = Image.new("RGB", (100, 100), color="red")
        img.save(train_dir / "test.jpg")

        dataset = TractorDataset(root_dir=tmp_path / "train")

        assert len(dataset) == 1
        assert dataset.model_classes == ["class_a"]
        assert dataset.model_to_idx == {"class_a": 0}

    def test_dataset_getitem(self, tmp_path):
        """Тест получения элемента из датасета."""
        train_dir = tmp_path / "train" / "class_a"
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
        train_dir = tmp_path / "train" / "class_a" / "clean"
        train_dir.mkdir(parents=True)

        img = Image.new("RGB", (100, 100), color="red")
        img.save(train_dir / "test_clean.jpg")

        dataset = TractorDataset(root_dir=tmp_path / "train", multi_task=True)
        sample = dataset[0]

        assert "image" in sample
        assert "model_label" in sample
        assert "state_label" in sample
        assert sample["state_label"] == 0

    def test_invalid_image_extensions(self, tmp_path):
        """Тест игнорирования файлов с неправильными расширениями."""
        train_dir = tmp_path / "train" / "class_a"
        train_dir.mkdir(parents=True)

        (train_dir / "test.txt").write_text("not an image")

        dataset = TractorDataset(root_dir=tmp_path / "train")
        assert len(dataset) == 0

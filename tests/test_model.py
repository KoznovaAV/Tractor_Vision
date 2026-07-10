"""Тесты для моделей."""

import torch

from src.models.classifier import CLASS_NAMES, TractorClassifier
from src.models.multi_task import MultiTaskTractorClassifier


class TestTractorClassifier:
    """Тесты для Single-Task классификатора."""

    def test_model_initialization(self):
        """Тест инициализации модели."""
        model = TractorClassifier(num_classes=len(CLASS_NAMES))
        assert model.num_classes == len(CLASS_NAMES)

    def test_forward_pass(self):
        """Тест прямого прохода."""
        model = TractorClassifier(num_classes=len(CLASS_NAMES))
        model.eval()

        batch_size = 2
        x = torch.randn(batch_size, 3, 224, 224)

        with torch.no_grad():
            logits = model(x)

        assert logits.shape == (batch_size, len(CLASS_NAMES))

    def test_output_probabilities(self):
        """Тест получения вероятностей."""
        model = TractorClassifier(num_classes=len(CLASS_NAMES))
        model.eval()

        x = torch.randn(1, 3, 224, 224)

        with torch.no_grad():
            logits = model(x)
            probs = torch.softmax(logits, dim=1)

        assert torch.allclose(probs.sum(dim=1), torch.tensor(1.0), atol=1e-5)


class TestMultiTaskClassifier:
    """Тесты для Multi-Task классификатора."""

    def test_model_initialization(self):
        """Тест инициализации модели."""
        model = MultiTaskTractorClassifier()
        assert model.num_model_classes == 5
        assert model.num_state_classes == 2

    def test_forward_pass(self):
        """Тест прямого прохода."""
        model = MultiTaskTractorClassifier()
        model.eval()

        batch_size = 2
        x = torch.randn(batch_size, 3, 224, 224)

        with torch.no_grad():
            model_logits, state_logits = model(x)

        assert model_logits.shape == (batch_size, 5)
        assert state_logits.shape == (batch_size, 2)

    def test_feature_extraction(self):
        """Тест извлечения признаков."""
        model = MultiTaskTractorClassifier()
        model.eval()

        x = torch.randn(1, 3, 224, 224)

        with torch.no_grad():
            features = model.backbone(x)

        assert features.shape[1] == 768

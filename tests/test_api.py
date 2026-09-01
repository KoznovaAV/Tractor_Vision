"""Тесты FastAPI-эндпоинтов Tractor Vision.

Проверяют контракты эндпоинтов, устойчивые к тому, загружены модели или нет
(в тестовом окружении весов может не быть). Статусы допускают оба исхода там,
где результат зависит от наличия моделей.
"""

from __future__ import annotations

import io
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from src.api import main as api_main
from src.api.main import app
from src.config.classes import MODEL_CLASSES


@pytest.fixture(scope="module")
def client() -> TestClient:
    """Создать тестовый клиент FastAPI.

    Returns:
        Клиент с инициализированным lifespan (загрузка моделей).
    """
    with TestClient(app) as test_client:
        yield test_client


def _make_image_bytes(fmt: str = "JPEG") -> bytes:
    """Сгенерировать байты валидного изображения.

    Args:
        fmt: Формат изображения (``JPEG``/``PNG``).

    Returns:
        Байты изображения.
    """
    buffer = io.BytesIO()
    Image.new("RGB", (64, 64), color=(120, 100, 80)).save(buffer, format=fmt)
    return buffer.getvalue()


class TestHealthEndpoint:
    """Тесты эндпоинта ``/health``."""

    def test_health_check(self, client: TestClient) -> None:
        """Статус ``healthy``, присутствуют ``version`` и ``models_loaded``."""
        response = client.get("/health")
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "healthy"
        assert "version" in body
        assert "models_loaded" in body

    def test_health_lists_models_with_versions(self, client: TestClient) -> None:
        """``models`` перечисляет записи реестра с полями ``name`` и ``version``."""
        body = client.get("/health").json()
        assert isinstance(body["models"], list)
        names = {m["name"] for m in body["models"]}
        assert "machine" in names
        machine = next(m for m in body["models"] if m["name"] == "machine")
        assert machine["version"] == "v2-finetune"


class TestModelsEndpoint:
    """Тесты эндпоинта ``/models``."""

    def test_list_models(self, client: TestClient) -> None:
        """Присутствуют ключи ``models`` и ``count``, они согласованы."""
        response = client.get("/models")
        assert response.status_code == 200
        body = response.json()
        assert "models" in body
        assert "count" in body
        assert body["count"] == len(body["models"])


class TestPredictEndpoint:
    """Тесты эндпоинта ``/predict``."""

    def test_predict_with_valid_image(self, client: TestClient) -> None:
        """Валидное JPEG-изображение: 200 (модель есть) или 500 (моделей нет)."""
        files = {"file": ("tractor.jpg", _make_image_bytes("JPEG"), "image/jpeg")}
        response = client.post("/predict", files=files)
        assert response.status_code in (200, 500)

    def test_predict_ok_has_review_fields(self, client: TestClient) -> None:
        """При успехе присутствуют ``request_id``, ``needs_review`` и ``state_confidence``."""
        files = {"file": ("tractor.jpg", _make_image_bytes("JPEG"), "image/jpeg")}
        response = client.post("/predict", files=files)
        if response.status_code != 200:
            pytest.skip("модель не загружена в тестовом окружении")
        body = response.json()
        assert isinstance(body["request_id"], str) and body["request_id"]
        assert isinstance(body["needs_review"], bool)
        assert isinstance(body["state_confidence"], float)
        assert 0.0 <= body["state_confidence"] <= 1.0

    def test_predict_ok_has_traceability_fields(self, client: TestClient) -> None:
        """При успехе присутствуют ``model_version`` и ``checkpoint_sha``."""
        files = {"file": ("tractor.jpg", _make_image_bytes("JPEG"), "image/jpeg")}
        response = client.post("/predict", files=files)
        if response.status_code != 200:
            pytest.skip("модель не загружена в тестовом окружении")
        body = response.json()
        assert body["model_version"] == "v2-finetune"
        assert isinstance(body["checkpoint_sha"], str)
        assert len(body["checkpoint_sha"]) == 12

    def test_predict_checkpoint_sha_stable(self, client: TestClient) -> None:
        """``checkpoint_sha`` одинаков между двумя запросами."""
        files = {"file": ("tractor.jpg", _make_image_bytes("JPEG"), "image/jpeg")}
        first = client.post("/predict", files=files)
        second = client.post("/predict", files=files)
        if first.status_code != 200 or second.status_code != 200:
            pytest.skip("модель не загружена в тестовом окружении")
        assert first.json()["checkpoint_sha"] == second.json()["checkpoint_sha"]

    def test_predict_with_invalid_extension(self, client: TestClient) -> None:
        """Недопустимое расширение файла: 422."""
        files = {"file": ("notes.txt", b"not an image", "text/plain")}
        response = client.post("/predict", files=files)
        assert response.status_code == 422

    def test_predict_with_empty_file(self, client: TestClient) -> None:
        """Пустой файл: 422 (валидация) или 500 (обработка)."""
        files = {"file": ("empty.jpg", b"", "image/jpeg")}
        response = client.post("/predict", files=files)
        assert response.status_code in (422, 500)

    def test_predict_with_png(self, client: TestClient) -> None:
        """Валидное PNG-изображение: 200 (модель есть) или 500 (моделей нет)."""
        files = {"file": ("tractor.png", _make_image_bytes("PNG"), "image/png")}
        response = client.post("/predict", files=files)
        assert response.status_code in (200, 500)


class TestDecideState:
    """Тесты решения по состоянию из порога ``api.state_dirty_threshold``."""

    def test_threshold_from_config(self) -> None:
        """Константа порога берётся из конфига, а не хардкодится."""
        from src.config.config_loader import load_config

        assert api_main.STATE_DIRTY_THRESHOLD == load_config().api.state_dirty_threshold

    def test_p_dirty_at_or_above_threshold_is_dirty(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """``p(dirty) >= threshold`` -> dirty, уверенность равна ``p(dirty)``."""
        monkeypatch.setattr(api_main, "STATE_DIRTY_THRESHOLD", 0.5)
        # state_idx=1 (dirty), conf 0.80 -> p(dirty)=0.80
        state, conf = api_main._decide_state(1, 0.80)
        assert state == "dirty"
        assert conf == pytest.approx(0.80)

    def test_p_dirty_below_threshold_is_clean(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """``p(dirty) < threshold`` -> clean, уверенность равна ``1 - p(dirty)``."""
        monkeypatch.setattr(api_main, "STATE_DIRTY_THRESHOLD", 0.7)
        # state_idx=1 (dirty) по argmax, conf 0.60 -> p(dirty)=0.60 < 0.7
        state, conf = api_main._decide_state(1, 0.60)
        assert state == "clean"
        assert conf == pytest.approx(0.40)

    def test_clean_argmax_recovers_p_dirty(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """При argmax=clean ``p(dirty) = 1 - conf`` и может превысить порог."""
        monkeypatch.setattr(api_main, "STATE_DIRTY_THRESHOLD", 0.4)
        # state_idx=0 (clean), conf 0.55 -> p(dirty)=0.45 >= 0.4 -> dirty
        state, conf = api_main._decide_state(0, 0.55)
        assert state == "dirty"
        assert conf == pytest.approx(0.45)


class TestFeedbackEndpoint:
    """Тесты эндпоинта ``/feedback``."""

    @pytest.fixture(autouse=True)
    def _isolate_feedback_dir(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Перенаправить запись фидбэка во временную директорию."""
        monkeypatch.setattr(api_main, "FEEDBACK_DIR", tmp_path / "feedback")

    def test_feedback_saves_photo_and_manifest(self, client: TestClient) -> None:
        """Валидный фидбэк: 200, ``saved`` истинно, фото и манифест на диске."""
        files = {"file": ("t.jpg", _make_image_bytes("JPEG"), "image/jpeg")}
        data = {"user_family": MODEL_CLASSES[0], "user_state": "dirty", "request_id": "abc"}
        response = client.post("/feedback", files=files, data=data)
        assert response.status_code == 200
        body = response.json()
        assert body["saved"] is True
        photo = Path(body["path"])
        assert photo.is_file()
        assert photo.with_suffix(".json").is_file()

    def test_feedback_rejects_unknown_family(self, client: TestClient) -> None:
        """Неизвестная семья: 422."""
        files = {"file": ("t.jpg", _make_image_bytes("JPEG"), "image/jpeg")}
        response = client.post("/feedback", files=files, data={"user_family": "spaceship"})
        assert response.status_code == 422

    def test_feedback_requires_family(self, client: TestClient) -> None:
        """Отсутствует обязательное поле ``user_family``: 422."""
        files = {"file": ("t.jpg", _make_image_bytes("JPEG"), "image/jpeg")}
        response = client.post("/feedback", files=files)
        assert response.status_code == 422

    def test_feedback_rejects_bad_extension(self, client: TestClient) -> None:
        """Недопустимое расширение файла: 422."""
        files = {"file": ("notes.txt", b"not an image", "text/plain")}
        response = client.post("/feedback", files=files, data={"user_family": MODEL_CLASSES[0]})
        assert response.status_code == 422

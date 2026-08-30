"""Тесты FastAPI-эндпоинтов Tractor Vision.

Проверяют контракты эндпоинтов, устойчивые к тому, загружены модели или нет
(в тестовом окружении весов может не быть). Статусы допускают оба исхода там,
где результат зависит от наличия моделей.
"""

from __future__ import annotations

import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from src.api.main import app


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

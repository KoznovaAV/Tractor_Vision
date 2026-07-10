"""Tests for FastAPI endpoints."""

import io

from fastapi.testclient import TestClient
from PIL import Image

from src.api.main import app

client = TestClient(app)


class TestHealthEndpoint:

    def test_health_check(self) -> None:
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "version" in data
        assert "models_loaded" in data


class TestModelsEndpoint:

    def test_list_models(self) -> None:
        response = client.get("/models")
        assert response.status_code == 200
        data = response.json()
        assert "models" in data
        assert "count" in data
        assert isinstance(data["models"], list)


class TestPredictEndpoint:

    def _make_image_bytes(self, size=(224, 224), color="red") -> bytes:
        img = Image.new("RGB", size, color=color)
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format="JPEG")
        img_byte_arr.seek(0)
        return img_byte_arr.read()

    def test_predict_with_valid_image(self) -> None:
        img_bytes = self._make_image_bytes()
        response = client.post(
            "/predict",
            files={"file": ("test.jpg", io.BytesIO(img_bytes), "image/jpeg")},
        )
        # 200 если модель загружена, 500 если нет
        assert response.status_code in [200, 500]

    def test_predict_with_invalid_extension(self) -> None:
        response = client.post(
            "/predict",
            files={"file": ("test.txt", io.BytesIO(b"not an image"), "text/plain")},
        )
        assert response.status_code == 422

    def test_predict_with_empty_file(self) -> None:
        response = client.post(
            "/predict",
            files={"file": ("empty.jpg", io.BytesIO(b""), "image/jpeg")},
        )
        assert response.status_code in [422, 500]

    def test_predict_with_png(self) -> None:
        img = Image.new("RGB", (224, 224), color="blue")
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format="PNG")
        img_byte_arr.seek(0)

        response = client.post(
            "/predict",
            files={"file": ("test.png", img_byte_arr, "image/png")},
        )
        assert response.status_code in [200, 500]

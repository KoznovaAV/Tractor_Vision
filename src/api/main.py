#!/usr/bin/env python3
"""FastAPI-приложение инференса Tractor Vision.

Сервис поднимает единственную multi-task модель (семья трактора + состояние) и
отдаёт эндпоинты ``/health``, ``/models``, ``/predict``.

Параметры (``image_size``, пути к весам, ``version``, ``max_file_size``) берутся
из ``config.yaml`` через :mod:`src.config.config_loader`; accuracy модели — из
метаданных чекпоинта через :func:`resolve_accuracy` с fallback из конфига. Имена
классов — из :mod:`src.config.classes`. Загрузка весов и инференс — общие утилиты
из :mod:`src.models.loader` и :mod:`src.models.predict`.
"""

from __future__ import annotations

import io
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image

from src.api.schemas import HealthResponse, ModelInfo, PredictionResponse
from src.config.classes import MODEL_CLASSES, STATE_CLASSES
from src.config.config_loader import load_config, resolve_accuracy
from src.data.transforms import get_val_transforms
from src.models.loader import load_multi_task_model, resolve_working_checkpoint
from src.models.multi_task import MultiTaskTractorClassifier
from src.models.predict import predict_image

# Конфигурация читается один раз при импорте модуля.
_CONFIG = load_config()

IMAGE_SIZE: int = _CONFIG.image_size
MAX_FILE_SIZE: int = _CONFIG.api.max_file_size_bytes
ALLOWED_EXTENSIONS: frozenset[str] = frozenset(_CONFIG.api.allowed_extensions)
API_VERSION: str = _CONFIG.api.version

# Валидационная трансформация на едином размере.
transform = get_val_transforms(IMAGE_SIZE)

# Глобальная ссылка на загруженную модель.
multi_task_model: MultiTaskTractorClassifier | None = None


def _load_multi_task() -> MultiTaskTractorClassifier | None:
    """Загрузить multi-task модель, если доступен рабочий чекпоинт.

    Returns:
        Модель в режиме eval либо ``None``, если веса недоступны.
    """
    try:
        checkpoint_path = resolve_working_checkpoint()
    except FileNotFoundError:
        return None
    return load_multi_task_model(checkpoint_path)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Загрузить модели при старте приложения.

    Args:
        app: Экземпляр FastAPI.

    Yields:
        Управление на время жизни приложения.
    """
    global multi_task_model
    multi_task_model = _load_multi_task()
    yield
    multi_task_model = None


app = FastAPI(title="Tractor Vision API", version=API_VERSION, lifespan=lifespan)


@app.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Проверка здоровья сервиса.

    Returns:
        Статус сервиса, версия и флаг загруженности моделей.
    """
    models_loaded = multi_task_model is not None
    return HealthResponse(
        status="healthy",
        version=API_VERSION,
        models_loaded=models_loaded,
    )


@app.get("/models")
async def list_models() -> dict[str, Any]:
    """Список загруженных моделей с их метриками.

    Accuracy читается из метаданных чекпоинта, иначе из fallback конфига.

    Returns:
        Словарь с ключами ``models`` (список) и ``count`` (число).
    """
    models: list[ModelInfo] = []

    if multi_task_model is not None:
        accuracy = resolve_accuracy(
            _CONFIG.weights.multi_task,
            _CONFIG.fallback_accuracy.get("multi_task"),
        )
        models.append(
            ModelInfo(
                name="Multi-Task Classifier",
                num_classes=len(MODEL_CLASSES),
                accuracy=accuracy if accuracy is not None else 0.0,
                weights_path=str(_CONFIG.weights.multi_task),
            )
        )

    return {
        "models": [m.model_dump() for m in models],
        "count": len(models),
    }


def _validate_upload(file: UploadFile) -> None:
    """Проверить расширение загружаемого файла.

    Args:
        file: Загружаемый файл.

    Raises:
        HTTPException: 422, если расширение не разрешено.
    """
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=422,
            detail=f"Недопустимое расширение: {suffix!r}. "
            f"Разрешены: {sorted(ALLOWED_EXTENSIONS)}.",
        )


def _run_inference(image: Image.Image) -> tuple[str, float, str]:
    """Выполнить инференс multi-task моделью.

    Args:
        image: Открытое изображение ``PIL.Image``.

    Returns:
        Кортеж ``(model_class, confidence, state)``.

    Raises:
        HTTPException: 500, если модель не загружена.
    """
    if multi_task_model is None:
        raise HTTPException(status_code=500, detail="No models loaded")

    model_idx, confidence, state_idx = predict_image(multi_task_model, image, transform)
    return MODEL_CLASSES[model_idx], confidence, STATE_CLASSES[state_idx]


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)) -> PredictionResponse:
    """Классифицировать трактор по изображению.

    Args:
        file: Загружаемое изображение (JPEG/PNG, максимум из конфига).

    Returns:
        Предсказание с классом модели, уверенностью, состоянием и таймингом.

    Raises:
        HTTPException: 422 при невалидном/пустом/слишком большом файле; 500 при
            отсутствии моделей или ошибке обработки изображения.
    """
    _validate_upload(file)

    contents = await file.read()
    if len(contents) == 0:
        raise HTTPException(status_code=422, detail="Пустой файл.")
    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=422,
            detail=f"Файл превышает лимит {MAX_FILE_SIZE} байт.",
        )

    start = time.perf_counter()
    try:
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Ошибка обработки изображения: {exc}") from exc

    model_class, confidence, state = _run_inference(image)
    processing_time = time.perf_counter() - start

    return PredictionResponse(
        model_class=model_class,
        confidence=confidence,
        state=state,
        processing_time=processing_time,
        timestamp=datetime.now(),
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)

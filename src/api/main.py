#!/usr/bin/env python3
"""FastAPI-приложение инференса Tractor Vision.

Сервис поднимает единственную multi-task модель (семья трактора + состояние) и
отдаёт эндпоинты ``/health``, ``/models``, ``/predict``.

Набор моделей задаётся разделом ``models`` в ``config.yaml`` и разбирается в
реестр (:mod:`src.models.registry`): ``lifespan`` грузит по записи реестра каждую
доступную модель, ``/models`` перечисляет реестр, ``/predict`` использует запись
``machine`` (переопределяется параметром ``?model=``).

Параметры (``image_size``, пути к весам, ``version``, ``max_file_size``) берутся
из ``config.yaml`` через :mod:`src.config.config_loader`; accuracy модели — из
метаданных чекпоинта через :func:`resolve_accuracy` с fallback из конфига. Имена
классов — из :mod:`src.config.classes`. Инференс — общая утилита из
:mod:`src.models.predict`.
"""

from __future__ import annotations

import io
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from PIL import Image

from src.api.schemas import HealthResponse, ModelInfo, PredictionResponse
from src.config.classes import MODEL_CLASSES, STATE_CLASSES
from src.config.config_loader import load_config, resolve_accuracy
from src.data.transforms import get_val_transforms
from src.models.multi_task import MultiTaskTractorClassifier
from src.models.predict import predict_image
from src.models.registry import build_registry, get_model

# Конфигурация и реестр моделей читаются один раз при импорте модуля.
_CONFIG = load_config()
_REGISTRY = build_registry(_CONFIG)

# Модель по умолчанию для /predict, если параметр ?model= не задан.
DEFAULT_MODEL: str = "machine"

IMAGE_SIZE: int = _CONFIG.image_size
MAX_FILE_SIZE: int = _CONFIG.api.max_file_size_bytes
ALLOWED_EXTENSIONS: frozenset[str] = frozenset(_CONFIG.api.allowed_extensions)
API_VERSION: str = _CONFIG.api.version

# Валидационная трансформация на едином размере.
transform = get_val_transforms(IMAGE_SIZE)

# Модели, загруженные при старте: имя записи реестра -> модель в режиме eval.
_loaded_models: dict[str, MultiTaskTractorClassifier] = {}


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Загрузить модели реестра при старте приложения.

    Каждая запись реестра грузится независимо; записи без доступного чекпоинта
    или с неподдерживаемым типом пропускаются.

    Args:
        app: Экземпляр FastAPI.

    Yields:
        Управление на время жизни приложения.
    """
    for name, entry in _REGISTRY.items():
        try:
            _loaded_models[name] = get_model(entry)
        except (FileNotFoundError, ValueError):
            continue
    yield
    _loaded_models.clear()
    get_model.cache_clear()


app = FastAPI(title="Tractor Vision API", version=API_VERSION, lifespan=lifespan)


@app.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Проверка здоровья сервиса.

    Returns:
        Статус сервиса, версия и флаг загруженности моделей.
    """
    models_loaded = bool(_loaded_models)
    return HealthResponse(
        status="healthy",
        version=API_VERSION,
        models_loaded=models_loaded,
    )


@app.get("/models")
async def list_models() -> dict[str, Any]:
    """Список моделей реестра, загруженных при старте, с их метриками.

    Accuracy читается из метаданных чекпоинта записи, иначе из fallback конфига
    по типу модели.

    Returns:
        Словарь с ключами ``models`` (список) и ``count`` (число).
    """
    models: list[ModelInfo] = []

    for name, entry in _REGISTRY.items():
        if name not in _loaded_models:
            continue
        accuracy = resolve_accuracy(
            entry.checkpoint,
            _CONFIG.fallback_accuracy.get(entry.type),
        )
        models.append(
            ModelInfo(
                name=entry.name,
                num_classes=len(MODEL_CLASSES),
                accuracy=accuracy if accuracy is not None else 0.0,
                weights_path=str(entry.checkpoint),
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


def _run_inference(image: Image.Image, model_name: str) -> tuple[str, float, str]:
    """Выполнить инференс моделью реестра по имени.

    Args:
        image: Открытое изображение ``PIL.Image``.
        model_name: Имя записи реестра (параметр ``?model=``).

    Returns:
        Кортеж ``(model_class, confidence, state)``.

    Raises:
        HTTPException: 422, если модель не значится в реестре; 500, если модель
            есть в реестре, но не загружена.
    """
    if model_name not in _REGISTRY:
        raise HTTPException(
            status_code=422,
            detail=f"Неизвестная модель: {model_name!r}. Доступны: {sorted(_REGISTRY)}.",
        )

    model = _loaded_models.get(model_name)
    if model is None:
        raise HTTPException(status_code=500, detail="No models loaded")

    model_idx, confidence, state_idx = predict_image(model, image, transform)
    return MODEL_CLASSES[model_idx], confidence, STATE_CLASSES[state_idx]


@app.post("/predict", response_model=PredictionResponse)
async def predict(
    file: UploadFile = File(...),
    model: str = Query(default=DEFAULT_MODEL),
) -> PredictionResponse:
    """Классифицировать трактор по изображению.

    Args:
        file: Загружаемое изображение (JPEG/PNG, максимум из конфига).
        model: Имя модели из реестра; по умолчанию ``machine``.

    Returns:
        Предсказание с классом модели, уверенностью, состоянием и таймингом.

    Raises:
        HTTPException: 422 при невалидном/пустом/слишком большом файле или
            неизвестной модели; 500 при отсутствии моделей или ошибке обработки
            изображения.
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

    model_class, confidence, state = _run_inference(image, model)
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

#!/usr/bin/env python3
"""FastAPI-приложение инференса Tractor Vision.

Все параметры, ранее захардкоженные в этом модуле, вынесены в ``config.yaml`` и
читаются через :mod:`src.config.config_loader`:

* ``image_size`` — единый размер (384) для train/eval/инференса; раньше здесь
  было 224, что резало точность в проде;
* пути к весам — из конфига, а не строковые литералы ``"weights/..."``;
* ``version`` и ``max_file_size`` — из конфига;
* accuracy моделей — через :func:`resolve_accuracy` (метаданные чекпоинта, иначе
  fallback из конфига); хардкод ``0.9149`` / ``0.7917`` убран.

Имена классов берутся из :mod:`src.config.classes`. Контракты эндпоинтов
(``/health``, ``/models``, ``/predict``) и Pydantic-схемы не изменены.
"""

from __future__ import annotations

import io
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image

from src.api.schemas import HealthResponse, ModelInfo, PredictionResponse
from src.config.classes import MODEL_CLASSES, STATE_CLASSES
from src.config.config_loader import load_config, resolve_accuracy
from src.data.transforms import get_val_transforms
from src.models.classifier import TractorClassifier
from src.models.multi_task import MultiTaskTractorClassifier

# Конфигурация читается один раз при импорте модуля.
_CONFIG = load_config()

IMAGE_SIZE: int = _CONFIG.image_size  # 384 — единый размер
MAX_FILE_SIZE: int = _CONFIG.api.max_file_size_bytes
ALLOWED_EXTENSIONS: frozenset[str] = frozenset(_CONFIG.api.allowed_extensions)
API_VERSION: str = _CONFIG.api.version

# Валидационная трансформация на едином размере.
transform = get_val_transforms(IMAGE_SIZE)

# Глобальные ссылки на загруженные модели.
single_task_model: TractorClassifier | None = None
multi_task_model: MultiTaskTractorClassifier | None = None


def _load_single_task() -> TractorClassifier | None:
    """Загрузить single-task модель, если её чекпоинт существует.

    Returns:
        Модель в режиме eval либо ``None``, если веса недоступны.
    """
    path = _CONFIG.weights.single_task
    if not Path(path).is_file():
        return None
    model = TractorClassifier(num_classes=len(MODEL_CLASSES))
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("state_dict", checkpoint)
    cleaned = {k.removeprefix("model."): v for k, v in state_dict.items()}
    model.load_state_dict(cleaned, strict=False)
    model.eval()
    return model


def _load_multi_task() -> MultiTaskTractorClassifier | None:
    """Загрузить multi-task модель, если её чекпоинт существует.

    Returns:
        Модель в режиме eval либо ``None``, если веса недоступны.
    """
    path = _CONFIG.weights.multi_task
    if not Path(path).is_file():
        return None
    model = MultiTaskTractorClassifier(
        num_model_classes=len(MODEL_CLASSES),
        num_state_classes=len(STATE_CLASSES),
    )
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("state_dict", checkpoint)
    cleaned = {k.removeprefix("model."): v for k, v in state_dict.items()}
    model.load_state_dict(cleaned, strict=False)
    model.eval()
    return model


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Загрузить модели при старте приложения.

    Args:
        app: Экземпляр FastAPI.

    Yields:
        Управление на время жизни приложения.
    """
    global single_task_model, multi_task_model
    single_task_model = _load_single_task()
    multi_task_model = _load_multi_task()
    yield
    single_task_model = None
    multi_task_model = None


app = FastAPI(title="Tractor Vision API", version=API_VERSION, lifespan=lifespan)


@app.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Проверка здоровья сервиса.

    Returns:
        Статус сервиса, версия и флаг загруженности моделей.
    """
    models_loaded = single_task_model is not None or multi_task_model is not None
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

    if single_task_model is not None:
        accuracy = resolve_accuracy(
            _CONFIG.weights.single_task,
            _CONFIG.fallback_accuracy.get("single_task"),
        )
        models.append(
            ModelInfo(
                name="Single-Task Classifier",
                num_classes=len(MODEL_CLASSES),
                accuracy=accuracy if accuracy is not None else 0.0,
                weights_path=str(_CONFIG.weights.single_task),
            )
        )

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


def _run_inference(input_tensor: torch.Tensor) -> tuple[str, float, str | None]:
    """Выполнить инференс доступной моделью.

    Приоритет у multi-task модели; при её отсутствии используется single-task.

    Args:
        input_tensor: Батч-тензор ``(1, 3, H, W)``.

    Returns:
        Кортеж ``(model_class, confidence, state)``; ``state`` — ``None`` для
        single-task модели.

    Raises:
        HTTPException: 500, если не загружено ни одной модели.
    """
    if multi_task_model is not None:
        model_logits, state_logits = multi_task_model(input_tensor)
        model_probs = torch.softmax(model_logits, dim=1)
        state_probs = torch.softmax(state_logits, dim=1)
        model_idx = int(torch.argmax(model_probs, dim=1).item())
        state_idx = int(torch.argmax(state_probs, dim=1).item())
        model_class = MODEL_CLASSES[model_idx]
        confidence = float(model_probs[0, model_idx].item())
        state = STATE_CLASSES[state_idx]
        return model_class, confidence, state

    if single_task_model is not None:
        logits = single_task_model(input_tensor)
        probs = torch.softmax(logits, dim=1)
        model_idx = int(torch.argmax(probs, dim=1).item())
        model_class = MODEL_CLASSES[model_idx]
        confidence = float(probs[0, model_idx].item())
        return model_class, confidence, None

    raise HTTPException(status_code=500, detail="No models loaded")


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

    import numpy as np

    tensor = transform(image=np.array(image))["image"].unsqueeze(0)
    model_class, confidence, state = _run_inference(tensor)
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

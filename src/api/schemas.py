"""Pydantic-схемы запросов и ответов API."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class PredictionResponse(BaseModel):
    """Ответ эндпоинта ``/predict``.

    ``state_confidence`` — softmax-уверенность в возвращённом состоянии с учётом
    порога ``api.state_dirty_threshold`` (``p(dirty)`` для ``dirty``,
    ``1 - p(dirty)`` для ``clean``).

    ``model_version`` и ``checkpoint_sha`` идентифицируют модель, выдавшую
    предсказание: метка версии из ``config.yaml`` (``None``, если не задана) и
    первые 12 hex-символов SHA-256 файла чекпоинта.
    """

    model_class: str
    confidence: float = Field(ge=0.0, le=1.0)
    state: str | None = None
    state_confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    processing_time: float
    timestamp: datetime
    request_id: str
    needs_review: bool
    model_version: str | None = None
    checkpoint_sha: str | None = None


class BatchItemResult(BaseModel):
    """Результат одного файла в ответе ``/predict_batch``.

    ``status`` — ``ok`` (тогда заполнено ``prediction``) либо ``error`` (тогда
    заполнено ``error`` с текстом причины).
    """

    file_name: str
    status: Literal["ok", "error"]
    error: str | None = None
    prediction: PredictionResponse | None = None


class BatchPredictionResponse(BaseModel):
    """Ответ эндпоинта ``/predict_batch``.

    ``processed`` — число успешно обработанных файлов, ``failed`` — число файлов
    с ошибкой; их сумма равна длине ``results``. ``model_version`` и
    ``checkpoint_sha`` берутся из первого успешного предсказания (``None``, если
    успешных нет).
    """

    results: list[BatchItemResult]
    processed: int
    failed: int
    model_version: str | None = None
    checkpoint_sha: str | None = None


class FeedbackResponse(BaseModel):
    """Ответ эндпоинта ``/feedback``."""

    saved: bool
    path: str


class HealthModelVersion(BaseModel):
    """Имя и метка версии одной модели реестра для ``/health``."""

    name: str
    version: str | None = None


class HealthResponse(BaseModel):
    """Ответ эндпоинта ``/health``.

    ``models`` перечисляет модели реестра с их метками версий (``version`` —
    ``None``, если в конфиге не задана).
    """

    status: str = "healthy"
    version: str = "1.0.0"
    models_loaded: bool = False
    models: list[HealthModelVersion] = Field(default_factory=list)


class ModelInfo(BaseModel):
    """Информация об одной загруженной модели для ``/models``."""

    name: str
    num_classes: int
    accuracy: float
    weights_path: str | None = None

"""Pydantic-схемы запросов и ответов API."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class PredictionResponse(BaseModel):
    """Ответ эндпоинта ``/predict``."""

    model_class: str
    confidence: float = Field(ge=0.0, le=1.0)
    state: str | None = None
    processing_time: float
    timestamp: datetime
    request_id: str
    needs_review: bool


class FeedbackResponse(BaseModel):
    """Ответ эндпоинта ``/feedback``."""

    saved: bool
    path: str


class HealthResponse(BaseModel):
    """Ответ эндпоинта ``/health``."""

    status: str = "healthy"
    version: str = "1.0.0"
    models_loaded: bool = False


class ModelInfo(BaseModel):
    """Информация об одной загруженной модели для ``/models``."""

    name: str
    num_classes: int
    accuracy: float
    weights_path: str | None = None

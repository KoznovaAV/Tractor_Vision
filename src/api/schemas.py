from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class PredictionResponse(BaseModel):

    model_class: str = Field(..., description="Название модели трактора")
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Уверенность предсказания"
    )
    state: Optional[str] = Field(None, description="Состояние: clean/dirty")
    processing_time: float = Field(..., description="Время обработки в секундах")
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class HealthResponse(BaseModel):

    status: str = "healthy"
    version: str = "1.0.0"
    models_loaded: bool = False


class ModelInfo(BaseModel):

    name: str
    num_classes: int
    accuracy: float
    weights_path: Optional[str] = None

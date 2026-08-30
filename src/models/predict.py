"""Единый инференс multi-task модели по одному изображению.

Заменяет разрозненные копии «открыть картинку → трансформ → softmax → argmax» в
API, демо-скрипте, псевдоразметке и оценке на реальной грязи.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from src.models.multi_task import MultiTaskTractorClassifier


def predict_image(
    model: MultiTaskTractorClassifier,
    image: str | Path | Image.Image,
    transform: Any,
) -> tuple[int, float, int]:
    """Прогнать одно изображение через multi-task модель.

    Args:
        model: Модель в режиме eval (на любом устройстве).
        image: Путь к файлу либо уже открытое ``PIL.Image``.
        transform: Валидационная трансформация Albumentations (принимает ``image=``).

    Returns:
        Кортеж ``(model_idx, model_conf, state_idx)``: индекс класса техники,
        его softmax-уверенность и индекс класса состояния (clean/dirty).
    """
    if isinstance(image, (str, Path)):
        image = Image.open(image)
    image_array = np.array(image.convert("RGB"))

    device = next(model.parameters()).device
    tensor = transform(image=image_array)["image"].unsqueeze(0).to(device)

    with torch.no_grad():
        model_logits, state_logits = model(tensor)

    model_probs = torch.softmax(model_logits, dim=1)
    state_probs = torch.softmax(state_logits, dim=1)
    model_idx = int(torch.argmax(model_probs, dim=1).item())
    state_idx = int(torch.argmax(state_probs, dim=1).item())
    model_conf = float(model_probs[0, model_idx].item())
    return model_idx, model_conf, state_idx


__all__ = ["predict_image"]

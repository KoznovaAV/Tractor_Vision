"""Единая загрузка multi-task чекпоинта и выбор рабочего чекпоинта.

Собирает все варианты чтения весов, которые раньше были продублированы в API,
демо-скрипте, псевдоразметке и обоих скриптах оценки: снятие Lightning-префикса
``model.`` со ``state_dict``, загрузка с ``weights_only=False``, отбрасывание
ключей вне архитектуры (параметры балансировки потерь) и перевод модели в
``eval``.

:func:`resolve_working_checkpoint` выбирает актуальный чекпоинт: сначала рабочий
файл ``config.weights.multi_task``, а при его отсутствии — самый свежий по
времени изменения среди ``weights/multi-task-best-*.ckpt`` (сортировка по имени
не используется: строка ``"19"`` больше ``"13"`` и выбирала бы не ту эпоху).
"""

from __future__ import annotations

from pathlib import Path

import torch

from src.config.classes import MODEL_CLASSES, STATE_CLASSES
from src.config.config_loader import load_config
from src.models.multi_task import MultiTaskTractorClassifier

CHECKPOINT_GLOB: str = "multi-task-best-*.ckpt"


def load_multi_task_model(
    checkpoint_path: Path,
    device: str | torch.device = "cpu",
) -> MultiTaskTractorClassifier:
    """Загрузить multi-task модель из чекпоинта в режиме eval.

    Поддерживает Lightning-чекпоинты (ключ ``state_dict`` с префиксом ``model.``)
    и сырые ``state_dict``. Ключи, которых нет в архитектуре модели (обучаемые
    ``log_var``/``task_weights`` из балансировки потерь), отбрасываются.

    Args:
        checkpoint_path: Путь к ``.ckpt``.
        device: Устройство, на которое переносится модель.

    Returns:
        Модель в режиме eval на указанном устройстве.

    Raises:
        FileNotFoundError: Если чекпоинт не найден.
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Чекпоинт не найден: {checkpoint_path}")

    model = MultiTaskTractorClassifier(
        num_model_classes=len(MODEL_CLASSES),
        num_state_classes=len(STATE_CLASSES),
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    raw_state = (
        checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    )

    target = model.state_dict()
    cleaned = {
        stripped: value
        for key, value in raw_state.items()
        if (stripped := key.removeprefix("model.")) in target
    }
    model.load_state_dict(cleaned, strict=True)
    model.eval()
    return model.to(device)


def resolve_working_checkpoint(weights_dir: Path | None = None) -> Path:
    """Выбрать актуальный чекпоинт multi-task модели.

    Порядок: рабочий файл ``config.weights.multi_task``; при его отсутствии —
    самый свежий по ``st_mtime`` среди ``<weights_dir>/multi-task-best-*.ckpt``.

    Args:
        weights_dir: Директория поиска best-чекпоинтов. ``None`` — ``config.weights.dir``.

    Returns:
        Путь к существующему чекпоинту.

    Raises:
        FileNotFoundError: Если ни один источник не дал чекпоинт.
    """
    config = load_config()
    working = Path(config.weights.multi_task)
    if working.is_file():
        return working

    search_dir = Path(weights_dir) if weights_dir is not None else Path(config.weights.dir)
    candidates = list(search_dir.glob(CHECKPOINT_GLOB))
    if not candidates:
        raise FileNotFoundError(
            f"Рабочий чекпоинт не найден: ни {working}, "
            f"ни файлов {search_dir / CHECKPOINT_GLOB}"
        )
    return max(candidates, key=lambda path: path.stat().st_mtime)


__all__ = ["CHECKPOINT_GLOB", "load_multi_task_model", "resolve_working_checkpoint"]

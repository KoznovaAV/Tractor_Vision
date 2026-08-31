"""Реестр моделей инференса, управляемый разделом ``models`` из ``config.yaml``.

Раздел ``models`` конфигурации превращается в набор :class:`ModelEntry` через
:func:`build_registry`. :func:`get_model` загружает веса для записи и кэширует
результат: повторный вызов с той же записью возвращает тот же объект модели без
повторного чтения чекпоинта с диска.

Поддерживается тип загрузчика ``multi_task`` (:func:`src.models.loader.
load_multi_task_model`). Если чекпоинт записи отсутствует, выбирается актуальный
рабочий чекпоинт через :func:`src.models.loader.resolve_working_checkpoint`.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from src.config.config_loader import AppConfig
from src.models.loader import load_multi_task_model, resolve_working_checkpoint
from src.models.multi_task import MultiTaskTractorClassifier

# Типы загрузчиков, для которых get_model умеет строить модель.
_MULTI_TASK_TYPE: str = "multi_task"


@dataclass(frozen=True)
class ModelEntry:
    """Запись реестра: описание модели без загруженных весов.

    Attributes:
        name: Имя модели (ключ в API, параметр ``?model=``).
        checkpoint: Путь к чекпоинту весов.
        type: Тип загрузчика (``multi_task``).
        tasks: Задачи, которые решает модель (``family``, ``state`` и т. п.).
    """

    name: str
    checkpoint: Path
    type: str
    tasks: tuple[str, ...]


def build_registry(config: AppConfig) -> dict[str, ModelEntry]:
    """Построить реестр моделей из конфигурации приложения.

    Args:
        config: Загруженная конфигурация с разделом ``models``.

    Returns:
        Словарь ``имя модели -> ModelEntry``.
    """
    return {
        name: ModelEntry(
            name=spec.name,
            checkpoint=Path(spec.checkpoint),
            type=spec.type,
            tasks=tuple(spec.tasks),
        )
        for name, spec in config.models.items()
    }


@lru_cache(maxsize=None)
def get_model(entry: ModelEntry) -> MultiTaskTractorClassifier:
    """Загрузить (и закэшировать) модель для записи реестра.

    Кэш ключуется по самой записи (``ModelEntry`` — неизменяемый dataclass),
    поэтому повторный вызов возвращает тот же объект модели.

    Args:
        entry: Запись реестра.

    Returns:
        Модель в режиме eval на CPU.

    Raises:
        ValueError: Если тип загрузчика записи не поддерживается.
        FileNotFoundError: Если для записи не удалось найти чекпоинт.
    """
    if entry.type != _MULTI_TASK_TYPE:
        raise ValueError(
            f"Неподдерживаемый тип модели {entry.type!r} для записи {entry.name!r}. "
            f"Ожидался {_MULTI_TASK_TYPE!r}."
        )

    checkpoint = entry.checkpoint
    if not checkpoint.is_file():
        checkpoint = resolve_working_checkpoint()
    return load_multi_task_model(checkpoint)


__all__ = ["ModelEntry", "build_registry", "get_model"]

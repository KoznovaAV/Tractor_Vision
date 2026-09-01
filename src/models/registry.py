"""Реестр моделей инференса, управляемый разделом ``models`` из ``config.yaml``.

Раздел ``models`` конфигурации превращается в набор :class:`ModelEntry` через
:func:`build_registry`. :func:`get_model` загружает веса для записи и кэширует
результат: повторный вызов с той же записью возвращает тот же объект модели без
повторного чтения чекпоинта с диска. :func:`get_model_meta` возвращает вместе с
моделью её метку версии и первые 12 hex-символов SHA-256 файла чекпоинта
(тоже кэшируются) — для трассируемости предсказаний.

Поддерживается тип загрузчика ``multi_task`` (:func:`src.models.loader.
load_multi_task_model`). Если чекпоинт записи отсутствует, выбирается актуальный
рабочий чекпоинт через :func:`src.models.loader.resolve_working_checkpoint`.
"""

from __future__ import annotations

import hashlib
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
        version: Метка версии модели из конфига либо ``None``.
    """

    name: str
    checkpoint: Path
    type: str
    tasks: tuple[str, ...]
    version: str | None = None


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
            version=spec.version,
        )
        for name, spec in config.models.items()
    }


def _resolve_checkpoint(entry: ModelEntry) -> Path:
    """Вернуть путь к чекпоинту записи, откатываясь на актуальный рабочий.

    Args:
        entry: Запись реестра.

    Returns:
        Путь к чекпоинту записи, если файл существует, иначе результат
        :func:`resolve_working_checkpoint`.
    """
    checkpoint = entry.checkpoint
    if not checkpoint.is_file():
        checkpoint = resolve_working_checkpoint()
    return checkpoint


@lru_cache(maxsize=None)
def _checkpoint_sha(checkpoint: Path) -> str:
    """Первые 12 hex-символов SHA-256 файла чекпоинта (кэшируется по пути).

    Args:
        checkpoint: Путь к существующему файлу чекпоинта.

    Returns:
        Усечённый до 12 символов hex-дайджест SHA-256.
    """
    digest = hashlib.sha256()
    with Path(checkpoint).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()[:12]


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

    return load_multi_task_model(_resolve_checkpoint(entry))


def get_model_meta(
    entry: ModelEntry,
) -> tuple[MultiTaskTractorClassifier, str | None, str]:
    """Загруженная модель вместе с её версией и хешем чекпоинта.

    Args:
        entry: Запись реестра.

    Returns:
        Кортеж ``(model, version, checkpoint_sha)``: модель из :func:`get_model`,
        метка версии записи (``entry.version``) и первые 12 hex-символов SHA-256
        фактически загруженного файла чекпоинта.

    Raises:
        ValueError: Если тип загрузчика записи не поддерживается.
        FileNotFoundError: Если для записи не удалось найти чекпоинт.
    """
    model = get_model(entry)
    return model, entry.version, _checkpoint_sha(_resolve_checkpoint(entry))


__all__ = ["ModelEntry", "build_registry", "get_model", "get_model_meta"]

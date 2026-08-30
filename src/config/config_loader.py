"""Загрузка конфигурации проекта и метрик из метаданных чекпоинтов.

Централизует чтение ``config.yaml`` (пути, ``image_size``, число классов) и
извлечение accuracy из чекпоинтов Lightning: точность ищется в
``hyper_parameters`` и в корне чекпоинта по набору известных ключей
(:data:`_ACCURACY_KEYS`), с fallback на значения из ``config.yaml``.

Пример::

    from src.config.config_loader import load_config, read_checkpoint_accuracy

    cfg = load_config()
    size = cfg.image_size                      # единый размер для train, eval, API
    acc = read_checkpoint_accuracy(cfg.weights.multi_task)  # из чекпоинта
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

# Кандидаты-ключи внутри чекпоинта, где может лежать accuracy. Проверяются по
# порядку; берётся первое найденное числовое значение.
_ACCURACY_KEYS: tuple[str, ...] = (
    "test_accuracy",
    "val_model_acc",
    "val_acc",
    "accuracy",
)

# Расположение config.yaml по умолчанию: корень проекта (на два уровня выше
# этого файла: src/config/config_loader.py -> src -> корень).
_DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config.yaml"


@dataclass(frozen=True)
class DataPaths:
    """Пути к деревьям данных.

    Attributes:
        processed_dir: Дерево чистых изображений (train/val/test) по классам.
        dirty_clean_dir: Multi-task дерево (train/val/test, уровень clean/dirty).
        collected_dir: Сырой сбор коллектора до prepare_dataset.
    """

    processed_dir: Path
    dirty_clean_dir: Path
    collected_dir: Path


@dataclass(frozen=True)
class WeightPaths:
    """Пути к весам моделей.

    Attributes:
        dir: Директория весов.
        multi_task: Чекпоинт multi-task модели.
    """

    dir: Path
    multi_task: Path


@dataclass(frozen=True)
class ApiConfig:
    """Параметры API.

    Attributes:
        max_file_size_bytes: Максимальный размер загружаемого файла в байтах.
        allowed_extensions: Разрешённые расширения изображений.
        version: Версия сервиса.
    """

    max_file_size_bytes: int
    allowed_extensions: tuple[str, ...]
    version: str


@dataclass(frozen=True)
class AppConfig:
    """Полная конфигурация приложения.

    Attributes:
        image_size: Единый размер изображения (px) для train/eval/API.
        num_model_classes: Число классов модели трактора.
        num_state_classes: Число классов состояния.
        data: Пути к данным.
        weights: Пути к весам.
        api: Параметры API.
        fallback_accuracy: Резервные метрики, если их нет в чекпоинте.
    """

    image_size: int
    num_model_classes: int
    num_state_classes: int
    data: DataPaths
    weights: WeightPaths
    api: ApiConfig
    fallback_accuracy: dict[str, float | None]


def _parse_config(raw: dict[str, Any]) -> AppConfig:
    """Преобразовать сырой словарь YAML в типизированный :class:`AppConfig`.

    Args:
        raw: Разобранное содержимое ``config.yaml``.

    Returns:
        Типизированная конфигурация.
    """
    data_raw = raw["data"]
    weights_raw = raw["weights"]
    api_raw = raw["api"]

    data = DataPaths(
        processed_dir=Path(data_raw["processed_dir"]),
        dirty_clean_dir=Path(data_raw["dirty_clean_dir"]),
        collected_dir=Path(data_raw["collected_dir"]),
    )
    weights = WeightPaths(
        dir=Path(weights_raw["dir"]),
        multi_task=Path(weights_raw["multi_task"]),
    )
    api = ApiConfig(
        max_file_size_bytes=int(api_raw["max_file_size_mb"]) * 1024 * 1024,
        allowed_extensions=tuple(api_raw["allowed_extensions"]),
        version=str(api_raw["version"]),
    )
    return AppConfig(
        image_size=int(raw["image_size"]),
        num_model_classes=int(raw["num_model_classes"]),
        num_state_classes=int(raw["num_state_classes"]),
        data=data,
        weights=weights,
        api=api,
        fallback_accuracy=dict(raw.get("fallback_accuracy", {})),
    )


@lru_cache(maxsize=1)
def load_config(config_path: Path | None = None) -> AppConfig:
    """Загрузить и закэшировать конфигурацию проекта.

    Args:
        config_path: Путь к ``config.yaml``. ``None`` — путь по умолчанию
            (корень проекта).

    Returns:
        Типизированная конфигурация приложения.

    Raises:
        FileNotFoundError: Если файл конфигурации не найден.
    """
    path = config_path or _DEFAULT_CONFIG_PATH
    if not path.is_file():
        raise FileNotFoundError(f"Файл конфигурации не найден: {path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    return _parse_config(raw)


def read_checkpoint_accuracy(checkpoint_path: Path) -> float | None:
    """Извлечь accuracy из метаданных чекпоинта Lightning.

    Ищет значение в ``hyper_parameters`` и в корне чекпоинта по ключам-кандидатам
    (:data:`_ACCURACY_KEYS`). Чекпоинт загружается на CPU без загрузки весов
    модели в память сверх необходимого.

    Args:
        checkpoint_path: Путь к ``.ckpt`` файлу.

    Returns:
        Accuracy как ``float`` в диапазоне ``[0, 1]`` либо ``None``, если
        метаданные отсутствуют или файл недоступен.
    """
    if not Path(checkpoint_path).is_file():
        return None

    # Ленивый импорт torch — модуль конфигурации не должен тянуть torch без нужды.
    import torch

    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except (RuntimeError, EOFError, OSError):
        return None

    if not isinstance(checkpoint, dict):
        return None

    search_scopes: list[dict[str, Any]] = [checkpoint]
    hparams = checkpoint.get("hyper_parameters")
    if isinstance(hparams, dict):
        search_scopes.append(hparams)

    for scope in search_scopes:
        for key in _ACCURACY_KEYS:
            value = scope.get(key)
            if isinstance(value, (int, float)):
                return float(value)

    return None


def resolve_accuracy(
    checkpoint_path: Path,
    fallback: float | None,
) -> float | None:
    """Вернуть accuracy из чекпоинта, откатываясь на fallback из конфига.

    Args:
        checkpoint_path: Путь к чекпоинту.
        fallback: Резервное значение из ``config.yaml``.

    Returns:
        Accuracy из чекпоинта, иначе fallback, иначе ``None``.
    """
    from_checkpoint = read_checkpoint_accuracy(checkpoint_path)
    if from_checkpoint is not None:
        return from_checkpoint
    return fallback


__all__ = [
    "ApiConfig",
    "AppConfig",
    "DataPaths",
    "WeightPaths",
    "load_config",
    "read_checkpoint_accuracy",
    "resolve_accuracy",
]

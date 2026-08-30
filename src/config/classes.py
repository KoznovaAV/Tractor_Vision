"""Единый источник правды по классам проекта Tractor Vision.

Датасет, модель, API и тесты импортируют имена классов и их индексы только
отсюда. В проекте 4 класса модели трактора и 2 класса состояния (clean/dirty).

Порядок классов зафиксирован и является контрактом: индекс класса в кортеже
соответствует индексу выходного логита соответствующей головы модели. При
изменении порядка или состава классов голова классификатора переобучается
с нуля.

Пример использования::

    from src.config.classes import MODEL_CLASSES, class_to_idx

    idx = class_to_idx("mtz_belarus")  # -> 3
"""

from __future__ import annotations

from typing import Final

# ---------------------------------------------------------------------------
# Классы модели трактора. ПОРЯДОК ФИКСИРОВАН — контракт с весами головы.
# ---------------------------------------------------------------------------
MODEL_CLASSES = ("chtz", "johndeere", "kirovets", "mtz_belarus")
NUM_MODEL_CLASSES: Final[int] = len(MODEL_CLASSES)

# ---------------------------------------------------------------------------
# Классы состояния (clean/dirty). Порядок фиксирован: clean=0, dirty=1.
# ---------------------------------------------------------------------------
STATE_CLASSES: Final[tuple[str, ...]] = ("clean", "dirty")
NUM_STATE_CLASSES: Final[int] = len(STATE_CLASSES)

# ---------------------------------------------------------------------------
# Псевдонимы имён, сводимые к каноническому классу при разметке и сборе данных
# (исторические имена моделей семейства МТЗ).
# ---------------------------------------------------------------------------
CLASS_ALIASES: Final[dict[str, str]] = {
    "mtz_1221": "mtz_belarus",
    "mtz_82": "mtz_belarus",
}

_MODEL_CLASS_TO_IDX: Final[dict[str, int]] = {name: idx for idx, name in enumerate(MODEL_CLASSES)}
_STATE_CLASS_TO_IDX: Final[dict[str, int]] = {name: idx for idx, name in enumerate(STATE_CLASSES)}


def canonical_class(name: str) -> str:
    """Свести имя класса к каноническому с учётом слияний."""
    return CLASS_ALIASES.get(name, name)


def class_to_idx(name: str) -> int:
    """Индекс класса модели трактора по имени (с учётом псевдонимов)."""
    canonical = canonical_class(name)
    if canonical not in _MODEL_CLASS_TO_IDX:
        raise KeyError(
            f"Неизвестный класс модели: {name!r} "
            f"(канонический: {canonical!r}). Ожидались: {MODEL_CLASSES}."
        )
    return _MODEL_CLASS_TO_IDX[canonical]


def state_to_idx(name: str) -> int:
    """Индекс класса состояния (clean/dirty) по имени."""
    if name not in _STATE_CLASS_TO_IDX:
        raise KeyError(f"Неизвестный класс состояния: {name!r}. Ожидались: {STATE_CLASSES}.")
    return _STATE_CLASS_TO_IDX[name]


__all__ = [
    "CLASS_ALIASES",
    "MODEL_CLASSES",
    "NUM_MODEL_CLASSES",
    "NUM_STATE_CLASSES",
    "STATE_CLASSES",
    "canonical_class",
    "class_to_idx",
    "state_to_idx",
]

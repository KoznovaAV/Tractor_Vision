#!/usr/bin/env python3
"""Оценка модели на held-out наборе реальной грязи (``data/real_dirty_val``).

Считает метрики для сравнения до/после дообучения головы состояния. Все фото в
наборе — грязные (реальная грязь), поэтому истинное состояние всегда ``dirty``;
ключевая метрика — **dirty recall** (доля грязных фото, распознанных как dirty),
цель проекта ≥ 0.90. Дополнительно считается accuracy классификации техники
(она уже сильная и не должна просесть от дообучения состояния).

Набор организован как ``<val_root>/<class>/*.jpg`` (метка класса — имя папки,
метка состояния — всегда dirty). Классы берутся из :mod:`src.config.classes`.

Пример::

    python -m scripts.eval_real_dirty \\
        --val-dir data/real_dirty_val \\
        --checkpoint weights/multi_task_best.ckpt
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch

from src.config.classes import class_to_idx, state_to_idx
from src.config.config_loader import load_config
from src.data.transforms import get_val_transforms
from src.models.loader import load_multi_task_model
from src.models.multi_task import MultiTaskTractorClassifier
from src.models.predict import predict_image

IMAGE_EXTENSIONS: frozenset[str] = frozenset({".jpg", ".jpeg", ".png", ".webp"})
DIRTY_STATE: str = "dirty"


def _iter_val_images(val_root: Path) -> list[tuple[Path, str]]:
    """Собрать пары ``(путь, класс)`` из held-out набора.

    Класс — имя папки первого уровня. Все фото считаются состоянием ``dirty``.

    Args:
        val_root: Корень набора ``data/real_dirty_val``.

    Returns:
        Список пар ``(путь_изображения, имя_класса)``.
    """
    pairs: list[tuple[Path, str]] = []
    if not val_root.is_dir():
        return pairs
    for class_dir in sorted(val_root.iterdir()):
        if not class_dir.is_dir():
            continue
        for image_path in sorted(class_dir.rglob("*")):
            if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTENSIONS:
                pairs.append((image_path, class_dir.name))
    return pairs


def evaluate(
    model: MultiTaskTractorClassifier,
    pairs: list[tuple[Path, str]],
    transform: Any,
) -> dict[str, Any]:
    """Прогнать модель по набору и посчитать метрики состояния и класса.

    Args:
        model: Multi-task модель в режиме eval.
        pairs: Пары ``(путь, класс)``.
        transform: Валидационная трансформация.

    Returns:
        Словарь метрик: ``dirty_recall``, ``model_accuracy``, счётчики и
        поклассовый dirty recall.
    """
    dirty_idx = state_to_idx(DIRTY_STATE)
    total = 0
    state_correct = 0  # предсказано dirty
    model_correct = 0  # верный класс техники
    per_class_total: dict[str, int] = defaultdict(int)
    per_class_dirty: dict[str, int] = defaultdict(int)

    for image_path, class_name in pairs:
        model_pred, _, state_pred, _ = predict_image(model, image_path, transform)

        total += 1
        per_class_total[class_name] += 1
        if state_pred == dirty_idx:
            state_correct += 1
            per_class_dirty[class_name] += 1
        if model_pred == class_to_idx(class_name):
            model_correct += 1

    dirty_recall = state_correct / total if total else 0.0
    model_accuracy = model_correct / total if total else 0.0
    per_class_recall = {
        cls: per_class_dirty[cls] / per_class_total[cls] for cls in sorted(per_class_total)
    }

    return {
        "total": total,
        "dirty_recall": dirty_recall,
        "model_accuracy": model_accuracy,
        "dirty_predicted": state_correct,
        "per_class_recall": per_class_recall,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Разобрать аргументы командной строки.

    Args:
        argv: Аргументы (по умолчанию ``sys.argv[1:]``).

    Returns:
        Пространство имён аргументов.
    """
    config = load_config()
    parser = argparse.ArgumentParser(
        description="Оценка модели на held-out наборе реальной грязи.",
    )
    parser.add_argument(
        "--val-dir",
        type=Path,
        default=Path("data/real_dirty_val"),
        help="Корень held-out набора реальной грязи.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=config.weights.multi_task,
        help="Чекпоинт multi-task модели (по умолчанию из config.yaml).",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=config.image_size,
        help="Размер изображения (по умолчанию из config.yaml).",
    )
    parser.add_argument(
        "--target-recall",
        type=float,
        default=0.90,
        help="Целевой dirty recall для итогового вердикта.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Точка входа оценки.

    Args:
        argv: Аргументы командной строки (для тестируемости).

    Returns:
        Код возврата процесса: 0 при достижении целевого recall, иначе 2.
    """
    args = parse_args(argv)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    pairs = _iter_val_images(args.val_dir)
    if not pairs:
        print(f"[ошибка] Нет изображений в {args.val_dir}")
        return 1

    model = load_multi_task_model(args.checkpoint, device)
    transform = get_val_transforms(args.image_size)

    results = evaluate(model, pairs, transform)

    print("=" * 55)
    print(f"ОЦЕНКА НА РЕАЛЬНОЙ ГРЯЗИ ({args.val_dir})")
    print("=" * 55)
    print(f"Всего фото:        {results['total']}")
    print(
        f"Dirty recall:      {results['dirty_recall']:.4f} "
        f"({results['dirty_predicted']}/{results['total']} распознаны как dirty)"
    )
    print(f"Model accuracy:    {results['model_accuracy']:.4f}")
    print("\nDirty recall по классам:")
    for cls, recall in results["per_class_recall"].items():
        print(f"    {cls:<16} {recall:.4f}")

    reached = results["dirty_recall"] >= args.target_recall
    verdict = "ДОСТИГНУТА" if reached else "НЕ достигнута"
    print(
        f"\nЦель dirty recall >= {args.target_recall:.2f}: {verdict} "
        f"(факт {results['dirty_recall']:.4f})"
    )
    return 0 if reached else 2


if __name__ == "__main__":
    sys.exit(main())

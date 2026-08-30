#!/usr/bin/env python3
"""Псевдоразметка собранных грязных фото по классам через обученную модель.

Использует ``weights/multi_task_best.ckpt`` (путь берётся из ``config.yaml``) для
классификации техники на собранных грязных фото. Работает по голове КЛАССА
модели (не состояния — состояние на этих фото как раз слабое, ради него всё и
затевается). Раскладка:

* уверенность класса ``>= threshold`` (по умолчанию 0.8) -> в
  ``data/real_dirty_labeled/<class>/``;
* иначе -> ``data/real_dirty_labeled/to_review/`` для ручной проверки.

Скрипт идемпотентен: имя файла назначения детерминировано (по имени источника),
повторный прогон не плодит копии. Классы берутся из :mod:`src.config.classes`.

Пример::

    python -m scripts.pseudo_label_dirty \\
        --input-dir data/real_dirty_raw/unsorted \\
        --output-dir data/real_dirty_labeled \\
        --threshold 0.8
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.config.classes import MODEL_CLASSES, STATE_CLASSES
from src.config.config_loader import load_config
from src.data.transforms import get_val_transforms
from src.models.multi_task import MultiTaskTractorClassifier

IMAGE_EXTENSIONS: frozenset[str] = frozenset({".jpg", ".jpeg", ".png", ".webp"})
REVIEW_SUBDIR: str = "to_review"


def load_multi_task_model(
    checkpoint_path: Path,
) -> MultiTaskTractorClassifier:
    """Загрузить multi-task модель из чекпоинта в режиме eval.

    Поддерживает Lightning-чекпоинты (ключ ``state_dict`` с префиксом ``model.``)
    и сырые ``state_dict``.

    Args:
        checkpoint_path: Путь к ``.ckpt``.

    Returns:
        Модель в режиме eval.

    Raises:
        FileNotFoundError: Если чекпоинт не найден.
    """
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Чекпоинт не найден: {checkpoint_path}")

    model = MultiTaskTractorClassifier(
        num_model_classes=len(MODEL_CLASSES),
        num_state_classes=len(STATE_CLASSES),
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("state_dict", checkpoint)
    cleaned = {key.removeprefix("model."): value for key, value in state_dict.items()}
    model.load_state_dict(cleaned, strict=False)
    model.eval()
    return model


def predict_class(
    model: MultiTaskTractorClassifier,
    image_path: Path,
    transform: Any,
    device: torch.device,
) -> tuple[str, float]:
    """Предсказать класс техники и уверенность для одного изображения.

    Args:
        model: Multi-task модель.
        image_path: Путь к изображению.
        transform: Валидационная трансформация.
        device: Устройство.

    Returns:
        Кортеж ``(имя_класса, уверенность)``.
    """
    from PIL import Image

    image = np.array(Image.open(image_path).convert("RGB"))
    tensor = transform(image=image)["image"].unsqueeze(0).to(device)
    with torch.no_grad():
        model_logits, _ = model(tensor)
        probs = torch.softmax(model_logits, dim=1)
        idx = int(torch.argmax(probs, dim=1).item())
        confidence = float(probs[0, idx].item())
    return MODEL_CLASSES[idx], confidence


def _iter_images(input_dir: Path) -> list[Path]:
    """Собрать изображения из входной директории (рекурсивно).

    Args:
        input_dir: Директория с фото (например, ``real_dirty_raw/unsorted``).

    Returns:
        Отсортированный список путей.
    """
    if not input_dir.is_dir():
        return []
    return sorted(
        p for p in input_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def pseudo_label(
    model: MultiTaskTractorClassifier,
    input_dir: Path,
    output_dir: Path,
    transform: Any,
    device: torch.device,
    threshold: float,
) -> dict[str, int]:
    """Разложить изображения по классам согласно предсказанию с порогом.

    Идемпотентно: имя файла назначения детерминировано по имени источника,
    существующий файл не копируется повторно.

    Args:
        model: Multi-task модель.
        input_dir: Директория с собранными фото.
        output_dir: Корень выходного дерева ``real_dirty_labeled``.
        transform: Валидационная трансформация.
        device: Устройство.
        threshold: Порог уверенности для авто-раскладки.

    Returns:
        Счётчик по классам плюс ключ ``to_review``.
    """
    counts: dict[str, int] = {name: 0 for name in MODEL_CLASSES}
    counts[REVIEW_SUBDIR] = 0

    for image_path in _iter_images(input_dir):
        model_class, confidence = predict_class(model, image_path, transform, device)
        if confidence >= threshold:
            dest_dir = output_dir / model_class
            counts[model_class] += 1
        else:
            dest_dir = output_dir / REVIEW_SUBDIR
            counts[REVIEW_SUBDIR] += 1

        dest_dir.mkdir(parents=True, exist_ok=True)
        # Детерминированное имя -> идемпотентность.
        dest = dest_dir / image_path.name
        if not dest.exists():
            shutil.copy2(image_path, dest)

    return counts


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Разобрать аргументы командной строки.

    Args:
        argv: Аргументы (по умолчанию ``sys.argv[1:]``).

    Returns:
        Пространство имён аргументов.
    """
    config = load_config()
    parser = argparse.ArgumentParser(
        description="Псевдоразметка грязных фото по классам через multi-task модель.",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/real_dirty_raw/unsorted"),
        help="Директория с собранными грязными фото.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/real_dirty_labeled"),
        help="Корень дерева с разложенными по классам фото.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=config.weights.multi_task,
        help="Чекпоинт multi-task модели (по умолчанию из config.yaml).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.8,
        help="Порог уверенности класса для авто-раскладки (иначе to_review).",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=config.image_size,
        help="Размер изображения (по умолчанию из config.yaml).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Точка входа псевдоразметки.

    Args:
        argv: Аргументы командной строки (для тестируемости).

    Returns:
        Код возврата процесса (0 — успех).
    """
    args = parse_args(argv)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = load_multi_task_model(args.checkpoint).to(device)
    transform = get_val_transforms(args.image_size)

    print(f"Модель: {args.checkpoint} | порог: {args.threshold}")
    counts = pseudo_label(
        model=model,
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        transform=transform,
        device=device,
        threshold=args.threshold,
    )

    print("\n" + "=" * 50)
    print("ИТОГ ПСЕВДОРАЗМЕТКИ")
    print("=" * 50)
    total = sum(counts.values())
    for name, count in counts.items():
        print(f"  {name:<16} {count}")
    print(f"Всего размечено: {total}")
    print(f"\nПроверить вручную: {args.output_dir / REVIEW_SUBDIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

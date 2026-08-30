#!/usr/bin/env python3
"""Разбиение реальных грязных фото на train-добавку и held-out валидацию.

Объединяет размеченные по классам реальные грязные фото из нескольких источников
(например, ``data/collected_dirty/<class>`` и ``data/real_dirty_labeled/<class>``
после ручной проверки ``to_review``) и стратифицированно по классам делит 80/20:

* 80% копируются в ``data/dirty_clean/train/<class>/dirty/`` с префиксом ``real_``
  — они вливаются в обучающую выборку головы состояния;
* 20% копируются в ``data/real_dirty_val/<class>/`` — held-out набор, НЕ
  участвующий в обучении, для честной оценки dirty recall до/после дообучения.

Префикс ``real_`` позволяет отличать реальные грязные фото от синтетических
(``*_dirty0``) внутри общей папки ``dirty/`` и при необходимости удалять/считать
их отдельно. Стратификация и content-hash дедупликация переиспользуют подход из
``prepare_dataset.py``. Классы берутся из :mod:`src.config.classes`.

Идемпотентно: имена файлов детерминированы (префикс + content-hash), повторный
прогон не плодит копии.

Пример::

    python -m scripts.split_real_dirty \\
        --sources data/collected_dirty data/real_dirty_labeled \\
        --train-dir data/dirty_clean \\
        --val-dir data/real_dirty_val \\
        --val-ratio 0.2 --seed 42
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.config.classes import MODEL_CLASSES

IMAGE_EXTENSIONS: frozenset[str] = frozenset({".jpg", ".jpeg", ".png", ".webp"})
REAL_PREFIX: str = "real_"
# Служебные папки, которые не считаются классами при обходе источников.
SKIP_DIRS: frozenset[str] = frozenset({"to_review", "unsorted"})


@dataclass(frozen=True)
class DirtyRecord:
    """Одно реальное грязное фото с классом и хешом содержимого.

    Attributes:
        path: Путь к исходному файлу.
        model_class: Канонический класс модели трактора.
        content_hash: MD5 содержимого (для дедупликации между источниками).
    """

    path: Path
    model_class: str
    content_hash: str


def _content_hash(path: Path) -> str:
    """Вычислить MD5 содержимого файла.

    Args:
        path: Путь к файлу.

    Returns:
        Шестнадцатеричный дайджест MD5.
    """
    hasher = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _iter_source(source: Path) -> list[DirtyRecord]:
    """Собрать записи одного источника, итерируя классы явно из MODEL_CLASSES.

    Args:
        source: Корень источника (``<class>/`` подпапки).

    Returns:
        Список записей с классом и хешом.
    """
    records: list[DirtyRecord] = []
    if not source.is_dir():
        print(f"[skip] источник не найден: {source}")
        return records

    for model_class in MODEL_CLASSES:
        class_dir = source / model_class
        if not class_dir.is_dir() or class_dir.name in SKIP_DIRS:
            continue
        for image_path in sorted(class_dir.rglob("*")):
            if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTENSIONS:
                records.append(
                    DirtyRecord(
                        path=image_path,
                        model_class=model_class,
                        content_hash=_content_hash(image_path),
                    )
                )
    return records


def _deduplicate(records: list[DirtyRecord]) -> list[DirtyRecord]:
    """Убрать дубликаты по content-hash, сохраняя первое вхождение.

    Args:
        records: Список записей.

    Returns:
        Список без повторяющегося содержимого.
    """
    seen: set[str] = set()
    unique: list[DirtyRecord] = []
    for record in records:
        if record.content_hash in seen:
            continue
        seen.add(record.content_hash)
        unique.append(record)
    return unique


def _stratified_split(
    records: list[DirtyRecord],
    val_ratio: float,
    seed: int,
) -> tuple[list[DirtyRecord], list[DirtyRecord]]:
    """Стратифицированно по классам разбить записи на train и val.

    Args:
        records: Все уникальные записи.
        val_ratio: Доля в held-out val (например, 0.2).
        seed: Зерно ГПСЧ.

    Returns:
        Кортеж ``(train_records, val_records)``.
    """
    rng = np.random.default_rng(seed)
    by_class: dict[str, list[DirtyRecord]] = defaultdict(list)
    for record in records:
        by_class[record.model_class].append(record)

    train: list[DirtyRecord] = []
    val: list[DirtyRecord] = []
    for model_class in sorted(by_class):
        items = by_class[model_class]
        indices = rng.permutation(len(items))
        n_val = int(round(len(items) * val_ratio))
        val_idx = set(indices[:n_val].tolist())
        for i, record in enumerate(items):
            (val if i in val_idx else train).append(record)
    return train, val


def _copy_train(records: list[DirtyRecord], train_root: Path) -> int:
    """Скопировать train-записи в ``<train_root>/train/<class>/dirty/`` с префиксом.

    Args:
        records: Train-записи.
        train_root: Корень multi-task дерева (``data/dirty_clean``).

    Returns:
        Число фактически скопированных файлов.
    """
    copied = 0
    for record in records:
        dest_dir = train_root / "train" / record.model_class / "dirty"
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / f"{REAL_PREFIX}{record.content_hash[:16]}{record.path.suffix.lower()}"
        if not dest.exists():
            shutil.copy2(record.path, dest)
            copied += 1
    return copied


def _copy_val(records: list[DirtyRecord], val_root: Path) -> int:
    """Скопировать val-записи в ``<val_root>/<class>/`` (вне обучения).

    Args:
        records: Val-записи.
        val_root: Корень held-out набора (``data/real_dirty_val``).

    Returns:
        Число фактически скопированных файлов.
    """
    copied = 0
    for record in records:
        dest_dir = val_root / record.model_class
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / f"{REAL_PREFIX}{record.content_hash[:16]}{record.path.suffix.lower()}"
        if not dest.exists():
            shutil.copy2(record.path, dest)
            copied += 1
    return copied


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Разобрать аргументы командной строки.

    Args:
        argv: Аргументы (по умолчанию ``sys.argv[1:]``).

    Returns:
        Пространство имён аргументов.
    """
    parser = argparse.ArgumentParser(
        description="Разбиение реальных грязных фото 80/20 (train-добавка + held-out val).",
    )
    parser.add_argument(
        "--sources",
        type=Path,
        nargs="+",
        default=[Path("data/collected_dirty"), Path("data/real_dirty_labeled")],
        help="Корни источников с раскладкой по классам.",
    )
    parser.add_argument(
        "--train-dir",
        type=Path,
        default=Path("data/dirty_clean"),
        help="Корень multi-task дерева (train/<class>/dirty/).",
    )
    parser.add_argument(
        "--val-dir",
        type=Path,
        default=Path("data/real_dirty_val"),
        help="Корень held-out набора реальной грязи.",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.2,
        help="Доля в held-out val.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Зерно ГПСЧ для воспроизводимого разбиения.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Точка входа разбиения.

    Args:
        argv: Аргументы командной строки (для тестируемости).

    Returns:
        Код возврата процесса (0 — успех).
    """
    args = parse_args(argv)

    if not 0.0 < args.val_ratio < 1.0:
        print(f"[ошибка] val-ratio должно быть в (0, 1), получено {args.val_ratio}")
        return 1

    all_records: list[DirtyRecord] = []
    for source in args.sources:
        source_records = _iter_source(source)
        print(f"  {source}: {len(source_records)} фото")
        all_records.extend(source_records)

    if not all_records:
        print("[ошибка] Не найдено ни одного грязного фото в источниках.")
        return 1

    before = len(all_records)
    all_records = _deduplicate(all_records)
    print(f"Дедупликация по содержимому: {before} -> {len(all_records)}")

    train, val = _stratified_split(all_records, args.val_ratio, args.seed)
    train_copied = _copy_train(train, args.train_dir)
    val_copied = _copy_val(val, args.val_dir)

    # Распределение по классам для контроля стратификации.
    def _by_class(records: list[DirtyRecord]) -> dict[str, int]:
        counts: dict[str, int] = defaultdict(int)
        for record in records:
            counts[record.model_class] += 1
        return counts

    print("\n" + "=" * 55)
    print("ИТОГ РАЗБИЕНИЯ РЕАЛЬНОЙ ГРЯЗИ")
    print("=" * 55)
    print(f"train (-> {args.train_dir}/train/<class>/dirty/, префикс {REAL_PREFIX}):")
    for cls, count in sorted(_by_class(train).items()):
        print(f"    {cls:<16} {count}")
    print(f"  скопировано: {train_copied}")
    print(f"val (-> {args.val_dir}/<class>/, held-out):")
    for cls, count in sorted(_by_class(val).items()):
        print(f"    {cls:<16} {count}")
    print(f"  скопировано: {val_copied}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

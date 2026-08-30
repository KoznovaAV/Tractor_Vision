#!/usr/bin/env python3
"""Объединение источников данных и стратифицированная пересборка сплитов.

Сливает существующий датасет (после слияния классов) и свежесобранные
изображения из ``data/collected`` в единое дерево, затем заново нарезает сплиты
train/val/test в пропорции 70/15/15 **стратифицированно**, чтобы распределение
классов (и состояний, если они присутствуют) сохранялось в каждом сплите.

Порядок в общем пайплайне данных::

    collect_tractors.py      # сбор -> data/collected
    prepare_dataset.py       # объединение + пересборка 70/15/15  (ЭТОТ скрипт)
    generate_dirty_dataset.py  # генерация грязи поверх финального processed

Стратификация адаптивна к структуре входа:

* если под классом есть уровень ``clean/dirty`` — страта = ``(класс, состояние)``
  и выходное дерево сохраняет уровень состояния (multi-task форма);
* если уровня состояния нет — страта = ``класс`` (single-task форма).

Классы берутся явно из :data:`MODEL_CLASSES`, поэтому служебные папки
(``to_review`` и т.п.) в пересборку не попадают. Дедупликация между источниками
выполняется по content-hash (md5 содержимого файла), чтобы одно и то же
изображение из разных источников не оказалось одновременно в train и test.

Пример::

    python -m scripts.prepare_dataset \\
        --sources data/processed data/collected \\
        --output-dir data/processed_rebuilt \\
        --ratios 0.7 0.15 0.15 \\
        --seed 42
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

from src.config.classes import MODEL_CLASSES, STATE_CLASSES

IMAGE_EXTENSIONS: frozenset[str] = frozenset({".jpg", ".jpeg", ".png", ".webp", ".bmp"})
SPLIT_NAMES: tuple[str, str, str] = ("train", "val", "test")
# Служебные директории, которые никогда не считаются классом/состоянием.
SKIP_DIRS: frozenset[str] = frozenset({"to_review"}) | frozenset(SPLIT_NAMES)


@dataclass(frozen=True)
class ImageRecord:
    """Одно изображение с его стратой.

    Attributes:
        path: Путь к исходному файлу.
        model_class: Канонический класс модели трактора.
        state: Состояние (``clean``/``dirty``) либо ``None`` для single-task.
        content_hash: MD5 содержимого (для дедупликации между источниками).
    """

    path: Path
    model_class: str
    state: str | None
    content_hash: str


def _is_image(path: Path) -> bool:
    """Проверить, является ли путь файлом-изображением.

    Args:
        path: Проверяемый путь.

    Returns:
        ``True`` для поддерживаемых расширений.
    """
    return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS


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


def _iter_source_images(source: Path) -> list[ImageRecord]:
    """Собрать изображения одного источника с определением страты.

    Источник может быть как «плоским» деревом ``<class>/`` или
    ``<class>/<state>/``, так и уже разбитым на сплиты ``<split>/<class>/...`` —
    в последнем случае сплиты обходятся прозрачно (существующее разбиение
    игнорируется, данные пересобираются заново).

    Args:
        source: Корень источника данных.

    Returns:
        Список записей изображений с проставленной стратой.
    """
    records: list[ImageRecord] = []
    if not source.is_dir():
        print(f"[skip] источник не найден: {source}")
        return records

    # Определяем, разбит ли источник на сплиты верхнего уровня.
    top_level = {child.name for child in source.iterdir() if child.is_dir()}
    split_roots: list[Path]
    if top_level & set(SPLIT_NAMES):
        split_roots = [source / name for name in SPLIT_NAMES if (source / name).is_dir()]
    else:
        split_roots = [source]

    for split_root in split_roots:
        for model_class in MODEL_CLASSES:
            class_dir = split_root / model_class
            if not class_dir.is_dir():
                continue

            # Есть ли уровень состояния clean/dirty?
            state_subdirs = [state for state in STATE_CLASSES if (class_dir / state).is_dir()]

            if state_subdirs:
                for state in state_subdirs:
                    for image_path in sorted((class_dir / state).rglob("*")):
                        if _is_image(image_path):
                            records.append(
                                ImageRecord(
                                    path=image_path,
                                    model_class=model_class,
                                    state=state,
                                    content_hash=_content_hash(image_path),
                                )
                            )
            else:
                for image_path in sorted(class_dir.rglob("*")):
                    # Пропускаем возможные state-подпапки, уже обойдённые выше,
                    # и любые служебные директории в пути.
                    if not _is_image(image_path):
                        continue
                    parts = set(image_path.relative_to(class_dir).parts[:-1])
                    if parts & SKIP_DIRS:
                        continue
                    records.append(
                        ImageRecord(
                            path=image_path,
                            model_class=model_class,
                            state=None,
                            content_hash=_content_hash(image_path),
                        )
                    )
    return records


def _deduplicate(records: list[ImageRecord]) -> list[ImageRecord]:
    """Убрать дубликаты по content-hash, сохраняя первое вхождение.

    Args:
        records: Список записей (возможно с дубликатами между источниками).

    Returns:
        Список без повторяющегося содержимого.
    """
    seen: set[str] = set()
    unique: list[ImageRecord] = []
    for record in records:
        if record.content_hash in seen:
            continue
        seen.add(record.content_hash)
        unique.append(record)
    return unique


def _stratified_split(
    records: list[ImageRecord],
    ratios: tuple[float, float, float],
    seed: int,
) -> dict[str, list[ImageRecord]]:
    """Разбить записи на train/val/test стратифицированно.

    Страта — ключ ``(model_class, state)``; в каждой страте записи перемешиваются
    и режутся по заданным пропорциям. Это сохраняет баланс классов и состояний
    во всех сплитах даже при неравномерном исходном распределении.

    Args:
        records: Все уникальные записи.
        ratios: Доли train/val/test (сумма должна быть ~1.0).
        seed: Зерно ГПСЧ.

    Returns:
        Словарь ``{split: [records]}``.
    """
    rng = np.random.default_rng(seed)
    buckets: dict[tuple[str, str | None], list[ImageRecord]] = defaultdict(list)
    for record in records:
        buckets[(record.model_class, record.state)].append(record)

    result: dict[str, list[ImageRecord]] = {name: [] for name in SPLIT_NAMES}
    train_ratio, val_ratio, _ = ratios

    for stratum, items in sorted(buckets.items(), key=lambda kv: str(kv[0])):
        indices = rng.permutation(len(items))
        n_total = len(items)
        n_train = int(round(n_total * train_ratio))
        n_val = int(round(n_total * val_ratio))
        # Остаток уходит в test, чтобы суммарно сходилось при округлении.
        n_train = min(n_train, n_total)
        n_val = min(n_val, n_total - n_train)

        train_idx = indices[:n_train]
        val_idx = indices[n_train : n_train + n_val]
        test_idx = indices[n_train + n_val :]

        for split_name, split_indices in zip(SPLIT_NAMES, (train_idx, val_idx, test_idx)):
            result[split_name].extend(items[i] for i in split_indices)

    return result


def _write_split(
    split_name: str,
    records: list[ImageRecord],
    output_root: Path,
) -> None:
    """Скопировать записи одного сплита в выходное дерево.

    Выходной путь сохраняет уровень состояния, если оно задано:
    ``<split>/<class>/<state>/`` для multi-task формы или ``<split>/<class>/``
    для single-task.

    Args:
        split_name: Имя сплита.
        records: Записи сплита.
        output_root: Корень выходного дерева.
    """
    for record in records:
        if record.state is not None:
            dest_dir = output_root / split_name / record.model_class / record.state
        else:
            dest_dir = output_root / split_name / record.model_class
        dest_dir.mkdir(parents=True, exist_ok=True)

        # Имя по content-hash предотвращает коллизии между источниками.
        dest = dest_dir / f"{record.content_hash[:16]}{record.path.suffix.lower()}"
        if not dest.exists():
            shutil.copy2(record.path, dest)


def _print_distribution(splits: dict[str, list[ImageRecord]]) -> None:
    """Напечатать распределение классов/состояний по сплитам.

    Args:
        splits: Словарь ``{split: [records]}``.
    """
    print("\nРаспределение по сплитам:")
    for split_name in SPLIT_NAMES:
        records = splits[split_name]
        by_class: dict[str, int] = defaultdict(int)
        by_state: dict[str, int] = defaultdict(int)
        for record in records:
            by_class[record.model_class] += 1
            if record.state is not None:
                by_state[record.state] += 1
        class_str = ", ".join(f"{k}={v}" for k, v in sorted(by_class.items()))
        state_str = (
            " | " + ", ".join(f"{k}={v}" for k, v in sorted(by_state.items())) if by_state else ""
        )
        print(f"  {split_name:<6} всего={len(records):<5} [{class_str}]{state_str}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Разобрать аргументы командной строки.

    Args:
        argv: Аргументы (по умолчанию ``sys.argv[1:]``).

    Returns:
        Пространство имён с аргументами.
    """
    parser = argparse.ArgumentParser(
        description="Объединение источников и стратифицированная пересборка сплитов.",
    )
    parser.add_argument(
        "--sources",
        type=Path,
        nargs="+",
        default=[Path("data/processed"), Path("data/collected")],
        help="Корни источников данных для объединения.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed_rebuilt"),
        help="Корень выходного дерева с пересобранными сплитами.",
    )
    parser.add_argument(
        "--ratios",
        type=float,
        nargs=3,
        default=[0.7, 0.15, 0.15],
        metavar=("TRAIN", "VAL", "TEST"),
        help="Доли train/val/test (сумма ~1.0).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Зерно ГПСЧ для воспроизводимого разбиения.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Точка входа подготовки датасета.

    Args:
        argv: Аргументы командной строки (для тестируемости).

    Returns:
        Код возврата процесса (0 — успех).
    """
    args = parse_args(argv)

    ratios = tuple(args.ratios)
    if abs(sum(ratios) - 1.0) > 1e-6:
        print(f"[ошибка] Сумма долей должна быть 1.0, получено {sum(ratios):.4f}")
        return 1

    print(f"Источники: {[str(s) for s in args.sources]}")
    all_records: list[ImageRecord] = []
    for source in args.sources:
        source_records = _iter_source_images(source)
        print(f"  {source}: {len(source_records)} изображений")
        all_records.extend(source_records)

    if not all_records:
        print("[ошибка] Не найдено ни одного изображения в источниках.")
        return 1

    before = len(all_records)
    all_records = _deduplicate(all_records)
    print(f"Дедупликация по содержимому: {before} -> {len(all_records)}")

    splits = _stratified_split(all_records, ratios, args.seed)  # type: ignore[arg-type]

    print(f"\nЗапись в {args.output_dir} ...")
    for split_name in SPLIT_NAMES:
        _write_split(split_name, splits[split_name], args.output_dir)

    _print_distribution(splits)

    total = sum(len(v) for v in splits.values())
    print(f"\nГотово. Всего записано: {total} изображений.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

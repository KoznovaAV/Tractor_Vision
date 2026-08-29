#!/usr/bin/env python3
"""Слияние классов ``mtz_1221`` -> ``mtz_82`` во всех сплитах датасета.

Класс ``mtz_1221`` визуально неотличим от ``mtz_82``, поэтому изображения
переносятся в папку ``mtz_82``, а исходная папка ``mtz_1221`` удаляется. Данные
не выбрасываются — они переиспользуются в целевом классе.

Скрипт обходит **фактически существующие** сплит-директории в каждом дереве,
ничего не хардкодя:

* single-task дерево ``data/processed/`` содержит ``train/val/test``;
* multi-task дерево ``data/dirty_clean/`` содержит ``train/val`` и под каждым
  классом ещё уровень ``clean/dirty``.

Оба дерева обрабатываются одинаковой рекурсивной логикой: ищем на любой глубине
директории с именем ``mtz_1221`` (или любым псевдонимом из
:data:`CLASS_ALIASES`) и сливаем их с соседней канонической папкой.

Пример::

    python -m scripts.merge_mtz1221 \\
        --processed-dir data/processed \\
        --dirty-clean-dir data/dirty_clean

    # Предпросмотр без изменений на диске:
    python -m scripts.merge_mtz1221 --dry-run
"""

from __future__ import annotations

import argparse
import shutil
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

# Скрипт запускается как ``python -m scripts.merge_mtz1221`` из корня проекта,
# поэтому пакет ``src`` доступен для импорта.
from src.config.classes import CLASS_ALIASES, canonical_class

# Расширения, считающиеся изображениями (для подсчёта статистики).
IMAGE_EXTENSIONS: frozenset[str] = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".webp"})

# Имена сплит-директорий, ожидаемые на верхнем уровне дерева датасета.
KNOWN_SPLITS: tuple[str, ...] = ("train", "val", "test")


@dataclass
class MergeStats:
    """Статистика слияния для отчёта до/после.

    Attributes:
        moved_files: Количество перенесённых файлов-изображений по (дерево, сплит).
        collisions: Количество файлов, переименованных из-за коллизии имён.
        removed_dirs: Список удалённых директорий-источников.
        before_counts: Счётчик изображений в целевом классе ДО слияния.
        after_counts: Счётчик изображений в целевом классе ПОСЛЕ слияния.
    """

    moved_files: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    collisions: int = 0
    removed_dirs: list[Path] = field(default_factory=list)
    before_counts: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    after_counts: dict[str, int] = field(default_factory=lambda: defaultdict(int))


def _count_images(directory: Path) -> int:
    """Рекурсивно подсчитать изображения в директории.

    Args:
        directory: Директория для обхода. Может не существовать.

    Returns:
        Число файлов с расширением из :data:`IMAGE_EXTENSIONS`.
    """
    if not directory.is_dir():
        return 0
    return sum(
        1
        for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def _unique_destination(dest_dir: Path, filename: str) -> Path:
    """Подобрать неконфликтующее имя файла в целевой директории.

    Если файл с таким именем уже существует, к имени добавляется суффикс
    ``_merged_N`` до тех пор, пока не будет найдено свободное имя.

    Args:
        dest_dir: Целевая директория.
        filename: Исходное имя файла.

    Returns:
        Путь назначения, гарантированно не существующий на диске.
    """
    candidate = dest_dir / filename
    if not candidate.exists():
        return candidate

    stem = Path(filename).stem
    suffix = Path(filename).suffix
    counter = 1
    while True:
        candidate = dest_dir / f"{stem}_merged_{counter}{suffix}"
        if not candidate.exists():
            return candidate
        counter += 1


def _merge_directory(
    source_dir: Path,
    dest_dir: Path,
    tree_split_key: str,
    stats: MergeStats,
    dry_run: bool,
) -> None:
    """Перенести всё содержимое ``source_dir`` в ``dest_dir`` и удалить источник.

    Переносятся файлы на верхнем уровне источника. Обработка вложенных
    структур (например, ``clean/dirty``) выполняется на уровне вызывающего кода
    через рекурсивный поиск, поэтому здесь достаточно переносить непосредственное
    содержимое, сохраняя вложенные поддиректории целиком.

    Args:
        source_dir: Директория-источник (``mtz_1221`` или её вложенность).
        dest_dir: Целевая директория (каноническая, ``mtz_82``).
        tree_split_key: Человекочитаемый ключ ``"<дерево>/<сплит>"`` для статистики.
        stats: Объект статистики, обновляется на месте.
        dry_run: Если ``True`` — только логировать, ничего не менять на диске.
    """
    if not dry_run:
        dest_dir.mkdir(parents=True, exist_ok=True)

    for item in sorted(source_dir.iterdir()):
        if item.is_file():
            destination = _unique_destination(dest_dir, item.name)
            if destination.name != item.name:
                stats.collisions += 1
            if item.suffix.lower() in IMAGE_EXTENSIONS:
                stats.moved_files[tree_split_key] += 1
            print(
                f"    move: {item.relative_to(source_dir.parent)} "
                f"-> {destination.relative_to(dest_dir.parent)}"
            )
            if not dry_run:
                shutil.move(str(item), str(destination))
        elif item.is_dir():
            # Вложенная поддиректория (например, clean/ или dirty/):
            # сливаем её содержимое с одноимённой поддиректорией назначения.
            _merge_directory(
                item,
                dest_dir / item.name,
                tree_split_key,
                stats,
                dry_run,
            )

    # После переноса содержимого удаляем опустевший источник.
    if not dry_run:
        remaining = list(source_dir.iterdir())
        if not remaining:
            source_dir.rmdir()
            stats.removed_dirs.append(source_dir)
    else:
        stats.removed_dirs.append(source_dir)


def _find_alias_dirs(split_root: Path) -> list[Path]:
    """Найти все директории-псевдонимы под корнем сплита.

    Ищет на любой глубине директории, чьё имя присутствует в
    :data:`CLASS_ALIASES` (то есть подлежит слиянию).

    Args:
        split_root: Корневая директория одного сплита (например,
            ``data/processed/train``).

    Returns:
        Список путей директорий-источников для слияния.
    """
    alias_names = set(CLASS_ALIASES.keys())
    return [
        path
        for path in split_root.rglob("*")
        if path.is_dir() and path.name in alias_names
    ]


def _process_tree(
    tree_root: Path, tree_label: str, stats: MergeStats, dry_run: bool
) -> None:
    """Обработать одно дерево датасета целиком.

    Args:
        tree_root: Корень дерева (``data/processed`` или ``data/dirty_clean``).
        tree_label: Метка дерева для логов и статистики.
        stats: Объект статистики, обновляется на месте.
        dry_run: Флаг предпросмотра без изменений.
    """
    if not tree_root.is_dir():
        print(f"[skip] Дерево не найдено: {tree_root}")
        return

    print(f"\n=== Дерево: {tree_label} ({tree_root}) ===")

    # Динамически определяем фактически существующие сплиты, ничего не хардкодя.
    split_dirs = sorted(child for child in tree_root.iterdir() if child.is_dir())
    if not split_dirs:
        print(f"[skip] Нет сплит-директорий в {tree_root}")
        return

    for split_root in split_dirs:
        split_name = split_root.name
        key = f"{tree_label}/{split_name}"

        # Целевой (канонический) класс для подсчёта до/после.
        # Все псевдонимы сводятся к одному канону, поэтому берём его один раз.
        canonical_targets = {canonical_class(name) for name in CLASS_ALIASES}

        alias_dirs = _find_alias_dirs(split_root)

        # Счётчик "до": суммарно по всем целевым каноническим папкам в этом сплите.
        before = sum(
            _count_images(dest)
            for target in canonical_targets
            for dest in split_root.rglob(target)
            if dest.is_dir()
        )
        stats.before_counts[key] = before

        if not alias_dirs:
            print(f"  [{split_name}] нет папок для слияния")
            stats.after_counts[key] = before
            continue

        print(f"  [{split_name}] найдено источников: {len(alias_dirs)}")
        for source_dir in alias_dirs:
            target_name = canonical_class(source_dir.name)
            dest_dir = source_dir.parent / target_name
            print(
                f"  сливаю {source_dir.relative_to(tree_root)} "
                f"-> {dest_dir.relative_to(tree_root)}"
            )
            _merge_directory(source_dir, dest_dir, key, stats, dry_run)

        # Счётчик "после".
        after = sum(
            _count_images(dest)
            for target in canonical_targets
            for dest in split_root.rglob(target)
            if dest.is_dir()
        )
        # В режиме dry-run файлы не перемещались, поэтому оцениваем результат.
        stats.after_counts[key] = (
            after
            if not dry_run
            else before + sum(_count_images(src) for src in alias_dirs)
        )


def _print_report(stats: MergeStats, dry_run: bool) -> None:
    """Напечатать итоговый отчёт до/после.

    Args:
        stats: Собранная статистика.
        dry_run: Флаг предпросмотра (влияет на заголовок отчёта).
    """
    mode = "DRY-RUN (изменения не применены)" if dry_run else "ПРИМЕНЕНО"
    print("\n" + "=" * 60)
    print(f"ИТОГОВЫЙ ОТЧЁТ  [{mode}]")
    print("=" * 60)

    total_moved = sum(stats.moved_files.values())
    print(f"Перенесено изображений: {total_moved}")
    print(f"Коллизий имён (переименовано): {stats.collisions}")
    print(f"Удалено директорий-источников: {len(stats.removed_dirs)}")

    print("\nПо сплитам (изображений в целевом классе mtz_82):")
    all_keys = sorted(set(stats.before_counts) | set(stats.after_counts))
    for key in all_keys:
        before = stats.before_counts.get(key, 0)
        after = stats.after_counts.get(key, 0)
        delta = after - before
        print(f"  {key:<28} до={before:<6} после={after:<6} (+{delta})")

    if stats.removed_dirs:
        print("\nУдалённые источники:")
        for path in stats.removed_dirs:
            print(f"  - {path}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Разобрать аргументы командной строки.

    Args:
        argv: Список аргументов (по умолчанию ``sys.argv[1:]``).

    Returns:
        Пространство имён с разобранными аргументами.
    """
    parser = argparse.ArgumentParser(
        description="Слияние классов mtz_1221 -> mtz_82 по всем сплитам датасета.",
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=Path("data/processed"),
        help="Корень single-task дерева (train/val/test). По умолчанию data/processed.",
    )
    parser.add_argument(
        "--dirty-clean-dir",
        type=Path,
        default=Path("data/dirty_clean"),
        help="Корень multi-task дерева (train/val). По умолчанию data/dirty_clean.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Показать план слияния без изменений на диске.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Точка входа скрипта слияния.

    Args:
        argv: Аргументы командной строки (для тестируемости).

    Returns:
        Код возврата процесса (0 — успех).
    """
    args = parse_args(argv)
    stats = MergeStats()

    print(f"Псевдонимы для слияния: {CLASS_ALIASES}")
    if args.dry_run:
        print(">>> Режим DRY-RUN: диск не изменяется.\n")

    _process_tree(args.processed_dir, "processed", stats, args.dry_run)
    _process_tree(args.dirty_clean_dir, "dirty_clean", stats, args.dry_run)

    _print_report(stats, args.dry_run)
    return 0


if __name__ == "__main__":
    sys.exit(main())

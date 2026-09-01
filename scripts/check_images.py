#!/usr/bin/env python3
"""Массовая проверка целостности изображений в дереве данных.

Рекурсивно обходит указанный корень и для каждого ``*.jpg`` / ``*.jpeg`` /
``*.png`` выполняет двухступенчатую проверку через Pillow:

* ``Image.verify()`` — быстрая валидация структуры контейнера без полного
  декодирования пикселей (ловит битые заголовки и явно обрезанные файлы);
* повторное открытие и ``Image.load()`` — полное декодирование (ловит
  повреждённые строки развёртки, на которые ``verify()`` не реагирует).

``ImageFile.LOAD_TRUNCATED_IMAGES`` принудительно выключается, иначе Pillow
молча дорисовывает обрезанные JPEG серым и битый файл проходит проверку.

Обход распараллелен по файлам через ``ThreadPoolExecutor`` (декодирование в
Pillow отпускает GIL). Результат — JSON в stdout: корень, число проверенных
файлов и список битых (путь + класс ошибки). Код возврата ``1``, если найден
хотя бы один битый файл, иначе ``0`` — удобно для CI и pre-commit.

Пример::

    python -m scripts.check_images --root data/ --workers 8
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image, ImageFile

# Обрезанные изображения должны падать, а не дорисовываться серым.
ImageFile.LOAD_TRUNCATED_IMAGES = False

IMAGE_EXTENSIONS: frozenset[str] = frozenset({".jpg", ".jpeg", ".png"})


def check_image(path: Path) -> str | None:
    """Проверить один файл изображения на целостность.

    Args:
        path: Путь к файлу изображения.

    Returns:
        ``None``, если файл читается и декодируется без ошибок, иначе строку
        вида ``"<ТипИсключения>: <сообщение>"``.
    """
    try:
        with Image.open(path) as image:
            image.verify()
    except Exception as exc:  # noqa: BLE001 — любая ошибка Pillow означает битый файл
        return f"{type(exc).__name__}: {exc}"

    # verify() «расходует» файловый объект — для полного декодирования открываем заново.
    try:
        with Image.open(path) as image:
            image.load()
    except Exception as exc:  # noqa: BLE001
        return f"{type(exc).__name__}: {exc}"

    return None


def iter_images(root: Path) -> list[Path]:
    """Собрать отсортированный список изображений под корнем (рекурсивно).

    Args:
        root: Корневая директория обхода.

    Returns:
        Отсортированный список путей с поддерживаемыми расширениями.
    """
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def check_tree(root: Path, workers: int) -> list[dict[str, str]]:
    """Проверить все изображения под корнем параллельно.

    Args:
        root: Корневая директория обхода.
        workers: Число потоков-воркеров.

    Returns:
        Список ``{"path": ..., "error": ...}`` для битых файлов (в порядке обхода).
    """
    paths = iter_images(root)
    if not paths:
        return []

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        errors = pool.map(check_image, paths)

    return [
        {"path": path.as_posix(), "error": error}
        for path, error in zip(paths, errors)
        if error is not None
    ]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Разобрать аргументы командной строки.

    Args:
        argv: Аргументы (по умолчанию ``sys.argv[1:]``).

    Returns:
        Пространство имён аргументов.
    """
    parser = argparse.ArgumentParser(
        description="Массовая проверка целостности JPG/PNG в дереве данных.",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("data"),
        help="Корень дерева для проверки (по умолчанию data/).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Число параллельных воркеров (по умолчанию 4).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Точка входа проверки.

    Args:
        argv: Аргументы командной строки (для тестируемости).

    Returns:
        ``1``, если найден хотя бы один битый файл или корень не существует,
        иначе ``0``.
    """
    args = parse_args(argv)

    if not args.root.is_dir():
        print(
            json.dumps({"error": f"корень не найден: {args.root.as_posix()}"}, ensure_ascii=False)
        )
        return 1

    checked = len(iter_images(args.root))
    broken = check_tree(args.root, args.workers)

    report = {
        "root": args.root.as_posix(),
        "checked": checked,
        "broken_count": len(broken),
        "broken": broken,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 1 if broken else 0


if __name__ == "__main__":
    sys.exit(main())

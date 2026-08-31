#!/usr/bin/env python3
"""Приём пользовательского фидбэка в обучающую выборку головы состояния.

Эндпоинт ``/feedback`` складывает присланные пользователем фото с исправленной
семьёй в ``config.api.feedback_dir/<family>/`` рядом с JSON-манифестом
``<stem>.json`` (поля ``user_family``, ``ts``, опционально ``user_state``). Этот
скрипт разбирает накопленный каталог фидбэка и вливает валидные фото в
multi-task дерево ``data/dirty_clean/train/<family>/<state>/`` с префиксом
``feedback_``.

Конвейер по каждому фото:

1. Валидация: семья каталога входит в :data:`MODEL_CLASSES`; файл — изображение
   (``.jpg``/``.jpeg``/``.png``); рядом лежит манифест с полями ``user_family``
   и ``ts``.
2. Дедупликация по content-hash (:func:`src.data.utils.compute_content_hash`)
   против уже разложенных данных в ``data/processed`` и ``data/dirty_clean`` —
   повторно пришедшее фото пропускается.
3. Состояние: берётся из ``manifest.user_state`` (если задано и входит в
   :data:`STATE_CLASSES`), иначе предсказывается multi-task моделью
   (:func:`src.models.predict.predict_image`) по рабочему чекпоинту.
4. Фото и манифест КОПИРУЮТСЯ (не перемещаются) в
   ``data/dirty_clean/train/<family>/<state>/`` с префиксом ``feedback_``.

По умолчанию скрипт работает в режиме ``--dry-run`` и печатает только сводку
(найдено/валидных/дубли/ошибок). Реальное копирование выполняется с ``--apply``.
Код возврата: ``0`` — ошибок валидации не было, ``1`` — были.

Пример::

    python -m scripts.ingest_feedback --verbose            # сухой прогон
    python -m scripts.ingest_feedback --apply              # реальное вливание
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from src.config.classes import MODEL_CLASSES, STATE_CLASSES
from src.config.config_loader import load_config
from src.data.utils import compute_content_hash

IMAGE_EXTENSIONS: frozenset[str] = frozenset({".jpg", ".jpeg", ".png"})
MANIFEST_SUFFIX: str = ".json"
FEEDBACK_PREFIX: str = "feedback_"
# Обязательные поля манифеста, без которых фидбэк не принимается.
REQUIRED_MANIFEST_FIELDS: tuple[str, ...] = ("user_family", "ts")

StatePredictor = Callable[[Path], str]


@dataclass(frozen=True)
class FeedbackItem:
    """Валидная единица фидбэка: фото + манифест + вычисленный хеш.

    Attributes:
        photo: Путь к присланному изображению.
        manifest_path: Путь к JSON-манифесту рядом с фото.
        family: Семья трактора (имя каталога, проверено по ``MODEL_CLASSES``).
        manifest: Разобранное содержимое манифеста.
        content_hash: MD5 содержимого фото (для дедупликации).
    """

    photo: Path
    manifest_path: Path
    family: str
    manifest: dict
    content_hash: str


@dataclass
class IngestStats:
    """Счётчики прогона.

    Attributes:
        found: Всего фото-кандидатов в каталоге фидбэка.
        valid: Прошли валидацию.
        duplicates: Из валидных — уже присутствуют в датасете по хешу.
        errors: Не прошли валидацию.
        copied: Фактически скопировано (только при ``--apply``).
        by_state_source: Разбивка новых фото по источнику состояния.
    """

    found: int = 0
    valid: int = 0
    duplicates: int = 0
    errors: int = 0
    copied: int = 0
    by_state_source: dict[str, int] = field(default_factory=lambda: {"манифест": 0, "прогноз": 0})


def _iter_candidates(feedback_dir: Path) -> list[tuple[str, Path]]:
    """Собрать пары ``(семья, путь_к_фото)`` из каталога фидбэка.

    Обходятся только подкаталоги первого уровня (``<feedback_dir>/<family>/``);
    JSON-манифесты и файлы в корне каталога кандидатами не считаются.

    Args:
        feedback_dir: Корень каталога фидбэка.

    Returns:
        Отсортированный список пар ``(имя_каталога, путь_к_файлу)``.
    """
    if not feedback_dir.is_dir():
        return []
    candidates: list[tuple[str, Path]] = []
    for family_dir in sorted(p for p in feedback_dir.iterdir() if p.is_dir()):
        for path in sorted(family_dir.iterdir()):
            if path.is_file() and path.suffix.lower() != MANIFEST_SUFFIX:
                candidates.append((family_dir.name, path))
    return candidates


def _validate(family: str, photo: Path) -> tuple[FeedbackItem | None, str | None]:
    """Проверить один кандидат фидбэка.

    Args:
        family: Имя каталога, в котором лежит фото.
        photo: Путь к файлу-кандидату.

    Returns:
        Пара ``(item, None)`` при успехе либо ``(None, сообщение_об_ошибке)``.
    """
    if family not in MODEL_CLASSES:
        return None, f"{photo}: неизвестная семья {family!r} (ожидались {sorted(MODEL_CLASSES)})"

    if photo.suffix.lower() not in IMAGE_EXTENSIONS:
        return None, f"{photo}: недопустимое расширение {photo.suffix!r}"

    manifest_path = photo.with_suffix(MANIFEST_SUFFIX)
    if not manifest_path.is_file():
        return None, f"{photo}: нет манифеста {manifest_path.name}"

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        return None, f"{manifest_path}: манифест не читается ({exc})"
    if not isinstance(manifest, dict):
        return None, f"{manifest_path}: манифест не является объектом"

    missing = [key for key in REQUIRED_MANIFEST_FIELDS if not manifest.get(key)]
    if missing:
        return None, f"{manifest_path}: в манифесте нет полей {missing}"

    return (
        FeedbackItem(
            photo=photo,
            manifest_path=manifest_path,
            family=family,
            manifest=manifest,
            content_hash=compute_content_hash(photo),
        ),
        None,
    )


def _collect_known_hashes(*roots: Path) -> set[str]:
    """Собрать content-hash всех изображений в указанных деревьях данных.

    Args:
        *roots: Корни деревьев (``data/processed``, ``data/dirty_clean`` и т. п.).

    Returns:
        Множество шестнадцатеричных MD5-дайджестов.
    """
    hashes: set[str] = set()
    for root in roots:
        if not root.is_dir():
            continue
        for path in root.rglob("*"):
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                hashes.add(compute_content_hash(path))
    return hashes


def _manifest_state(manifest: dict) -> str | None:
    """Вернуть валидное состояние из манифеста либо ``None``.

    Args:
        manifest: Разобранный манифест фидбэка.

    Returns:
        Значение ``user_state``, если оно входит в ``STATE_CLASSES``, иначе ``None``.
    """
    user_state = manifest.get("user_state")
    if isinstance(user_state, str) and user_state in STATE_CLASSES:
        return user_state
    return None


def _load_state_predictor(image_size: int, checkpoint: Path | None) -> StatePredictor:
    """Собрать предсказатель состояния на основе multi-task модели.

    Тяжёлые зависимости (torch, веса) импортируются и грузятся только здесь —
    сухой прогон и фидбэк с заполненным ``user_state`` модель не трогают.

    Args:
        image_size: Размер валидационной трансформации.
        checkpoint: Явный путь к чекпоинту либо ``None`` (автовыбор рабочего).

    Returns:
        Функция ``photo -> state``, возвращающая имя класса состояния.
    """
    import torch

    from src.data.transforms import get_val_transforms
    from src.models.loader import load_multi_task_model, resolve_working_checkpoint
    from src.models.predict import predict_image

    resolved = Path(checkpoint) if checkpoint is not None else resolve_working_checkpoint()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_multi_task_model(resolved, device)
    transform = get_val_transforms(image_size)

    def _predict(photo: Path) -> str:
        _, _, state_idx, _ = predict_image(model, photo, transform)
        return STATE_CLASSES[state_idx]

    return _predict


def _copy_item(item: FeedbackItem, state: str, dirty_clean_dir: Path) -> bool:
    """Скопировать фото и манифест фидбэка в multi-task дерево.

    Args:
        item: Валидная единица фидбэка.
        state: Класс состояния (подкаталог назначения).
        dirty_clean_dir: Корень дерева ``data/dirty_clean``.

    Returns:
        ``True``, если фото было скопировано; ``False``, если оно уже на месте.
    """
    dest_dir = dirty_clean_dir / "train" / item.family / state
    dest_dir.mkdir(parents=True, exist_ok=True)

    photo_dest = dest_dir / f"{FEEDBACK_PREFIX}{item.photo.name}"
    manifest_dest = dest_dir / f"{FEEDBACK_PREFIX}{item.manifest_path.name}"

    copied = False
    if not photo_dest.exists():
        shutil.copy2(item.photo, photo_dest)
        copied = True
    if not manifest_dest.exists():
        shutil.copy2(item.manifest_path, manifest_dest)
    return copied


def ingest_feedback(
    feedback_dir: Path,
    processed_dir: Path,
    dirty_clean_dir: Path,
    image_size: int,
    checkpoint: Path | None = None,
    apply: bool = False,
    verbose: bool = False,
) -> IngestStats:
    """Разобрать каталог фидбэка и (опционально) влить его в датасет.

    Args:
        feedback_dir: Каталог, куда ``/feedback`` складывает фото и манифесты.
        processed_dir: Дерево чистых данных для дедупликации.
        dirty_clean_dir: Multi-task дерево — источник дедупликации и назначение.
        image_size: Размер валидационной трансформации для предсказания состояния.
        checkpoint: Явный чекпоинт модели либо ``None`` (автовыбор).
        apply: ``True`` — реально копировать; ``False`` — только сводка.
        verbose: Печатать строку по каждому файлу.

    Returns:
        Счётчики прогона.
    """
    stats = IngestStats()
    candidates = _iter_candidates(feedback_dir)
    stats.found = len(candidates)

    valid_items: list[FeedbackItem] = []
    for family, photo in candidates:
        item, error = _validate(family, photo)
        if error is not None:
            stats.errors += 1
            if verbose:
                print(f"[err] {error}")
            continue
        stats.valid += 1
        valid_items.append(item)

    known_hashes = _collect_known_hashes(processed_dir, dirty_clean_dir)

    fresh: list[tuple[FeedbackItem, str]] = []
    seen_this_run: set[str] = set()
    for item in valid_items:
        if item.content_hash in known_hashes or item.content_hash in seen_this_run:
            stats.duplicates += 1
            if verbose:
                print(f"[dup] {item.photo}: хеш уже в датасете")
            continue
        seen_this_run.add(item.content_hash)

        state = _manifest_state(item.manifest)
        source = "манифест" if state is not None else "прогноз"
        fresh.append((item, state))  # state может быть None до загрузки модели
        stats.by_state_source[source] += 1
        if verbose and not apply:
            target = state or "?"
            print(f"[ok]  {item.photo} -> {item.family}/{target} (состояние: {source})")

    if not apply:
        return stats

    need_predict = any(state is None for _, state in fresh)
    predictor = _load_state_predictor(image_size, checkpoint) if need_predict else None

    for item, state in fresh:
        resolved_state = state if state is not None else predictor(item.photo)  # type: ignore[misc]
        copied = _copy_item(item, resolved_state, dirty_clean_dir)
        if copied:
            stats.copied += 1
        if verbose:
            action = "скопировано" if copied else "уже на месте"
            print(f"[ok]  {item.photo} -> {item.family}/{resolved_state} ({action})")

    return stats


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Разобрать аргументы командной строки.

    Args:
        argv: Аргументы (по умолчанию ``sys.argv[1:]``).

    Returns:
        Пространство имён аргументов.
    """
    config = load_config()
    parser = argparse.ArgumentParser(
        description="Вливание пользовательского фидбэка в обучающую выборку головы состояния.",
    )
    parser.add_argument(
        "--feedback-dir",
        type=Path,
        default=config.api.feedback_dir,
        help="Каталог фидбэка (по умолчанию из config.api.feedback_dir).",
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=config.data.processed_dir,
        help="Дерево чистых данных для дедупликации (по умолчанию из config.yaml).",
    )
    parser.add_argument(
        "--dirty-clean-dir",
        type=Path,
        default=config.data.dirty_clean_dir,
        help="Multi-task дерево: дедупликация и назначение (по умолчанию из config.yaml).",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Чекпоинт multi-task модели для предсказания состояния (по умолчанию — рабочий).",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=config.image_size,
        help="Размер изображения (по умолчанию из config.yaml).",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--dry-run",
        action="store_true",
        default=True,
        help="Только сводка, без копирования (режим по умолчанию).",
    )
    mode.add_argument(
        "--apply",
        action="store_true",
        help="Реально копировать валидные фото и манифесты в датасет.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Детальный вывод по каждому файлу.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Точка входа вливания фидбэка.

    Args:
        argv: Аргументы командной строки (для тестируемости).

    Returns:
        ``0`` — ошибок валидации не было, ``1`` — были.
    """
    args = parse_args(argv)

    stats = ingest_feedback(
        feedback_dir=args.feedback_dir,
        processed_dir=args.processed_dir,
        dirty_clean_dir=args.dirty_clean_dir,
        image_size=args.image_size,
        checkpoint=args.checkpoint,
        apply=args.apply,
        verbose=args.verbose,
    )

    fresh = stats.valid - stats.duplicates
    print("\n" + "=" * 50)
    print("ИТОГ ВЛИВАНИЯ ФИДБЭКА")
    print("=" * 50)
    print(f"  каталог фидбэка : {args.feedback_dir}")
    print(f"  найдено         : {stats.found}")
    print(f"  валидных        : {stats.valid}")
    print(f"  дубли           : {stats.duplicates}")
    print(f"  ошибок          : {stats.errors}")
    print(
        f"  к вливанию      : {fresh} "
        f"(состояние: манифест={stats.by_state_source['манифест']}, "
        f"прогноз={stats.by_state_source['прогноз']})"
    )
    if args.apply:
        print(
            f"  скопировано     : {stats.copied} -> {args.dirty_clean_dir}/train/<family>/<state>/"
        )
    else:
        print("  режим сухого прогона — запустите с --apply для копирования")

    return 1 if stats.errors else 0


if __name__ == "__main__":
    sys.exit(main())

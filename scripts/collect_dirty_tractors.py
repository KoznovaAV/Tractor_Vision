#!/usr/bin/env python3
"""Сбор реальных «грязных» фотографий тракторов для дообучения головы состояния.

Модель хорошо распознаёт класс техники (99%), но состояние clean/dirty слабое на
реальной экстремальной грязи, потому что обучалось на синтетике (капли/пыль
поверх чистого фото), а не на реальной корке ила. Этот скрипт собирает реальные
грязные фото по «грязным» запросам, чтобы дополнить обучающую выборку.

Переиспользует проверенные компоненты коллектора чистых фото
(:mod:`scripts.collect_tractors`): интерфейс :class:`SourceAdapter` с дефолтным
DuckDuckGo, дедупликацию perceptual hashing, CLIP zero-shot фильтр «трактор ли
это» и скачиватель с rate limiting. Отличие — набор запросов и целевая
директория; классовая раскладка здесь НЕ выполняется (все фото падают в
``unsorted``) и делается отдельно в ``pseudo_label_dirty.py``.

Сохранение в ``data/real_dirty_raw/unsorted/`` с манифестом (исходный URL,
уверенность CLIP, запрос).

Пример::

    python -m scripts.collect_dirty_tractors --output-dir data/real_dirty_raw --limit 300
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from scripts.collect_tractors import (
    ClipLabeler,
    DuckDuckGoAdapter,
    PerceptualDeduplicator,
    RateLimitedDownloader,
    SourceAdapter,
    _decode_image,
    _save_image,
)

# «Грязные» запросы: русские и английские. Подобраны так, чтобы возвращать фото
# техники под реальной грязью/илом, а не студийные чистые снимки.
DIRTY_QUERIES: tuple[str, ...] = (
    "грязный трактор",
    "трактор в грязи",
    "грязный МТЗ",
    "трактор застрял в грязи",
    "трактор Беларус грязный",
    "muddy tractor",
    "tractor stuck in mud",
    "dirty farm tractor mud",
    "tractor covered in mud",
)

# Целевой поддиректорий: все собранные фото падают сюда без классовой раскладки.
UNSORTED_SUBDIR: str = "unsorted"


def collect_dirty(
    output_root: Path,
    limit: int,
    source: SourceAdapter,
    downloader: RateLimitedDownloader,
    deduplicator: PerceptualDeduplicator,
    labeler: ClipLabeler | None,
    clip_threshold: float,
    manifest: list[dict[str, Any]],
) -> dict[str, int]:
    """Собрать грязные фото по всем запросам в единую папку ``unsorted``.

    Args:
        output_root: Корень дерева (``data/real_dirty_raw``).
        limit: Целевое число принятых уникальных изображений.
        source: Источник изображений.
        downloader: Скачиватель с rate limiting.
        deduplicator: Дедупликатор phash (общий на весь сбор).
        labeler: CLIP-фильтр «трактор ли это» или ``None``.
        clip_threshold: Порог уверенности CLIP для принятия.
        manifest: Общий список записей манифеста (обновляется на месте).

    Returns:
        Счётчик этапов ``{found, downloaded, deduped, rejected_no_tractor, accepted}``.
    """
    stats = {
        "found": 0,
        "downloaded": 0,
        "deduped": 0,
        "rejected_no_tractor": 0,
        "accepted": 0,
    }
    unsorted_dir = output_root / UNSORTED_SUBDIR
    per_query = max(20, (limit * 3) // len(DIRTY_QUERIES))

    for query in DIRTY_QUERIES:
        if stats["accepted"] >= limit:
            break
        print(f"  запрос: '{query}' (до {per_query})")
        candidates = source.search(query, per_query)
        stats["found"] += len(candidates)

        for candidate in candidates:
            if stats["accepted"] >= limit:
                break

            data = downloader.download(candidate.url)
            if data is None:
                continue
            image = _decode_image(data)
            if image is None:
                continue
            stats["downloaded"] += 1

            image_hash = deduplicator.compute_hash(image)
            if deduplicator.is_duplicate(image_hash):
                stats["deduped"] += 1
                continue

            clip_conf = 1.0
            clip_prompt = "<clip disabled>"
            if labeler is not None:
                verdict = labeler.classify(image)
                clip_conf = verdict.confidence
                clip_prompt = verdict.best_prompt
                # Фильтр только «трактор ли это» — грязь тут не оценивается.
                if not verdict.is_tractor or verdict.confidence < clip_threshold:
                    stats["rejected_no_tractor"] += 1
                    continue

            deduplicator.add(image_hash)
            saved_path = _save_image(image, data, unsorted_dir, candidate.url)
            stats["accepted"] += 1

            manifest.append(
                {
                    "path": str(saved_path),
                    "url": candidate.url,
                    "query": candidate.source_query,
                    "title": candidate.title,
                    "clip_confidence": round(clip_conf, 4),
                    "clip_best_prompt": clip_prompt,
                    "source": source.name,
                }
            )

    return stats


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Разобрать аргументы командной строки.

    Args:
        argv: Аргументы (по умолчанию ``sys.argv[1:]``).

    Returns:
        Пространство имён аргументов.
    """
    parser = argparse.ArgumentParser(
        description="Сбор реальных грязных фото тракторов по грязным запросам.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/real_dirty_raw"),
        help="Корень дерева; фото сохраняются в <output-dir>/unsorted/.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=300,
        help="Целевое число принятых уникальных изображений.",
    )
    parser.add_argument(
        "--clip-threshold",
        type=float,
        default=0.7,
        help="Порог уверенности CLIP-фильтра 'трактор ли это'.",
    )
    parser.add_argument(
        "--hamming-threshold",
        type=int,
        default=6,
        help="Порог расстояния Хэмминга для дедупликации phash.",
    )
    parser.add_argument(
        "--rate-limit",
        type=float,
        default=1.0,
        help="Минимальный интервал между сетевыми запросами в секундах.",
    )
    parser.add_argument(
        "--no-clip",
        action="store_true",
        help="Отключить CLIP-фильтр (только поиск + дедуп + скачивание).",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Путь к JSON-манифесту. По умолчанию <output-dir>/dirty_manifest.json.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Точка входа сбора грязных фото.

    Args:
        argv: Аргументы командной строки (для тестируемости).

    Returns:
        Код возврата процесса (0 — успех).
    """
    args = parse_args(argv)

    source = DuckDuckGoAdapter()
    downloader = RateLimitedDownloader(min_interval=args.rate_limit)
    deduplicator = PerceptualDeduplicator(hamming_threshold=args.hamming_threshold)
    labeler = None if args.no_clip else ClipLabeler()

    manifest_path = args.manifest or (args.output_dir / "dirty_manifest.json")
    manifest: list[dict[str, Any]] = []
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            print(f"[warn] манифест повреждён, начинаю заново: {manifest_path}")
            manifest = []

    print(f"Источник: {source.name} | CLIP: {'off' if args.no_clip else 'on'}")
    print(f"Цель: {args.limit} уникальных грязных фото")

    stats = collect_dirty(
        output_root=args.output_dir,
        limit=args.limit,
        source=source,
        downloader=downloader,
        deduplicator=deduplicator,
        labeler=labeler,
        clip_threshold=args.clip_threshold,
        manifest=manifest,
    )

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n" + "=" * 60)
    print("ИТОГ СБОРА ГРЯЗНЫХ ФОТО")
    print("=" * 60)
    print(
        f"найдено={stats['found']} скачано={stats['downloaded']} "
        f"дубли={stats['deduped']} не_трактор={stats['rejected_no_tractor']} "
        f"принято={stats['accepted']}"
    )
    print(f"Сохранено в: {args.output_dir / UNSORTED_SUBDIR}")
    print(f"Манифест: {manifest_path} (записей: {len(manifest)})")
    print(
        "\nДалее: ручная раскладка unsorted/ по классам ИЛИ "
        "pseudo_label_dirty.py для авторазметки."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Error analysis головы состояния (clean/dirty) на размеченных наборах.

Прогоняет рабочую multi-task модель через :func:`src.models.predict.predict_image`
по трём источникам с известной меткой состояния и показывает, где голова
состояния промахивается:

* ``data/dirty_clean/test/<class>/<state>/`` — синтетика, известны обе метки;
* ``data/real_dirty_val/<class>/`` — реальная грязь, состояние всегда ``dirty``;
* ``data/real_clean_probe/<class>/`` — реальная чистая техника, состояние всегда
  ``clean``; если набор пуст, источник пропускается.

Считаются: false-dirty rate на чистых фото (доля чистых, распознанных как dirty)
в разрезе источников и классов; false-clean rate на грязных фото; dirty recall на
``data/real_dirty_val``. Промахи копируются в ``<out-dir>/clean_as_dirty/`` и
``<out-dir>/dirty_as_clean/`` под именем ``pred_<state>_conf<conf>_<исходное имя>``
(``conf`` — уверенность головы класса, единственная, что отдаёт ``predict_image``).
Полный отчёт пишется в ``<out-dir>/report.json``, сводная таблица — в stdout.

Наборы ``data/`` и ``weights/`` только читаются, результат пишется в ``output/``.

Пример::

    python scripts/error_analysis_state.py
    python scripts/error_analysis_state.py \\
        --out-dir output/error_analysis \\
        --checkpoint weights/multi_task_best.ckpt
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse  # noqa: E402
import json  # noqa: E402
import shutil  # noqa: E402
from collections import Counter, defaultdict  # noqa: E402
from collections.abc import Iterator  # noqa: E402
from typing import Any, NamedTuple  # noqa: E402

import torch  # noqa: E402

from src.config.classes import MODEL_CLASSES, STATE_CLASSES, state_to_idx  # noqa: E402
from src.config.config_loader import load_config  # noqa: E402
from src.data.transforms import get_val_transforms  # noqa: E402
from src.models.loader import load_multi_task_model, resolve_working_checkpoint  # noqa: E402
from src.models.multi_task import MultiTaskTractorClassifier  # noqa: E402
from src.models.predict import predict_image  # noqa: E402

IMAGE_EXTENSIONS: frozenset[str] = frozenset({".jpg", ".jpeg", ".png", ".webp"})

SYNTHETIC_SOURCE: str = "synthetic"
REAL_DIRTY_SOURCE: str = "real_dirty_val"
REAL_CLEAN_SOURCE: str = "real_clean_probe"


class Sample(NamedTuple):
    """Одно размеченное изображение.

    Attributes:
        source: Имя источника (``synthetic``/``real_dirty_val``/``real_clean_probe``).
        path: Путь к файлу изображения.
        family: Класс техники (имя папки).
        true_state: Истинное состояние (``clean``/``dirty``).
    """

    source: str
    path: Path
    family: str
    true_state: str


def _iter_images(directory: Path) -> Iterator[Path]:
    """Отдать пути к изображениям внутри директории (рекурсивно, отсортированно)."""
    if not directory.is_dir():
        return
    for path in sorted(directory.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            yield path


def collect_synthetic(root: Path) -> list[Sample]:
    """Собрать размеченные пары из синтетического дерева ``<root>/<class>/<state>/``.

    Args:
        root: Корень набора (например, ``data/dirty_clean/test``).

    Returns:
        Список :class:`Sample` с известными классом и состоянием.
    """
    samples: list[Sample] = []
    for family in MODEL_CLASSES:
        for state in STATE_CLASSES:
            for path in _iter_images(root / family / state):
                samples.append(Sample(SYNTHETIC_SOURCE, path, family, state))
    return samples


def collect_flat(root: Path, source: str, state: str) -> list[Sample]:
    """Собрать пары из плоского дерева ``<root>/<class>/`` с фиксированным состоянием.

    Args:
        root: Корень набора (например, ``data/real_dirty_val``).
        source: Имя источника для отчёта.
        state: Истинное состояние всех фото набора (``clean``/``dirty``).

    Returns:
        Список :class:`Sample`.
    """
    samples: list[Sample] = []
    for family in MODEL_CLASSES:
        for path in _iter_images(root / family):
            samples.append(Sample(source, path, family, state))
    return samples


def _miss_subdir(true_state: str, pred_state: str) -> str:
    """Имя папки для промаха: ``<истина>_as_<предсказание>``."""
    return f"{true_state}_as_{pred_state}"


def _copy_miss(sample: Sample, pred_state: str, conf: float, out_dir: Path) -> Path:
    """Скопировать промах в ``<out_dir>/<true>_as_<pred>/`` с информативным именем.

    Args:
        sample: Промахнувшийся образец.
        pred_state: Предсказанное (неверное) состояние.
        conf: Уверенность головы класса из ``predict_image``.
        out_dir: Корневая директория результатов.

    Returns:
        Путь к созданной копии.
    """
    dest_dir = out_dir / _miss_subdir(sample.true_state, pred_state)
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / f"pred_{pred_state}_conf{conf:.2f}_{sample.path.name}"
    shutil.copy2(sample.path, dest)
    return dest


def _summarize(pred_counts: Counter[str], true_state: str) -> dict[str, Any]:
    """Свести счётчик предсказаний по группе к метрикам ошибок состояния.

    Args:
        pred_counts: Счётчик предсказанных состояний внутри группы.
        true_state: Истинное состояние группы.

    Returns:
        Словарь с ``total`` и долей ошибок (``false_dirty*`` для чистых,
        ``false_clean*`` и ``dirty_recall`` для грязных).
    """
    total = sum(pred_counts.values())
    wrong_state = STATE_CLASSES[1 - state_to_idx(true_state)]
    wrong = pred_counts.get(wrong_state, 0)
    metric = "false_dirty" if true_state == "clean" else "false_clean"
    summary: dict[str, Any] = {
        "total": total,
        metric: wrong,
        f"{metric}_rate": (wrong / total if total else 0.0),
    }
    if true_state == "dirty":
        summary["dirty_recall"] = (pred_counts.get("dirty", 0) / total) if total else 0.0
    return summary


def analyze(
    model: MultiTaskTractorClassifier,
    samples: list[Sample],
    transform: Any,
    out_dir: Path,
) -> dict[str, Any]:
    """Прогнать модель по образцам, посчитать метрики и скопировать промахи.

    Args:
        model: Multi-task модель в режиме eval.
        samples: Размеченные образцы из всех источников.
        transform: Валидационная трансформация.
        out_dir: Директория результатов (создаётся при необходимости).

    Returns:
        Словарь отчёта (пишется в ``report.json``): метрики по источникам и
        классам, агрегаты и список промахов.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    by_group: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    by_class: dict[tuple[str, str, str], Counter[str]] = defaultdict(Counter)
    misses: dict[str, list[dict[str, Any]]] = {
        _miss_subdir("clean", "dirty"): [],
        _miss_subdir("dirty", "clean"): [],
    }

    for sample in samples:
        _, conf, state_idx = predict_image(model, sample.path, transform)
        pred_state = STATE_CLASSES[state_idx]

        by_group[(sample.source, sample.true_state)][pred_state] += 1
        by_class[(sample.source, sample.true_state, sample.family)][pred_state] += 1

        if pred_state != sample.true_state:
            dest = _copy_miss(sample, pred_state, conf, out_dir)
            misses[_miss_subdir(sample.true_state, pred_state)].append(
                {
                    "source": sample.source,
                    "family": sample.family,
                    "path": str(sample.path),
                    "pred_state": pred_state,
                    "family_conf": round(conf, 4),
                    "copied_as": dest.name,
                }
            )

    sources: dict[str, Any] = {}
    for (source, true_state), counts in sorted(by_group.items()):
        node = sources.setdefault(source, {})
        node[true_state] = _summarize(counts, true_state)
        node[true_state]["per_class"] = {
            family: _summarize(by_class[(source, true_state, family)], true_state)
            for family in MODEL_CLASSES
            if (source, true_state, family) in by_class
        }

    clean_counts: Counter[str] = Counter()
    dirty_counts: Counter[str] = Counter()
    for (_, true_state), counts in by_group.items():
        (clean_counts if true_state == "clean" else dirty_counts).update(counts)

    real_dirty = by_group.get((REAL_DIRTY_SOURCE, "dirty"), Counter())
    totals = {
        "clean_false_dirty_rate": _summarize(clean_counts, "clean")["false_dirty_rate"],
        "dirty_false_clean_rate": _summarize(dirty_counts, "dirty")["false_clean_rate"],
        "real_dirty_recall": _summarize(real_dirty, "dirty").get("dirty_recall", 0.0),
    }

    return {
        "samples": len(samples),
        "sources": sources,
        "totals": totals,
        "misses": misses,
    }


def _print_table(report: dict[str, Any]) -> None:
    """Напечатать сводную таблицу отчёта в stdout."""
    print("=" * 72)
    print("ERROR ANALYSIS ГОЛОВЫ СОСТОЯНИЯ")
    print("=" * 72)
    print(f"Всего размеченных фото: {report['samples']}")

    for source, node in report["sources"].items():
        print(f"\n[{source}]")
        for true_state, summary in node.items():
            if true_state == "clean":
                head = (
                    f"  clean: {summary['false_dirty']}/{summary['total']} "
                    f"как dirty  (false-dirty rate {summary['false_dirty_rate']:.4f})"
                )
            else:
                head = (
                    f"  dirty: {summary['false_clean']}/{summary['total']} "
                    f"как clean  (false-clean rate {summary['false_clean_rate']:.4f}, "
                    f"dirty recall {summary['dirty_recall']:.4f})"
                )
            print(head)
            for family, per in summary["per_class"].items():
                if true_state == "clean":
                    tail = (
                        f"false-dirty {per['false_dirty_rate']:.4f} "
                        f"({per['false_dirty']}/{per['total']})"
                    )
                else:
                    tail = (
                        f"recall {per['dirty_recall']:.4f} "
                        f"({per['total'] - per['false_clean']}/{per['total']})"
                    )
                print(f"      {family:<14} {tail}")

    totals = report["totals"]
    print("\nИТОГО:")
    print(f"  false-dirty rate на чистых:  {totals['clean_false_dirty_rate']:.4f}")
    print(f"  false-clean rate на грязных: {totals['dirty_false_clean_rate']:.4f}")
    print(f"  dirty recall (real_dirty_val): {totals['real_dirty_recall']:.4f}")

    clean_as_dirty = len(report["misses"][_miss_subdir("clean", "dirty")])
    dirty_as_clean = len(report["misses"][_miss_subdir("dirty", "clean")])
    print(
        f"\nПромахи скопированы: clean_as_dirty={clean_as_dirty}, dirty_as_clean={dirty_as_clean}"
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Разобрать аргументы командной строки.

    Args:
        argv: Аргументы (по умолчанию ``sys.argv[1:]``).

    Returns:
        Пространство имён аргументов.
    """
    config = load_config()
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("output/error_analysis"),
        help="Куда писать report.json и копии промахов (по умолчанию output/error_analysis).",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Чекпоинт multi-task модели (по умолчанию — рабочий из config.yaml).",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=config.image_size,
        help="Размер изображения (по умолчанию из config.yaml).",
    )
    parser.add_argument(
        "--synthetic-dir",
        type=Path,
        default=Path("data/dirty_clean/test"),
        help="Синтетическое дерево <class>/<state>/.",
    )
    parser.add_argument(
        "--real-dirty-dir",
        type=Path,
        default=Path("data/real_dirty_val"),
        help="Набор реальной грязи <class>/ (все dirty).",
    )
    parser.add_argument(
        "--real-clean-dir",
        type=Path,
        default=Path("data/real_clean_probe"),
        help="Набор реальной чистой техники <class>/ (все clean); пустой — пропускается.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Точка входа error analysis.

    Args:
        argv: Аргументы командной строки (для тестируемости).

    Returns:
        Код возврата процесса: 0 при успехе, 1 если не нашлось ни одного фото.
    """
    args = parse_args(argv)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint = args.checkpoint or resolve_working_checkpoint()

    samples = collect_synthetic(args.synthetic_dir)
    samples += collect_flat(args.real_dirty_dir, REAL_DIRTY_SOURCE, "dirty")
    clean_probe = collect_flat(args.real_clean_dir, REAL_CLEAN_SOURCE, "clean")
    if clean_probe:
        samples += clean_probe
    else:
        print(f"[инфо] {args.real_clean_dir} пуст — источник real_clean_probe пропущен.")

    if not samples:
        print("[ошибка] Не нашлось ни одного размеченного фото.")
        return 1

    model = load_multi_task_model(checkpoint, device)
    transform = get_val_transforms(args.image_size)

    report = analyze(model, samples, transform, args.out_dir)
    report = {"checkpoint": str(checkpoint), "image_size": args.image_size, **report}

    report_path = args.out_dir / "report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    _print_table(report)
    print(f"\nОтчёт: {report_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

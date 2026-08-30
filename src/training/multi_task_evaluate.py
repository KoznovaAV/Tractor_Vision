#!/usr/bin/env python3
"""Оценка Multi-Task модели на val-наборе.

Показывает accuracy обеих задач (класс техники и состояние). Чекпоинт берётся из
``--checkpoint``; без него — рабочий файл через
:func:`src.models.loader.resolve_working_checkpoint` (сначала
``config.weights.multi_task``, иначе свежий по времени изменения best-чекпоинт).

С флагом ``--out-dir`` дополнительно сохраняет ``confusion_matrix.png`` (матрица
ошибок по классу техники) и ``misclassified_examples.png`` (сетка неверно
классифицированных фото).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.config.classes import MODEL_CLASSES
from src.config.config_loader import load_config
from src.data.dataloader import get_dataloader
from src.models.loader import load_multi_task_model, resolve_working_checkpoint


def resolve_checkpoint(explicit: Path | None) -> Path:
    """Выбрать чекпоинт: явный аргумент либо рабочий файл проекта.

    Args:
        explicit: Путь из ``--checkpoint`` либо ``None``.

    Returns:
        Путь к существующему чекпоинту.

    Raises:
        FileNotFoundError: Если явный путь не существует либо рабочий чекпоинт
            не найден.
    """
    if explicit is not None:
        if not explicit.is_file():
            raise FileNotFoundError(f"Чекпоинт не найден: {explicit}")
        return explicit
    return resolve_working_checkpoint()


def evaluate_model(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, list[int]]:
    """Прогнать модель по загрузчику и собрать предсказания обеих задач.

    Args:
        model: Multi-task модель в режиме eval.
        loader: Загрузчик val-набора (``shuffle=False``).
        device: Устройство инференса.

    Returns:
        Словарь со списками предсказаний и меток для задач модели и состояния.
    """
    model_preds: list[int] = []
    model_labels: list[int] = []
    state_preds: list[int] = []
    state_labels: list[int] = []

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            m_logits, s_logits = model(images)
            model_preds.extend(m_logits.argmax(dim=1).cpu().tolist())
            model_labels.extend(batch["model_label"].tolist())
            state_preds.extend(s_logits.argmax(dim=1).cpu().tolist())
            state_labels.extend(batch["state_label"].tolist())

    return {
        "model_preds": model_preds,
        "model_labels": model_labels,
        "state_preds": state_preds,
        "state_labels": state_labels,
    }


def _accuracy(preds: list[int], labels: list[int]) -> float:
    """Доля совпадений предсказаний и меток."""
    if not labels:
        return 0.0
    return float((np.array(preds) == np.array(labels)).mean())


def save_visualizations(
    out_dir: Path,
    predictions: dict[str, list[int]],
    sample_paths: list[Path],
) -> int:
    """Сохранить матрицу ошибок и сетку неверно классифицированных фото.

    Args:
        out_dir: Директория для картинок (создаётся при необходимости).
        predictions: Результат :func:`evaluate_model` (порядок совпадает с
            ``sample_paths``, так как val-загрузчик не перемешивается).
        sample_paths: Пути образцов в порядке обхода набора.

    Returns:
        Число неверно классифицированных по классу техники образцов.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    from PIL import Image
    from sklearn.metrics import confusion_matrix

    out_dir.mkdir(parents=True, exist_ok=True)
    model_preds = predictions["model_preds"]
    model_labels = predictions["model_labels"]

    cm = confusion_matrix(model_labels, model_preds, labels=list(range(len(MODEL_CLASSES))))
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=MODEL_CLASSES,
        yticklabels=MODEL_CLASSES,
        ax=ax,
    )
    ax.set_xlabel("Предсказано")
    ax.set_ylabel("Истина")
    ax.set_title("Матрица ошибок: модель трактора (val)")
    fig.tight_layout()
    fig.savefig(out_dir / "confusion_matrix.png", dpi=150)
    plt.close(fig)

    misclassified = [
        (sample_paths[i], model_labels[i], model_preds[i])
        for i in range(len(model_preds))
        if model_preds[i] != model_labels[i]
    ]
    if misclassified:
        n = min(8, len(misclassified))
        fig, axes = plt.subplots(1, n, figsize=(3 * n, 3.2))
        axes = [axes] if n == 1 else list(axes)
        for ax, (path, true_idx, pred_idx) in zip(axes, misclassified[:n]):
            ax.imshow(Image.open(path))
            ax.set_title(
                f"истина: {MODEL_CLASSES[true_idx]}\nпред: {MODEL_CLASSES[pred_idx]}",
                fontsize=8,
            )
            ax.axis("off")
        fig.tight_layout()
        fig.savefig(out_dir / "misclassified_examples.png", dpi=150)
        plt.close(fig)

    return len(misclassified)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Разобрать аргументы командной строки."""
    config = load_config()
    parser = argparse.ArgumentParser(description="Оценка Multi-Task модели на val-наборе.")
    parser.add_argument("--data-dir", type=Path, default=config.data.dirty_clean_dir)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--image-size", type=int, default=config.image_size)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Куда сохранить confusion_matrix.png и misclassified_examples.png.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Точка входа оценки."""
    args = parse_args(argv)
    checkpoint_path = resolve_checkpoint(args.checkpoint)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Запуск оценки Multi-Task модели...")
    print(f"Датасет: {args.data_dir / 'val'}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Устройство: {device}")
    print("-" * 50)

    model = load_multi_task_model(checkpoint_path, device)
    loader = get_dataloader(
        data_dir=args.data_dir,
        split="val",
        image_size=args.image_size,
        num_workers=0,
        multi_task=True,
    )

    print(f"Оценка на {len(loader.dataset)} фото...")
    predictions = evaluate_model(model, loader, device)
    model_accuracy = _accuracy(predictions["model_preds"], predictions["model_labels"])
    state_accuracy = _accuracy(predictions["state_preds"], predictions["state_labels"])

    print("\n" + "=" * 50)
    print("РЕЗУЛЬТАТЫ ОЦЕНКИ MULTI-TASK МОДЕЛИ")
    print("=" * 50)
    print(f"Задача 1 (модель трактора):  {model_accuracy:.2%}")
    print(f"Задача 2 (грязный/чистый):   {state_accuracy:.2%}")
    print("=" * 50)

    if args.out_dir is not None:
        sample_paths = [sample.path for sample in loader.dataset.samples]
        n_bad = save_visualizations(args.out_dir, predictions, sample_paths)
        print(f"Картинки сохранены в {args.out_dir} (ошибок по классу: {n_bad}).")

    return 0


if __name__ == "__main__":
    sys.exit(main())

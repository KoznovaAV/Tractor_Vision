#!/usr/bin/env python3
"""Оценка Single-Task модели на test-наборе с метриками и визуализацией.

Число классов и их имена берутся из :mod:`src.config.classes`; размер
изображения по умолчанию — из ``config.yaml``. Локальный ``num_classes=5``
удалён.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchmetrics import Accuracy, ConfusionMatrix, F1Score, Precision, Recall

from src.config.classes import MODEL_CLASSES, NUM_MODEL_CLASSES
from src.config.config_loader import load_config
from src.data.dataloader import get_dataloader
from src.models.classifier import TractorClassifier

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
MAX_MISCLASSIFIED: int = 9


def load_model(checkpoint_path: Path, num_classes: int = NUM_MODEL_CLASSES) -> torch.nn.Module:
    """Загрузить обученную single-task модель из чекпоинта.

    Поддерживает как «сырые» ``state_dict``, так и Lightning-чекпоинты (ключ
    ``state_dict`` с префиксом ``model.``).

    Args:
        checkpoint_path: Путь к ``.ckpt``.
        num_classes: Число классов (по умолчанию из config.classes).

    Returns:
        Модель в режиме ``eval``.
    """
    model = TractorClassifier(num_classes=num_classes)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    state_dict = checkpoint.get("state_dict", checkpoint)
    # Снять префикс "model." от Lightning-обёртки, если он есть.
    cleaned = {key.removeprefix("model."): value for key, value in state_dict.items()}
    model.load_state_dict(cleaned, strict=False)
    model.eval()
    return model


def create_test_loader(
    data_dir: Path,
    batch_size: int = 16,
    image_size: int | None = None,
) -> DataLoader:
    """Создать загрузчик test-сплита.

    Args:
        data_dir: Корень датасета.
        batch_size: Размер батча.
        image_size: Размер изображения; ``None`` — взять из ``config.yaml``.

    Returns:
        DataLoader test-сплита.
    """
    size = image_size if image_size is not None else load_config().image_size
    return get_dataloader(
        data_dir=data_dir,
        split="test",
        batch_size=batch_size,
        image_size=size,
        num_workers=2,
        multi_task=False,
    )


def create_metrics(num_classes: int, device: torch.device) -> dict[str, Any]:
    """Создать набор метрик оценки.

    Args:
        num_classes: Число классов.
        device: Устройство.

    Returns:
        Словарь метрик.
    """
    return {
        "accuracy": Accuracy(task="multiclass", num_classes=num_classes).to(device),
        "precision": Precision(task="multiclass", num_classes=num_classes, average="macro").to(
            device
        ),
        "recall": Recall(task="multiclass", num_classes=num_classes, average="macro").to(device),
        "f1": F1Score(task="multiclass", num_classes=num_classes, average="macro").to(device),
        "confusion": ConfusionMatrix(task="multiclass", num_classes=num_classes).to(device),
    }


def evaluate_model(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, Any]:
    """Прогнать модель по загрузчику и посчитать метрики.

    Args:
        model: Оцениваемая модель.
        loader: Загрузчик данных.
        device: Устройство.

    Returns:
        Словарь со скалярными метриками и матрицей ошибок.
    """
    model = model.to(device)
    metrics = create_metrics(NUM_MODEL_CLASSES, device)

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            labels = batch["model_label"].to(device)
            logits = model(images)
            preds = torch.argmax(logits, dim=1)
            for name, metric in metrics.items():
                metric.update(preds if name == "confusion" else logits, labels)

    return {
        "accuracy": float(metrics["accuracy"].compute()),
        "precision": float(metrics["precision"].compute()),
        "recall": float(metrics["recall"].compute()),
        "f1": float(metrics["f1"].compute()),
        "confusion": metrics["confusion"].compute().cpu().numpy(),
    }


def plot_confusion_matrix(conf_matrix: np.ndarray, save_path: Path) -> None:
    """Отрисовать и сохранить матрицу ошибок.

    Args:
        conf_matrix: Матрица ошибок ``(C, C)``.
        save_path: Путь сохранения PNG.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        conf_matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=list(MODEL_CLASSES),
        yticklabels=list(MODEL_CLASSES),
    )
    plt.xlabel("Предсказано")
    plt.ylabel("Истинно")
    plt.title("Матрица ошибок")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def denormalize_image(image: torch.Tensor) -> np.ndarray:
    """Обратная нормализация тензора изображения в ``uint8`` RGB.

    Args:
        image: Тензор ``(3, H, W)`` после ImageNet-нормализации.

    Returns:
        Массив ``(H, W, 3)`` типа ``uint8``.
    """
    array = image.cpu().numpy().transpose(1, 2, 0)
    array = array * IMAGENET_STD + IMAGENET_MEAN
    return (np.clip(array, 0.0, 1.0) * 255).astype(np.uint8)


def find_misclassified_examples(
    loader: DataLoader,
    model: torch.nn.Module,
    device: torch.device,
    max_examples: int = MAX_MISCLASSIFIED,
) -> list[dict[str, Any]]:
    """Найти примеры ошибочной классификации.

    Args:
        loader: Загрузчик данных.
        model: Модель.
        device: Устройство.
        max_examples: Максимум примеров.

    Returns:
        Список записей с изображением, истинным и предсказанным классом.
    """
    model = model.to(device).eval()
    errors: list[dict[str, Any]] = []

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            labels = batch["model_label"].to(device)
            preds = torch.argmax(model(images), dim=1)
            for i in range(len(labels)):
                if preds[i] != labels[i]:
                    errors.append(
                        {
                            "image": denormalize_image(images[i]),
                            "true": MODEL_CLASSES[int(labels[i])],
                            "pred": MODEL_CLASSES[int(preds[i])],
                        }
                    )
                    if len(errors) >= max_examples:
                        return errors
    return errors


def plot_misclassified_examples(errors: list[dict[str, Any]], save_path: Path) -> None:
    """Отрисовать сетку ошибочно классифицированных примеров.

    Args:
        errors: Список записей ошибок.
        save_path: Путь сохранения PNG.
    """
    import matplotlib.pyplot as plt

    if not errors:
        print("Ошибок классификации не найдено — нечего визуализировать.")
        return

    save_path.parent.mkdir(parents=True, exist_ok=True)
    count = len(errors)
    cols = 3
    rows = (count + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(12, 4 * rows))
    axes_flat = np.array(axes).reshape(-1)

    for idx, error in enumerate(errors):
        ax = axes_flat[idx]
        ax.imshow(error["image"])
        ax.set_title(f"ист: {error['true']}\nпред: {error['pred']}", fontsize=9)
        ax.axis("off")
    for idx in range(count, len(axes_flat)):
        axes_flat[idx].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Разобрать аргументы командной строки.

    Args:
        argv: Аргументы (по умолчанию ``sys.argv[1:]``).

    Returns:
        Пространство имён аргументов.
    """
    config = load_config()
    parser = argparse.ArgumentParser(description="Оценка Single-Task модели на test-наборе.")
    parser.add_argument("--data-dir", type=Path, default=config.data.processed_dir)
    parser.add_argument("--checkpoint", type=Path, default=config.weights.single_task)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=config.image_size)
    parser.add_argument("--output-dir", type=Path, default=Path("output"))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Точка входа оценки.

    Args:
        argv: Аргументы командной строки.

    Returns:
        Код возврата процесса.
    """
    args = parse_args(argv)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not args.checkpoint.is_file():
        print(f"[ошибка] Чекпоинт не найден: {args.checkpoint}")
        return 1

    model = load_model(args.checkpoint, num_classes=NUM_MODEL_CLASSES)
    loader = create_test_loader(args.data_dir, args.batch_size, args.image_size)

    results = evaluate_model(model, loader, device)
    print(f"Accuracy:  {results['accuracy']:.4f}")
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall:    {results['recall']:.4f}")
    print(f"F1:        {results['f1']:.4f}")

    plot_confusion_matrix(results["confusion"], args.output_dir / "confusion_matrix.png")
    errors = find_misclassified_examples(loader, model, device)
    plot_misclassified_examples(errors, args.output_dir / "misclassified_examples.png")
    print(f"Артефакты оценки сохранены в {args.output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

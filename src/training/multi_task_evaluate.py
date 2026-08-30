#!/usr/bin/env python3
"""Оценка Multi-Task модели на val-наборе.

Показывает accuracy для обеих задач: модель и состояние.
Чекпоинт берётся из ``--checkpoint``; по умолчанию — рабочий файл
``weights/multi_task_best.ckpt`` из ``config.yaml``, а при его отсутствии —
свежий по времени изменения best-чекпоинт. Сортировка по имени файла больше
не используется (она ошибочно выбирала ``epoch=19`` вместо ``epoch=13``,
так как строка "19" > "13").
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.config.config_loader import load_config
from src.data.dataset import TractorDataset
from src.data.transforms import get_val_transforms
from src.models.multi_task import MultiTaskTractorClassifier

WEIGHTS_DIR = Path("weights")


def load_model(checkpoint_path: Path) -> torch.nn.Module:
    """Загружает Multi-Task модель из checkpoint."""
    model = MultiTaskTractorClassifier()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = {
        k.replace("model.", ""): v
        for k, v in checkpoint["state_dict"].items()
        if k.startswith("model.")
    }
    model.load_state_dict(state_dict)
    model.eval()
    return model


def resolve_checkpoint(explicit: Path | None) -> Path:
    """Выбрать чекпоинт: явный аргумент, иначе рабочий файл, иначе свежий best.

    Args:
        explicit: Путь из ``--checkpoint`` либо ``None``.

    Returns:
        Путь к существующему чекпоинту.

    Raises:
        FileNotFoundError: Если ни один источник не дал чекпоинт.
    """
    config = load_config()
    if explicit is not None:
        if not explicit.is_file():
            raise FileNotFoundError(f"Чекпоинт не найден: {explicit}")
        return explicit
    working = Path(config.weights.multi_task)
    if working.is_file():
        return working
    candidates = list(WEIGHTS_DIR.glob("multi-task-best-*.ckpt"))
    if not candidates:
        raise FileNotFoundError(f"Нет Multi-Task checkpoints в {WEIGHTS_DIR}")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def create_val_loader(
    data_dir: Path,
    batch_size: int = 16,
    image_size: int = 384,
) -> DataLoader:
    """Создаёт DataLoader для val-набора."""
    dataset = TractorDataset(
        root_dir=data_dir / "val",
        transform=get_val_transforms(image_size),
        multi_task=True,
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)


def evaluate_model(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    """Оценивает модель на обеих задачах."""
    model_preds, model_labels = [], []
    state_preds, state_labels = [], []

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            m_labels = batch["model_label"].to(device)
            s_labels = batch["state_label"].to(device)

            m_logits, s_logits = model(images)

            model_preds.extend(m_logits.argmax(dim=1).cpu().numpy())
            model_labels.extend(m_labels.cpu().numpy())
            state_preds.extend(s_logits.argmax(dim=1).cpu().numpy())
            state_labels.extend(s_labels.cpu().numpy())

    return {
        "model_accuracy": float((np.array(model_preds) == np.array(model_labels)).mean()),
        "state_accuracy": float((np.array(state_preds) == np.array(state_labels)).mean()),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Разобрать аргументы командной строки."""
    config = load_config()
    parser = argparse.ArgumentParser(description="Оценка Multi-Task модели на val-наборе.")
    parser.add_argument("--data-dir", type=Path, default=config.data.dirty_clean_dir)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--image-size", type=int, default=config.image_size)
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

    model = load_model(checkpoint_path).to(device)
    loader = create_val_loader(args.data_dir, image_size=args.image_size)

    print(f"Оценка на {len(loader.dataset)} фото...")
    metrics = evaluate_model(model, loader, device)

    print("\n" + "=" * 50)
    print("РЕЗУЛЬТАТЫ ОЦЕНКИ MULTI-TASK МОДЕЛИ")
    print("=" * 50)
    print(f"Задача 1 (модель трактора):  {metrics['model_accuracy']:.2%}")
    print(f"Задача 2 (грязный/чистый):   {metrics['state_accuracy']:.2%}")
    print("=" * 50)
    return 0


if __name__ == "__main__":
    sys.exit(main())

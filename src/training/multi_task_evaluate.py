"""
Скрипт для оценки Multi-Task модели на val-наборе.
Показывает accuracy для обеих задач: модель и состояние.
"""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.dataset import TractorDataset
from src.data.transforms import get_val_transforms
from src.models.multi_task import MultiTaskTractorClassifier

OUTPUT_DIR = Path("output")
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


def create_val_loader(
    data_dir: Path, batch_size: int = 16, image_size: int = 384
) -> DataLoader:
    """Создаёт DataLoader для val-набора."""
    dataset = TractorDataset(
        root_dir=data_dir / "val",
        transform=get_val_transforms(image_size),
        multi_task=True,
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)


def evaluate_model(
    model: torch.nn.Module, loader: DataLoader, device: torch.device
) -> dict:
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
        "model_accuracy": (np.array(model_preds) == np.array(model_labels)).mean(),
        "state_accuracy": (np.array(state_preds) == np.array(state_labels)).mean(),
    }


def find_best_checkpoint() -> Path:
    """Находит последний Multi-Task checkpoint."""
    checkpoints = sorted(WEIGHTS_DIR.glob("multi-task-*.ckpt"))
    if not checkpoints:
        raise FileNotFoundError(f"Нет Multi-Task checkpoints в {WEIGHTS_DIR}")
    return checkpoints[-1]


def main():
    print("Запуск оценки Multi-Task модели...")

    data_dir = Path("data/dirty_clean")
    checkpoint_path = find_best_checkpoint()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Датасет: {data_dir / 'val'}")
    print(f"️Checkpoint: {checkpoint_path}")
    print(f"Устройство: {device}")
    print("-" * 50)

    model = load_model(checkpoint_path).to(device)
    loader = create_val_loader(data_dir)

    print(f"Оценка на {len(loader.dataset)} фото...")
    metrics = evaluate_model(model, loader, device)

    print("\n" + "=" * 50)
    print("РЕЗУЛЬТАТЫ ОЦЕНКИ MULTI-TASK МОДЕЛИ")
    print("=" * 50)
    print(f"Задача 1 (модель трактора):  {metrics['model_accuracy']:.2%}")
    print(f"Задача 2 (грязный/чистый):   {metrics['state_accuracy']:.2%}")
    print("=" * 50)


if __name__ == "__main__":
    main()

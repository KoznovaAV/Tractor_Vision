"""
Оценка обученной модели на test-наборе.
"""

import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from torch.utils.data import DataLoader
from torchmetrics import Accuracy, ConfusionMatrix, F1Score, Precision, Recall

from src.data.dataset import TractorDataset
from src.data.transforms import get_val_transforms
from src.models.classifier import CLASS_NAMES, TractorClassifier

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

OUTPUT_DIR = Path("output")
WEIGHTS_DIR = Path("weights")
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])


def load_model(
    checkpoint_path: Path, num_classes: int = len(CLASS_NAMES)
) -> torch.nn.Module:
    """Загружает модель из checkpoint."""
    model = TractorClassifier(num_classes=num_classes)

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = {
        k.replace("model.", ""): v
        for k, v in checkpoint["state_dict"].items()
        if k.startswith("model.")
    }
    model.load_state_dict(state_dict)
    model.eval()

    logger.info("Модель загружена: %s", checkpoint_path)
    return model


def create_test_loader(
    data_dir: Path, batch_size: int = 16, image_size: int = 224
) -> DataLoader:
    """Создаёт DataLoader для test-набора."""
    dataset = TractorDataset(
        root_dir=data_dir / "test",
        transform=get_val_transforms(image_size),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)


def create_metrics(num_classes: int, device: torch.device) -> dict:
    """Создаёт метрики для оценки."""
    return {
        "accuracy": Accuracy(task="multiclass", num_classes=num_classes).to(device),
        "precision": Precision(
            task="multiclass", num_classes=num_classes, average="macro"
        ).to(device),
        "recall": Recall(
            task="multiclass", num_classes=num_classes, average="macro"
        ).to(device),
        "f1": F1Score(task="multiclass", num_classes=num_classes, average="macro").to(
            device
        ),
        "confusion_matrix": ConfusionMatrix(
            task="multiclass", num_classes=num_classes
        ).to(device),
    }


def evaluate_model(
    model: torch.nn.Module, loader: DataLoader, device: torch.device
) -> dict:
    """Оценивает модель на test-наборе."""
    metrics = create_metrics(len(CLASS_NAMES), device)
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)

            logits = model(images)
            preds = torch.softmax(logits, dim=1)

            for metric in metrics.values():
                metric(preds, labels)

            all_preds.extend(preds.argmax(dim=1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return {
        "accuracy": metrics["accuracy"].compute().item(),
        "precision": metrics["precision"].compute().item(),
        "recall": metrics["recall"].compute().item(),
        "f1": metrics["f1"].compute().item(),
        "confusion_matrix": metrics["confusion_matrix"].compute().cpu().numpy(),
        "predictions": np.array(all_preds),
        "labels": np.array(all_labels),
    }


def plot_confusion_matrix(conf_matrix: np.ndarray, save_path: Path) -> None:
    """Строит и сохраняет confusion matrix."""
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        conf_matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix (Test Set)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    logger.info("Confusion matrix сохранена: %s", save_path)


def find_misclassified_examples(
    loader: DataLoader,
    model: torch.nn.Module,
    device: torch.device,
    max_examples: int = 9,
) -> list[dict]:
    """Находит примеры ошибок классификации."""
    model.eval()
    errors = []

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)

            logits = model(images)
            preds = logits.argmax(dim=1)

            for i in range(len(labels)):
                if preds[i] != labels[i]:
                    errors.append(
                        {
                            "image": batch["image"][i].permute(1, 2, 0).numpy(),
                            "true": CLASS_NAMES[labels[i].item()],
                            "pred": CLASS_NAMES[preds[i].item()],
                        }
                    )

                    if len(errors) >= max_examples:
                        return errors

    return errors


def denormalize_image(image: np.ndarray) -> np.ndarray:
    """Денормализует изображение для визуализации."""
    return (image * IMAGENET_STD + IMAGENET_MEAN).clip(0, 1)


def plot_misclassified_examples(errors: list[dict], save_path: Path) -> None:
    """Строит и сохраняет примеры ошибок."""
    if not errors:
        logger.info("Ошибок не найдено — модель идеальна!")
        return

    cols = 3
    rows = (len(errors) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
    axes = axes.flatten() if len(errors) > 1 else [axes]

    for i, error in enumerate(errors):
        ax = axes[i]
        img = denormalize_image(error["image"])

        ax.imshow(img)
        ax.set_title(f"True: {error['true']}\nPred: {error['pred']}", color="red")
        ax.axis("off")

    for j in range(len(errors), len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    logger.info("Примеры ошибок сохранены: %s", save_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Оценка модели на test-наборе")
    parser.add_argument("--data_dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--image_size", type=int, default=224)
    return parser.parse_args()


def find_best_checkpoint() -> Path:
    """Находит лучший checkpoint по val_acc."""
    checkpoints = sorted(WEIGHTS_DIR.glob("best-*.ckpt"))
    if not checkpoints:
        raise FileNotFoundError(f"Нет checkpoints в {WEIGHTS_DIR}")
    return checkpoints[-1]


def main():
    args = parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    checkpoint_path = args.checkpoint or find_best_checkpoint()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info("Устройство: %s", device)
    logger.info("Checkpoint: %s", checkpoint_path)

    model = load_model(checkpoint_path).to(device)
    loader = create_test_loader(args.data_dir, args.batch_size, args.image_size)

    logger.info("Оценка на test-наборе (%d фото)...", len(loader.dataset))
    metrics = evaluate_model(model, loader, device)

    print("\n" + "=" * 60)
    print("РЕЗУЛЬТАТЫ ОЦЕНКИ НА TEST-НАБОРЕ")
    print("=" * 60)
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1-score:  {metrics['f1']:.4f}")
    print("=" * 60)

    plot_confusion_matrix(
        metrics["confusion_matrix"], OUTPUT_DIR / "confusion_matrix.png"
    )

    errors = find_misclassified_examples(loader, model, device)
    plot_misclassified_examples(errors, OUTPUT_DIR / "misclassified_examples.png")

    logger.info("Готово! Результаты в папке %s/", OUTPUT_DIR)


if __name__ == "__main__":
    main()

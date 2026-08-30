"""Матрица ошибок и примеры ошибок multi-task модели на val-сплите."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns
import torch
from PIL import Image
from sklearn.metrics import confusion_matrix

from src.config.classes import MODEL_CLASSES, STATE_CLASSES
from src.data.dataset import TractorDataset
from src.data.transforms import get_val_transforms
from src.models.multi_task import MultiTaskTractorClassifier


def load_model(checkpoint_path: Path, device: str) -> MultiTaskTractorClassifier:
    """Загрузить веса из Lightning-чекпоинта в чистую модель."""
    model = MultiTaskTractorClassifier()
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = ckpt.get("state_dict", ckpt)
    # Убираем префикс "model." от Lightning-обёртки и лишние ключи.
    cleaned = {(k.removeprefix("model.")): v for k, v in state_dict.items()}
    target = model.state_dict()
    cleaned = {k: v for k, v in cleaned.items() if k in target}
    model.load_state_dict(cleaned, strict=True)
    return model.to(device).eval()


def main(argv: list[str] | None = None) -> int:
    """Точка входа: построить матрицу ошибок и сетку ошибок."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--data-dir", type=Path, default=Path("data/dirty_clean"))
    parser.add_argument("--image-size", type=int, default=384)
    parser.add_argument("--out-dir", type=Path, default=Path("output"))
    args = parser.parse_args(argv)

    if args.checkpoint is None:
        candidates = sorted(Path("weights").glob("multi-task-best-*.ckpt"))
        if not candidates:
            print("[ошибка] best-чекпоинт не найден в weights/")
            return 1
        args.checkpoint = candidates[-1]
    print(f"Чекпоинт: {args.checkpoint}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(args.checkpoint, device)
    dataset = TractorDataset(
        root_dir=args.data_dir / "val",
        transform=get_val_transforms(args.image_size),
        multi_task=True,
    )

    model_preds: list[int] = []
    model_trues: list[int] = []
    state_preds: list[int] = []
    state_trues: list[int] = []
    misclassified: list[tuple[Path, int, int]] = []

    with torch.no_grad():
        for idx in range(len(dataset)):
            item = dataset[idx]
            x = item["image"].unsqueeze(0).to(device)
            model_logits, state_logits = model(x)
            mp = int(model_logits.argmax(dim=1))
            sp = int(state_logits.argmax(dim=1))
            mt = int(item["model_label"])
            st = int(item["state_label"])
            model_preds.append(mp)
            model_trues.append(mt)
            state_preds.append(sp)
            state_trues.append(st)
            if mp != mt:
                misclassified.append((dataset.samples[idx].path, mt, mp))

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Матрица ошибок по модели.
    cm = confusion_matrix(model_trues, model_preds, labels=list(range(len(MODEL_CLASSES))))
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
    fig.savefig(args.out_dir / "confusion_matrix.png", dpi=150)
    plt.close(fig)

    # Матрица по состоянию — текстом.
    cm_state = confusion_matrix(state_trues, state_preds, labels=list(range(len(STATE_CLASSES))))
    print("Матрица по состоянию (clean/dirty):")
    print(cm_state)
    print(f"Ошибок по модели: {len(misclassified)} из {len(dataset)}")

    # Сетка примеров ошибок.
    if misclassified:
        n = min(8, len(misclassified))
        fig, axes = plt.subplots(1, n, figsize=(3 * n, 3.2))
        axes = [axes] if n == 1 else list(axes)
        for ax, (path, mt, mp) in zip(axes, misclassified[:n]):
            ax.imshow(Image.open(path))
            ax.set_title(
                f"истина: {MODEL_CLASSES[mt]}\nпред: {MODEL_CLASSES[mp]}",
                fontsize=8,
            )
            ax.axis("off")
        fig.tight_layout()
        fig.savefig(args.out_dir / "misclassified_examples.png", dpi=150)
        plt.close(fig)
    else:
        print("Ошибок по модели нет — сетка примеров не создана.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

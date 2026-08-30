"""Локальное демо: модель трактора + состояние по произвольным фото."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from scripts.eval_visualize import load_model
from src.config.classes import MODEL_CLASSES, STATE_CLASSES
from src.data.transforms import get_val_transforms

IMAGE_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".webp"})


def iter_images(paths: list[Path]) -> list[Path]:
    """Развернуть папки в список файлов изображений."""
    result: list[Path] = []
    for p in paths:
        if p.is_dir():
            result.extend(sorted(f for f in p.iterdir() if f.suffix.lower() in IMAGE_EXTENSIONS))
        else:
            result.append(p)
    return result


def main(argv: list[str] | None = None) -> int:
    """Прогнать фото через best-чекпоинт и напечатать предсказания."""
    parser = argparse.ArgumentParser(description="Демо-предсказание по фото.")
    parser.add_argument("images", nargs="+", type=Path, help="Файлы или папки с фото.")
    parser.add_argument("--image-size", type=int, default=384)
    args = parser.parse_args(argv)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(Path("weights/multi_task_best.ckpt"), device)
    transform = get_val_transforms(args.image_size)

    for path in iter_images(args.images):
        image = Image.open(path).convert("RGB")
        x = transform(image=np.array(image))["image"].unsqueeze(0).to(device)
        with torch.no_grad():
            m_logits, s_logits = model(x)
        m_probs = torch.softmax(m_logits, dim=1).squeeze(0).cpu().tolist()
        s_probs = torch.softmax(s_logits, dim=1).squeeze(0).cpu().tolist()
        mi = int(max(range(len(m_probs)), key=lambda i: m_probs[i]))
        si = int(max(range(len(s_probs)), key=lambda i: s_probs[i]))
        second = int(max((i for i in range(len(m_probs)) if i != mi), key=lambda i: m_probs[i]))
        print(f"\n{path.name}")
        print(f"  модель:     {MODEL_CLASSES[mi]} ({m_probs[mi]:.0%})")
        print(f"  альтернат.: {MODEL_CLASSES[second]} ({m_probs[second]:.0%})")
        print(f"  состояние:  {STATE_CLASSES[si]} ({s_probs[si]:.0%})")
    return 0


if __name__ == "__main__":
    sys.exit(main())

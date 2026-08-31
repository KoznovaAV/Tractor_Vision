"""Локальное демо: модель трактора + состояние по произвольным фото.

Берёт рабочий чекпоинт через :func:`src.models.loader.resolve_working_checkpoint`
и прогоняет каждое фото единым инференсом :func:`src.models.predict.predict_image`.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

from src.config.classes import MODEL_CLASSES, STATE_CLASSES
from src.config.config_loader import load_config
from src.data.transforms import get_val_transforms
from src.models.loader import load_multi_task_model, resolve_working_checkpoint
from src.models.predict import predict_image

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
    """Прогнать фото через рабочий чекпоинт и напечатать предсказания."""
    config = load_config()
    parser = argparse.ArgumentParser(description="Демо-предсказание по фото.")
    parser.add_argument("images", nargs="+", type=Path, help="Файлы или папки с фото.")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--image-size", type=int, default=config.image_size)
    args = parser.parse_args(argv)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint = args.checkpoint or resolve_working_checkpoint()
    model = load_multi_task_model(checkpoint, device)
    transform = get_val_transforms(args.image_size)
    print(f"Чекпоинт: {checkpoint}")

    for path in iter_images(args.images):
        model_idx, model_conf, state_idx, state_conf = predict_image(model, path, transform)
        print(f"\n{path.name}")
        print(f"  модель:     {MODEL_CLASSES[model_idx]} ({model_conf:.0%})")
        print(f"  состояние:  {STATE_CLASSES[state_idx]} ({state_conf:.0%})")
    return 0


if __name__ == "__main__":
    sys.exit(main())

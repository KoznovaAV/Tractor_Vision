#!/usr/bin/env python3
"""Обучение Single-Task модели классификации тракторов (PyTorch Lightning).

Число классов берётся из :mod:`src.config.classes`, размер изображения по
умолчанию — из ``config.yaml`` (единый ``image_size`` для train/eval/API).
Локальный ``num_classes=5`` удалён.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from torch import nn
from torchmetrics import Accuracy, F1Score

from src.config.classes import NUM_MODEL_CLASSES
from src.config.config_loader import load_config
from src.data.dataloader import get_dataloader
from src.models.classifier import TractorClassifier

DEFAULT_LR: float = 5e-4
DEFAULT_BATCH_SIZE: int = 8
DEFAULT_EPOCHS: int = 100
WEIGHT_DECAY: float = 1e-4
LABEL_SMOOTHING: float = 0.1
EARLY_STOPPING_PATIENCE: int = 15
LR_SCHEDULER_PATIENCE: int = 7
LR_REDUCE_FACTOR: float = 0.5
MIN_LR: float = 1e-6


class TractorLightningModule(pl.LightningModule):
    """Lightning-модуль обучения single-task классификатора."""

    def __init__(
        self,
        num_classes: int = NUM_MODEL_CLASSES,
        lr: float = DEFAULT_LR,
    ) -> None:
        """Инициализировать модуль.

        Args:
            num_classes: Число классов (по умолчанию из config.classes).
            lr: Learning rate.
        """
        super().__init__()
        self.save_hyperparameters()
        self.model = TractorClassifier(num_classes=num_classes)
        self.criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
        self.lr = lr

        self.train_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_f1 = F1Score(task="multiclass", num_classes=num_classes, average="macro")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Прямой проход.

        Args:
            x: Батч изображений.

        Returns:
            Логиты классов.
        """
        logits: torch.Tensor = self.model(x)
        return logits

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Шаг обучения.

        Args:
            batch: Батч с ключами ``image`` и ``model_label``.
            batch_idx: Индекс батча.

        Returns:
            Значение потери.
        """
        logits = self(batch["image"])
        loss = self.criterion(logits, batch["model_label"])
        self.train_acc(logits, batch["model_label"])
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", self.train_acc, prog_bar=True)
        return loss

    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Шаг валидации.

        Args:
            batch: Валидационный батч.
            batch_idx: Индекс батча.

        Returns:
            Значение потери.
        """
        logits = self(batch["image"])
        loss = self.criterion(logits, batch["model_label"])
        self.val_acc(logits, batch["model_label"])
        self.val_f1(logits, batch["model_label"])
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", self.val_acc, prog_bar=True)
        self.log("val_f1", self.val_f1, prog_bar=True)
        return loss

    def configure_optimizers(self) -> dict[str, Any]:
        """Собрать оптимизатор и планировщик.

        Returns:
            Конфигурация оптимизатора для Lightning.
        """
        trainable = [p for p in self.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable, lr=self.lr, weight_decay=WEIGHT_DECAY)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=LR_REDUCE_FACTOR,
            patience=LR_SCHEDULER_PATIENCE,
            min_lr=MIN_LR,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": "val_acc"},
        }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Разобрать аргументы командной строки.

    Args:
        argv: Аргументы (по умолчанию ``sys.argv[1:]``).

    Returns:
        Пространство имён аргументов.
    """
    config = load_config()
    parser = argparse.ArgumentParser(description="Обучение Single-Task классификатора тракторов.")
    parser.add_argument("--data-dir", type=Path, default=config.data.processed_dir)
    parser.add_argument("--weights-dir", type=Path, default=config.weights.dir)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--max-epochs", type=int, default=DEFAULT_EPOCHS)
    # image_size по умолчанию из config.yaml — единый размер train/eval/API.
    parser.add_argument("--image-size", type=int, default=config.image_size)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--accelerator", type=str, default="auto")
    return parser.parse_args(argv)


def create_dataloaders(args: argparse.Namespace) -> dict[str, Any]:
    """Создать train/val загрузчики.

    Args:
        args: Разобранные аргументы.

    Returns:
        Словарь с ключами ``train`` и ``val``.
    """
    return {
        "train": get_dataloader(
            data_dir=args.data_dir,
            split="train",
            batch_size=args.batch_size,
            image_size=args.image_size,
            num_workers=args.num_workers,
            multi_task=False,
        ),
        "val": get_dataloader(
            data_dir=args.data_dir,
            split="val",
            batch_size=args.batch_size,
            image_size=args.image_size,
            num_workers=args.num_workers,
            multi_task=False,
        ),
    }


def create_callbacks(weights_dir: Path) -> list[pl.Callback]:
    """Создать колбэки обучения.

    Args:
        weights_dir: Директория чекпоинтов.

    Returns:
        Список колбэков.
    """
    return [
        ModelCheckpoint(
            dirpath=weights_dir,
            filename="single-task-best-{epoch:02d}-{val_acc:.3f}",
            monitor="val_acc",
            mode="max",
            save_top_k=1,
            save_last=True,
        ),
        EarlyStopping(
            monitor="val_acc",
            patience=EARLY_STOPPING_PATIENCE,
            mode="max",
            verbose=True,
        ),
        LearningRateMonitor(logging_interval="epoch"),
    ]


def create_trainer(args: argparse.Namespace, callbacks: list[pl.Callback]) -> pl.Trainer:
    """Создать Lightning Trainer.

    Args:
        args: Аргументы.
        callbacks: Колбэки.

    Returns:
        Настроенный тренер.
    """
    return pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator=args.accelerator,
        callbacks=callbacks,
        log_every_n_steps=10,
    )


def main(argv: list[str] | None = None) -> int:
    """Точка входа обучения.

    Args:
        argv: Аргументы командной строки.

    Returns:
        Код возврата процесса.
    """
    args = parse_args(argv)
    args.weights_dir.mkdir(parents=True, exist_ok=True)

    loaders = create_dataloaders(args)
    module = TractorLightningModule(num_classes=NUM_MODEL_CLASSES, lr=args.lr)
    trainer = create_trainer(args, create_callbacks(args.weights_dir))
    trainer.fit(module, loaders["train"], loaders["val"])

    final_path = args.weights_dir / "final_model.ckpt"
    trainer.save_checkpoint(final_path)
    print(f"Финальный чекпоинт сохранён: {final_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""
Тренировочный скрипт для классификатора тракторов.
Использует PyTorch Lightning для упрощения цикла обучения.
"""

import argparse
import logging
from pathlib import Path

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from torch import nn
from torchmetrics import Accuracy, F1Score

from src.data.dataloader import get_dataloader
from src.models.classifier import CLASS_NAMES, TractorClassifier

torch.set_float32_matmul_precision("high")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

WEIGHTS_DIR = Path("weights")
EARLY_STOPPING_PATIENCE = 15
LR_SCHEDULER_PATIENCE = 7
LR_REDUCE_FACTOR = 0.5
MIN_LR = 1e-6
WEIGHT_DECAY = 1e-4
DEFAULT_LR = 5e-4
DEFAULT_BATCH_SIZE = 8
DEFAULT_EPOCHS = 100
DEFAULT_NUM_WORKERS = 0
DEFAULT_IMAGE_SIZE = 384
SPLITS = ("train", "val")


class TractorLightningModule(pl.LightningModule):
    """Lightning-обёртка для классификатора тракторов."""

    def __init__(self, num_classes: int = len(CLASS_NAMES), lr: float = DEFAULT_LR):
        super().__init__()
        self.save_hyperparameters()

        self.model = TractorClassifier(num_classes=num_classes)
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

        self.train_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_f1 = F1Score(
            task="multiclass", num_classes=num_classes, average="macro"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def training_step(
        self, batch: dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        images, labels = batch["image"], batch["label"]
        logits = self(images)
        loss = self.criterion(logits, labels)

        preds = torch.softmax(logits, dim=1)
        self.train_acc(preds, labels)

        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", self.train_acc, prog_bar=True)
        return loss

    def validation_step(
        self, batch: dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        images, labels = batch["image"], batch["label"]
        logits = self(images)
        loss = self.criterion(logits, labels)

        preds = torch.softmax(logits, dim=1)
        self.val_acc(preds, labels)
        self.val_f1(preds, labels)

        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", self.val_acc, prog_bar=True)
        self.log("val_f1", self.val_f1, prog_bar=True)
        return loss

    def configure_optimizers(self) -> dict:
        optimizer = torch.optim.AdamW(
            self.model.backbone.classifier.parameters(),
            lr=self.hparams.lr,
            weight_decay=WEIGHT_DECAY,
        )

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=LR_REDUCE_FACTOR,
            patience=LR_SCHEDULER_PATIENCE,
            min_lr=MIN_LR,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "val_acc",
        }


def parse_args() -> argparse.Namespace:
    """Парсит аргументы командной строки."""
    parser = argparse.ArgumentParser(description="Обучение классификатора тракторов")
    parser.add_argument("--data_dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--image_size", type=int, default=DEFAULT_IMAGE_SIZE)
    parser.add_argument("--num_workers", type=int, default=DEFAULT_NUM_WORKERS)
    return parser.parse_args()


def create_dataloaders(
    args: argparse.Namespace,
) -> dict[str, torch.utils.data.DataLoader]:
    """Создаёт train и val dataloaders."""
    return {
        split: get_dataloader(
            data_dir=args.data_dir,
            split=split,
            batch_size=args.batch_size,
            image_size=args.image_size,
            num_workers=args.num_workers,
        )
        for split in SPLITS
    }


def create_callbacks() -> list[pl.Callback]:
    """Создаёт callbacks для сохранения лучшей модели и ранней остановки."""
    checkpoint_callback = ModelCheckpoint(
        dirpath=WEIGHTS_DIR,
        filename="best-{epoch:02d}-{val_acc:.3f}",
        monitor="val_acc",
        mode="max",
        save_top_k=1,
        verbose=True,
    )
    early_stopping = EarlyStopping(
        monitor="val_acc",
        patience=EARLY_STOPPING_PATIENCE,
        mode="max",
        verbose=True,
    )
    return [checkpoint_callback, early_stopping]


def create_trainer(
    args: argparse.Namespace, callbacks: list[pl.Callback]
) -> pl.Trainer:
    """Создаёт PyTorch Lightning trainer."""
    return pl.Trainer(
        max_epochs=args.epochs,
        accelerator="auto",
        devices=1,
        logger=False,
        callbacks=callbacks,
        log_every_n_steps=1,
        deterministic=True,
    )


def main():
    """Главная функция обучения."""
    args = parse_args()

    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

    loaders = create_dataloaders(args)

    model = TractorLightningModule(
        num_classes=len(CLASS_NAMES),
        lr=args.lr,
    )

    trainer = create_trainer(args, create_callbacks())
    trainer.fit(model, loaders["train"], loaders["val"])

    final_path = WEIGHTS_DIR / "final_model.ckpt"
    trainer.save_checkpoint(final_path)
    logger.info("Финальная модель сохранена: %s", final_path)


if __name__ == "__main__":
    main()

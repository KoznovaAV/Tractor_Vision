"""
Обучение Multi-Task модели с автоматической балансировкой через uncertainty.
"""

import argparse
import logging
from pathlib import Path

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from torch import nn
from torchmetrics import Accuracy

from src.data.dataloader import get_dataloader
from src.models.multi_task import MODEL_CLASSES, STATE_CLASSES, MultiTaskTractorClassifier

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


class MultiTaskLightningModule(pl.LightningModule):
    """Lightning-обёртка для Multi-Task модели с uncertainty weighting."""

    def __init__(self, lr: float = DEFAULT_LR):
        super().__init__()
        self.save_hyperparameters()

        self.model = MultiTaskTractorClassifier()

        self.model_criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        self.state_criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

        # Логи стандартных отклонений для автоматической балансировки
        self.log_var_model = nn.Parameter(torch.zeros(1))
        self.log_var_state = nn.Parameter(torch.zeros(1))

        # Метрики
        self.train_model_acc = Accuracy(
            task="multiclass", num_classes=len(MODEL_CLASSES)
        )
        self.val_model_acc = Accuracy(task="multiclass", num_classes=len(MODEL_CLASSES))
        self.train_state_acc = Accuracy(
            task="multiclass", num_classes=len(STATE_CLASSES)
        )
        self.val_state_acc = Accuracy(task="multiclass", num_classes=len(STATE_CLASSES))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.model(x)

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        images = batch["image"]
        model_labels = batch["model_label"]
        state_labels = batch["state_label"]

        model_logits, state_logits = self(images)

        model_loss = self.model_criterion(model_logits, model_labels)
        state_loss = self.state_criterion(state_logits, state_labels)

        # Автоматическая балансировка через uncertainty
        precision_model = torch.exp(-self.log_var_model)
        precision_state = torch.exp(-self.log_var_state)

        loss = (
            precision_model * model_loss
            + precision_state * state_loss
            + self.log_var_model
            + self.log_var_state
        )

        # Метрики
        model_preds = torch.softmax(model_logits, dim=1)
        state_preds = torch.softmax(state_logits, dim=1)

        self.train_model_acc(model_preds, model_labels)
        self.train_state_acc(state_preds, state_labels)

        self.log("train_loss", loss, prog_bar=True)
        self.log("train_model_acc", self.train_model_acc, prog_bar=True)
        self.log("train_state_acc", self.train_state_acc, prog_bar=True)
        self.log("log_var_model", self.log_var_model, prog_bar=False)
        self.log("log_var_state", self.log_var_state, prog_bar=False)

        return loss

    def validation_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        images = batch["image"]
        model_labels = batch["model_label"]
        state_labels = batch["state_label"]

        model_logits, state_logits = self(images)

        model_loss = self.model_criterion(model_logits, model_labels)
        state_loss = self.state_criterion(state_logits, state_labels)

        loss = model_loss + state_loss  # Для валидации используем равные веса

        model_preds = torch.softmax(model_logits, dim=1)
        state_preds = torch.softmax(state_logits, dim=1)

        self.val_model_acc(model_preds, model_labels)
        self.val_state_acc(state_preds, state_labels)

        self.log("val_loss", loss, prog_bar=True)
        self.log("val_model_acc", self.val_model_acc, prog_bar=True)
        self.log("val_state_acc", self.val_state_acc, prog_bar=True)

        return loss

    def configure_optimizers(self) -> dict:
        optimizer = torch.optim.AdamW(
            [
                {"params": self.model.model_head.parameters()},
                {"params": self.model.state_head.parameters()},
                {"params": [self.log_var_model, self.log_var_state]},
            ],
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
            "monitor": "val_model_acc",
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Обучение Multi-Task модели")
    parser.add_argument("--data_dir", type=Path, default=Path("data/dirty_clean"))
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--image_size", type=int, default=DEFAULT_IMAGE_SIZE)
    parser.add_argument("--num_workers", type=int, default=DEFAULT_NUM_WORKERS)
    return parser.parse_args()


def create_dataloaders(args: argparse.Namespace) -> dict:
    return {
        split: get_dataloader(
            data_dir=args.data_dir,
            split=split,
            batch_size=args.batch_size,
            image_size=args.image_size,
            num_workers=args.num_workers,
            multi_task=True,
        )
        for split in SPLITS
    }


def create_callbacks() -> list[pl.Callback]:
    checkpoint_callback = ModelCheckpoint(
        dirpath=WEIGHTS_DIR,
        filename="multi-task-{epoch:02d}-{val_model_acc:.3f}",
        monitor="val_model_acc",
        mode="max",
        save_top_k=1,
        verbose=True,
    )
    early_stopping = EarlyStopping(
        monitor="val_model_acc",
        patience=EARLY_STOPPING_PATIENCE,
        mode="max",
        verbose=True,
    )
    return [checkpoint_callback, early_stopping]


def main():
    args = parse_args()

    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

    loaders = create_dataloaders(args)

    model = MultiTaskLightningModule(lr=args.lr)

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator="auto",
        devices=1,
        logger=False,
        callbacks=create_callbacks(),
        log_every_n_steps=1,
        deterministic=True,
    )

    trainer.fit(model, loaders["train"], loaders["val"])

    final_path = WEIGHTS_DIR / "multi_task_final.ckpt"
    trainer.save_checkpoint(final_path)
    logger.info("Финальная модель сохранена: %s", final_path)


if __name__ == "__main__":
    main()

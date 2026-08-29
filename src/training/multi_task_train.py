#!/usr/bin/env python3
"""Обучение Multi-Task модели с partial fine-tuning и балансировкой потерь.

Обновлённая recipe относительно прежней версии (полностью замороженный backbone
+ только uncertainty weighting):

* **Partial fine-tuning** последних 1–2 стадий torchvision ConvNeXt-Tiny. Backbone
  ``features`` — ``Sequential`` из 8 модулей (0–7); стадии-блоки находятся на
  индексах 1, 3, 5, 7, между ними downsample-слои на чётных индексах.
  Разморозка последних ``NUM_UNFROZEN_STAGES`` стадий захватывает и их
  downsample-префиксы (``n=1`` -> ``features[6:8]``, ``n=2`` -> ``features[4:8]``).
* **Дифференцированный LR**: размороженный backbone учится с LR в
  ``BACKBONE_LR_FACTOR`` (=10) раз меньше, чем головы и параметры балансировки.
* **Балансировка потерь**: два взаимоисключающих режима, выбираемых флагом
  ``--loss-balancing``:
  * ``uncertainty`` — обучаемые ``log_var`` по Kendall et al. (2018);
  * ``gradnorm`` — динамическая нормировка градиентов по Chen et al. (2018)
    с целевым выравниванием скоростей обучения задач.

Скрипт не переписывает саму модель: он размораживает стадии через публичную
структуру ``model.backbone.features`` снаружи, поэтому совместим с существующим
``MultiTaskTractorClassifier``.

Пример::

    python -m src.training.multi_task_train \\
        --data-dir data/dirty_clean \\
        --num-unfrozen-stages 2 \\
        --loss-balancing gradnorm \\
        --image-size 384 \\
        --max-epochs 100
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from torch import nn
from torchmetrics import Accuracy

from src.config.classes import NUM_MODEL_CLASSES, NUM_STATE_CLASSES
from src.data.dataloader import get_dataloader
from src.models.multi_task import MultiTaskTractorClassifier

# Индексы стадий-блоков в torchvision ConvNeXt-Tiny ``features`` (0..7).
CONVNEXT_STAGE_BLOCK_INDICES: tuple[int, ...] = (1, 3, 5, 7)

# Гиперпараметры по умолчанию.
DEFAULT_LR: float = 5e-4
BACKBONE_LR_FACTOR: float = 0.1  # backbone учится в 10 раз медленнее голов
DEFAULT_BATCH_SIZE: int = 8
DEFAULT_MAX_EPOCHS: int = 100
DEFAULT_IMAGE_SIZE: int = 384
WEIGHT_DECAY: float = 1e-4
LABEL_SMOOTHING: float = 0.1
EARLY_STOPPING_PATIENCE: int = 15
LR_SCHEDULER_PATIENCE: int = 7
LR_REDUCE_FACTOR: float = 0.5
MIN_LR: float = 1e-6
GRADNORM_ALPHA: float = 1.5  # сила восстановления баланса в GradNorm


def unfreeze_last_stages(model: MultiTaskTractorClassifier, num_stages: int) -> list[int]:
    """Разморозить последние ``num_stages`` стадий backbone ConvNeXt-Tiny.

    Сначала весь backbone замораживается, затем размораживаются хвостовые стадии
    вместе с их downsample-префиксами. Головы модели (``model_head``,
    ``state_head``) не затрагиваются — они всегда обучаемы.

    Args:
        model: Multi-task модель с backbone ``features`` (Sequential из 8 модулей).
        num_stages: Сколько последних стадий разморозить (0, 1 или 2).

    Returns:
        Список индексов размороженных модулей ``features``.

    Raises:
        ValueError: Если ``num_stages`` вне диапазона ``[0, 4]``.
    """
    if not 0 <= num_stages <= len(CONVNEXT_STAGE_BLOCK_INDICES):
        raise ValueError(
            f"num_stages должно быть в [0, {len(CONVNEXT_STAGE_BLOCK_INDICES)}], "
            f"получено {num_stages}."
        )

    features: nn.Sequential = model.backbone.features
    for param in features.parameters():
        param.requires_grad_(False)

    if num_stages == 0:
        return []

    tail_stages = CONVNEXT_STAGE_BLOCK_INDICES[-num_stages:]
    first_stage_idx = tail_stages[0]
    # Начинаем с downsample-слоя перед первой размораживаемой стадией
    # (для первой стадии индекса 1 это stem features[0]).
    start_idx = max(0, first_stage_idx - 1)

    unfrozen: list[int] = []
    for idx in range(start_idx, len(features)):
        for param in features[idx].parameters():
            param.requires_grad_(True)
        unfrozen.append(idx)
    return unfrozen


class MultiTaskLightningModule(pl.LightningModule):
    """Lightning-модуль multi-task обучения с partial fine-tuning.

    Attributes:
        model: Обёрнутая multi-task модель.
        lr: Базовый LR для голов и параметров балансировки.
        loss_balancing: Режим балансировки потерь (``uncertainty``/``gradnorm``).
    """

    def __init__(
        self,
        num_unfrozen_stages: int = 2,
        lr: float = DEFAULT_LR,
        backbone_lr_factor: float = BACKBONE_LR_FACTOR,
        weight_decay: float = WEIGHT_DECAY,
        loss_balancing: str = "uncertainty",
        gradnorm_alpha: float = GRADNORM_ALPHA,
    ) -> None:
        """Инициализировать модуль.

        Args:
            num_unfrozen_stages: Число размораживаемых последних стадий backbone.
            lr: Базовый learning rate.
            backbone_lr_factor: Множитель LR для размороженного backbone.
            weight_decay: L2-регуляризация.
            loss_balancing: ``uncertainty`` или ``gradnorm``.
            gradnorm_alpha: Параметр восстановления баланса GradNorm.

        Raises:
            ValueError: При неизвестном режиме балансировки.
        """
        super().__init__()
        if loss_balancing not in ("uncertainty", "gradnorm"):
            raise ValueError(
                f"loss_balancing должно быть 'uncertainty' или 'gradnorm', "
                f"получено {loss_balancing!r}."
            )

        self.save_hyperparameters()
        self.model = MultiTaskTractorClassifier(
            num_model_classes=NUM_MODEL_CLASSES,
            num_state_classes=NUM_STATE_CLASSES,
        )
        self.unfrozen_indices = unfreeze_last_stages(self.model, num_unfrozen_stages)

        self.lr = lr
        self.backbone_lr_factor = backbone_lr_factor
        self.weight_decay = weight_decay
        self.loss_balancing = loss_balancing
        self.gradnorm_alpha = gradnorm_alpha

        self.model_criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
        self.state_criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)

        # --- Параметры балансировки потерь ---
        if self.loss_balancing == "uncertainty":
            # log(sigma^2) для каждой задачи (Kendall et al., 2018).
            self.log_var_model = nn.Parameter(torch.zeros(1))
            self.log_var_state = nn.Parameter(torch.zeros(1))
        else:
            # GradNorm: обучаемые веса задач и запомненные начальные потери.
            self.task_weights = nn.Parameter(torch.ones(2))
            self.initial_losses: torch.Tensor | None = None
            # Ручная оптимизация нужна, чтобы делать два backward (задачи + GradNorm).
            self.automatic_optimization = False

        # Метрики.
        self.train_model_acc = Accuracy(task="multiclass", num_classes=NUM_MODEL_CLASSES)
        self.val_model_acc = Accuracy(task="multiclass", num_classes=NUM_MODEL_CLASSES)
        self.val_state_acc = Accuracy(task="multiclass", num_classes=NUM_STATE_CLASSES)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Прямой проход.

        Args:
            x: Батч изображений ``(B, 3, H, W)``.

        Returns:
            Кортеж ``(model_logits, state_logits)``.
        """
        model_logits, state_logits = self.model(x)
        return model_logits, state_logits

    def _shared_step(
        self, batch: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Общий проход: логиты и покомпонентные потери.

        Args:
            batch: Батч с ключами ``image``, ``model_label``, ``state_label``.

        Returns:
            Кортеж ``(model_logits, state_logits, model_loss, state_loss)``.
        """
        images = batch["image"]
        model_labels = batch["model_label"]
        state_labels = batch["state_label"]

        model_logits, state_logits = self(images)
        model_loss = self.model_criterion(model_logits, model_labels)
        state_loss = self.state_criterion(state_logits, state_labels)
        return model_logits, state_logits, model_loss, state_loss

    def _combine_uncertainty(
        self, model_loss: torch.Tensor, state_loss: torch.Tensor
    ) -> torch.Tensor:
        """Взвесить потери через uncertainty weighting (Kendall et al., 2018).

        Args:
            model_loss: Потеря задачи классификации модели.
            state_loss: Потеря задачи классификации состояния.

        Returns:
            Суммарная взвешенная потеря.
        """
        precision_model = torch.exp(-self.log_var_model)
        precision_state = torch.exp(-self.log_var_state)
        return (
            precision_model * model_loss
            + precision_state * state_loss
            + self.log_var_model
            + self.log_var_state
        ).squeeze()

    def _gradnorm_step(
        self,
        model_loss: torch.Tensor,
        state_loss: torch.Tensor,
    ) -> torch.Tensor:
        """Выполнить шаг обучения в режиме GradNorm (ручная оптимизация).

        Реализует нормировку градиентов задач по Chen et al. (2018): веса задач
        подстраиваются так, чтобы скорости обучения задач выравнивались
        относительно среднего темпа.

        Args:
            model_loss: Потеря задачи модели.
            state_loss: Потеря задачи состояния.

        Returns:
            Итоговая взвешенная потель задач (для логирования).
        """
        optimizer = self.optimizers()
        assert isinstance(optimizer, torch.optim.Optimizer)

        task_losses = torch.stack([model_loss, state_loss])
        if self.initial_losses is None:
            # Фиксируем начальные потери задач для нормировки темпа.
            self.initial_losses = task_losses.detach().clamp(min=1e-8)

        weighted_losses = self.task_weights * task_losses
        total_loss = weighted_losses.sum()

        optimizer.zero_grad()
        # Сохраняем граф — далее считаем градиенты по общему слою для GradNorm.
        self.manual_backward(total_loss, retain_graph=True)

        # Общий слой для нормировки градиентов — последний размороженный блок
        # backbone (или головы, если backbone полностью заморожен).
        shared_params = self._gradnorm_shared_parameters()
        grad_norms = []
        for i in range(task_losses.shape[0]):
            grads = torch.autograd.grad(
                self.task_weights[i] * task_losses[i],
                shared_params,
                retain_graph=True,
                create_graph=True,
                allow_unused=True,
            )
            flat = torch.cat([g.flatten() for g in grads if g is not None])
            grad_norms.append(torch.norm(flat))
        grad_norm_tensor = torch.stack(grad_norms)

        # Целевые нормы по относительной скорости обучения задач.
        loss_ratios = task_losses.detach() / self.initial_losses
        inverse_rates = loss_ratios / loss_ratios.mean()
        target_norms = grad_norm_tensor.mean().detach() * inverse_rates**self.gradnorm_alpha
        gradnorm_loss = F.l1_loss(grad_norm_tensor, target_norms.detach())

        # Градиент GradNorm-потери только по весам задач.
        weight_grad = torch.autograd.grad(gradnorm_loss, self.task_weights)[0]
        self.task_weights.grad = weight_grad

        optimizer.step()

        # Перенормировка весов, чтобы их сумма оставалась равной числу задач.
        with torch.no_grad():
            self.task_weights.data = (
                self.task_weights.data / self.task_weights.data.sum() * task_losses.shape[0]
            )

        return total_loss.detach()

    def _gradnorm_shared_parameters(self) -> list[nn.Parameter]:
        """Выбрать общий набор параметров для нормировки градиентов GradNorm.

        Возвращает параметры последнего размороженного блока backbone; если
        backbone полностью заморожен, откатывается на объединённые головы.

        Returns:
            Список обучаемых параметров общего слоя.
        """
        if self.unfrozen_indices:
            last_idx = self.unfrozen_indices[-1]
            features: nn.Sequential = self.model.backbone.features
            params = [p for p in features[last_idx].parameters() if p.requires_grad]
            if params:
                return params
        return [
            p
            for head in (self.model.model_head, self.model.state_head)
            for p in head.parameters()
            if p.requires_grad
        ]

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor | None:
        """Шаг обучения (авто- или ручная оптимизация в зависимости от режима).

        Args:
            batch: Обучающий батч.
            batch_idx: Индекс батча.

        Returns:
            Потеря (для uncertainty) или ``None`` (для gradnorm с ручной опт.).
        """
        model_logits, _, model_loss, state_loss = self._shared_step(batch)
        self.train_model_acc(model_logits, batch["model_label"])

        if self.loss_balancing == "uncertainty":
            loss = self._combine_uncertainty(model_loss, state_loss)
            self.log("train_loss", loss, prog_bar=True)
            self.log("train_model_loss", model_loss)
            self.log("train_state_loss", state_loss)
            self.log("train_model_acc", self.train_model_acc, prog_bar=True)
            return loss

        # GradNorm — ручная оптимизация.
        total = self._gradnorm_step(model_loss, state_loss)
        self.log("train_loss", total, prog_bar=True)
        self.log("train_model_loss", model_loss)
        self.log("train_state_loss", state_loss)
        self.log("train_model_acc", self.train_model_acc, prog_bar=True)
        self.log("task_weight_model", self.task_weights[0])
        self.log("task_weight_state", self.task_weights[1])

        # При ручной оптимизации шаг планировщика делаем в конце эпохи.
        if self.trainer.is_last_batch:
            self._manual_scheduler_step()
        return None

    def _manual_scheduler_step(self) -> None:
        """Сделать шаг ReduceLROnPlateau вручную (режим GradNorm).

        В ручной оптимизации Lightning не вызывает планировщик сам, поэтому шаг
        по метрике ``val_model_acc`` выполняется здесь в конце эпохи.
        """
        schedulers = self.lr_schedulers()
        if schedulers is None:
            return
        scheduler = schedulers[0] if isinstance(schedulers, list) else schedulers
        metric = self.trainer.callback_metrics.get("val_model_acc")
        metric_value = float(metric) if metric is not None else 0.0
        # ReduceLROnPlateau.step принимает метрику; у него сигнатура step(metrics).
        scheduler.step(metric_value)  # type: ignore[call-arg]

    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> None:
        """Шаг валидации: считает потери и метрики обеих задач.

        Args:
            batch: Валидационный батч.
            batch_idx: Индекс батча.
        """
        model_logits, state_logits, model_loss, state_loss = self._shared_step(batch)

        if self.loss_balancing == "uncertainty":
            val_loss = self._combine_uncertainty(model_loss, state_loss)
        else:
            val_loss = (self.task_weights.detach() * torch.stack([model_loss, state_loss])).sum()

        self.val_model_acc(model_logits, batch["model_label"])
        self.val_state_acc(state_logits, batch["state_label"])
        self.log("val_loss", val_loss, prog_bar=True)
        self.log("val_model_acc", self.val_model_acc, prog_bar=True)
        self.log("val_state_acc", self.val_state_acc, prog_bar=True)

    def configure_optimizers(self) -> dict[str, Any]:
        """Собрать оптимизатор с дифференцированным LR и планировщик.

        Параметры разбиваются на две группы: размороженный backbone (LR в
        ``backbone_lr_factor`` раз меньше) и всё остальное (головы + параметры
        балансировки) на базовом LR.

        Returns:
            Словарь конфигурации оптимизатора для Lightning.
        """
        backbone_params = [p for p in self.model.backbone.parameters() if p.requires_grad]
        backbone_ids = {id(p) for p in backbone_params}
        other_params = [
            p for p in self.parameters() if p.requires_grad and id(p) not in backbone_ids
        ]

        param_groups = [{"params": other_params, "lr": self.lr}]
        if backbone_params:
            param_groups.append(
                {
                    "params": backbone_params,
                    "lr": self.lr * self.backbone_lr_factor,
                }
            )

        optimizer = torch.optim.AdamW(param_groups, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=LR_REDUCE_FACTOR,
            patience=LR_SCHEDULER_PATIENCE,
            min_lr=MIN_LR,
        )

        # При ручной оптимизации (GradNorm) Lightning не управляет планировщиком
        # автоматически — шаг делается вручную в training_step.
        if not self.automatic_optimization:
            return {"optimizer": optimizer, "lr_scheduler": scheduler}

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_model_acc",
            },
        }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Разобрать аргументы командной строки.

    Args:
        argv: Аргументы (по умолчанию ``sys.argv[1:]``).

    Returns:
        Пространство имён с аргументами.
    """
    parser = argparse.ArgumentParser(
        description="Обучение Multi-Task модели с partial fine-tuning.",
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data/dirty_clean"))
    parser.add_argument("--weights-dir", type=Path, default=Path("weights"))
    parser.add_argument(
        "--num-unfrozen-stages",
        type=int,
        default=2,
        help="Число размораживаемых последних стадий backbone (0-2 рекоменд.).",
    )
    parser.add_argument(
        "--loss-balancing",
        type=str,
        choices=["uncertainty", "gradnorm"],
        default="uncertainty",
    )
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--max-epochs", type=int, default=DEFAULT_MAX_EPOCHS)
    parser.add_argument("--image-size", type=int, default=DEFAULT_IMAGE_SIZE)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument(
        "--accelerator",
        type=str,
        default="auto",
        help="Ускоритель Lightning (auto/gpu/cpu).",
    )
    return parser.parse_args(argv)


def create_callbacks(weights_dir: Path) -> list[pl.Callback]:
    """Создать колбэки обучения.

    Args:
        weights_dir: Директория для сохранения чекпоинтов.

    Returns:
        Список колбэков Lightning.
    """
    return [
        ModelCheckpoint(
            dirpath=weights_dir,
            filename="multi-task-best-{epoch:02d}-{val_model_acc:.3f}",
            monitor="val_model_acc",
            mode="max",
            save_top_k=1,
            save_last=True,
        ),
        EarlyStopping(
            monitor="val_model_acc",
            patience=EARLY_STOPPING_PATIENCE,
            mode="max",
            verbose=True,
        ),
        LearningRateMonitor(logging_interval="epoch"),
    ]


def main(argv: list[str] | None = None) -> int:
    """Точка входа обучения.

    Args:
        argv: Аргументы командной строки (для тестируемости).

    Returns:
        Код возврата процесса (0 — успех).
    """
    args = parse_args(argv)
    args.weights_dir.mkdir(parents=True, exist_ok=True)

    train_loader = get_dataloader(
        data_dir=args.data_dir,
        split="train",
        batch_size=args.batch_size,
        image_size=args.image_size,
        num_workers=args.num_workers,
        multi_task=True,
    )
    val_loader = get_dataloader(
        data_dir=args.data_dir,
        split="val",
        batch_size=args.batch_size,
        image_size=args.image_size,
        num_workers=args.num_workers,
        multi_task=True,
    )

    module = MultiTaskLightningModule(
        num_unfrozen_stages=args.num_unfrozen_stages,
        lr=args.lr,
        loss_balancing=args.loss_balancing,
    )
    print(
        f"Разморожены стадии backbone (features): {module.unfrozen_indices} | "
        f"балансировка: {args.loss_balancing}"
    )

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator=args.accelerator,
        callbacks=create_callbacks(args.weights_dir),
        log_every_n_steps=10,
    )
    trainer.fit(module, train_loader, val_loader)

    final_path = args.weights_dir / "multi_task_final.ckpt"
    trainer.save_checkpoint(final_path)
    print(f"Финальный чекпоинт сохранён: {final_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

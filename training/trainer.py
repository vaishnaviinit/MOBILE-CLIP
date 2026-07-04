"""
Main Trainer class — orchestrates the full training loop.

Responsibilities:
  - Phase 1 (epochs 1..freeze_backbone_epochs): frozen backbone, head-only training
  - Phase 2 (remaining epochs): differential LR fine-tuning of full model
  - Mixed precision (torch.amp.autocast + GradScaler)
  - Gradient clipping (after unscale, before optimizer step)
  - Per-epoch train + val loops with full metric logging
  - CSV + TensorBoard logging
  - Checkpoint saving (last_checkpoint.pt every epoch, best_model.pt on improvement)
  - Early stopping monitored on val_f2 (recall-heavy F-score)
  - Resume from any checkpoint
"""

from __future__ import annotations

import csv
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader

from models.classifier import PhishingClassifier
from models.focal_loss import build_loss
from training.callbacks import EarlyStopping, ModelCheckpoint, LRMonitor, CallbackResult
from utils.device import resolve_device, get_device_info, memory_summary

logger = logging.getLogger(__name__)


@dataclass
class TrainerConfig:
    """All training hyperparameters in one dataclass."""

    epochs: int = 50
    freeze_backbone_epochs: int = 5

    # Phase 1 (linear probe): head only
    lr_head: float = 1e-3

    # Phase 2 (fine-tune): differential LR
    lr_head_phase2: float = 1e-4   # reduced from phase 1
    lr_backbone: float = 1e-5

    weight_decay: float = 0.01
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8

    warmup_epochs: int = 2
    min_lr: float = 1e-7

    mixed_precision: bool = True
    gradient_clip: float = 1.0

    device: str = "auto"
    seed: int = 42

    checkpoint_dir: str = "outputs/checkpoints"
    log_dir: str = "outputs/logs"

    early_stopping_patience: int = 10
    early_stopping_monitor: str = "val_f2"

    loss_name: str = "focal"
    focal_gamma: float = 2.0
    label_smoothing: float = 0.0

    class_weights: Optional[Tensor] = field(default=None, repr=False)


class Trainer:
    """
    Trainer for PhishingClassifier.

    Usage:
        trainer = Trainer(model, train_loader, val_loader, cfg)
        result = trainer.train()
        # Resume:
        result = trainer.train(resume_from="outputs/checkpoints/last_checkpoint.pt")

    Args:
        model:        PhishingClassifier instance.
        train_loader: DataLoader for training split.
        val_loader:   DataLoader for validation split.
        cfg:          TrainerConfig dataclass.
    """

    def __init__(
        self,
        model: PhishingClassifier,
        train_loader: DataLoader,
        val_loader: DataLoader,
        cfg: TrainerConfig,
    ) -> None:
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.cfg = cfg

        self.device = resolve_device(cfg.device)
        self.model.to(self.device)

        # GradScaler only meaningful on CUDA; disabled on CPU/MPS
        amp_enabled = cfg.mixed_precision and self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
        self._amp_enabled = amp_enabled
        self._amp_device_type = self.device.type if self.device.type == "cuda" else "cpu"

        self._optimizer: Optional[Optimizer] = None
        self._scheduler: Optional[LRScheduler] = None
        self._loss_fn: Optional[nn.Module] = None
        self._start_epoch: int = 0
        self._in_phase2: bool = False

        self._early_stopping_cb: EarlyStopping
        self._checkpoint_cb: ModelCheckpoint
        self._lr_monitor: LRMonitor
        self._setup_callbacks()

        self._csv_path: Optional[Path] = None
        self._tb_writer = None
        self._setup_loggers()

        # Log device info
        info = get_device_info(self.device)
        logger.info(
            "Trainer ready | device=%s (%s) | AMP=%s",
            info["device"],
            info.get("name", ""),
            self._amp_enabled,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train(self, resume_from: Optional[str] = None) -> dict:
        """
        Run the full training loop.

        Two-phase strategy:
          Phase 1: epochs 1..freeze_backbone_epochs — backbone frozen, head learns.
          Phase 2: epochs (freeze_backbone_epochs+1)..epochs — full model with
                   warmup + cosine scheduler and differential LR.

        Args:
            resume_from: Path to last_checkpoint.pt to resume from. The Trainer
                         restores epoch, optimizer, scheduler, and callback states.

        Returns:
            {
              "best_checkpoint": str path to best_model.pt,
              "final_metrics":   dict of last epoch's metrics,
              "history":         list of per-epoch metric dicts,
            }
        """
        history: list[dict] = []

        # Build loss (done here so class_weights can be moved to device)
        self._loss_fn = self._build_loss()
        self._loss_fn.to(self.device)

        # Restore from checkpoint if requested
        if resume_from is not None:
            self._start_epoch = self._load_checkpoint(resume_from)
            logger.info("Resuming from epoch %d", self._start_epoch + 1)
        else:
            # Fresh run — start Phase 1
            self._setup_phase1()

        best_checkpoint_path: Optional[str] = None

        logger.info(
            "Training | epochs=%d | phase1=%d | warmup=%d | device=%s",
            self.cfg.epochs,
            self.cfg.freeze_backbone_epochs,
            self.cfg.warmup_epochs,
            self.device,
        )

        for epoch in range(self._start_epoch + 1, self.cfg.epochs + 1):

            # ── Phase transition ──────────────────────────────────────
            if epoch == self.cfg.freeze_backbone_epochs + 1 and not self._in_phase2:
                logger.info("━" * 60)
                logger.info("Epoch %d: switching to Phase 2 (full fine-tuning)", epoch)
                self._setup_phase2()

            # ── Train ─────────────────────────────────────────────────
            train_metrics = self._train_epoch(epoch)

            # ── Validate ──────────────────────────────────────────────
            val_metrics = self._val_epoch(epoch)

            # ── Merge + LR ────────────────────────────────────────────
            epoch_metrics: dict = {"epoch": epoch, **train_metrics, **val_metrics}
            lr_info = self._lr_monitor.step(epoch, self._optimizer, epoch_metrics)
            epoch_metrics.update(lr_info)

            # ── Scheduler step ────────────────────────────────────────
            self._scheduler.step()

            # ── Log ───────────────────────────────────────────────────
            self._log_epoch(epoch, epoch_metrics)
            history.append(epoch_metrics)

            # ── Checkpoint callback ───────────────────────────────────
            ckpt_result = self._checkpoint_cb.step(epoch, epoch_metrics)
            self._save_checkpoint(epoch, epoch_metrics, is_best=ckpt_result.is_best)
            if ckpt_result.is_best:
                best_checkpoint_path = str(
                    Path(self.cfg.checkpoint_dir) / "best_model.pt"
                )

            # ── Early stopping ────────────────────────────────────────
            es_result = self._early_stopping_cb.step(epoch, epoch_metrics)

            # ── Console summary ───────────────────────────────────────
            self._print_epoch_summary(epoch, epoch_metrics, ckpt_result, es_result)

            if es_result.stop_training:
                logger.info("Early stopping — training complete at epoch %d.", epoch)
                break

        self._finalize_logging()

        return {
            "best_checkpoint": best_checkpoint_path,
            "final_metrics": history[-1] if history else {},
            "history": history,
        }

    # ------------------------------------------------------------------
    # Per-epoch loops
    # ------------------------------------------------------------------

    def _train_epoch(self, epoch: int) -> dict[str, float]:
        """
        One full pass over the training DataLoader.

        Returns dict with keys: train_loss, train_acc.
        """
        self.model.train()
        total_loss = 0.0
        n_correct = 0
        n_total = 0

        for images, labels in self.train_loader:
            images: Tensor = images.to(self.device, non_blocking=True)
            labels: Tensor = labels.to(self.device, non_blocking=True)

            self._optimizer.zero_grad(set_to_none=True)

            # Mixed precision forward
            with torch.amp.autocast(
                device_type=self._amp_device_type,
                enabled=self._amp_enabled,
            ):
                outputs = self.model(images)
                loss: Tensor = self._loss_fn(outputs["logits"], labels)

            # Scaled backward
            self.scaler.scale(loss).backward()

            # Unscale before clipping (required by GradScaler contract)
            self.scaler.unscale_(self._optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.cfg.gradient_clip
            )

            self.scaler.step(self._optimizer)
            self.scaler.update()

            total_loss += loss.item()

            # Accuracy (at default 0.5 threshold — for training monitoring only)
            with torch.no_grad():
                preds = outputs["probs"][:, 1] >= 0.5
                n_correct += (preds == labels.bool()).sum().item()
                n_total += labels.size(0)

        avg_loss = total_loss / max(len(self.train_loader), 1)
        acc = n_correct / max(n_total, 1)

        return {"train_loss": avg_loss, "train_acc": acc}

    def _val_epoch(self, epoch: int) -> dict[str, float]:
        """
        One full pass over the validation DataLoader.

        Returns dict with keys: val_loss, val_acc, val_precision, val_recall,
        val_f1, val_f2, val_roc_auc.

        Metrics are computed using sklearn at threshold=0.5 for consistency.
        Full threshold optimization happens in the Evaluator (Step 6).
        """
        from sklearn.metrics import (
            accuracy_score,
            f1_score,
            fbeta_score,
            precision_score,
            recall_score,
            roc_auc_score,
        )

        self.model.eval()
        total_loss = 0.0
        all_probs: list[float] = []
        all_labels: list[int] = []

        with torch.no_grad():
            for images, labels in self.val_loader:
                images = images.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)

                with torch.amp.autocast(
                    device_type=self._amp_device_type,
                    enabled=self._amp_enabled,
                ):
                    outputs = self.model(images)
                    loss = self._loss_fn(outputs["logits"], labels)

                total_loss += loss.item()
                all_probs.extend(outputs["probs"][:, 1].cpu().tolist())
                all_labels.extend(labels.cpu().tolist())

        avg_loss = total_loss / max(len(self.val_loader), 1)

        probs = np.array(all_probs)
        labels_arr = np.array(all_labels)
        preds = (probs >= 0.5).astype(int)

        # Graceful handling of degenerate batches (all one class)
        kw = {"zero_division": 0}
        acc = float(accuracy_score(labels_arr, preds))
        prec = float(precision_score(labels_arr, preds, **kw))
        rec = float(recall_score(labels_arr, preds, **kw))
        f1 = float(f1_score(labels_arr, preds, **kw))
        f2 = float(fbeta_score(labels_arr, preds, beta=2, **kw))

        if len(np.unique(labels_arr)) > 1:
            roc_auc = float(roc_auc_score(labels_arr, probs))
        else:
            roc_auc = 0.0
            logger.debug("val_roc_auc: only one class present — set to 0.0")

        return {
            "val_loss": avg_loss,
            "val_acc": acc,
            "val_precision": prec,
            "val_recall": rec,
            "val_f1": f1,
            "val_f2": f2,
            "val_roc_auc": roc_auc,
        }

    # ------------------------------------------------------------------
    # Setup helpers
    # ------------------------------------------------------------------

    def _setup_phase1(self) -> None:
        """
        Phase 1: freeze backbone, train head only with constant LR.

        No warmup is needed here — the head is randomly initialized
        and benefits from training at full lr_head from the start.
        """
        logger.info("━" * 60)
        logger.info("Phase 1: Linear Probing (backbone FROZEN)")

        self.model.backbone.freeze()
        self._in_phase2 = False

        param_groups = self.model.get_optimizer_param_groups(
            lr_head=self.cfg.lr_head,
            lr_backbone=0.0,       # ignored — backbone is frozen
            weight_decay=self.cfg.weight_decay,
        )
        self._optimizer = self._build_optimizer(param_groups)

        # Constant LR for the linear probe phase
        self._scheduler = torch.optim.lr_scheduler.LambdaLR(
            self._optimizer, lr_lambda=lambda _epoch: 1.0
        )

    def _setup_phase2(self) -> None:
        """
        Phase 2: unfreeze backbone, rebuild optimizer with differential LR,
        and attach warmup + cosine scheduler.

        Head uses lr_head_phase2 (10x lower than phase 1) because the head
        has already adapted in phase 1 and only needs fine adjustment.
        Backbone uses lr_backbone (100x lower than phase 1 lr_head).
        """
        logger.info("Phase 2: Full Fine-Tuning (differential LR)")
        logger.info(
            "  lr_head=%.1e  lr_backbone=%.1e  warmup=%d epochs",
            self.cfg.lr_head_phase2,
            self.cfg.lr_backbone,
            self.cfg.warmup_epochs,
        )

        self.model.backbone.unfreeze()
        self._in_phase2 = True

        param_groups = self.model.get_optimizer_param_groups(
            lr_head=self.cfg.lr_head_phase2,
            lr_backbone=self.cfg.lr_backbone,
            weight_decay=self.cfg.weight_decay,
        )
        self._optimizer = self._build_optimizer(param_groups)

        remaining = self.cfg.epochs - self.cfg.freeze_backbone_epochs
        self._scheduler = self._build_scheduler(
            self._optimizer,
            total_epochs=max(remaining, 1),
            warmup_epochs=self.cfg.warmup_epochs,
        )

    def _build_optimizer(self, param_groups: list[dict]) -> Optimizer:
        """Construct AdamW with given parameter groups."""
        return torch.optim.AdamW(
            param_groups,
            betas=self.cfg.betas,
            eps=self.cfg.eps,
            # LR and weight_decay are set per group in param_groups
        )

    def _build_scheduler(
        self,
        optimizer: Optimizer,
        total_epochs: int,
        warmup_epochs: int,
    ) -> LRScheduler:
        """
        Construct Linear Warmup → CosineAnnealingLR scheduler.

        LinearLR ramps from (1/warmup_epochs) × base_lr to base_lr over
        warmup_epochs steps. CosineAnnealingLR then decays from base_lr
        to eta_min=min_lr over the remaining epochs.

        SequentialLR combines both with the milestone at warmup_epochs.
        """
        if warmup_epochs > 0:
            warmup = torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=1.0 / max(warmup_epochs, 1),
                end_factor=1.0,
                total_iters=warmup_epochs,
            )
            cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=max(total_epochs - warmup_epochs, 1),
                eta_min=self.cfg.min_lr,
            )
            return torch.optim.lr_scheduler.SequentialLR(
                optimizer,
                schedulers=[warmup, cosine],
                milestones=[warmup_epochs],
            )

        # No warmup — pure cosine decay
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=total_epochs,
            eta_min=self.cfg.min_lr,
        )

    def _build_loss(self) -> nn.Module:
        """Build loss function from TrainerConfig."""
        loss_cfg = {
            "name": self.cfg.loss_name,
            "gamma": self.cfg.focal_gamma,
            "label_smoothing": self.cfg.label_smoothing,
        }
        return build_loss(loss_cfg, self.cfg.class_weights)

    def _setup_callbacks(self) -> None:
        """Instantiate EarlyStopping, ModelCheckpoint, and LRMonitor."""
        self._early_stopping_cb = EarlyStopping(
            monitor=self.cfg.early_stopping_monitor,
            patience=self.cfg.early_stopping_patience,
            mode="max",
            min_delta=1e-4,
        )
        self._checkpoint_cb = ModelCheckpoint(
            checkpoint_dir=self.cfg.checkpoint_dir,
            monitor=self.cfg.early_stopping_monitor,
            mode="max",
        )
        self._lr_monitor = LRMonitor()

    def _setup_loggers(self) -> None:
        """Set up CSV writer path and TensorBoard SummaryWriter."""
        log_dir = Path(self.cfg.log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        self._csv_path = log_dir / "training_logs.csv"

        try:
            from torch.utils.tensorboard import SummaryWriter
            tb_dir = log_dir / "tensorboard"
            self._tb_writer = SummaryWriter(log_dir=str(tb_dir))
            logger.info("TensorBoard → %s  (run: tensorboard --logdir %s)", tb_dir, tb_dir)
        except ImportError:
            logger.info("tensorboard not installed — skipping TB logging")
            self._tb_writer = None

    # ------------------------------------------------------------------
    # Checkpoint I/O
    # ------------------------------------------------------------------

    def _save_checkpoint(
        self, epoch: int, metrics: dict, is_best: bool
    ) -> None:
        """
        Save last_checkpoint.pt every epoch.
        Save best_model.pt when is_best=True.

        The checkpoint includes full training state so that training can be
        resumed exactly from any epoch.
        """
        checkpoint_dir = Path(self.cfg.checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        state = {
            "epoch": epoch,
            "model_state": self.model.state_dict(),
            "optimizer_state": self._optimizer.state_dict() if self._optimizer else None,
            "scheduler_state": self._scheduler.state_dict() if self._scheduler else None,
            "scaler_state": self.scaler.state_dict(),
            "metrics": metrics,
            "phase": 2 if self._in_phase2 else 1,
            "early_stopping": self._early_stopping_cb.state_dict(),
            "model_checkpoint": self._checkpoint_cb.state_dict(),
        }

        last_path = checkpoint_dir / "last_checkpoint.pt"
        torch.save(state, last_path)

        if is_best:
            best_path = checkpoint_dir / "best_model.pt"
            torch.save(state, best_path)
            logger.info(
                "Saved best_model.pt | epoch=%d | %s=%.4f",
                epoch,
                self.cfg.early_stopping_monitor,
                metrics.get(self.cfg.early_stopping_monitor, float("nan")),
            )

    def _load_checkpoint(self, path: str) -> int:
        """
        Load checkpoint and restore all training states.

        Returns:
            The epoch stored in the checkpoint (training resumes from epoch+1).
        """
        ckpt_path = Path(path)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path.resolve()}")

        state = torch.load(ckpt_path, map_location="cpu", weights_only=False)

        # Restore model
        self.model.load_state_dict(state["model_state"])
        self.model.to(self.device)

        # Restore phase and rebuild optimizer/scheduler accordingly
        phase = state.get("phase", 1)
        if phase == 2:
            self._setup_phase2()
        else:
            self._setup_phase1()

        # Restore optimizer + scheduler state
        if self._optimizer and state.get("optimizer_state"):
            self._optimizer.load_state_dict(state["optimizer_state"])
        if self._scheduler and state.get("scheduler_state"):
            self._scheduler.load_state_dict(state["scheduler_state"])
        if state.get("scaler_state"):
            self.scaler.load_state_dict(state["scaler_state"])

        # Restore callback states
        if "early_stopping" in state:
            self._early_stopping_cb.load_state_dict(state["early_stopping"])
        if "model_checkpoint" in state:
            self._checkpoint_cb.load_state_dict(state["model_checkpoint"])

        epoch: int = state.get("epoch", 0)
        logger.info(
            "Resumed from %s | epoch=%d | phase=%d | best_%s=%.4f",
            ckpt_path.name,
            epoch,
            phase,
            self.cfg.early_stopping_monitor,
            state.get("metrics", {}).get(self.cfg.early_stopping_monitor, float("nan")),
        )
        return epoch

    # ------------------------------------------------------------------
    # Logging helpers
    # ------------------------------------------------------------------

    def _log_epoch(self, epoch: int, metrics: dict) -> None:
        """Write metrics to CSV and TensorBoard."""
        # ── CSV ──────────────────────────────────────────────────────
        if self._csv_path is not None:
            # Flatten to only serialisable scalar types
            row = {
                k: v for k, v in metrics.items()
                if isinstance(v, (int, float))
            }
            write_header = not self._csv_path.exists()
            with open(self._csv_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=sorted(row.keys()))
                if write_header:
                    writer.writeheader()
                writer.writerow(row)

        # ── TensorBoard ───────────────────────────────────────────────
        if self._tb_writer is not None:
            for key, value in metrics.items():
                if isinstance(value, (int, float)) and not math.isnan(value):
                    self._tb_writer.add_scalar(key, value, epoch)

    def _print_epoch_summary(
        self,
        epoch: int,
        metrics: dict,
        ckpt_result: CallbackResult,
        es_result: CallbackResult,
    ) -> None:
        """Compact one-line epoch summary to console."""
        phase_tag = "P2" if self._in_phase2 else "P1"
        best_tag = " ★" if ckpt_result.is_best else ""
        stop_tag = " [STOP]" if es_result.stop_training else ""

        mem = memory_summary(self.device)
        mem_str = f" | {mem}" if mem else ""

        logger.info(
            "Epoch %3d/%d [%s]%s  "
            "loss=%.4f/%.4f  acc=%.3f/%.3f  "
            "rec=%.3f  f2=%.3f  auc=%.3f%s%s",
            epoch,
            self.cfg.epochs,
            phase_tag,
            best_tag,
            metrics.get("train_loss", 0),
            metrics.get("val_loss", 0),
            metrics.get("train_acc", 0),
            metrics.get("val_acc", 0),
            metrics.get("val_recall", 0),
            metrics.get("val_f2", 0),
            metrics.get("val_roc_auc", 0),
            mem_str,
            stop_tag,
        )

    def _finalize_logging(self) -> None:
        """Flush and close TensorBoard writer."""
        if self._tb_writer is not None:
            self._tb_writer.flush()
            self._tb_writer.close()

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_device(device: str) -> torch.device:
        """Resolve "auto" to available device (delegates to utils.device)."""
        return resolve_device(device)

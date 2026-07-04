"""
Training callbacks: EarlyStopping, ModelCheckpoint, LRMonitor.

Callbacks are called by the Trainer at the end of each epoch.
They are stateful objects that track history across epochs.

Design: callbacks do NOT have direct access to the Trainer — they receive
only the epoch index and the metrics dict. This keeps them decoupled and
independently testable.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class CallbackResult:
    """Returned by each callback after an epoch."""

    stop_training: bool = False
    save_checkpoint: bool = False
    is_best: bool = False
    message: str = ""


class EarlyStopping:
    """
    Stop training when a monitored metric stops improving.

    The counter increments every epoch without a qualifying improvement.
    When counter >= patience, stop_training=True is returned.

    Args:
        monitor:   Metric key in the epoch metrics dict (e.g. "val_f2").
        patience:  Number of epochs without improvement before stopping.
        mode:      "max" (higher is better) or "min" (lower is better).
        min_delta: Minimum absolute change to count as improvement.

    Note:
        Monitors "val_f2" by default — F2 score weights recall 4× over
        precision, which aligns with the security requirement of minimizing
        False Negatives (phishing pages that slip through as legitimate).
    """

    def __init__(
        self,
        monitor: str = "val_f2",
        patience: int = 10,
        mode: str = "max",
        min_delta: float = 1e-4,
    ) -> None:
        if mode not in ("max", "min"):
            raise ValueError(f"mode must be 'max' or 'min', got '{mode}'")

        self.monitor = monitor
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta

        self._counter: int = 0
        self._best_value: float = float("-inf") if mode == "max" else float("inf")

    def step(self, epoch: int, metrics: dict[str, float]) -> CallbackResult:
        """
        Evaluate this epoch's metrics and decide whether to stop.

        Args:
            epoch:   Current epoch number (1-indexed).
            metrics: Dict of metric_name → float from the validation loop.

        Returns:
            CallbackResult with stop_training=True if patience is exhausted.
        """
        if self.monitor not in metrics:
            logger.warning(
                "EarlyStopping: monitor metric '%s' not in metrics dict. "
                "Available keys: %s",
                self.monitor,
                list(metrics.keys()),
            )
            return CallbackResult()

        current = float(metrics[self.monitor])

        if self._is_improvement(current):
            self._best_value = current
            self._counter = 0
            return CallbackResult(
                message=f"{self.monitor} improved to {current:.4f}"
            )

        self._counter += 1
        msg = (
            f"No improvement in {self.monitor} for {self._counter}/{self.patience} epochs "
            f"(best={self._best_value:.4f}, current={current:.4f})"
        )
        logger.debug("EarlyStopping: %s", msg)

        if self._counter >= self.patience:
            logger.info(
                "EarlyStopping triggered after %d epochs without improvement "
                "(best %s=%.4f)",
                self.patience,
                self.monitor,
                self._best_value,
            )
            return CallbackResult(stop_training=True, message=msg)

        return CallbackResult(message=msg)

    def _is_improvement(self, current: float) -> bool:
        """Return True if current value is a meaningful improvement over best."""
        if self.mode == "max":
            return current > self._best_value + self.min_delta
        return current < self._best_value - self.min_delta

    def state_dict(self) -> dict:
        """Serialize state for checkpoint saving."""
        return {"counter": self._counter, "best_value": self._best_value}

    def load_state_dict(self, state: dict) -> None:
        """Restore state from checkpoint."""
        self._counter = state["counter"]
        self._best_value = state["best_value"]


class ModelCheckpoint:
    """
    Track the best model checkpoint and decide when to save.

    The Trainer always saves last_checkpoint.pt every epoch.
    This callback tells the Trainer when the current epoch is also
    the best (i.e., the monitor metric improved), triggering a
    separate save to best_model.pt.

    Args:
        checkpoint_dir:      Directory to write checkpoint files.
        monitor:             Metric key to compare for "best" checkpoint.
        mode:                "max" or "min".
        save_every_n_epochs: If > 0, also flags a periodic checkpoint save.
    """

    def __init__(
        self,
        checkpoint_dir: str | Path,
        monitor: str = "val_f2",
        mode: str = "max",
        save_every_n_epochs: int = 0,
    ) -> None:
        if mode not in ("max", "min"):
            raise ValueError(f"mode must be 'max' or 'min', got '{mode}'")

        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.monitor = monitor
        self.mode = mode
        self.save_every_n_epochs = save_every_n_epochs
        self._best_value: float = float("-inf") if mode == "max" else float("inf")

    def step(self, epoch: int, metrics: dict[str, float]) -> CallbackResult:
        """
        Decide whether this epoch should be saved as the best checkpoint.

        Always returns save_checkpoint=True (Trainer always saves last.pt).
        Sets is_best=True only when the monitored metric improves.

        Args:
            epoch:   Current epoch number (1-indexed).
            metrics: Epoch metrics dict.

        Returns:
            CallbackResult with is_best set correctly.
        """
        is_best = False
        message = ""

        if self.monitor in metrics:
            current = float(metrics[self.monitor])
            if self._is_improvement(current):
                self._best_value = current
                is_best = True
                message = (
                    f"Best {self.monitor}={current:.4f} — saving best_model.pt"
                )
                logger.info("ModelCheckpoint: %s", message)
            else:
                logger.debug(
                    "ModelCheckpoint: %s did not improve (best=%.4f, current=%.4f)",
                    self.monitor,
                    self._best_value,
                    current,
                )

        # Also flag periodic saves
        save_periodic = (
            self.save_every_n_epochs > 0
            and epoch % self.save_every_n_epochs == 0
        )

        return CallbackResult(
            save_checkpoint=True,
            is_best=is_best or save_periodic,
            message=message,
        )

    def _is_improvement(self, current: float) -> bool:
        """Return True if current metric is the new best."""
        if self.mode == "max":
            return current > self._best_value
        return current < self._best_value

    @property
    def best_value(self) -> float:
        """Best metric value seen so far."""
        return self._best_value

    def state_dict(self) -> dict:
        """Serialize for checkpoint inclusion."""
        return {"best_value": self._best_value}

    def load_state_dict(self, state: dict) -> None:
        """Restore from checkpoint."""
        self._best_value = state["best_value"]


class LRMonitor:
    """
    Log learning rate(s) at the end of each epoch.

    Reads LR from the optimizer's param groups by name. If the group
    has a "name" key (set in get_optimizer_param_groups), it's used as
    the metric key prefix; otherwise falls back to "lr_group_{i}".
    """

    def __init__(self) -> None:
        self._history: list[dict[str, float]] = []

    def step(
        self,
        epoch: int,
        optimizer,
        metrics: dict[str, float],
    ) -> dict[str, float]:
        """
        Extract LR from all optimizer param groups.

        Args:
            epoch:     Current epoch number.
            optimizer: The active optimizer.
            metrics:   Current epoch metrics (not used, included for API symmetry).

        Returns:
            Dict of {"lr_{group_name}": float, ...} for all param groups.
        """
        lr_dict: dict[str, float] = {}

        for i, group in enumerate(optimizer.param_groups):
            group_name = group.get("name", f"group_{i}")
            lr_dict[f"lr_{group_name}"] = float(group["lr"])

        self._history.append(lr_dict)
        return lr_dict

    @property
    def history(self) -> list[dict[str, float]]:
        """Full LR history across all logged epochs."""
        return self._history

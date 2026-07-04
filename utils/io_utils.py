"""
Checkpoint and JSON I/O utilities.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

logger = logging.getLogger(__name__)


def save_checkpoint(
    path: str | Path,
    epoch: int,
    model: nn.Module,
    optimizer: Optimizer,
    scheduler: Optional[LRScheduler],
    scaler,
    metrics: dict[str, float],
    config: dict,
    extra: Optional[dict] = None,
) -> None:
    """Save a full training checkpoint to path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict() if scheduler else None,
        "scaler_state": scaler.state_dict() if scaler else None,
        "metrics": metrics,
        "config": config,
    }
    if extra:
        checkpoint.update(extra)

    torch.save(checkpoint, path)
    logger.debug("Checkpoint saved → %s (epoch=%d)", path, epoch)


def load_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: Optional[Optimizer] = None,
    scheduler: Optional[LRScheduler] = None,
    scaler=None,
    device: str = "cpu",
) -> dict:
    """
    Load a checkpoint and restore all states in place.

    Returns:
        The full checkpoint dict (epoch, metrics, config, etc.).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path.resolve()}")

    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])

    if optimizer and "optimizer_state" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state"])
    if scheduler and checkpoint.get("scheduler_state"):
        scheduler.load_state_dict(checkpoint["scheduler_state"])
    if scaler and checkpoint.get("scaler_state"):
        scaler.load_state_dict(checkpoint["scaler_state"])

    logger.info(
        "Checkpoint loaded from %s | epoch=%d | metrics=%s",
        path,
        checkpoint.get("epoch", -1),
        {k: f"{v:.4f}" for k, v in checkpoint.get("metrics", {}).items()
         if isinstance(v, float)},
    )
    return checkpoint


def save_json(data: Any, path: str | Path, indent: int = 2) -> None:
    """Save any JSON-serializable object to a file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, cls=NumpyEncoder)


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy scalars and arrays."""

    def default(self, obj: Any) -> Any:
        try:
            import numpy as np
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
        except ImportError:
            pass
        if isinstance(obj, torch.Tensor):
            return obj.tolist()
        return super().default(obj)

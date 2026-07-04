"""
Device resolution utilities.

Resolves the "auto" device string to the best available hardware:
  CUDA (NVIDIA GPU) > MPS (Apple Silicon) > CPU
"""

from __future__ import annotations

import logging

import torch

logger = logging.getLogger(__name__)


def resolve_device(device: str = "auto") -> torch.device:
    """
    Resolve device string to a torch.device.

    Args:
        device: "auto", "cuda", "cuda:N", "mps", or "cpu".

    Returns:
        torch.device
    """
    if device == "auto":
        if torch.cuda.is_available():
            resolved = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            resolved = torch.device("mps")
        else:
            resolved = torch.device("cpu")
        logger.info("Device auto-resolved → %s", resolved)
        return resolved

    resolved = torch.device(device)
    logger.info("Device set to %s", resolved)
    return resolved


def get_device_info(device: torch.device) -> dict[str, str]:
    """Return human-readable device metadata for logging."""
    info: dict[str, str] = {"device": str(device)}
    if device.type == "cuda":
        idx = device.index or 0
        props = torch.cuda.get_device_properties(idx)
        info["name"] = props.name
        info["memory_gb"] = f"{props.total_memory / 1e9:.1f}"
    elif device.type == "mps":
        info["name"] = "Apple MPS"
    else:
        info["name"] = "CPU"
    return info


def memory_summary(device: torch.device) -> str:
    """Return a brief GPU memory summary string for training logs."""
    if device.type != "cuda":
        return ""
    allocated = torch.cuda.memory_allocated(device) / 1e9
    reserved = torch.cuda.memory_reserved(device) / 1e9
    return f"GPU mem: {allocated:.2f}GB alloc / {reserved:.2f}GB reserved"

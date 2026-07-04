"""
Attention map visualization for the transformer stage of MobileCLIP.

MobileCLIP-S2 has a transformer stage after the CNN stem. The attention
weights from the last multi-head attention block show which image patches
the model attends to when deciding "phishing" vs "legitimate".

Complements GradCAM: GradCAM shows gradient-weighted features from the
CNN stem; attention maps show transformer-level patch-level focus.

This module extracts the CLS token attention from the last transformer
block and upsamples it to the original image resolution.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch import Tensor

from models.classifier import PhishingClassifier

logger = logging.getLogger(__name__)


class AttentionVisualizer:
    """
    Extract and visualize attention maps from MobileCLIP's transformer stage.

    Args:
        model:  Trained PhishingClassifier.
        device: Torch device string.
    """

    def __init__(
        self,
        model: PhishingClassifier,
        device: str = "auto",
    ) -> None:
        self.model = model
        self.device = self._resolve_device(device)
        self.model.to(self.device)
        self.model.eval()
        self._attention_weights: Optional[Tensor] = None
        self._hook = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(
        self,
        image_tensor: Tensor,
    ) -> tuple[np.ndarray, Tensor]:
        """
        Extract CLS token attention from last transformer block.

        Args:
            image_tensor: [1, 3, 256, 256] normalized tensor.

        Returns:
            (attention_map, raw_attention):
              attention_map: [H, W] float32 in [0, 1], resized to image resolution.
              raw_attention: raw [n_heads, n_patches] attention tensor.
        """
        ...

    def visualize_and_save(
        self,
        image_path: str | Path,
        image_tensor: Tensor,
        output_path: str | Path,
        label: Optional[int] = None,
        predicted_class: Optional[int] = None,
    ) -> Path:
        """
        Generate attention overlay and save to disk.

        Args:
            image_path:      Path to original image (for overlay).
            image_tensor:    Preprocessed tensor.
            output_path:     Where to save the visualization.
            label:           Ground truth class (for title).
            predicted_class: Model prediction (for title).

        Returns:
            Path to saved visualization file.
        """
        ...

    def cleanup(self) -> None:
        """Remove registered attention hook."""
        if self._hook is not None:
            self._hook.remove()
            self._hook = None

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _find_last_attention_block(self) -> Optional[nn.Module]:
        """Walk backbone modules to find last multi-head attention layer."""
        ...

    def _register_attention_hook(self, attention_module: nn.Module) -> None:
        """Register a forward hook to capture attention weights."""
        ...

    def _upsample_attention(
        self,
        attention: Tensor,
        target_size: tuple[int, int],
        patch_size: int = 16,
    ) -> np.ndarray:
        """Upsample [n_heads, n_patches] → [H, W] image resolution."""
        ...

    @staticmethod
    def _resolve_device(device: str) -> torch.device:
        """Resolve "auto" to available device."""
        ...

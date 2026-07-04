"""
GradCAM visualization for MobileCLIP phishing classifier.

GradCAM computes the gradient of the target class score with respect to
the feature maps of a target convolutional/attention layer. Regions with
high positive gradient × activation indicate the pixels that contributed
most to the "phishing" prediction.

For MobileCLIP-S2 (a hybrid CNN-ViT architecture), the target layer is
the last convolutional block in the CNN stem before the transformer stage.

Visualization outputs saved to:
  outputs/visualizations/gradcam/
    ├── phishing_correct/     ← true phishing, correctly classified
    ├── phishing_missed/      ← false negatives (most important to analyze)
    ├── legitimate_correct/   ← true legitimate
    └── legitimate_wrong/     ← false positives
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


class GradCAMVisualizer:
    """
    GradCAM over the last convolutional layer of MobileCLIP.

    Args:
        model:        Trained PhishingClassifier.
        target_layer: The nn.Module whose activations to hook.
                      If None, auto-detected from backbone architecture.
        device:       Torch device.
    """

    def __init__(
        self,
        model: PhishingClassifier,
        target_layer: Optional[nn.Module] = None,
        device: str = "auto",
    ) -> None:
        self.model = model
        self.device = self._resolve_device(device)
        self.model.to(self.device)
        self.model.eval()

        self._activations: Optional[Tensor] = None
        self._gradients: Optional[Tensor] = None
        self._hooks: list = []

        self.target_layer = target_layer or self._auto_detect_target_layer()
        self._register_hooks()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(
        self,
        image_tensor: Tensor,
        target_class: Optional[int] = None,
    ) -> tuple[np.ndarray, dict]:
        """
        Generate GradCAM heatmap for a single image.

        Args:
            image_tensor: [1, 3, 256, 256] normalized image tensor.
            target_class: Class index to explain. If None, uses the predicted class.

        Returns:
            (heatmap, metadata):
              heatmap: [H, W] float32 array in [0, 1], same size as input image.
              metadata: dict with predicted_class, confidence, target_class.
        """
        ...

    def visualize_batch(
        self,
        image_paths: list[str],
        labels: list[int],
        threshold: float,
        output_dir: str | Path,
        max_samples: int = 20,
    ) -> list[Path]:
        """
        Generate and save GradCAM overlays for a batch of images.

        Organizes outputs into subdirectories by classification outcome.

        Args:
            image_paths: List of image file paths.
            labels:      Ground truth labels.
            threshold:   Decision threshold.
            output_dir:  Root directory for saving overlays.
            max_samples: Maximum number of samples to visualize per category.

        Returns:
            List of saved file paths.
        """
        ...

    def cleanup(self) -> None:
        """Remove all registered hooks."""
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _auto_detect_target_layer(self) -> nn.Module:
        """
        Find the last convolutional layer in the MobileCLIP backbone.

        Walks the backbone's named modules and returns the last nn.Conv2d.
        """
        ...

    def _register_hooks(self) -> None:
        """Register forward and backward hooks on the target layer."""
        ...

    def _compute_cam(self) -> np.ndarray:
        """
        Compute the class activation map from stored activations and gradients.

        Algorithm:
          1. Global average pool the gradients: alpha_k = mean(grad_k)
          2. Weighted sum of activation maps: L = ReLU(sum_k alpha_k * A_k)
          3. Normalize to [0, 1]
        """
        ...

    @staticmethod
    def _overlay_heatmap(
        original_image: Image.Image,
        heatmap: np.ndarray,
        alpha: float = 0.4,
    ) -> Image.Image:
        """Blend the heatmap (jet colormap) over the original image."""
        ...

    @staticmethod
    def _resolve_device(device: str) -> torch.device:
        """Resolve "auto" to available device."""
        ...

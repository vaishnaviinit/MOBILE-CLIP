"""
Diagnostic plotting utilities.

Generates all evaluation plots saved to outputs/visualizations/:
  - confusion_matrix.png
  - roc_curve.png
  - pr_curve.png
  - training_history.png
  - threshold_sweep.png
  - misclassified_samples.png

All plots use a consistent dark-background style appropriate for
security tooling dashboards.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

logger = logging.getLogger(__name__)

# Consistent color scheme across all plots
PHISHING_COLOR = "#E74C3C"      # red
LEGITIMATE_COLOR = "#2ECC71"    # green
THRESHOLD_COLOR = "#F39C12"     # orange
BACKGROUND_COLOR = "#1A1A2E"
GRID_COLOR = "#2D2D44"


def plot_confusion_matrix(
    tp: int,
    tn: int,
    fp: int,
    fn: int,
    output_path: str | Path,
    threshold: float = 0.5,
    normalize: bool = True,
) -> Path:
    """
    Plot and save a confusion matrix with FNR/FPR annotations.

    Annotations include absolute counts and normalized percentages.
    FNR is highlighted in red to emphasize the security-critical metric.

    Args:
        tp, tn, fp, fn: Confusion matrix values.
        output_path:    Path to save the PNG.
        threshold:      Decision threshold (shown in title).
        normalize:      If True, show percentages alongside counts.

    Returns:
        Path to saved figure.
    """
    ...


def plot_roc_curve(
    labels: np.ndarray,
    phishing_probs: np.ndarray,
    output_path: str | Path,
    threshold: Optional[float] = None,
) -> Path:
    """
    Plot ROC curve with AUC annotation and operating point marker.

    Args:
        labels:         Ground truth integer labels [N].
        phishing_probs: Phishing class probability [N].
        output_path:    Path to save the PNG.
        threshold:      If provided, mark this operating point on the curve.

    Returns:
        Path to saved figure.
    """
    ...


def plot_pr_curve(
    labels: np.ndarray,
    phishing_probs: np.ndarray,
    output_path: str | Path,
    threshold: Optional[float] = None,
) -> Path:
    """
    Plot Precision-Recall curve with AUC annotation.

    PR curves are more informative than ROC for imbalanced datasets
    because they directly show how precision degrades as recall increases.

    Args:
        labels:         Ground truth integer labels [N].
        phishing_probs: Phishing class probability [N].
        output_path:    Path to save the PNG.
        threshold:      If provided, mark this operating point.

    Returns:
        Path to saved figure.
    """
    ...


def plot_training_history(
    history: list[dict[str, float]],
    output_path: str | Path,
) -> Path:
    """
    Plot train/val loss, accuracy, F2, and recall across epochs.

    Four subplots:
      - Loss (train vs val)
      - Accuracy (train vs val)
      - F2 Score (val) + Recall (val)
      - Learning Rate

    Args:
        history:     List of epoch metric dicts from Trainer.
        output_path: Path to save the PNG.

    Returns:
        Path to saved figure.
    """
    ...


def plot_threshold_sweep(
    thresholds: np.ndarray,
    f1_scores: np.ndarray,
    f2_scores: np.ndarray,
    fnr_scores: np.ndarray,
    precision_scores: np.ndarray,
    recall_scores: np.ndarray,
    f1_threshold: float,
    f2_threshold: float,
    min_fnr_threshold: float,
    output_path: str | Path,
) -> Path:
    """
    Plot all metrics vs threshold to visualize operating point tradeoffs.

    Two subplots:
      - Top:    F1, F2, Precision, Recall vs threshold
      - Bottom: FNR vs threshold with minimum FNR target line

    Vertical lines mark each of the three operating points.

    Args:
        thresholds:       Array of threshold values swept.
        f1_scores:        F1 at each threshold.
        f2_scores:        F2 at each threshold.
        fnr_scores:       FNR at each threshold.
        precision_scores: Precision at each threshold.
        recall_scores:    Recall at each threshold.
        f1_threshold:     Optimal F1 threshold (vertical line).
        f2_threshold:     Optimal F2 threshold (vertical line).
        min_fnr_threshold: Minimum FNR threshold (vertical line).
        output_path:      Path to save the PNG.

    Returns:
        Path to saved figure.
    """
    ...
"""
Focal Loss for imbalanced phishing detection.

Reference: Lin et al., "Focal Loss for Dense Object Detection", ICCV 2017.

Why Focal Loss over Weighted Cross Entropy:
  Weighted CE assigns fixed per-class penalties regardless of prediction
  confidence. Focal Loss additionally down-weights easy correct predictions
  (confident legitimate pages) and up-weights hard examples (phishing pages
  that look legitimate or legitimate pages that look suspicious).

  FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

  gamma = 0 → reduces to weighted CE (no focusing effect)
  gamma = 2 → standard Focal Loss; standard for moderately hard tasks
  gamma = 3 → stronger focus; use if many phishing pages look clearly legit

For phishing detection with 5.34:1 imbalance:
  alpha_phishing  = 3.168  (from compute_class_weights)
  alpha_legit     = 0.594
  gamma = 2.0 (default)

Mathematical note on gradients:
  The gradient dFL/dp_t flows through BOTH the focal weight (1-p_t)^gamma
  and the CE term -log(p_t). This is the standard implementation approach
  used in detectron2 and OpenMMLab. The combined gradient naturally
  encourages aggressive learning on hard misclassified phishing examples.
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

logger = logging.getLogger(__name__)


class FocalLoss(nn.Module):
    """
    Multi-class Focal Loss with per-class alpha weighting.

    Args:
        alpha:           Per-class weights tensor of shape [num_classes].
                         If None, no class weighting is applied.
        gamma:           Focusing parameter gamma >= 0.
                         0 = standard weighted CE (no focusing).
                         2 = standard Focal Loss (default).
        reduction:       "mean" (default), "sum", or "none".
        label_smoothing: Epsilon for label smoothing in [0, 1).
                         0.0 = disabled (default).
                         When > 0, the target distribution is smoothed:
                         q_c = (1 - eps) * one_hot + eps / C.
                         The focal weight still uses the hard p_t.
    """

    def __init__(
        self,
        alpha: Tensor | None = None,
        gamma: float = 2.0,
        reduction: str = "mean",
        label_smoothing: float = 0.0,
    ) -> None:
        super().__init__()

        if gamma < 0:
            raise ValueError(f"gamma must be >= 0, got {gamma}")
        if reduction not in ("mean", "sum", "none"):
            raise ValueError(f"reduction must be 'mean', 'sum', or 'none', got '{reduction}'")
        if not (0.0 <= label_smoothing < 1.0):
            raise ValueError(f"label_smoothing must be in [0, 1), got {label_smoothing}")

        self.gamma = gamma
        self.reduction = reduction
        self.label_smoothing = label_smoothing

        if alpha is not None:
            self.register_buffer("alpha", alpha.float())
        else:
            self.alpha = None  # type: ignore[assignment]

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        """
        Compute focal loss for a batch.

        Step-by-step:
          1. log_softmax for numerical stability.
          2. CE loss per sample (with optional label smoothing).
          3. p_t = softmax probability for each sample's true class.
          4. focal_weight = (1 - p_t)^gamma  [down-weights easy examples]
          5. alpha_t = per-class alpha for each sample's true class.
          6. loss = alpha_t * focal_weight * ce_loss  (per sample)
          7. Reduce.

        Args:
            logits:  [B, C] raw unnormalized logits from the classifier.
            targets: [B] integer class indices in {0, 1, ..., C-1}.

        Returns:
            Scalar loss (if reduction="mean"/"sum") or [B] tensor (if "none").
        """
        n_classes: int = logits.size(-1)

        # Step 1 — numerically stable log-probabilities
        log_probs: Tensor = F.log_softmax(logits, dim=-1)  # [B, C]

        # Step 2 — cross-entropy loss per sample
        if self.label_smoothing > 0.0:
            # Smoothed target distribution: q_c = eps/C everywhere,
            # +=(1-eps) on the true class.
            eps = self.label_smoothing
            # Build smoothed distribution
            smooth = torch.full_like(log_probs, eps / n_classes)
            smooth.scatter_(
                dim=1,
                index=targets.unsqueeze(1),
                value=1.0 - eps + eps / n_classes,
            )
            # CE = -sum_c q_c * log p_c
            ce_loss: Tensor = -(smooth * log_probs).sum(dim=-1)  # [B]
        else:
            # Standard CE = -log p_{true class}
            ce_loss = -log_probs.gather(
                dim=1, index=targets.unsqueeze(1)
            ).squeeze(1)  # [B]

        # Step 3 — p_t: model confidence in the true class
        # Used only for the focal weight; we detach to avoid computing
        # second-order gradients through the modulation factor.
        probs: Tensor = log_probs.exp()  # [B, C]
        p_t: Tensor = probs.gather(
            dim=1, index=targets.unsqueeze(1)
        ).squeeze(1).detach()  # [B]
        p_t = p_t.clamp(min=1e-8, max=1.0)  # numerical guard

        # Step 4 — focal modulation weight
        focal_weight: Tensor = (1.0 - p_t).pow(self.gamma)  # [B]

        # Step 5 — per-class alpha
        if self.alpha is not None:
            alpha_t: Tensor = self.alpha[targets]  # [B]
            focal_weight = focal_weight * alpha_t

        # Step 6 — combine
        loss: Tensor = focal_weight * ce_loss  # [B]

        # Step 7 — reduce
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss  # "none": [B]

    def extra_repr(self) -> str:
        """Shown in print(model) output."""
        alpha_str = (
            f"[{', '.join(f'{a:.4f}' for a in self.alpha.tolist())}]"
            if self.alpha is not None
            else "None"
        )
        return (
            f"gamma={self.gamma}, alpha={alpha_str}, "
            f"reduction={self.reduction!r}, label_smoothing={self.label_smoothing}"
        )


def build_loss(config: dict, class_weights: Tensor | None = None) -> nn.Module:
    """
    Factory function: construct the loss module from config.yaml.

    Args:
        config:        The training config dict (config["training"]["loss"]).
        class_weights: [2] tensor from PhishingDataset.compute_class_weights().
                       Passed as Focal Loss alpha, or CE weight depending on loss type.

    Returns:
        Configured loss nn.Module.

    Supported loss names (config["name"]):
      "focal"       → FocalLoss with class_weights as alpha (recommended)
      "ce"          → nn.CrossEntropyLoss, no class weighting
      "weighted_ce" → nn.CrossEntropyLoss with class_weights
    """
    name: str = config.get("name", "focal").lower().strip()
    gamma: float = float(config.get("gamma", 2.0))
    label_smoothing: float = float(config.get("label_smoothing", 0.0))

    if name == "focal":
        loss_fn = FocalLoss(
            alpha=class_weights,   # [2]: [legit_weight, phishing_weight]
            gamma=gamma,
            reduction="mean",
            label_smoothing=label_smoothing,
        )
        alpha_str = (
            f"{class_weights.tolist()}" if class_weights is not None else "None"
        )
        logger.info(
            "Loss: FocalLoss | gamma=%.1f | alpha=%s | label_smoothing=%.2f",
            gamma,
            alpha_str,
            label_smoothing,
        )
        return loss_fn

    if name in ("ce", "crossentropy", "cross_entropy"):
        loss_fn = nn.CrossEntropyLoss(
            label_smoothing=label_smoothing,
            reduction="mean",
        )
        logger.info(
            "Loss: CrossEntropyLoss | label_smoothing=%.2f | (no class weighting)",
            label_smoothing,
        )
        return loss_fn

    if name in ("weighted_ce", "weighted_crossentropy"):
        if class_weights is None:
            logger.warning(
                "weighted_ce requested but class_weights is None — "
                "falling back to unweighted CrossEntropyLoss"
            )
        loss_fn = nn.CrossEntropyLoss(
            weight=class_weights,
            label_smoothing=label_smoothing,
            reduction="mean",
        )
        weight_str = (
            f"{class_weights.tolist()}" if class_weights is not None else "None"
        )
        logger.info(
            "Loss: Weighted CrossEntropyLoss | weight=%s | label_smoothing=%.2f",
            weight_str,
            label_smoothing,
        )
        return loss_fn

    raise ValueError(
        f"Unknown loss name: '{name}'. "
        f"Supported: 'focal', 'ce', 'weighted_ce'."
    )

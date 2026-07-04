"""
Comprehensive metrics suite for phishing classification evaluation.

Computes all metrics required for security-critical binary classification,
with emphasis on recall-oriented metrics (the cost of False Negatives
-- missing phishing pages -- is much higher than False Positives).

Metrics computed:
  Standard:   Accuracy, Loss
  PR:         Precision, Recall, F1, F2, PR-AUC
  ROC:        ROC-AUC
  Confusion:  TP, TN, FP, FN, FPR, FNR, Specificity, Sensitivity
  Composite:  Balanced Accuracy, Matthews Correlation Coefficient (MCC)
"""

from __future__ import annotations

import dataclasses
import logging
import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
from torch import Tensor

logger = logging.getLogger(__name__)

PHISHING_IDX = 1
LEGITIMATE_IDX = 0


@dataclass
class MetricResult:
    """
    All classification metrics for one evaluation pass.

    phishing is always treated as the positive class.
    FNR is the primary security metric: a high FNR means many phishing
    pages are slipping through as legitimate -- the worst outcome.
    """

    # Core
    loss: float = 0.0
    accuracy: float = 0.0
    balanced_accuracy: float = 0.0

    # Precision-Recall (phishing = positive)
    precision: float = 0.0
    recall: float = 0.0       # = Sensitivity = TPR
    f1: float = 0.0
    f2: float = 0.0           # F-beta with beta=2 (recall-heavy)
    pr_auc: float = 0.0

    # ROC
    roc_auc: float = 0.0

    # Confusion matrix raw counts
    tp: int = 0
    tn: int = 0
    fp: int = 0
    fn: int = 0

    # Derived rates
    fpr: float = 0.0          # False Positive Rate = FP / (FP + TN)
    fnr: float = 0.0          # False Negative Rate = FN / (FN + TP) < minimize this
    specificity: float = 0.0  # = TNR = 1 - FPR
    sensitivity: float = 0.0  # = Recall = TPR = 1 - FNR

    # Matthews Correlation Coefficient
    mcc: float = 0.0

    # Threshold used for confusion-matrix-derived metrics
    threshold: float = 0.5

    def as_dict(self) -> dict[str, float]:
        """Return all metrics as a flat dict (all values cast to float)."""
        return {k: float(v) for k, v in dataclasses.asdict(self).items()}

    def summary(self) -> str:
        """
        Compact multi-line security-focused summary.

        Highlights FNR prominently -- it's the metric operators care about most.
        """
        sep = "-" * 64
        return (
            f"{sep}\n"
            f"  Threshold : {self.threshold:.3f}\n"
            f"  Accuracy  : {self.accuracy:.4f}   Balanced Acc : {self.balanced_accuracy:.4f}\n"
            f"  Precision : {self.precision:.4f}   Recall       : {self.recall:.4f}\n"
            f"  F1        : {self.f1:.4f}   F2 (^rec)    : {self.f2:.4f}\n"
            f"  ROC-AUC   : {self.roc_auc:.4f}   PR-AUC       : {self.pr_auc:.4f}\n"
            f"  MCC       : {self.mcc:.4f}\n"
            f"  -- Confusion ------------------------------------------\n"
            f"  TP={self.tp:5d}  TN={self.tn:5d}  FP={self.fp:5d}  FN={self.fn:5d}\n"
            f"  FNR={self.fnr:.4f} (phishing missed)  "
            f"FPR={self.fpr:.4f} (legit flagged)\n"
            f"  Sensitivity={self.sensitivity:.4f}  Specificity={self.specificity:.4f}\n"
            f"{sep}"
        )


class MetricsCalculator:
    """
    Accumulates per-batch predictions and computes the full MetricResult.

    Usage (in the Evaluator):
        calc = MetricsCalculator()
        for images, labels in test_loader:
            outputs = model(images)
            calc.update(outputs["probs"], labels, loss.item())
        result = calc.compute(threshold=0.35)
        calc.reset()

    Args:
        n_classes: Number of output classes (2 for binary phishing/legitimate).
    """

    def __init__(self, n_classes: int = 2) -> None:
        self.n_classes = n_classes
        self._all_probs: list[np.ndarray] = []
        self._all_labels: list[np.ndarray] = []
        self._total_loss: float = 0.0
        self._n_batches: int = 0

    # ------------------------------------------------------------------
    # Accumulation interface
    # ------------------------------------------------------------------

    def update(
        self,
        probs: Tensor,
        labels: Tensor,
        loss: float = 0.0,
    ) -> None:
        """
        Accumulate one batch.

        Args:
            probs:  [B, 2] softmax probabilities (CPU or GPU tensor).
            labels: [B]    integer ground-truth class indices.
            loss:   Scalar batch loss (averaged into a running mean).
        """
        self._all_probs.append(probs.detach().cpu().numpy())
        self._all_labels.append(labels.detach().cpu().numpy())
        self._total_loss += float(loss)
        self._n_batches += 1

    def compute(self, threshold: float = 0.5) -> MetricResult:
        """
        Compute all metrics from accumulated state.

        Args:
            threshold: Decision threshold on phishing probability (column 1).
                       Samples with P(phishing) >= threshold → predicted phishing.

        Returns:
            Fully populated MetricResult.
        """
        if not self._all_probs:
            logger.warning("MetricsCalculator.compute() called with no data accumulated.")
            return MetricResult(threshold=threshold)

        probs: np.ndarray = np.concatenate(self._all_probs, axis=0)   # [N, 2]
        labels: np.ndarray = np.concatenate(self._all_labels, axis=0) # [N]

        phish_probs = probs[:, PHISHING_IDX]                           # [N]
        preds = (phish_probs >= threshold).astype(int)                 # [N]

        avg_loss = self._total_loss / max(self._n_batches, 1)

        # -- Confusion matrix ------------------------------------------
        tp, tn, fp, fn = self._compute_confusion_matrix(preds, labels)

        # -- Point metrics ---------------------------------------------
        acc = (tp + tn) / max(tp + tn + fp + fn, 1)
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)       # sensitivity / TPR
        fpr = fp / max(fp + tn, 1)
        fnr = fn / max(fn + tp, 1)
        specificity = tn / max(tn + fp, 1)
        bal_acc = (rec + specificity) / 2.0

        f1 = self._compute_f_beta(prec, rec, beta=1.0)
        f2 = self._compute_f_beta(prec, rec, beta=2.0)

        mcc = self._compute_mcc(tp, tn, fp, fn)

        # -- Curve-based metrics (require both classes present) --------
        n_unique = len(np.unique(labels))
        if n_unique > 1:
            from sklearn.metrics import roc_auc_score, average_precision_score
            roc_auc = float(roc_auc_score(labels, phish_probs))
            pr_auc = float(average_precision_score(labels, phish_probs))
        else:
            roc_auc = 0.0
            pr_auc = 0.0
            logger.warning(
                "Only one class present in batch -- ROC-AUC and PR-AUC set to 0.0"
            )

        return MetricResult(
            loss=float(avg_loss),
            accuracy=float(acc),
            balanced_accuracy=float(bal_acc),
            precision=float(prec),
            recall=float(rec),
            f1=float(f1),
            f2=float(f2),
            pr_auc=float(pr_auc),
            roc_auc=float(roc_auc),
            tp=int(tp),
            tn=int(tn),
            fp=int(fp),
            fn=int(fn),
            fpr=float(fpr),
            fnr=float(fnr),
            specificity=float(specificity),
            sensitivity=float(rec),
            mcc=float(mcc),
            threshold=float(threshold),
        )

    def reset(self) -> None:
        """Clear all accumulated state."""
        self._all_probs.clear()
        self._all_labels.clear()
        self._total_loss = 0.0
        self._n_batches = 0

    # ------------------------------------------------------------------
    # Private computation helpers
    # ------------------------------------------------------------------

    def _compute_confusion_matrix(
        self,
        preds: np.ndarray,
        labels: np.ndarray,
    ) -> tuple[int, int, int, int]:
        """
        Return (TP, TN, FP, FN) with phishing (1) as the positive class.

        TP: phishing correctly detected
        TN: legitimate correctly passed
        FP: legitimate incorrectly flagged as phishing  (annoying, but safe)
        FN: phishing incorrectly passed as legitimate   (dangerous)
        """
        tp = int(np.sum((preds == 1) & (labels == 1)))
        tn = int(np.sum((preds == 0) & (labels == 0)))
        fp = int(np.sum((preds == 1) & (labels == 0)))
        fn = int(np.sum((preds == 0) & (labels == 1)))
        return tp, tn, fp, fn

    def _compute_f_beta(
        self,
        precision: float,
        recall: float,
        beta: float,
    ) -> float:
        """
        F_beta = (1 + beta²) · P · R / (beta² · P + R)

        beta=1 → F1 (harmonic mean of P and R)
        beta=2 → F2 (weights recall 4× over precision)
        """
        b2 = beta * beta
        denom = b2 * precision + recall
        if denom == 0.0:
            return 0.0
        return (1.0 + b2) * precision * recall / denom

    def _compute_mcc(
        self, tp: int, tn: int, fp: int, fn: int
    ) -> float:
        """
        Matthews Correlation Coefficient.

        MCC = (TP·TN − FP·FN) / sqrt((TP+FP)(TP+FN)(TN+FP)(TN+FN))

        MCC = +1 → perfect prediction
        MCC =  0 → random
        MCC = -1 → perfect inverse prediction

        More informative than accuracy on imbalanced datasets: a classifier
        that always predicts "legitimate" gets MCC ≈ 0, not MCC = 0.85.
        """
        numerator = float(tp * tn - fp * fn)
        denominator = math.sqrt(
            float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        )
        if denominator == 0.0:
            return 0.0
        return numerator / denominator


"""
Post-training threshold optimization.

Default 0.5 is almost never the optimal threshold for imbalanced
security-critical classifiers. This module sweeps the full probability
range and reports three operating points.

Three operating points:
  1. Max F1 threshold:   balanced precision/recall
  2. Max F2 threshold:   recall-heavy (recommended for deployment)
  3. Min FNR threshold:  highest threshold that still achieves FNR <= target

Key insight on FNR and threshold direction:
  Lower threshold → predict phishing more aggressively → lower FNR, higher FPR
  Higher threshold → predict phishing more conservatively → higher FNR, lower FPR

  The "Min FNR" operating point finds the HIGHEST threshold where
  FNR <= min_fnr_target. This maximises precision while still meeting
  the security constraint (e.g. catch >=95% of phishing pages).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ThresholdReport:
    """Results of a full threshold sweep with three operating points."""

    # Sweep metadata
    n_thresholds_swept: int = 0
    sweep_range: tuple[float, float] = (0.05, 0.95)
    step: float = 0.005

    # -- Operating point 1: Max F1 ----------------------------------
    f1_threshold: float = 0.5
    f1_at_threshold: float = 0.0
    precision_at_f1: float = 0.0
    recall_at_f1: float = 0.0
    fnr_at_f1: float = 0.0

    # -- Operating point 2: Max F2 (recommended for deployment) -----
    f2_threshold: float = 0.5
    f2_at_threshold: float = 0.0
    precision_at_f2: float = 0.0
    recall_at_f2: float = 0.0
    fnr_at_f2: float = 0.0

    # -- Operating point 3: Min FNR (maximum security mode) ---------
    min_fnr_threshold: float = 0.5
    fnr_at_min_fnr: float = 0.0
    precision_at_min_fnr: float = 0.0
    recall_at_min_fnr: float = 0.0

    # Full sweep arrays (kept for plotting)
    thresholds: np.ndarray = field(default_factory=lambda: np.array([]))
    f1_scores: np.ndarray = field(default_factory=lambda: np.array([]))
    f2_scores: np.ndarray = field(default_factory=lambda: np.array([]))
    fnr_scores: np.ndarray = field(default_factory=lambda: np.array([]))
    precision_scores: np.ndarray = field(default_factory=lambda: np.array([]))
    recall_scores: np.ndarray = field(default_factory=lambda: np.array([]))

    def recommended_threshold(self, strategy: str = "f2") -> float:
        """
        Return the threshold for the given deployment strategy.

        Args:
            strategy: "f1"     → balanced precision/recall
                      "f2"     → recall-heavy (default, recommended)
                      "min_fnr"→ maximum security (guaranteed recall)
        """
        mapping = {
            "f1": self.f1_threshold,
            "f2": self.f2_threshold,
            "min_fnr": self.min_fnr_threshold,
        }
        if strategy not in mapping:
            raise ValueError(
                f"Unknown strategy '{strategy}'. "
                f"Supported: {list(mapping.keys())}"
            )
        return mapping[strategy]

    def as_table(self) -> str:
        """
        Formatted comparison table of all three operating points.

        Example output:
          Operating Point      Threshold       F1       F2  Precision   Recall      FNR
          -----------------------------------------------------------------------------
          Max F1                   0.380   0.7841        -     0.8103   0.7600   0.2400
          Max F2 (recommended)     0.280   0.7523   0.8120     0.7214   0.8200   0.1800
          Min FNR (<= 5%)           0.120   0.6312   0.7440     0.5830   0.9500   0.0500
        """
        w = (20, 10, 8, 8, 10, 8, 8)
        header = (
            f"{'Operating Point':<{w[0]}} {'Threshold':>{w[1]}} "
            f"{'F1':>{w[2]}} {'F2':>{w[3]}} {'Precision':>{w[4]}} "
            f"{'Recall':>{w[5]}} {'FNR':>{w[6]}}"
        )
        sep = "-" * sum(w + (len(w) - 1,))

        rows = [
            (
                f"{'Max F1':<{w[0]}} {self.f1_threshold:>{w[1]}.3f} "
                f"{self.f1_at_threshold:>{w[2]}.4f} {'-':>{w[3]}} "
                f"{self.precision_at_f1:>{w[4]}.4f} "
                f"{self.recall_at_f1:>{w[5]}.4f} "
                f"{self.fnr_at_f1:>{w[6]}.4f}"
            ),
            (
                f"{'Max F2 (recommended)':<{w[0]}} {self.f2_threshold:>{w[1]}.3f} "
                f"{'-':>{w[2]}} {self.f2_at_threshold:>{w[3]}.4f} "
                f"{self.precision_at_f2:>{w[4]}.4f} "
                f"{self.recall_at_f2:>{w[5]}.4f} "
                f"{self.fnr_at_f2:>{w[6]}.4f}"
            ),
            (
                f"{'Min FNR (security)':<{w[0]}} {self.min_fnr_threshold:>{w[1]}.3f} "
                f"{'-':>{w[2]}} {'-':>{w[3]}} "
                f"{self.precision_at_min_fnr:>{w[4]}.4f} "
                f"{self.recall_at_min_fnr:>{w[5]}.4f} "
                f"{self.fnr_at_min_fnr:>{w[6]}.4f}"
            ),
        ]
        return "\n".join([header, sep] + rows)


class ThresholdOptimizer:
    """
    Sweep decision threshold over [sweep_min, sweep_max] and find all
    three operating points via vectorised numpy operations.

    The full sweep (180 thresholds × N test samples) runs in <1ms on CPU
    using broadcasting -- no Python loop over thresholds.

    Args:
        sweep_min:      Start of threshold sweep (default 0.05).
        sweep_max:      End of threshold sweep (default 0.95).
        step:           Step size (default 0.005 → 180 thresholds).
        min_fnr_target: FNR ceiling for the security-mode operating point
                        (default 0.05 → catch >=95% of phishing pages).
    """

    def __init__(
        self,
        sweep_min: float = 0.05,
        sweep_max: float = 0.95,
        step: float = 0.005,
        min_fnr_target: float = 0.05,
    ) -> None:
        self.sweep_min = sweep_min
        self.sweep_max = sweep_max
        self.step = step
        self.min_fnr_target = min_fnr_target

    def optimize(
        self,
        phishing_probs: np.ndarray,
        labels: np.ndarray,
    ) -> ThresholdReport:
        """
        Run the vectorised threshold sweep and populate a ThresholdReport.

        Args:
            phishing_probs: [N] float array -- P(phishing) for each test sample.
            labels:         [N] int array   -- ground truth (0=legit, 1=phishing).

        Returns:
            ThresholdReport with all three operating points and full sweep arrays.
        """
        thresholds = np.arange(
            self.sweep_min,
            self.sweep_max + self.step / 2,  # +step/2 avoids float rounding drop
            self.step,
        )

        # -- Vectorised confusion matrix across all thresholds ---------
        # preds[i, j] = 1 iff phishing_probs[j] >= thresholds[i]
        # Shape: [n_thresholds, N]
        preds = (
            phishing_probs[np.newaxis, :] >= thresholds[:, np.newaxis]
        ).astype(np.int32)

        labs = labels[np.newaxis, :]  # [1, N] broadcast

        tp = np.sum((preds == 1) & (labs == 1), axis=1).astype(float)
        tn = np.sum((preds == 0) & (labs == 0), axis=1).astype(float)
        fp = np.sum((preds == 1) & (labs == 0), axis=1).astype(float)
        fn = np.sum((preds == 0) & (labs == 1), axis=1).astype(float)

        # Safe division helpers
        precision = np.where(tp + fp > 0, tp / (tp + fp), 0.0)
        recall    = np.where(tp + fn > 0, tp / (tp + fn), 0.0)
        fnr       = np.where(tp + fn > 0, fn / (tp + fn), 1.0)

        # F1 = 2PR / (P + R)
        f1 = np.where(
            precision + recall > 0,
            2.0 * precision * recall / (precision + recall),
            0.0,
        )
        # F2 = 5PR / (4P + R)
        f2 = np.where(
            4.0 * precision + recall > 0,
            5.0 * precision * recall / (4.0 * precision + recall),
            0.0,
        )

        # -- Operating point 1: Max F1 ------------------------------
        f1_idx = int(np.argmax(f1))

        # -- Operating point 2: Max F2 ------------------------------
        f2_idx = int(np.argmax(f2))

        # -- Operating point 3: Min FNR ------------------------------
        # Highest threshold where FNR <= target (most selective while safe)
        eligible = np.where(fnr <= self.min_fnr_target)[0]
        if len(eligible) > 0:
            min_fnr_idx = int(eligible[-1])   # highest threshold that meets FNR target
        else:
            min_fnr_idx = int(np.argmin(fnr))
            logger.warning(
                "No threshold achieves FNR <= %.2f -- "
                "using lowest FNR threshold (%.3f, FNR=%.4f)",
                self.min_fnr_target,
                thresholds[min_fnr_idx],
                fnr[min_fnr_idx],
            )

        report = ThresholdReport(
            n_thresholds_swept=len(thresholds),
            sweep_range=(float(self.sweep_min), float(self.sweep_max)),
            step=float(self.step),

            # Operating point 1
            f1_threshold=float(thresholds[f1_idx]),
            f1_at_threshold=float(f1[f1_idx]),
            precision_at_f1=float(precision[f1_idx]),
            recall_at_f1=float(recall[f1_idx]),
            fnr_at_f1=float(fnr[f1_idx]),

            # Operating point 2
            f2_threshold=float(thresholds[f2_idx]),
            f2_at_threshold=float(f2[f2_idx]),
            precision_at_f2=float(precision[f2_idx]),
            recall_at_f2=float(recall[f2_idx]),
            fnr_at_f2=float(fnr[f2_idx]),

            # Operating point 3
            min_fnr_threshold=float(thresholds[min_fnr_idx]),
            fnr_at_min_fnr=float(fnr[min_fnr_idx]),
            precision_at_min_fnr=float(precision[min_fnr_idx]),
            recall_at_min_fnr=float(recall[min_fnr_idx]),

            # Full arrays for plotting
            thresholds=thresholds,
            f1_scores=f1,
            f2_scores=f2,
            fnr_scores=fnr,
            precision_scores=precision,
            recall_scores=recall,
        )

        logger.info(
            "Threshold sweep: %d points | F2-opt=%.3f (FNR=%.4f) | "
            "F1-opt=%.3f | MinFNR=%.3f (FNR=%.4f)",
            len(thresholds),
            report.f2_threshold,
            report.fnr_at_f2,
            report.f1_threshold,
            report.min_fnr_threshold,
            report.fnr_at_min_fnr,
        )

        return report

    # ------------------------------------------------------------------
    # Private helpers (kept for external callers who want single-threshold metrics)
    # ------------------------------------------------------------------

    def _metrics_at_threshold(
        self,
        phishing_probs: np.ndarray,
        labels: np.ndarray,
        threshold: float,
    ) -> dict[str, float]:
        """Compute P, R, F1, F2, FNR, FPR at a single threshold."""
        preds = (phishing_probs >= threshold).astype(int)

        tp = int(np.sum((preds == 1) & (labels == 1)))
        tn = int(np.sum((preds == 0) & (labels == 0)))
        fp = int(np.sum((preds == 1) & (labels == 0)))
        fn = int(np.sum((preds == 0) & (labels == 1)))

        precision = tp / max(tp + fp, 1)
        recall    = tp / max(tp + fn, 1)
        fnr       = fn / max(fn + tp, 1)
        fpr       = fp / max(fp + tn, 1)

        return {
            "precision": precision,
            "recall":    recall,
            "fnr":       fnr,
            "fpr":       fpr,
            "f1":        self._f_beta(precision, recall, beta=1.0),
            "f2":        self._f_beta(precision, recall, beta=2.0),
        }

    @staticmethod
    def _f_beta(precision: float, recall: float, beta: float) -> float:
        """F-beta score. beta=2 weights recall 4× over precision."""
        b2 = beta * beta
        denom = b2 * precision + recall
        return 0.0 if denom == 0.0 else (1.0 + b2) * precision * recall / denom


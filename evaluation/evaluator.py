"""
Full evaluation suite -- runs after training completes on the held-out test set.

Orchestrates:
  1. Full inference pass → (phishing_probs, labels, image_paths)
  2. MetricResult at threshold=0.5
  3. ThresholdOptimizer → three operating points
  4. MetricResult at recommended threshold
  5. sklearn classification report
  6. Per-sample predictions JSON
  7. Misclassified samples (FN sorted by confidence -- most dangerous first)
  8. Diagnostic plots (confusion matrix, ROC, PR, threshold sweep)

All artifacts are written to output_dir/predictions/ and output_dir/visualizations/.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from evaluation.metrics import MetricsCalculator, MetricResult
from evaluation.threshold import ThresholdOptimizer, ThresholdReport
from models.classifier import PhishingClassifier
from utils.device import resolve_device
from utils.io_utils import save_json

logger = logging.getLogger(__name__)

# Class name map for human-readable output
IDX_TO_CLASS = {0: "legitimate", 1: "phishing"}


@dataclass
class EvaluationReport:
    """
    Full evaluation output -- all metrics, thresholds, and artifact paths.
    """

    # Metrics at default threshold (0.5)
    metrics_default: MetricResult = field(default_factory=MetricResult)

    # Metrics at recommended threshold (strategy-dependent)
    metrics_recommended: MetricResult = field(default_factory=MetricResult)

    # Threshold analysis
    threshold_report: Optional[ThresholdReport] = None
    recommended_threshold: float = 0.5

    # Per-sample predictions (list of dicts)
    predictions: list[dict] = field(default_factory=list)

    # Misclassification analysis
    false_negatives: list[dict] = field(default_factory=list)
    false_positives: list[dict] = field(default_factory=list)

    # Artifact paths filled by Evaluator.evaluate()
    artifact_paths: dict[str, str] = field(default_factory=dict)

    def summary(self) -> str:
        """Multi-line human-readable evaluation summary."""
        lines = [
            "=" * 64,
            "  EVALUATION REPORT",
            "=" * 64,
            "",
            f"  Recommended threshold : {self.recommended_threshold:.3f}",
            f"  False Negatives       : {len(self.false_negatives)} "
            f"(phishing pages missed -- DANGER)",
            f"  False Positives       : {len(self.false_positives)} "
            f"(legitimate pages flagged)",
            "",
            "  -- Metrics @ threshold=0.5 --",
            self.metrics_default.summary(),
            "",
        ]
        if self.metrics_recommended.threshold != 0.5:
            lines += [
                f"  -- Metrics @ recommended threshold={self.recommended_threshold:.3f} --",
                self.metrics_recommended.summary(),
                "",
            ]
        if self.threshold_report is not None:
            lines += [
                "  -- Threshold Operating Points --",
                self.threshold_report.as_table(),
                "",
            ]
        if self.artifact_paths:
            lines += ["  -- Saved Artifacts --"]
            for name, path in self.artifact_paths.items():
                lines.append(f"  {name:<30} {path}")
        lines.append("=" * 64)
        return "\n".join(lines)

    def to_json(self) -> str:
        """Serialize scalar metrics to a JSON string."""
        data = {
            "recommended_threshold": self.recommended_threshold,
            "metrics_default": self.metrics_default.as_dict(),
            "metrics_recommended": self.metrics_recommended.as_dict(),
            "n_false_negatives": len(self.false_negatives),
            "n_false_positives": len(self.false_positives),
            "artifact_paths": self.artifact_paths,
        }
        return json.dumps(data, indent=2)


class Evaluator:
    """
    Runs the full evaluation pipeline on the held-out test set.

    Args:
        model:               Trained PhishingClassifier in eval mode.
        test_loader:         DataLoader for the test split (no shuffle).
        output_dir:          Root directory for artifacts.
        threshold_strategy:  "f1", "f2", or "min_fnr".
        min_fnr_target:      FNR ceiling for "min_fnr" strategy (default 0.05).
        device:              "auto", "cuda", "cpu", or "mps".
    """

    def __init__(
        self,
        model: PhishingClassifier,
        test_loader: DataLoader,
        output_dir: str | Path = "outputs",
        threshold_strategy: str = "f2",
        min_fnr_target: float = 0.05,
        device: str = "auto",
    ) -> None:
        self.model = model
        self.test_loader = test_loader
        self.output_dir = Path(output_dir)
        self.threshold_strategy = threshold_strategy
        self.min_fnr_target = min_fnr_target
        self.device = resolve_device(device)

        self.pred_dir = self.output_dir / "predictions"
        self.viz_dir = self.output_dir / "visualizations"
        self.pred_dir.mkdir(parents=True, exist_ok=True)
        self.viz_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(self) -> EvaluationReport:
        """
        Run the full evaluation suite.

        Steps:
          1. Collect predictions (probs, labels, paths) over the full test set.
          2. Compute MetricResult at threshold=0.5.
          3. Run ThresholdOptimizer to find F1-opt, F2-opt, MinFNR points.
          4. Compute MetricResult at the recommended threshold.
          5. Save classification report text file.
          6. Save per-sample predictions JSON.
          7. Identify and rank misclassifications.
          8. Generate diagnostic plots (ROC, PR, confusion matrix, sweep).
          9. Assemble and return EvaluationReport.

        Returns:
            EvaluationReport with all fields populated.
        """
        logger.info("Starting evaluation on %d test samples …", len(self.test_loader.dataset))

        report = EvaluationReport()

        # -- Step 1: Collect predictions -------------------------------
        phishing_probs, labels, image_paths = self._collect_predictions()
        logger.info(
            "Collected predictions | n=%d | phishing=%d | legitimate=%d",
            len(labels),
            int(labels.sum()),
            int((labels == 0).sum()),
        )

        # -- Step 2: Metrics at default threshold=0.5 -----------------
        calc = MetricsCalculator()
        # Feed as a single "batch" -- calc just needs the arrays
        calc._all_probs = [
            np.stack([1 - phishing_probs, phishing_probs], axis=1)
        ]
        calc._all_labels = [labels]
        report.metrics_default = calc.compute(threshold=0.5)
        logger.info("Metrics @ 0.5:\n%s", report.metrics_default.summary())

        # -- Step 3: Threshold optimization ----------------------------
        optimizer = ThresholdOptimizer(
            min_fnr_target=self.min_fnr_target,
        )
        t_report = optimizer.optimize(phishing_probs, labels)
        report.threshold_report = t_report
        logger.info("\n%s", t_report.as_table())

        rec_threshold = t_report.recommended_threshold(self.threshold_strategy)
        report.recommended_threshold = rec_threshold

        # -- Step 4: Metrics at recommended threshold ------------------
        report.metrics_recommended = calc.compute(threshold=rec_threshold)
        logger.info(
            "Metrics @ recommended threshold=%.3f:\n%s",
            rec_threshold,
            report.metrics_recommended.summary(),
        )

        # -- Step 5: Classification report -----------------------------
        preds_at_rec = (phishing_probs >= rec_threshold).astype(int)
        clf_report_path = self.pred_dir / "classification_report.txt"
        self._save_classification_report(labels, preds_at_rec, clf_report_path)
        report.artifact_paths["classification_report"] = str(clf_report_path)

        # -- Step 6: Per-sample predictions JSON -----------------------
        pred_json_path = self.pred_dir / "predictions.json"
        report.predictions = self._build_predictions_list(
            phishing_probs, labels, image_paths, rec_threshold
        )
        save_json(report.predictions, pred_json_path)
        report.artifact_paths["predictions_json"] = str(pred_json_path)

        # -- Step 7: Misclassification analysis ------------------------
        report.false_negatives, report.false_positives = (
            self._extract_misclassifications(
                phishing_probs, labels, image_paths, rec_threshold
            )
        )
        fn_path = self.pred_dir / "false_negatives.json"
        fp_path = self.pred_dir / "false_positives.json"
        save_json(report.false_negatives, fn_path)
        save_json(report.false_positives, fp_path)
        report.artifact_paths["false_negatives"] = str(fn_path)
        report.artifact_paths["false_positives"] = str(fp_path)
        logger.info(
            "Misclassifications | FN=%d (phishing missed) | FP=%d (legit flagged)",
            len(report.false_negatives),
            len(report.false_positives),
        )

        # -- Step 8: Diagnostic plots ----------------------------------
        self._generate_plots(
            report=report,
            phishing_probs=phishing_probs,
            labels=labels,
            rec_threshold=rec_threshold,
        )

        # -- Step 9: Save full report JSON -----------------------------
        report_path = self.pred_dir / "evaluation_report.json"
        save_json(json.loads(report.to_json()), report_path)
        report.artifact_paths["evaluation_report"] = str(report_path)

        logger.info("Evaluation complete. Artifacts in %s", self.output_dir)
        return report

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _collect_predictions(
        self,
    ) -> tuple[np.ndarray, np.ndarray, list[str]]:
        """
        Run inference over the full test_loader (no gradient computation).

        Returns:
            phishing_probs: [N] float32 -- P(phishing) for each sample.
            labels:         [N] int32   -- ground-truth class index.
            image_paths:    [N] str     -- file paths from dataset.records.
        """
        self.model.eval()
        self.model.to(self.device)

        all_probs: list[float] = []
        all_labels: list[int] = []

        with torch.no_grad():
            for images, batch_labels in self.test_loader:
                images = images.to(self.device, non_blocking=True)
                outputs = self.model(images)
                all_probs.extend(outputs["probs"][:, 1].cpu().tolist())
                all_labels.extend(batch_labels.tolist())

        # Retrieve paths in dataset order -- test loader must NOT shuffle
        dataset = self.test_loader.dataset
        if hasattr(dataset, "get_image_paths"):
            image_paths: list[str] = dataset.get_image_paths()[: len(all_probs)]
        else:
            image_paths = [""] * len(all_probs)

        return (
            np.array(all_probs, dtype=np.float32),
            np.array(all_labels, dtype=np.int32),
            image_paths,
        )

    def _save_classification_report(
        self,
        labels: np.ndarray,
        preds: np.ndarray,
        path: Path,
    ) -> None:
        """Write sklearn classification_report to a text file."""
        from sklearn.metrics import classification_report

        report_str = classification_report(
            labels,
            preds,
            target_names=["legitimate", "phishing"],
            digits=4,
        )
        path.write_text(report_str, encoding="utf-8")
        logger.info("Classification report → %s", path)

    def _build_predictions_list(
        self,
        phishing_probs: np.ndarray,
        labels: np.ndarray,
        image_paths: list[str],
        threshold: float,
    ) -> list[dict]:
        """Build list of per-sample prediction dicts for JSON export."""
        preds = (phishing_probs >= threshold).astype(int)
        results = []
        for i, (prob, label, pred, path) in enumerate(
            zip(phishing_probs, labels, preds, image_paths)
        ):
            results.append({
                "index": i,
                "path": path,
                "true_label": IDX_TO_CLASS[int(label)],
                "predicted_label": IDX_TO_CLASS[int(pred)],
                "phishing_probability": round(float(prob), 6),
                "legitimate_probability": round(float(1.0 - prob), 6),
                "correct": bool(pred == label),
                "threshold": float(threshold),
            })
        return results

    def _extract_misclassifications(
        self,
        phishing_probs: np.ndarray,
        labels: np.ndarray,
        image_paths: list[str],
        threshold: float,
    ) -> tuple[list[dict], list[dict]]:
        """
        Identify False Negatives and False Positives, ranked by confidence.

        False Negatives are sorted by descending phishing_probability:
        the ones with highest P(phishing) that were still missed at this
        threshold are the most instructive for debugging.

        False Positives are sorted by descending phishing_probability:
        the most confidently-wrong legitimate pages are ranked first.

        Returns:
            (false_negatives, false_positives)
        """
        preds = (phishing_probs >= threshold).astype(int)
        false_negatives: list[dict] = []
        false_positives: list[dict] = []

        for prob, label, pred, path in zip(
            phishing_probs, labels, preds, image_paths
        ):
            label, pred = int(label), int(pred)
            if label == 1 and pred == 0:
                # Phishing page missed -- most dangerous error
                false_negatives.append({
                    "path": path,
                    "true_label": "phishing",
                    "predicted_label": "legitimate",
                    "phishing_probability": round(float(prob), 6),
                    "note": "MISSED PHISHING -- increase threshold sensitivity",
                })
            elif label == 0 and pred == 1:
                # Legitimate page incorrectly flagged
                false_positives.append({
                    "path": path,
                    "true_label": "legitimate",
                    "predicted_label": "phishing",
                    "phishing_probability": round(float(prob), 6),
                    "note": "FALSE ALARM -- legitimate page flagged",
                })

        # Sort FN by prob descending (highest-prob misses → most instructive)
        false_negatives.sort(key=lambda x: x["phishing_probability"], reverse=True)
        # Sort FP by prob descending (most confident wrong predictions first)
        false_positives.sort(key=lambda x: x["phishing_probability"], reverse=True)

        return false_negatives, false_positives

    def _generate_plots(
        self,
        report: EvaluationReport,
        phishing_probs: np.ndarray,
        labels: np.ndarray,
        rec_threshold: float,
    ) -> None:
        """
        Generate all diagnostic plots.

        Each plot function is called inside a try/except so that a failure
        in one plot does not abort the rest of the evaluation.
        """
        try:
            from visualization.plot_utils import (
                plot_confusion_matrix,
                plot_roc_curve,
                plot_pr_curve,
                plot_threshold_sweep,
            )
        except ImportError:
            logger.warning("visualization.plot_utils not available -- skipping plots")
            return

        m = report.metrics_recommended
        t = report.threshold_report

        _plots: list[tuple[str, callable, tuple]] = [
            (
                "confusion_matrix",
                plot_confusion_matrix,
                (m.tp, m.tn, m.fp, m.fn,
                 self.viz_dir / "confusion_matrix.png",
                 rec_threshold),
            ),
            (
                "roc_curve",
                plot_roc_curve,
                (labels, phishing_probs,
                 self.viz_dir / "roc_curve.png",
                 rec_threshold),
            ),
            (
                "pr_curve",
                plot_pr_curve,
                (labels, phishing_probs,
                 self.viz_dir / "pr_curve.png",
                 rec_threshold),
            ),
        ]
        if t is not None and len(t.thresholds) > 0:
            _plots.append((
                "threshold_sweep",
                plot_threshold_sweep,
                (t.thresholds, t.f1_scores, t.f2_scores, t.fnr_scores,
                 t.precision_scores, t.recall_scores,
                 t.f1_threshold, t.f2_threshold, t.min_fnr_threshold,
                 self.viz_dir / "threshold_sweep.png"),
            ))

        for name, fn, args in _plots:
            try:
                result = fn(*args)
                if result is not None:
                    report.artifact_paths[name] = str(result)
                    logger.info("Plot saved: %s → %s", name, result)
            except NotImplementedError:
                logger.debug("Plot '%s' not yet implemented (stub)", name)
            except Exception as exc:
                logger.warning("Plot '%s' failed: %s", name, exc)


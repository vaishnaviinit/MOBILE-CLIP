"""
WeightedRandomSampler factory for class-imbalanced phishing dataset.

Why WeightedRandomSampler (not oversampling or undersampling):
  - Oversampling (repeat phishing images): risks overfitting on a small set.
  - Undersampling (drop legitimate images): wastes 70% of the training data.
  - WeightedRandomSampler: draws from the full dataset but samples phishing
    images more frequently in each epoch, effectively balancing the batch
    composition without duplicating or discarding data.

Combined with Focal Loss alpha-weighting, this gives double protection:
  - Sampler:    ensures roughly balanced batches at the data level.
  - Focal Loss: still penalizes phishing errors more at the loss level,
                catching any remaining imbalance the sampler doesn't fully fix.
"""

from __future__ import annotations

import logging
from collections import Counter

import torch
from torch.utils.data import WeightedRandomSampler

from datasets.phishing_dataset import ImageRecord

logger = logging.getLogger(__name__)


def compute_sample_weights(records: list[ImageRecord]) -> list[float]:
    """
    Compute per-sample inverse-frequency weights for WeightedRandomSampler.

    Algorithm:
      count[c] = number of records with label c
      weight[i] = 1.0 / count[records[i].label]

    Effect: with dataset of 2435 legit and 385 phishing in the train split,
      weight_legit   = 1 / 2435 ≈ 0.000411
      weight_phishing = 1 / 385 ≈ 0.002597
    → phishing is drawn ~6.3x more per step.

    Args:
        records: List of ImageRecord objects (training split only).

    Returns:
        List of float weights, one per record, same order as records.
    """
    counts: Counter = Counter(r.label for r in records)

    if 0 not in counts or 1 not in counts:
        logger.warning(
            "Training split is missing a class! counts=%s — "
            "sampler may not work correctly.",
            dict(counts),
        )

    weights: list[float] = [1.0 / counts[r.label] for r in records]
    return weights


def build_weighted_sampler(
    records: list[ImageRecord],
    replacement: bool = True,
) -> WeightedRandomSampler:
    """
    Build a WeightedRandomSampler that balances legitimate and phishing samples.

    num_samples is set to len(records) so one sampler epoch processes roughly
    the same number of gradient steps as a standard epoch — the class
    composition is balanced, not the total sample count.

    Args:
        records:     List of ImageRecord (training split from DatasetSplitter).
        replacement: Must be True for oversampling the minority class to work.

    Returns:
        WeightedRandomSampler ready to pass to DataLoader(sampler=...).
    """
    weights = compute_sample_weights(records)

    counts: Counter = Counter(r.label for r in records)
    logger.info(
        "WeightedRandomSampler | legit=%d (w=%.5f) | phishing=%d (w=%.5f) | "
        "num_samples=%d | replacement=%s",
        counts[0],
        1.0 / max(counts[0], 1),
        counts[1],
        1.0 / max(counts[1], 1),
        len(records),
        replacement,
    )

    return WeightedRandomSampler(
        weights=torch.tensor(weights, dtype=torch.float64),
        num_samples=len(records),
        replacement=replacement,
    )

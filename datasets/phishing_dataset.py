"""
Auto-discovering dataset for phishing website screenshot classification.

Recursively walks dataset/legitimate/ and dataset/phishing/ to find all images.
Never hardcodes brand names or folder counts — scales to any dataset size.
Performs brand-aware train/val/test splitting to prevent brand memorization.
"""

from __future__ import annotations

import logging
import random
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import torch
from PIL import Image
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

# Class index contract — shared across the entire codebase
LABEL_MAP: dict[str, int] = {"legitimate": 0, "phishing": 1}
IDX_TO_CLASS: dict[int, str] = {v: k for k, v in LABEL_MAP.items()}
PHISHING_CLASS_IDX: int = 1
LEGITIMATE_CLASS_IDX: int = 0


@dataclass
class ImageRecord:
    """Single dataset record."""

    path: Path
    label: int          # 0 = legitimate, 1 = phishing
    class_name: str     # "legitimate" or "phishing"
    brand: str          # brand/subfolder name — used for stratified splitting


@dataclass
class DatasetStats:
    """Summary statistics printed after dataset discovery."""

    total: int
    n_legitimate: int
    n_phishing: int
    n_brands_legitimate: int
    n_brands_phishing: int
    class_weights: dict[str, float] = field(default_factory=dict)

    def __str__(self) -> str:
        ratio = self.n_legitimate / max(self.n_phishing, 1)
        weight_str = " | ".join(
            f"{k} weight: {v:.4f}" for k, v in self.class_weights.items()
        )
        return (
            f"Dataset: {self.total} images\n"
            f"  Legitimate : {self.n_legitimate:>5} images  ({self.n_brands_legitimate} brands)\n"
            f"  Phishing   : {self.n_phishing:>5} images  ({self.n_brands_phishing} brands)\n"
            f"  Imbalance  : {ratio:.2f}:1 (legitimate:phishing)\n"
            f"  {weight_str}"
        )


class PhishingDataset(Dataset):
    """
    PyTorch Dataset for phishing screenshot classification.

    Recursively discovers all PNG/JPG/JPEG/WEBP images under:
      dataset_root/legitimate/**
      dataset_root/phishing/**

    The dataset is designed to work at any scale — from the current
    ~2,300 images to 10,000+ without any code changes.

    Args:
        records:   List of ImageRecord objects (produced by DatasetSplitter).
        transform: Callable applied to each PIL image before returning.
        augment:   If True, heavy train augmentations are expected in transform.
    """

    VALID_EXTENSIONS: frozenset[str] = frozenset(
        {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
    )

    def __init__(
        self,
        records: list[ImageRecord],
        transform: Optional[Callable] = None,
        augment: bool = False,
    ) -> None:
        self.records = records
        self.transform = transform
        self.augment = augment
        self._label_tensor_cache: dict[int, torch.Tensor] = {}

    # ------------------------------------------------------------------
    # PyTorch Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (image_tensor, label_tensor) for a given index."""
        record = self.records[idx]
        image = self._load_image(record.path)

        if self.transform is not None:
            image = self.transform(image)

        return image, self._label_to_tensor(record.label)

    # ------------------------------------------------------------------
    # Class-level helpers
    # ------------------------------------------------------------------

    @classmethod
    def discover(cls, dataset_root: str | Path) -> list[ImageRecord]:
        """
        Walk dataset_root and return all ImageRecords.

        Expected layout:
          dataset_root/
            legitimate/<brand>/<image>
            phishing/<brand>/<image>

        Raises:
            FileNotFoundError: if dataset_root does not exist.
            ValueError: if no images are found in either class.
        """
        root = Path(dataset_root)
        if not root.exists():
            raise FileNotFoundError(f"Dataset root not found: {root.resolve()}")

        records: list[ImageRecord] = []

        for class_name, label in LABEL_MAP.items():
            class_dir = root / class_name
            if not class_dir.exists():
                raise FileNotFoundError(
                    f"Class directory not found: {class_dir.resolve()}\n"
                    f"Expected 'legitimate/' and 'phishing/' under {root.resolve()}"
                )

            class_images = 0
            # rglob for all files, filter by extension
            for path in sorted(class_dir.rglob("*")):
                if not path.is_file():
                    continue
                if path.suffix.lower() not in cls.VALID_EXTENSIONS:
                    continue

                # Brand = immediate subfolder under class_dir
                # e.g. legitimate/amazon/homepage.png → brand = "amazon"
                # e.g. phishing/generic/xyz.png      → brand = "generic"
                parts = path.relative_to(class_dir).parts
                brand = parts[0] if len(parts) > 1 else class_name

                records.append(
                    ImageRecord(
                        path=path,
                        label=label,
                        class_name=class_name,
                        brand=brand,
                    )
                )
                class_images += 1

            if class_images == 0:
                raise ValueError(
                    f"No images found in {class_dir.resolve()}. "
                    f"Supported extensions: {cls.VALID_EXTENSIONS}"
                )

            logger.info("Discovered %d %s images", class_images, class_name)

        logger.info("Total discovered: %d images", len(records))
        return records

    @classmethod
    def compute_stats(cls, records: list[ImageRecord]) -> DatasetStats:
        """Compute and return DatasetStats for a list of records."""
        n_legit = sum(1 for r in records if r.label == LEGITIMATE_CLASS_IDX)
        n_phish = sum(1 for r in records if r.label == PHISHING_CLASS_IDX)

        legit_brands = {r.brand for r in records if r.label == LEGITIMATE_CLASS_IDX}
        phish_brands = {r.brand for r in records if r.label == PHISHING_CLASS_IDX}

        weights_tensor = cls.compute_class_weights(records)
        class_weights = {
            "legitimate": round(weights_tensor[LEGITIMATE_CLASS_IDX].item(), 4),
            "phishing": round(weights_tensor[PHISHING_CLASS_IDX].item(), 4),
        }

        return DatasetStats(
            total=len(records),
            n_legitimate=n_legit,
            n_phishing=n_phish,
            n_brands_legitimate=len(legit_brands),
            n_brands_phishing=len(phish_brands),
            class_weights=class_weights,
        )

    @classmethod
    def compute_class_weights(cls, records: list[ImageRecord]) -> torch.Tensor:
        """
        Return a [2] tensor of inverse-frequency class weights.

        weight[c] = total / (n_classes * count[c])

        Used to initialize Focal Loss alpha weights.
        """
        n_total = len(records)
        n_classes = len(LABEL_MAP)
        counts = [0] * n_classes

        for r in records:
            counts[r.label] += 1

        weights = []
        for c in range(n_classes):
            if counts[c] == 0:
                logger.warning("Class %d has 0 samples — weight set to 0.0", c)
                weights.append(0.0)
            else:
                weights.append(n_total / (n_classes * counts[c]))

        return torch.tensor(weights, dtype=torch.float32)

    # ------------------------------------------------------------------
    # Accessors used by Trainer and Evaluator
    # ------------------------------------------------------------------

    def get_labels(self) -> list[int]:
        """Return all labels in dataset order — used by WeightedRandomSampler."""
        return [r.label for r in self.records]

    def get_image_paths(self) -> list[str]:
        """Return all image paths as strings — used by Evaluator for misclassification analysis."""
        return [str(r.path) for r in self.records]

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_image(self, path: Path) -> Image.Image:
        """Load image as RGB PIL Image; raises on corrupt files."""
        try:
            img = Image.open(path)
            # Convert to RGB — handles RGBA screenshots (transparency → white),
            # grayscale screenshots, and palette-mode images.
            return img.convert("RGB")
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load image: {path}\nOriginal error: {exc}"
            ) from exc

    def _label_to_tensor(self, label: int) -> torch.Tensor:
        """Cache scalar label tensors to avoid repeated allocation per batch."""
        if label not in self._label_tensor_cache:
            # dtype=long is required by nn.CrossEntropyLoss and FocalLoss
            self._label_tensor_cache[label] = torch.tensor(label, dtype=torch.long)
        return self._label_tensor_cache[label]


class DatasetSplitter:
    """
    Performs brand-aware stratified train/val/test splitting.

    Brand-awareness: all screenshots of the same brand (same subfolder)
    are kept in the same split, preventing the model from memorizing
    brand-specific pixel patterns rather than phishing visual cues.

    Concretely: if amazon/homepage.png is in train, amazon/signup.png
    is also in train — never split across sets.

    For phishing/generic (single flat folder, no sub-brands), splits
    are done at the file level with the same seed.

    Args:
        records:     Full list of ImageRecord objects from PhishingDataset.discover().
        train_frac:  Fraction of data for training (default 0.70).
        val_frac:    Fraction of data for validation (default 0.15).
        seed:        Random seed for reproducible splits.
    """

    def __init__(
        self,
        records: list[ImageRecord],
        train_frac: float = 0.70,
        val_frac: float = 0.15,
        seed: int = 42,
    ) -> None:
        if not (0 < train_frac < 1 and 0 < val_frac < 1):
            raise ValueError("train_frac and val_frac must be in (0, 1)")
        if train_frac + val_frac >= 1.0:
            raise ValueError("train_frac + val_frac must be < 1.0")

        self.records = records
        self.train_frac = train_frac
        self.val_frac = val_frac
        self.test_frac = 1.0 - train_frac - val_frac
        self.seed = seed

    def split(self) -> tuple[list[ImageRecord], list[ImageRecord], list[ImageRecord]]:
        """
        Return (train_records, val_records, test_records).

        Algorithm:
          1. Separate legitimate and phishing records.
          2. For legitimate: group by brand, shuffle brands, assign whole
             brand groups to train/val/test greedily by cumulative count.
          3. For phishing: file-level shuffle and split (no sub-brand structure).
          4. Merge and log split statistics.
        """
        legit = [r for r in self.records if r.label == LEGITIMATE_CLASS_IDX]
        phish = [r for r in self.records if r.label == PHISHING_CLASS_IDX]

        legit_groups = self._group_by_brand(legit)
        legit_train, legit_val, legit_test = self._split_by_brand(legit_groups)
        phish_train, phish_val, phish_test = self._split_by_file(phish)

        train = legit_train + phish_train
        val = legit_val + phish_val
        test = legit_test + phish_test

        self._log_split_stats(train, val, test)
        return train, val, test

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _group_by_brand(
        self, records: list[ImageRecord]
    ) -> dict[str, list[ImageRecord]]:
        """Group records by brand name."""
        groups: dict[str, list[ImageRecord]] = defaultdict(list)
        for r in records:
            groups[r.brand].append(r)
        return dict(groups)

    def _split_by_brand(
        self, brand_groups: dict[str, list[ImageRecord]]
    ) -> tuple[list[ImageRecord], list[ImageRecord], list[ImageRecord]]:
        """
        Assign brand groups to splits without breaking groups apart.

        Greedy algorithm:
          - Sort brand names (determinism), then shuffle with seed.
          - Accumulate brands into train until we reach train_frac of total.
          - Then accumulate into val until we reach val_frac of total.
          - Remainder goes to test.
        """
        rng = random.Random(self.seed)
        # Sort first so the shuffle is deterministic across Python versions
        brands = sorted(brand_groups.keys())
        rng.shuffle(brands)

        total = sum(len(v) for v in brand_groups.values())
        n_train_target = round(total * self.train_frac)
        n_val_target = round(total * self.val_frac)

        train: list[ImageRecord] = []
        val: list[ImageRecord] = []
        test: list[ImageRecord] = []
        n_train = 0
        n_val = 0

        for brand in brands:
            group = brand_groups[brand]
            if n_train < n_train_target:
                train.extend(group)
                n_train += len(group)
            elif n_val < n_val_target:
                val.extend(group)
                n_val += len(group)
            else:
                test.extend(group)

        return train, val, test

    def _split_by_file(
        self, records: list[ImageRecord]
    ) -> tuple[list[ImageRecord], list[ImageRecord], list[ImageRecord]]:
        """
        File-level split for phishing records (no meaningful brand structure).

        Uses seed + 1 to avoid correlation with the brand-level split.
        Sorts by path first for cross-platform determinism.
        """
        rng = random.Random(self.seed + 1)
        shuffled = sorted(records, key=lambda r: str(r.path))
        rng.shuffle(shuffled)

        n = len(shuffled)
        n_train = round(n * self.train_frac)
        n_val = round(n * self.val_frac)

        train = shuffled[:n_train]
        val = shuffled[n_train : n_train + n_val]
        test = shuffled[n_train + n_val :]

        return train, val, test

    @staticmethod
    def _log_split_stats(
        train: list[ImageRecord],
        val: list[ImageRecord],
        test: list[ImageRecord],
    ) -> None:
        """Log per-split class counts to verify the split is sensible."""
        for name, records in [("Train", train), ("Val", val), ("Test", test)]:
            n_legit = sum(1 for r in records if r.label == LEGITIMATE_CLASS_IDX)
            n_phish = sum(1 for r in records if r.label == PHISHING_CLASS_IDX)
            total = len(records)
            pct = total / max(sum(map(len, [train, val, test])), 1) * 100
            logger.info(
                "%s: %d images (%.1f%%) | legit=%d  phishing=%d",
                name,
                total,
                pct,
                n_legit,
                n_phish,
            )

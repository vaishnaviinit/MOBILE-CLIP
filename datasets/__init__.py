"""Dataset package: auto-discovering PhishingDataset and transform pipelines."""

from datasets.phishing_dataset import PhishingDataset, DatasetSplitter, DatasetStats
from datasets.transforms import build_train_transforms, build_val_transforms

__all__ = [
    "PhishingDataset",
    "DatasetSplitter",
    "DatasetStats",
    "build_train_transforms",
    "build_val_transforms",
]

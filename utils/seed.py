"""
Reproducibility: seed all random sources.

Seeds: Python random, NumPy, PyTorch (CPU + CUDA), and optionally sets
torch deterministic mode.

Note on determinism vs performance:
  torch.use_deterministic_algorithms(True) can slow down some CUDA
  operations (especially certain conv kernels). We enable it by default
  for reproducibility. Pass allow_nondeterministic=True to skip it when
  training speed matters more than bit-exact reproducibility.
"""

from __future__ import annotations

import logging
import os
import random

logger = logging.getLogger(__name__)


def seed_everything(seed: int = 42, allow_nondeterministic: bool = False) -> None:
    """
    Set all random seeds for full reproducibility.

    Args:
        seed:                  Integer seed. Config default: 42.
        allow_nondeterministic: If True, skip deterministic algorithm enforcement.
    """
    import numpy as np
    import torch

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False  # disable auto-tuner for reproducibility

    if not allow_nondeterministic:
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except AttributeError:
            pass  # torch < 1.8

    logger.info("Seeded everything with seed=%d", seed)

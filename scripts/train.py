"""
Training entry point.

Usage:
    python scripts/train.py
    python scripts/train.py --config configs/config.yaml
    python scripts/train.py --resume outputs/checkpoints/last_checkpoint.pt
    python scripts/train.py --override training.batch_size=64 model.dropout=0.4

All hyperparameters are read from configs/config.yaml.
Override any dotted config path with --override key=value:
    python scripts/train.py --override training.epochs=30 training.optimizer.lr_head=5e-4
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import yaml
from torch.utils.data import DataLoader

from datasets.phishing_dataset import PhishingDataset, DatasetSplitter
from datasets.transforms import build_transforms_from_config
from models.backbone import MobileCLIPBackbone
from models.classifier import PhishingClassifier, ClassifierConfig
from training.sampler import build_weighted_sampler
from training.trainer import Trainer, TrainerConfig
from utils.logging_utils import setup_logging, get_logger
from utils.seed import seed_everything

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train MobileCLIP phishing screenshot classifier",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config", type=str, default="configs/config.yaml",
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--resume", type=str, default=None,
        help="Path to last_checkpoint.pt to resume training from",
    )
    parser.add_argument(
        "--override", nargs="*", default=[],
        metavar="KEY=VALUE",
        help="Override config values (e.g. training.batch_size=64)",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def apply_overrides(config: dict, overrides: list[str]) -> dict:
    """
    Apply --override key=value pairs to the parsed config dict.

    Keys use dot notation to address nested fields:
        "training.batch_size=64"
        "training.optimizer.lr_head=5e-4"
        "model.dropout=0.4"
    """
    for override in overrides:
        if "=" not in override:
            raise ValueError(
                f"Override must be in 'key=value' format, got: '{override}'"
            )
        key_path, value_str = override.split("=", 1)
        keys = [k.strip() for k in key_path.strip().split(".")]

        # Navigate to the parent dict
        node = config
        for key in keys[:-1]:
            if key not in node:
                raise KeyError(
                    f"Override key path '{key_path}' — '{key}' not found in config. "
                    f"Available: {list(node.keys())}"
                )
            node = node[key]

        # Parse the value: try int → float → bool → str
        final_key = keys[-1]
        value: int | float | bool | str
        value_lower = value_str.strip().lower()
        if value_lower in ("true", "yes"):
            value = True
        elif value_lower in ("false", "no"):
            value = False
        else:
            try:
                value = int(value_str)
            except ValueError:
                try:
                    value = float(value_str)
                except ValueError:
                    value = value_str

        old = node.get(final_key, "<not set>")
        node[final_key] = value
        logger.info("Override: %s = %r  (was: %r)", key_path, value, old)

    return config


# ---------------------------------------------------------------------------
# Component factories
# ---------------------------------------------------------------------------

def build_dataloaders(
    config: dict,
) -> tuple[DataLoader, DataLoader, DataLoader, torch.Tensor]:
    """
    Discover dataset, split, build transforms, and assemble DataLoaders.

    Returns:
        (train_loader, val_loader, test_loader, class_weights [2])
    """
    data_cfg = config["data"]
    path_cfg = config["paths"]
    train_cfg = config["training"]

    # -- Dataset discovery and splitting --------------------------------
    records = PhishingDataset.discover(path_cfg["dataset_root"])
    stats = PhishingDataset.compute_stats(records)
    logger.info("Dataset:\n%s", stats)

    splitter = DatasetSplitter(
        records,
        train_frac=data_cfg["train_split"],
        val_frac=data_cfg["val_split"],
        seed=config["project"]["seed"],
    )
    train_rec, val_rec, test_rec = splitter.split()

    # Class weights from training split only (not val/test)
    class_weights = PhishingDataset.compute_class_weights(train_rec)
    logger.info(
        "Class weights (from train split): legit=%.4f  phishing=%.4f",
        class_weights[0].item(),
        class_weights[1].item(),
    )

    # -- Transforms -----------------------------------------------------
    train_tfm, val_tfm = build_transforms_from_config(config)

    # -- PyTorch Datasets -----------------------------------------------
    train_ds = PhishingDataset(train_rec, transform=train_tfm, augment=True)
    val_ds   = PhishingDataset(val_rec,   transform=val_tfm)
    test_ds  = PhishingDataset(test_rec,  transform=val_tfm)

    # -- WeightedRandomSampler (for train only) -------------------------
    sampler = build_weighted_sampler(train_rec)

    # -- DataLoader config ----------------------------------------------
    batch_size   = int(train_cfg["batch_size"])
    num_workers  = int(data_cfg.get("num_workers", 0))
    pin_memory   = bool(data_cfg.get("pin_memory", False))
    # persistent_workers requires num_workers > 0
    persistent   = num_workers > 0

    dl_kwargs = dict(
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        sampler=sampler,     # sampler is mutually exclusive with shuffle
        **dl_kwargs,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        **dl_kwargs,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        **dl_kwargs,
    )

    logger.info(
        "DataLoaders ready | train=%d batches | val=%d batches | test=%d batches",
        len(train_loader),
        len(val_loader),
        len(test_loader),
    )
    return train_loader, val_loader, test_loader, class_weights


def build_model(config: dict) -> PhishingClassifier:
    """Construct a fresh PhishingClassifier from config."""
    model_cfg = config["model"]

    backbone = MobileCLIPBackbone(
        model_name=model_cfg["backbone"],
        pretrained=model_cfg["pretrained"],
        embedding_dim=int(model_cfg["embedding_dim"]),
        normalize=True,
    )
    head_cfg = ClassifierConfig(
        embedding_dim=int(model_cfg["embedding_dim"]),
        hidden_dim=int(model_cfg["hidden_dim"]),
        num_classes=2,
        dropout=float(model_cfg["dropout"]),
        use_layer_norm=True,
    )
    model = PhishingClassifier(backbone, head_cfg)

    param_counts = model.n_parameters(trainable_only=False)
    logger.info(
        "Model built | backbone=%.1fM | head=%.1fM | total=%.1fM",
        param_counts["backbone"] / 1e6,
        param_counts["head"] / 1e6,
        param_counts["total"] / 1e6,
    )
    return model


def build_trainer_config(
    config: dict,
    class_weights: torch.Tensor,
) -> TrainerConfig:
    """Assemble TrainerConfig from the parsed YAML config dict."""
    train_cfg = config["training"]
    opt_cfg   = train_cfg["optimizer"]
    sch_cfg   = train_cfg["scheduler"]
    loss_cfg  = train_cfg["loss"]
    es_cfg    = train_cfg["early_stopping"]
    path_cfg  = config["paths"]

    return TrainerConfig(
        epochs=int(train_cfg["epochs"]),
        freeze_backbone_epochs=int(config["model"]["freeze_backbone_epochs"]),

        lr_head=float(opt_cfg["lr_head"]),
        # Phase 2 head LR: 10x lower than phase 1 (already adapated)
        lr_head_phase2=float(opt_cfg.get("lr_head_phase2", opt_cfg["lr_head"] / 10)),
        lr_backbone=float(opt_cfg["lr_backbone"]),
        weight_decay=float(opt_cfg["weight_decay"]),
        betas=(float(opt_cfg["betas"][0]), float(opt_cfg["betas"][1])),
        eps=float(opt_cfg["eps"]),

        warmup_epochs=int(sch_cfg["warmup_epochs"]),
        min_lr=float(sch_cfg["min_lr"]),

        mixed_precision=bool(train_cfg["mixed_precision"]),
        gradient_clip=float(train_cfg["gradient_clip"]),

        device=str(train_cfg.get("device", "auto")),
        seed=int(config["project"]["seed"]),

        checkpoint_dir=str(path_cfg["checkpoint_dir"]),
        log_dir=str(path_cfg["log_dir"]),

        early_stopping_patience=int(es_cfg["patience"]),
        early_stopping_monitor=str(es_cfg["monitor"]),

        loss_name=str(loss_cfg["name"]),
        focal_gamma=float(loss_cfg.get("gamma", 2.0)),
        label_smoothing=float(loss_cfg.get("label_smoothing", 0.0)),

        class_weights=class_weights,
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    with open(args.config, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if args.override:
        config = apply_overrides(config, args.override)

    setup_logging(
        log_dir=config["paths"]["log_dir"],
        level=config["logging"]["level"],
        console=bool(config["logging"]["console"]),
    )
    seed_everything(int(config["project"]["seed"]))

    logger.info(
        "=" * 60 + "\n  Phishing CLIP Classifier — Training\n" + "=" * 60
    )
    logger.info("Config: %s", args.config)
    if args.resume:
        logger.info("Resuming from: %s", args.resume)

    # Build all components
    train_loader, val_loader, test_loader, class_weights = build_dataloaders(config)
    model = build_model(config)
    trainer_cfg = build_trainer_config(config, class_weights)

    # Train
    trainer = Trainer(model, train_loader, val_loader, trainer_cfg)
    result  = trainer.train(resume_from=args.resume)

    logger.info("Training complete.")
    logger.info("Best checkpoint : %s", result["best_checkpoint"])
    logger.info(
        "Final metrics   : %s",
        {k: f"{v:.4f}" for k, v in result["final_metrics"].items()
         if isinstance(v, float)},
    )

    # Optionally run evaluation on the test set immediately
    logger.info("Run evaluation with: python scripts/evaluate.py")


if __name__ == "__main__":
    main()

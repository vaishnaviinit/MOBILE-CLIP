"""
Evaluation entry point — run the full test-set evaluation suite.

Usage:
    python scripts/evaluate.py
    python scripts/evaluate.py --checkpoint outputs/checkpoints/best_model.pt
    python scripts/evaluate.py --threshold 0.35
    python scripts/evaluate.py --strategy min_fnr
    python scripts/evaluate.py --gradcam
    python scripts/evaluate.py --attention

Outputs:
    outputs/predictions/predictions.json
    outputs/predictions/false_negatives.json
    outputs/predictions/false_positives.json
    outputs/predictions/classification_report.txt
    outputs/predictions/evaluation_report.json
    outputs/visualizations/confusion_matrix.png
    outputs/visualizations/roc_curve.png
    outputs/visualizations/pr_curve.png
    outputs/visualizations/threshold_sweep.png
    outputs/visualizations/gradcam/   (if --gradcam)
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
from evaluation.evaluator import Evaluator
from models.backbone import MobileCLIPBackbone
from models.classifier import PhishingClassifier, ClassifierConfig
from utils.device import resolve_device
from utils.logging_utils import setup_logging, get_logger
from utils.seed import seed_everything

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate trained MobileCLIP phishing classifier on the test set",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config", type=str, default="configs/config.yaml",
    )
    parser.add_argument(
        "--checkpoint", type=str,
        default="outputs/checkpoints/best_model.pt",
        help="Path to best_model.pt checkpoint",
    )
    parser.add_argument(
        "--threshold", type=float, default=None,
        help="Hard-override the decision threshold (skips optimization)",
    )
    parser.add_argument(
        "--strategy", type=str, default=None,
        choices=["f1", "f2", "min_fnr"],
        help="Threshold selection strategy (overrides config)",
    )
    parser.add_argument(
        "--gradcam", action="store_true",
        help="Run GradCAM on misclassified samples",
    )
    parser.add_argument(
        "--attention", action="store_true",
        help="Run attention-map visualization on misclassified samples",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Component factories (mirror train.py — keeping scripts self-contained)
# ---------------------------------------------------------------------------

def _load_model(config: dict, checkpoint_path: str, device: torch.device) -> PhishingClassifier:
    """Build the model architecture and restore fine-tuned weights."""
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
    )
    model = PhishingClassifier(backbone, head_cfg)

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()

    epoch = ckpt.get("epoch", "?")
    val_f2 = ckpt.get("metrics", {}).get("val_f2", float("nan"))
    logger.info(
        "Loaded checkpoint: %s | epoch=%s | val_f2=%.4f",
        Path(checkpoint_path).name,
        epoch,
        val_f2,
    )
    return model


def _build_test_loader(config: dict) -> DataLoader:
    """Reproduce the same split used during training and return the test DataLoader."""
    data_cfg = config["data"]
    path_cfg = config["paths"]

    records = PhishingDataset.discover(path_cfg["dataset_root"])
    splitter = DatasetSplitter(
        records,
        train_frac=data_cfg["train_split"],
        val_frac=data_cfg["val_split"],
        seed=int(config["project"]["seed"]),
    )
    _, _, test_rec = splitter.split()

    _, val_tfm = build_transforms_from_config(config)
    test_ds = PhishingDataset(test_rec, transform=val_tfm)

    num_workers = int(data_cfg.get("num_workers", 0))
    return DataLoader(
        test_ds,
        batch_size=int(config["training"]["batch_size"]),
        shuffle=False,      # MUST be False — Evaluator reads paths by dataset order
        num_workers=num_workers,
        pin_memory=bool(data_cfg.get("pin_memory", False)),
        persistent_workers=num_workers > 0,
    )


# ---------------------------------------------------------------------------
# GradCAM / Attention helpers
# ---------------------------------------------------------------------------

def _run_gradcam(
    model: PhishingClassifier,
    report,
    config: dict,
    device: torch.device,
) -> None:
    """Run GradCAM on false-negative and false-positive samples."""
    try:
        from visualization.gradcam import GradCAMVisualizer
        from datasets.transforms import build_val_transforms
        from PIL import Image
        import torch

        output_dir = Path(config["paths"]["visualization_dir"]) / "gradcam"
        output_dir.mkdir(parents=True, exist_ok=True)

        vis = GradCAMVisualizer(model, device=str(device))
        val_tfm = build_val_transforms(int(config["data"]["image_size"]))

        samples = (
            report.false_negatives[:10] +   # top 10 missed phishing
            report.false_positives[:5]        # top 5 false alarms
        )
        for record in samples:
            if not record["path"]:
                continue
            img = Image.open(record["path"]).convert("RGB")
            tensor = val_tfm(img).unsqueeze(0).to(device)
            heatmap, meta = vis.generate(tensor)
            logger.info(
                "GradCAM: %s → %s (P=%.3f)",
                Path(record["path"]).name,
                meta.get("predicted_class", "?"),
                record["phishing_probability"],
            )
        vis.cleanup()
    except NotImplementedError:
        logger.info("GradCAM visualization not yet implemented (stub)")
    except Exception as exc:
        logger.warning("GradCAM failed: %s", exc)


def _run_attention(
    model: PhishingClassifier,
    report,
    config: dict,
    device: torch.device,
) -> None:
    """Run attention-map visualization on false-negative samples."""
    try:
        from visualization.attention_vis import AttentionVisualizer
        from datasets.transforms import build_val_transforms
        from PIL import Image

        output_dir = Path(config["paths"]["visualization_dir"]) / "attention"
        output_dir.mkdir(parents=True, exist_ok=True)

        vis = AttentionVisualizer(model, device=str(device))
        val_tfm = build_val_transforms(int(config["data"]["image_size"]))

        for record in report.false_negatives[:10]:
            if not record["path"]:
                continue
            img = Image.open(record["path"]).convert("RGB")
            tensor = val_tfm(img).unsqueeze(0).to(device)
            out_path = output_dir / (Path(record["path"]).stem + "_attn.png")
            vis.visualize_and_save(record["path"], tensor, out_path)
        vis.cleanup()
    except NotImplementedError:
        logger.info("Attention visualization not yet implemented (stub)")
    except Exception as exc:
        logger.warning("Attention visualization failed: %s", exc)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    with open(args.config, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    setup_logging(
        log_dir=config["paths"]["log_dir"],
        level="INFO",
        console=True,
    )
    seed_everything(int(config["project"]["seed"]))

    logger.info(
        "=" * 60 + "\n  Phishing CLIP Classifier -- Evaluation\n" + "=" * 60
    )
    logger.info("Checkpoint: %s", args.checkpoint)

    device = resolve_device(config["training"].get("device", "auto"))

    # -- Load model -------------------------------------------------------
    model = _load_model(config, args.checkpoint, device)

    # -- Build test DataLoader --------------------------------------------
    test_loader = _build_test_loader(config)
    logger.info("Test set: %d samples", len(test_loader.dataset))

    # -- Threshold strategy -----------------------------------------------
    eval_cfg = config["evaluation"]
    strategy = args.strategy or str(eval_cfg.get("threshold_strategy", "f2"))
    min_fnr_target = float(eval_cfg.get("min_fnr_target", 0.05))

    # -- Run Evaluator ----------------------------------------------------
    evaluator = Evaluator(
        model=model,
        test_loader=test_loader,
        output_dir=config["paths"]["output_dir"],
        threshold_strategy=strategy,
        min_fnr_target=min_fnr_target,
        device=str(device),
    )

    report = evaluator.evaluate()

    # -- Override threshold if explicitly provided -------------------------
    if args.threshold is not None:
        logger.info(
            "Threshold overridden by --threshold %.3f (optimizer chose %.3f)",
            args.threshold,
            report.recommended_threshold,
        )
        report.recommended_threshold = args.threshold

    # -- Optional: interpretability visualizations ------------------------
    if args.gradcam:
        logger.info("Running GradCAM visualizations …")
        _run_gradcam(model, report, config, device)

    if args.attention:
        logger.info("Running attention-map visualizations …")
        _run_attention(model, report, config, device)

    # -- Print final report -----------------------------------------------
    print(report.summary())

    logger.info(
        "Evaluation done. Recommended threshold for inference: %.4f",
        report.recommended_threshold,
    )
    logger.info(
        "To run inference with this threshold:\n"
        "  python scripts/infer.py --image <path> --threshold %.4f",
        report.recommended_threshold,
    )


if __name__ == "__main__":
    main()

"""
Inference entry point — classify a single website screenshot.

Usage:
    python scripts/infer.py --image path/to/screenshot.png
    python scripts/infer.py --image path/to/screenshot.png --threshold 0.35
    python scripts/infer.py --image path/to/screenshot.png --gradcam
    python scripts/infer.py --image path/to/screenshot.png --embed-only

Output (stdout JSON):
    {
      "image": "path/to/screenshot.png",
      "predicted_class": "phishing",
      "confidence": 0.9412,
      "phishing_probability": 0.9412,
      "legitimate_probability": 0.0588,
      "threshold_used": 0.350,
      "inference_time_ms": 18.4,
      "model": "MobileCLIP-S2",
      "embedding_dim": 512
    }

Threshold priority:
  1. --threshold CLI argument (highest)
  2. outputs/predictions/evaluation_report.json recommended_threshold (if exists)
  3. configs/config.yaml evaluation.threshold (fallback, default 0.5)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import yaml
from PIL import Image

from datasets.transforms import build_val_transforms
from models.backbone import MobileCLIPBackbone
from models.classifier import PhishingClassifier, ClassifierConfig
from utils.device import resolve_device
from utils.logging_utils import setup_logging, get_logger

logger = get_logger(__name__)

IDX_TO_CLASS = {0: "legitimate", 1: "phishing"}


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Classify a single website screenshot as phishing or legitimate",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--image", type=str, required=True,
        help="Path to the screenshot (PNG/JPG/WEBP) to classify",
    )
    parser.add_argument(
        "--config", type=str, default="configs/config.yaml",
    )
    parser.add_argument(
        "--checkpoint", type=str,
        default="outputs/checkpoints/best_model.pt",
        help="Trained model checkpoint",
    )
    parser.add_argument(
        "--threshold", type=float, default=None,
        help="Decision threshold (overrides auto-detected value)",
    )
    parser.add_argument(
        "--gradcam", action="store_true",
        help="Generate and save a GradCAM heatmap",
    )
    parser.add_argument(
        "--gradcam-output", type=str, default=None,
        help="Path for GradCAM overlay image (default: <image_name>_gradcam.png)",
    )
    parser.add_argument(
        "--embed-only", action="store_true",
        help="Only output the embedding vector (for downstream ensemble use)",
    )
    parser.add_argument(
        "--no-json", action="store_true",
        help="Print human-readable output instead of JSON",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(
    config: dict,
    checkpoint_path: str,
    device: torch.device,
) -> PhishingClassifier:
    """
    Build the model architecture from config and restore fine-tuned weights.

    Note: The backbone is initialised with pretrained OpenCLIP weights first,
    then the full fine-tuned state_dict from the checkpoint overwrites them.
    This is unavoidable given the current architecture but is a one-time cost.
    """
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

    ckpt_path = Path(checkpoint_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt_path.resolve()}\n"
            f"Train first with: python scripts/train.py"
        )

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()

    logger.debug(
        "Loaded %s (epoch %s)", ckpt_path.name, ckpt.get("epoch", "?")
    )
    return model


# ---------------------------------------------------------------------------
# Threshold resolution
# ---------------------------------------------------------------------------

def resolve_threshold(
    cli_threshold: float | None,
    config: dict,
) -> tuple[float, str]:
    """
    Determine the decision threshold with a clear priority chain.

    Returns:
        (threshold, source_description)
    """
    # Priority 1: explicit CLI override
    if cli_threshold is not None:
        return cli_threshold, "CLI --threshold"

    # Priority 2: recommended threshold from a previous evaluate.py run
    report_path = Path(config["paths"]["prediction_dir"]) / "evaluation_report.json"
    if report_path.exists():
        try:
            with open(report_path, encoding="utf-8") as f:
                eval_report = json.load(f)
            rec = float(eval_report.get("recommended_threshold", 0.5))
            return rec, f"evaluation_report.json (strategy: {config['evaluation'].get('threshold_strategy', 'f2')})"
        except (KeyError, ValueError, json.JSONDecodeError):
            pass

    # Priority 3: config default
    fallback = float(config["evaluation"].get("threshold", 0.5))
    return fallback, "config.yaml (default)"


# ---------------------------------------------------------------------------
# Core inference
# ---------------------------------------------------------------------------

def run_inference(
    model: PhishingClassifier,
    image_path: str,
    threshold: float,
    device: torch.device,
    image_size: int = 256,
) -> dict:
    """
    Classify a single image and return a structured result dict.

    Timing covers the full model forward pass including device transfer
    but excludes image loading and preprocessing (those are I/O costs
    that vary independently of model complexity).

    Args:
        model:      PhishingClassifier in eval mode, on the target device.
        image_path: Path to the screenshot file.
        threshold:  Decision threshold on P(phishing).
        device:     Torch device.
        image_size: Input resolution (must match model training config).

    Returns:
        Dict with prediction, probabilities, timing, and embedding shape.
    """
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path.resolve()}")

    # -- Preprocessing ------------------------------------------------
    try:
        image = Image.open(path).convert("RGB")
    except Exception as exc:
        raise RuntimeError(f"Failed to open image {path}: {exc}") from exc

    transform = build_val_transforms(image_size)
    tensor: torch.Tensor = transform(image).unsqueeze(0).to(device)  # [1, 3, H, W]

    # -- Forward pass (timed) -----------------------------------------
    t_start = time.perf_counter()
    with torch.no_grad():
        outputs = model(tensor)
    t_end = time.perf_counter()

    inference_ms = (t_end - t_start) * 1000.0

    # -- Result assembly -----------------------------------------------
    phishing_prob = float(outputs["probs"][0, 1])
    legit_prob    = float(outputs["probs"][0, 0])
    predicted_idx = int(phishing_prob >= threshold)
    predicted_cls = IDX_TO_CLASS[predicted_idx]
    confidence    = phishing_prob if predicted_idx == 1 else legit_prob

    return {
        "image": str(path),
        "predicted_class": predicted_cls,
        "confidence": round(confidence, 6),
        "phishing_probability": round(phishing_prob, 6),
        "legitimate_probability": round(legit_prob, 6),
        "threshold_used": round(threshold, 6),
        "inference_time_ms": round(inference_ms, 2),
        "model": model.backbone.model_name,
        "embedding_dim": int(outputs["embeddings"].shape[-1]),
    }


def run_embed_only(
    model: PhishingClassifier,
    image_path: str,
    device: torch.device,
    image_size: int = 256,
) -> dict:
    """
    Extract only the visual embedding vector (for ensemble downstream use).

    Returns:
        Dict with 'embedding' list[float] of length 512, plus timing.
    """
    path = Path(image_path)
    image = Image.open(path).convert("RGB")
    transform = build_val_transforms(image_size)
    tensor = transform(image).unsqueeze(0).to(device)

    t_start = time.perf_counter()
    with torch.no_grad():
        embedding = model.backbone.extract_features(tensor)  # [1, 512]
    t_ms = (time.perf_counter() - t_start) * 1000.0

    return {
        "image": str(path),
        "embedding": embedding[0].cpu().tolist(),
        "embedding_dim": embedding.shape[-1],
        "inference_time_ms": round(t_ms, 2),
        "model": model.backbone.model_name,
    }


# ---------------------------------------------------------------------------
# GradCAM helper
# ---------------------------------------------------------------------------

def run_gradcam(
    model: PhishingClassifier,
    image_path: str,
    output_path: str | None,
    device: torch.device,
    image_size: int = 256,
) -> str | None:
    """
    Generate and save a GradCAM heatmap overlay.

    Returns:
        Path to the saved overlay image, or None if not available.
    """
    try:
        from visualization.gradcam import GradCAMVisualizer

        transform = build_val_transforms(image_size)
        image = Image.open(image_path).convert("RGB")
        tensor = transform(image).unsqueeze(0).to(device)

        vis = GradCAMVisualizer(model, device=str(device))
        heatmap, meta = vis.generate(tensor)

        if output_path is None:
            stem = Path(image_path).stem
            output_path = str(Path(image_path).parent / f"{stem}_gradcam.png")

        vis.cleanup()
        logger.info("GradCAM saved to: %s", output_path)
        return output_path

    except NotImplementedError:
        logger.info("GradCAM not yet implemented (stub)")
        return None
    except Exception as exc:
        logger.warning("GradCAM failed: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    with open(args.config, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Suppress noise in inference mode
    setup_logging(
        log_dir=config["paths"]["log_dir"],
        level="WARNING",
        console=False,      # don't pollute stdout — output is JSON
    )

    device = resolve_device(config["training"].get("device", "auto"))

    # -- Load model ---------------------------------------------------
    model = load_model(config, args.checkpoint, device)

    # -- Resolve threshold --------------------------------------------
    threshold, threshold_source = resolve_threshold(args.threshold, config)
    logger.info("Threshold: %.4f (source: %s)", threshold, threshold_source)

    # -- Inference ----------------------------------------------------
    image_size = int(config["data"]["image_size"])

    if args.embed_only:
        result = run_embed_only(model, args.image, device, image_size)
    else:
        result = run_inference(model, args.image, threshold, device, image_size)
        result["threshold_source"] = threshold_source

    # -- GradCAM (optional) -------------------------------------------
    if args.gradcam and not args.embed_only:
        cam_path = run_gradcam(
            model, args.image, args.gradcam_output, device, image_size
        )
        if cam_path:
            result["gradcam_path"] = cam_path

    # -- Output -------------------------------------------------------
    if args.no_json:
        if args.embed_only:
            print(f"Embedding ({result['embedding_dim']}d) extracted in "
                  f"{result['inference_time_ms']:.1f}ms")
        else:
            verdict = result["predicted_class"].upper()
            prob    = result["phishing_probability"]
            conf    = result["confidence"]
            ms      = result["inference_time_ms"]
            print(
                f"\n{'='*40}\n"
                f"  Verdict  : {verdict}\n"
                f"  Phishing : {prob:.4f}\n"
                f"  Legit    : {result['legitimate_probability']:.4f}\n"
                f"  Confidence : {conf:.4f}\n"
                f"  Threshold  : {result['threshold_used']:.4f} ({threshold_source})\n"
                f"  Time       : {ms:.1f}ms\n"
                f"{'='*40}"
            )
    else:
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

"""
Transform pipelines for training and validation/inference.

Design principles:
  - Training: conservative augmentations that preserve webpage semantics.
  - Validation/Inference: deterministic — resize + center crop + normalize only.
  - All normalization uses OpenCLIP's published mean/std for MobileCLIP-S2.
  - No horizontal flip (webpage logos and layouts are directional).
  - No aggressive rotations (webpages are always upright, never sideways).
  - No MixUp/CutMix (blending two pages is semantically meaningless).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torchvision.transforms as T


# OpenCLIP publishes these for MobileCLIP — match the pretrained normalization.
CLIP_MEAN: tuple[float, float, float] = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD: tuple[float, float, float] = (0.26862954, 0.26130258, 0.27577711)


@dataclass
class AugmentationConfig:
    """Mirror of configs/config.yaml augmentation section."""

    image_size: int = 256

    # RandomResizedCrop
    rrc_scale_min: float = 0.85
    rrc_scale_max: float = 1.0
    rrc_ratio_min: float = 0.9
    rrc_ratio_max: float = 1.1

    # ColorJitter
    brightness: float = 0.15
    contrast: float = 0.15
    saturation: float = 0.10
    hue: float = 0.0          # intentionally 0 — brand colors must not shift

    # GaussianBlur
    blur_p: float = 0.15
    blur_kernel: int = 3
    blur_sigma_min: float = 0.1
    blur_sigma_max: float = 0.5

    # RandomGrayscale
    grayscale_p: float = 0.03

    # RandomErasing
    erasing_p: float = 0.10
    erasing_scale_min: float = 0.01
    erasing_scale_max: float = 0.08


def build_train_transforms(cfg: Optional[AugmentationConfig] = None) -> T.Compose:
    """
    Build the training augmentation pipeline.

    Transform order explanation:
      1. Resize (PIL)         — ensure minimum size before crop (avoids upscaling artifacts)
      2. RandomResizedCrop    — tight zoom variation; preserves layout, avoids distortion
      3. ColorJitter          — screen calibration / gamma differences across capture tools
      4. RandomGrayscale      — rare grayscale captures; forces model off pure color signals
      5. RandomApply(Blur)    — screenshot compression artifacts and low-res captures
      6. ToTensor             — PIL → [0, 1] float32 tensor
      7. Normalize            — shift to CLIP pretrained distribution
      8. RandomErasing        — partial page loads, browser chrome overlap, dynamic placeholders

    Intentionally excluded:
      - RandomHorizontalFlip:  logos/navbars are directional; flipping changes meaning
      - Rotation > 2°:         webpages are always upright
      - Perspective warp:      destroys grid layout alignment
      - Strong saturation/hue: brand colors are a legitimate visual signal

    Args:
        cfg: AugmentationConfig dataclass. Uses defaults if None.

    Returns:
        torchvision.transforms.Compose suitable for DataLoader.
    """
    cfg = cfg or AugmentationConfig()

    return T.Compose([
        # Step 1: Upscale to slightly above target before crop to avoid
        # introducing small black borders on screenshots with aspect ratio != 1.
        T.Resize(
            (int(cfg.image_size * 1.15), int(cfg.image_size * 1.15)),
            interpolation=T.InterpolationMode.BICUBIC,
            antialias=True,
        ),

        # Step 2: Random crop within tight scale/ratio bounds.
        # scale=(0.85, 1.0): at most 15% zoom in — keeps full layout visible.
        # ratio=(0.9, 1.1): near-square crops only — webpages aren't extreme panoramas.
        # BICUBIC: best quality for downsampling text and fine UI elements.
        T.RandomResizedCrop(
            cfg.image_size,
            scale=(cfg.rrc_scale_min, cfg.rrc_scale_max),
            ratio=(cfg.rrc_ratio_min, cfg.rrc_ratio_max),
            interpolation=T.InterpolationMode.BICUBIC,
            antialias=True,
        ),

        # Step 3: Mild color jitter.
        # brightness + contrast: monitor calibration and OS dark/light theme variation.
        # saturation: slight variation acceptable (not brand-critical).
        # hue=0.0: intentionally disabled — shifting PayPal blue to red would be
        #          semantically deceptive and could teach wrong color-class associations.
        T.ColorJitter(
            brightness=cfg.brightness,
            contrast=cfg.contrast,
            saturation=cfg.saturation,
            hue=cfg.hue,
        ),

        # Step 4: Very rare grayscale conversion (p=0.03).
        # Handles tools that capture screenshots without color (accessibility tools,
        # headless browsers in certain configs). Trains the model to use layout,
        # not just color, as a feature.
        T.RandomGrayscale(p=cfg.grayscale_p),

        # Step 5: Occasional mild blur (p=0.15).
        # Simulates JPEG compression artifacts from screenshot pipelines, low-DPI
        # captures, and network-throttled page renders. kernel_size=3 is minimal.
        T.RandomApply(
            [
                T.GaussianBlur(
                    kernel_size=cfg.blur_kernel,
                    sigma=(cfg.blur_sigma_min, cfg.blur_sigma_max),
                )
            ],
            p=cfg.blur_p,
        ),

        # Step 6: Convert PIL Image → [C, H, W] float32 in [0, 1].
        T.ToTensor(),

        # Step 7: Shift pixel distribution to match OpenCLIP pretraining.
        # Must match the pretrained MobileCLIP-S2 normalization exactly.
        T.Normalize(mean=CLIP_MEAN, std=CLIP_STD),

        # Step 8: Random rectangular patch erasing (p=0.10).
        # scale=(0.01, 0.08): small patches — simulates loading spinners,
        # cookie banners partially obscuring content, browser UI chrome overlap.
        # value='random': random noise, not black — less likely to create a
        # spurious training signal from a constant fill color.
        # Applied AFTER normalize so erased values don't break the color stats.
        T.RandomErasing(
            p=cfg.erasing_p,
            scale=(cfg.erasing_scale_min, cfg.erasing_scale_max),
            ratio=(0.3, 3.3),
            value="random",
        ),
    ])


def build_val_transforms(image_size: int = 256) -> T.Compose:
    """
    Build the deterministic validation/inference transform pipeline.

    No augmentation — only resize, center crop, and normalize.
    Identical behavior every call — required for reproducible evaluation.

    Resize to image_size * 1.14 then center-crop to image_size: this
    follows the standard CLIP evaluation protocol, preserving the center
    of the image (where the key UI elements tend to appear) while
    cropping peripheral whitespace or browser chrome.

    Args:
        image_size: Target square resolution (must match backbone input size).

    Returns:
        torchvision.transforms.Compose suitable for DataLoader.
    """
    # Resize slightly larger than target before center crop.
    # Ratio follows OpenCLIP's own evaluation preprocessing for MobileCLIP.
    resize_size = int(image_size * (292 / 256))  # ≈ 292 for image_size=256

    return T.Compose([
        T.Resize(
            resize_size,
            interpolation=T.InterpolationMode.BICUBIC,
            antialias=True,
        ),
        T.CenterCrop(image_size),
        T.ToTensor(),
        T.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
    ])


def build_transforms_from_config(config: dict) -> tuple[T.Compose, T.Compose]:
    """
    Convenience factory: parse YAML config dict and return (train_tfm, val_tfm).

    Reads from config["data"]["image_size"] and config["augmentation"].
    Gracefully falls back to AugmentationConfig defaults for any missing key.

    Args:
        config: Parsed config.yaml as a nested dict.

    Returns:
        (train_transforms, val_transforms)
    """
    image_size: int = config["data"]["image_size"]
    aug: dict = config.get("augmentation", {})

    rrc = aug.get("random_resized_crop", {})
    cj = aug.get("color_jitter", {})
    blur = aug.get("gaussian_blur", {})
    gs = aug.get("random_grayscale", {})
    erasing = aug.get("random_erasing", {})

    # Build from config, fall back to AugmentationConfig defaults per field
    defaults = AugmentationConfig()

    cfg = AugmentationConfig(
        image_size=image_size,
        rrc_scale_min=rrc.get("scale", [defaults.rrc_scale_min, defaults.rrc_scale_max])[0],
        rrc_scale_max=rrc.get("scale", [defaults.rrc_scale_min, defaults.rrc_scale_max])[1],
        rrc_ratio_min=rrc.get("ratio", [defaults.rrc_ratio_min, defaults.rrc_ratio_max])[0],
        rrc_ratio_max=rrc.get("ratio", [defaults.rrc_ratio_min, defaults.rrc_ratio_max])[1],
        brightness=cj.get("brightness", defaults.brightness),
        contrast=cj.get("contrast", defaults.contrast),
        saturation=cj.get("saturation", defaults.saturation),
        hue=cj.get("hue", defaults.hue),
        blur_p=blur.get("p", defaults.blur_p),
        blur_kernel=blur.get("kernel_size", defaults.blur_kernel),
        blur_sigma_min=blur.get("sigma", [defaults.blur_sigma_min, defaults.blur_sigma_max])[0],
        blur_sigma_max=blur.get("sigma", [defaults.blur_sigma_min, defaults.blur_sigma_max])[1],
        grayscale_p=gs.get("p", defaults.grayscale_p),
        erasing_p=erasing.get("p", defaults.erasing_p),
        erasing_scale_min=erasing.get("scale", [defaults.erasing_scale_min, defaults.erasing_scale_max])[0],
        erasing_scale_max=erasing.get("scale", [defaults.erasing_scale_min, defaults.erasing_scale_max])[1],
    )

    return build_train_transforms(cfg), build_val_transforms(image_size)

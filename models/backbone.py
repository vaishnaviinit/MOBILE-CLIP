"""
MobileCLIP image encoder backbone.

Wraps the OpenCLIP MobileCLIP-S2 image encoder as a standalone feature extractor.
The key contract is extract_features() — this is what the future ensemble model
will call to get visual embeddings alongside URL features.

Architecture:
  Input:  [B, 3, 256, 256] float32 (normalized by CLIP mean/std)
  Output: [B, 512] L2-normalized float32 embeddings

Two-phase fine-tuning is controlled by the Trainer:
  Phase 1: backbone.freeze()  — only the classifier head learns
  Phase 2: backbone.unfreeze() with a 100x lower LR than the head
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

logger = logging.getLogger(__name__)

# Patterns indicating parameters that should NOT receive weight decay.
# Biases and normalization layer parameters are excluded per standard
# transformer training practice (GPT, CLIP, ViT papers).
_NO_DECAY_PATTERNS: tuple[str, ...] = (
    "bias",
    "norm",
    "ln",
    "bn",
    "layernorm",
    "positional_embedding",
    "class_embedding",
    "logit_scale",
)


class MobileCLIPBackbone(nn.Module):
    """
    Frozen-then-fine-tuned MobileCLIP-S2 image encoder.

    This class intentionally exposes ONLY the image encoder (not the text
    encoder) since this is a pure vision classification task.

    Args:
        model_name:    OpenCLIP model identifier (e.g. "MobileCLIP2-S2").
        pretrained:    OpenCLIP pretrained checkpoint tag (e.g. "dfndr2b").
        embedding_dim: Expected output embedding dimension (512 for all MobileCLIP variants).
        normalize:     If True, L2-normalize the output embeddings (recommended).
    """

    SUPPORTED_MODELS: dict[str, dict] = {
        # MobileCLIP v1 — pretrained="datacompdr"
        "MobileCLIP-S1": {"embedding_dim": 512, "image_size": 256, "params_m": 21},
        "MobileCLIP-S2": {"embedding_dim": 512, "image_size": 256, "params_m": 35},
        "MobileCLIP-B":  {"embedding_dim": 512, "image_size": 224, "params_m": 86},
        # MobileCLIP v2 — pretrained="dfndr2b" (newer, stronger)
        "MobileCLIP2-S0": {"embedding_dim": 512, "image_size": 256, "params_m": 13},
        "MobileCLIP2-S2": {"embedding_dim": 512, "image_size": 256, "params_m": 36},
        "MobileCLIP2-B":  {"embedding_dim": 512, "image_size": 256, "params_m": 88},
    }

    def __init__(
        self,
        model_name: str = "MobileCLIP2-S2",
        pretrained: str = "dfndr2b",
        embedding_dim: int = 512,
        normalize: bool = True,
    ) -> None:
        super().__init__()
        self.model_name = model_name
        self.pretrained = pretrained
        self.embedding_dim = embedding_dim
        self.normalize = normalize
        self._frozen = False

        self._validate_model_name(model_name)
        self.encoder = self._load_encoder(model_name, pretrained)

        n_total = sum(p.numel() for p in self.encoder.parameters())
        logger.info(
            "MobileCLIPBackbone ready | model=%s | pretrained=%s | "
            "embedding_dim=%d | params=%.1fM",
            model_name,
            pretrained,
            embedding_dim,
            n_total / 1e6,
        )

    # ------------------------------------------------------------------
    # Core interface — this is the ensemble contract
    # ------------------------------------------------------------------

    def extract_features(self, images: Tensor) -> Tensor:
        """
        Extract visual embeddings from a batch of images.

        This method is the public contract for downstream ensemble models.
        Always returns L2-normalized embeddings when self.normalize=True.

        Args:
            images: [B, 3, H, W] float32 tensor, normalized by CLIP mean/std.

        Returns:
            [B, 512] float32 embedding tensor.
        """
        features: Tensor = self.encoder(images)

        # Flatten in case the encoder returns spatial features (unlikely for
        # MobileCLIP-S2 which pools to a vector, but defensive).
        if features.dim() > 2:
            features = features.flatten(1)

        if self.normalize:
            features = F.normalize(features, dim=-1, p=2)

        return features

    def forward(self, images: Tensor) -> Tensor:
        """Alias for extract_features — satisfies nn.Module convention."""
        return self.extract_features(images)

    # ------------------------------------------------------------------
    # Phase control — called by Trainer
    # ------------------------------------------------------------------

    def freeze(self) -> None:
        """
        Freeze all backbone parameters (Phase 1: linear probing).

        After calling, self.encoder.parameters() all have requires_grad=False.
        The Trainer calls this before constructing the Phase 1 optimizer so
        that only the head's parameters are updated.
        """
        for param in self.encoder.parameters():
            param.requires_grad = False
        self._frozen = True
        n_params = sum(p.numel() for p in self.encoder.parameters())
        logger.info(
            "Backbone FROZEN | %.1fM params frozen | Phase 1: linear probing only",
            n_params / 1e6,
        )

    def unfreeze(self) -> None:
        """
        Unfreeze all backbone parameters (Phase 2: full fine-tuning).

        After calling, the Trainer rebuilds the optimizer with two groups:
          backbone_params: lr = lr_backbone (e.g. 1e-5)
          head_params:     lr = lr_head     (e.g. 1e-4)
        """
        for param in self.encoder.parameters():
            param.requires_grad = True
        self._frozen = False
        n_params = sum(p.numel() for p in self.encoder.parameters())
        logger.info(
            "Backbone UNFROZEN | %.1fM params trainable | Phase 2: differential LR fine-tuning",
            n_params / 1e6,
        )

    def is_frozen(self) -> bool:
        """Return True if backbone is currently frozen."""
        return self._frozen

    def get_parameter_groups(
        self,
        lr_backbone: float,
        weight_decay: float,
    ) -> list[dict]:
        """
        Return AdamW parameter groups with backbone LR, split by weight-decay eligibility.

        Parameters whose name contains any pattern from _NO_DECAY_PATTERNS
        (biases, norms, positional embeddings) get weight_decay=0.0.
        All others get the specified weight_decay.

        This follows the standard practice from the GPT-3 / CLIP papers:
        applying weight decay to biases and norms hurts rather than helps.

        Args:
            lr_backbone:  Learning rate for this parameter group (e.g. 1e-5).
            weight_decay: Weight decay applied to non-bias/norm parameters.

        Returns:
            List of 1 or 2 param group dicts for AdamW.
        """
        decay_params: list[Tensor] = []
        no_decay_params: list[Tensor] = []

        for name, param in self.encoder.named_parameters():
            if not param.requires_grad:
                continue
            if any(nd in name.lower() for nd in _NO_DECAY_PATTERNS):
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        groups: list[dict] = []
        if decay_params:
            groups.append({
                "params": decay_params,
                "lr": lr_backbone,
                "weight_decay": weight_decay,
                "name": "backbone_decay",
            })
        if no_decay_params:
            groups.append({
                "params": no_decay_params,
                "lr": lr_backbone,
                "weight_decay": 0.0,
                "name": "backbone_no_decay",
            })

        return groups

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def output_dim(self) -> int:
        """Embedding output dimension."""
        return self.embedding_dim

    def n_parameters(self, trainable_only: bool = True) -> int:
        """Return parameter count (trainable only by default)."""
        return sum(
            p.numel() for p in self.parameters()
            if (p.requires_grad if trainable_only else True)
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @classmethod
    def _validate_model_name(cls, model_name: str) -> None:
        """Raise ValueError if model_name is not in SUPPORTED_MODELS."""
        if model_name not in cls.SUPPORTED_MODELS:
            supported = list(cls.SUPPORTED_MODELS.keys())
            raise ValueError(
                f"Unsupported model: '{model_name}'. "
                f"Supported models: {supported}"
            )

    @staticmethod
    def _load_encoder(model_name: str, pretrained: str) -> nn.Module:
        """
        Load MobileCLIP image encoder via OpenCLIP.

        Returns only the visual (image) encoder — the full model (including
        text encoder) is created, the visual module is extracted, then the
        full model reference is dropped so the text encoder can be GC'd.

        We deliberately do NOT use OpenCLIP's own preprocessing transforms
        because our datasets/transforms.py handles all preprocessing with
        the same CLIP mean/std normalization.

        Raises:
            ImportError: if open_clip_torch is not installed.
            RuntimeError: if the checkpoint download or model creation fails.
        """
        try:
            import open_clip
        except ImportError as exc:
            raise ImportError(
                "open_clip_torch is required but not installed.\n"
                "Install with: pip install open_clip_torch"
            ) from exc

        logger.info(
            "Downloading/loading %s checkpoint '%s' via OpenCLIP ...",
            model_name,
            pretrained,
        )

        try:
            # create_model_and_transforms returns (model, train_tfm, val_tfm).
            # We discard both transforms — our own pipeline is in transforms.py.
            full_model, _, _ = open_clip.create_model_and_transforms(
                model_name,
                pretrained=pretrained,
            )
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load '{model_name}' with pretrained='{pretrained}'.\n"
                f"Ensure you have an internet connection for the first download, "
                f"or that the checkpoint is cached at ~/.cache/huggingface/hub.\n"
                f"Original error: {exc}"
            ) from exc

        # Extract the image encoder; the text encoder is no longer referenced
        # and will be garbage-collected when full_model goes out of scope.
        encoder: nn.Module = full_model.visual

        return encoder
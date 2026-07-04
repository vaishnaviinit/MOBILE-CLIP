"""
Classification head and full phishing classifier.

Architecture:
  ClassificationHead:
    LayerNorm(512) → Linear(512, 256) → GELU → Dropout(0.3) → Linear(256, 2)

  PhishingClassifier:
    MobileCLIPBackbone → ClassificationHead

The two components are deliberately kept separate so that:
  1. Backbone embeddings can be exported without running the head.
  2. Future ensemble can swap in a different head (e.g. a linear probe
     trained jointly with URL features) while reusing the same backbone.

Forward output contract:
  {
    "logits":     [B, 2]   — raw logits (for loss computation)
    "probs":      [B, 2]   — softmax probabilities
    "embeddings": [B, 512] — backbone features (for ensemble consumption)
  }
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from models.backbone import MobileCLIPBackbone, _NO_DECAY_PATTERNS

logger = logging.getLogger(__name__)


@dataclass
class ClassifierConfig:
    """Configuration for the classification head."""

    embedding_dim: int = 512
    hidden_dim: int = 256
    num_classes: int = 2
    dropout: float = 0.3
    use_layer_norm: bool = True


class ClassificationHead(nn.Module):
    """
    Two-layer MLP classification head on top of the backbone embedding.

    Design rationale:
      LayerNorm:  Stabilizes the L2-normalized 512-dim inputs. Although
                  L2-norm constrains the vector to a unit hypersphere, the
                  magnitude can still vary slightly in practice (e.g. during
                  early fine-tuning). LayerNorm re-scales per sample.

      Linear(512 → 256):  Compressed projection — the 512-dim embedding
                  has far more capacity than needed for a binary task.
                  256 units force the head to identify the most discriminative
                  directions in embedding space.

      GELU:       Preferred over ReLU for transformer-adjacent architectures.
                  Smoother gradient flow; standard in CLIP fine-tuning heads.

      Dropout(0.3): Regularizes the small training set (~2.4k samples).
                  Applied between the two Linear layers where overfitting is
                  most likely.

      Linear(256 → 2):  Raw logit output. Softmax is applied in forward()
                  for interpretability (we need calibrated probabilities for
                  threshold optimization, not just argmax).

    Args:
        cfg: ClassifierConfig dataclass.
    """

    def __init__(self, cfg: Optional[ClassifierConfig] = None) -> None:
        super().__init__()
        cfg = cfg or ClassifierConfig()
        self.cfg = cfg
        self.net = self._build_head(cfg)

    def forward(self, embeddings: Tensor) -> Tensor:
        """
        Args:
            embeddings: [B, embedding_dim] float32 from backbone.

        Returns:
            [B, num_classes] raw logits (NOT softmax — loss functions
            prefer logit input for numerical stability).
        """
        return self.net(embeddings)

    def _build_head(self, cfg: ClassifierConfig) -> nn.Sequential:
        """Construct the MLP layers as a Sequential."""
        layers: list[nn.Module] = []

        if cfg.use_layer_norm:
            layers.append(nn.LayerNorm(cfg.embedding_dim))

        layers.extend([
            nn.Linear(cfg.embedding_dim, cfg.hidden_dim),
            nn.GELU(),
            nn.Dropout(p=cfg.dropout),
            nn.Linear(cfg.hidden_dim, cfg.num_classes),
        ])

        return nn.Sequential(*layers)


class PhishingClassifier(nn.Module):
    """
    Full phishing classifier: MobileCLIPBackbone + ClassificationHead.

    This is the top-level model used during training and inference.
    The backbone and head are accessible as separate attributes so the
    Trainer can freeze/unfreeze them independently.

    Args:
        backbone: Pretrained MobileCLIPBackbone instance.
        head_cfg: ClassifierConfig for the classification head.
    """

    def __init__(
        self,
        backbone: MobileCLIPBackbone,
        head_cfg: Optional[ClassifierConfig] = None,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.head = ClassificationHead(head_cfg or ClassifierConfig())

        # Log model summary
        n_backbone = sum(p.numel() for p in self.backbone.parameters())
        n_head = sum(p.numel() for p in self.head.parameters())
        logger.info(
            "PhishingClassifier | backbone=%.1fM params | head=%.1fM params | total=%.1fM params",
            n_backbone / 1e6,
            n_head / 1e6,
            (n_backbone + n_head) / 1e6,
        )

    def forward(self, images: Tensor) -> dict[str, Tensor]:
        """
        Full forward pass: image → embeddings → logits → probs.

        Args:
            images: [B, 3, 256, 256] normalized image tensor.

        Returns:
            Dict with:
              "logits":     [B, 2]   raw pre-softmax logits
              "probs":      [B, 2]   softmax probabilities
              "embeddings": [B, 512] L2-normalized backbone features
        """
        embeddings: Tensor = self.backbone.extract_features(images)  # [B, 512]
        logits: Tensor = self.head(embeddings)                        # [B, 2]
        probs: Tensor = F.softmax(logits, dim=-1)                     # [B, 2]

        return {
            "logits": logits,
            "probs": probs,
            "embeddings": embeddings,
        }

    @torch.no_grad()
    def predict(self, images: Tensor, threshold: float = 0.5) -> dict[str, Tensor]:
        """
        Inference-mode forward with threshold applied.

        Wraps forward() with torch.no_grad() and applies the decision
        threshold to the phishing probability (column 1 of probs).

        Args:
            images:    [B, 3, 256, 256] tensor.
            threshold: Decision threshold on P(phishing). Samples with
                       P(phishing) >= threshold are classified as phishing.
                       Default 0.5; use F2-optimized threshold in production.

        Returns:
            Dict with:
              "class_idx"  [B] int64 tensor  (0=legitimate, 1=phishing)
              "probs"      [B, 2] float32
              "embeddings" [B, 512] float32
        """
        self.eval()
        outputs = self.forward(images)

        phishing_prob: Tensor = outputs["probs"][:, 1]   # [B]
        class_idx: Tensor = (phishing_prob >= threshold).long()  # [B]

        return {
            "class_idx": class_idx,
            "probs": outputs["probs"],
            "embeddings": outputs["embeddings"],
        }

    def get_optimizer_param_groups(
        self,
        lr_head: float,
        lr_backbone: float,
        weight_decay: float,
    ) -> list[dict]:
        """
        Build AdamW parameter groups with differential learning rates.

        Structure:
          - If backbone is FROZEN: only head groups are returned.
            (Backbone has requires_grad=False so including it would give
            an AdamW group with zero parameters → error.)
          - If backbone is UNFROZEN: backbone groups (100x lower LR) +
            head groups.

        Both backbone and head are further split into decay / no-decay
        sub-groups following the standard transformer weight decay protocol.

        Args:
            lr_head:      Learning rate for classification head (e.g. 1e-3).
            lr_backbone:  Learning rate for backbone (e.g. 1e-5). Ignored
                          if backbone is frozen.
            weight_decay: Weight decay applied to non-bias/norm parameters.

        Returns:
            List of param group dicts ready for torch.optim.AdamW.
        """
        groups: list[dict] = []

        # Backbone groups (only when unfrozen and has trainable params)
        if not self.backbone.is_frozen():
            groups.extend(
                self.backbone.get_parameter_groups(lr_backbone, weight_decay)
            )

        # Head groups — always trainable
        head_decay: list[Tensor] = []
        head_no_decay: list[Tensor] = []

        for name, param in self.head.named_parameters():
            if not param.requires_grad:
                continue
            if any(nd in name.lower() for nd in _NO_DECAY_PATTERNS):
                head_no_decay.append(param)
            else:
                head_decay.append(param)

        if head_decay:
            groups.append({
                "params": head_decay,
                "lr": lr_head,
                "weight_decay": weight_decay,
                "name": "head_decay",
            })
        if head_no_decay:
            groups.append({
                "params": head_no_decay,
                "lr": lr_head,
                "weight_decay": 0.0,
                "name": "head_no_decay",
            })

        n_groups = len(groups)
        n_params = sum(
            sum(p.numel() for p in g["params"])
            for g in groups
        )
        logger.debug(
            "Optimizer param groups: %d groups | %.1fM params total",
            n_groups,
            n_params / 1e6,
        )

        return groups

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    @property
    def embedding_dim(self) -> int:
        """Embedding dimension from backbone."""
        return self.backbone.embedding_dim

    def n_parameters(self, trainable_only: bool = True) -> dict[str, int]:
        """Return parameter counts for backbone, head, and total."""
        def count(module: nn.Module) -> int:
            return sum(
                p.numel() for p in module.parameters()
                if (p.requires_grad if trainable_only else True)
            )
        return {
            "backbone": count(self.backbone),
            "head": count(self.head),
            "total": count(self),
        }

"""
The deep-learning model.  Two parts:

  1. FeatureExtractor      : a CNN (default MobileNetV2, ImageNet-pretrained) that turns ONE frame
                             into ONE feature vector.  It is NOT fine-tuned (small dataset).
  2. BiLSTMAttentionHead   : BiLSTM -> temporal Attention -> Dense -> (Softmax)
                             It reads the sequence of frame features and predicts the sentence class.

Shapes (B = batch, T = frames, D = feature size, C = classes):
    frames         (B, T, 3, H, W)
    CNN features   (B, T, D)              e.g. D = 1280 for MobileNetV2
    BiLSTM output  (B, T, 2*hidden)
    Attention      weights (B, T)  ->  context vector (B, 2*hidden)
    Dense          (B, dense_units)
    logits         (B, C)   ->  softmax  ->  probabilities (B, C)

NOTE: CNN + BiLSTM + Attention is an existing, well-known combination.  We use it as our
selected architecture; we do not claim it as a new model.
"""
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalAttention(nn.Module):
    """Learns how important each time step (frame) is.

    score_t  = v^T tanh(W h_t)          one number per frame
    weight_t = softmax(score_t)         weights over time, they add up to 1
    context  = sum_t weight_t * h_t     weighted summary of the whole sequence
    """

    def __init__(self, dim: int, attn_dim: int = 64):
        super().__init__()
        self.proj = nn.Linear(dim, attn_dim)
        self.score = nn.Linear(attn_dim, 1, bias=False)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: (B, T, D)
        scores = self.score(torch.tanh(self.proj(x))).squeeze(-1)   # (B, T)
        weights = torch.softmax(scores, dim=1)                       # (B, T)
        context = (weights.unsqueeze(-1) * x).sum(dim=1)             # (B, D)
        return context, weights


class BiLSTMAttentionHead(nn.Module):
    def __init__(self, input_dim: int, num_classes: int, hidden: int = 128, layers: int = 1,
                 dense_units: int = 128, dropout: float = 0.5, use_attention: bool = True):
        super().__init__()
        self.use_attention = use_attention
        self.norm = nn.LayerNorm(input_dim)
        self.in_drop = nn.Dropout(dropout)
        self.lstm = nn.LSTM(input_dim, hidden, num_layers=layers, batch_first=True, bidirectional=True,
                            dropout=dropout if layers > 1 else 0.0)
        self.attention = TemporalAttention(2 * hidden)
        self.dense = nn.Sequential(nn.Linear(2 * hidden, dense_units), nn.ReLU(), nn.Dropout(dropout))
        self.classifier = nn.Linear(dense_units, num_classes)   # softmax is applied in loss / inference

    def forward(self, x: torch.Tensor, return_attention: bool = False):
        x = self.in_drop(self.norm(x))
        seq, _ = self.lstm(x)                                   # (B, T, 2*hidden)
        if self.use_attention:
            ctx, weights = self.attention(seq)
        else:                                                   # ablation: plain average over time
            ctx = seq.mean(dim=1)
            weights = torch.full(seq.shape[:2], 1.0 / seq.shape[1], device=seq.device)
        logits = self.classifier(self.dense(ctx))
        return (logits, weights) if return_attention else logits

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        return F.softmax(self.forward(x), dim=-1)


def head_from_meta(meta: dict) -> BiLSTMAttentionHead:
    """Rebuild the head from the settings saved with a checkpoint."""
    return BiLSTMAttentionHead(
        input_dim=meta["input_dim"], num_classes=meta["num_classes"], hidden=meta["hidden"],
        layers=meta["layers"], dense_units=meta["dense_units"], dropout=meta["dropout"],
        use_attention=meta["use_attention"])


# ---------------------------------------------------------------- CNN feature extractor
def build_backbone(name: str, pretrained: bool):
    """Returns (feature_module, feature_dim). Kept lazy so importing this file is cheap."""
    import torchvision.models as tvm

    if name == "mobilenet_v2":
        w = tvm.MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
        return tvm.mobilenet_v2(weights=w).features, 1280
    if name == "resnet18":
        w = tvm.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        m = tvm.resnet18(weights=w)
        return nn.Sequential(*list(m.children())[:-2]), 512
    if name == "efficientnet_b0":
        w = tvm.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        return tvm.efficientnet_b0(weights=w).features, 1280
    raise ValueError(f"Unknown CNN_BACKBONE '{name}'. Use mobilenet_v2, resnet18 or efficientnet_b0.")


class FeatureExtractor(nn.Module):
    """Frame -> feature vector (global-average-pooled CNN features)."""

    def __init__(self, name: str = "mobilenet_v2", pretrained: bool = True):
        super().__init__()
        self.backbone, self.feature_dim = build_backbone(name, pretrained)
        self.eval()
        for p in self.parameters():
            p.requires_grad = False          # frozen: no fine-tuning at first

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, 3, H, W) -> (N, D)
        f = self.backbone(x)
        return F.adaptive_avg_pool2d(f, 1).flatten(1)

    @torch.no_grad()
    def extract(self, frames: torch.Tensor, chunk: int = 16) -> torch.Tensor:
        """frames (T, 3, H, W) -> features (T, D), processed in small chunks to save memory."""
        outs = [self.forward(frames[i:i + chunk]) for i in range(0, frames.shape[0], chunk)]
        return torch.cat(outs, dim=0)

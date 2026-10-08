"""Temporal classifier for normalized MediaPipe landmark sequences."""
from __future__ import annotations

from app.models.signbridge_model import BiLSTMAttentionHead

LANDMARK_INPUT_DIM = 258
LANDMARK_NUM_FRAMES = 30


def build_landmark_classifier(num_classes: int = 101, dropout: float = 0.3) -> BiLSTMAttentionHead:
    """Build the project's established BiLSTM-attention head for 30 x 258 inputs."""
    return BiLSTMAttentionHead(
        input_dim=LANDMARK_INPUT_DIM,
        num_classes=num_classes,
        hidden=128,
        layers=1,
        dense_units=128,
        dropout=dropout,
        use_attention=True,
    )

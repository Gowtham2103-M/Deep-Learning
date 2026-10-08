"""Experiment 4 BiLSTM-attention head for landmark position and motion sequences."""
from __future__ import annotations

from app.models.signbridge_model import BiLSTMAttentionHead
from app.preprocessing.landmark_sequence_exp4 import FEATURE_DIM, NUM_FRAMES


def build_landmark_motion_classifier(num_classes: int = 101, dropout: float = 0.3) -> BiLSTMAttentionHead:
    """Keep the Experiment 3 temporal architecture; change only its input dimension."""
    return BiLSTMAttentionHead(
        input_dim=FEATURE_DIM,
        num_classes=num_classes,
        hidden=128,
        layers=1,
        dense_units=128,
        dropout=dropout,
        use_attention=True,
    )

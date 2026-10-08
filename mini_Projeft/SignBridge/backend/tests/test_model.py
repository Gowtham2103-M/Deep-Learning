"""Model architecture and splitting."""
import numpy as np
import pandas as pd
import torch

from app.models.signbridge_model import BiLSTMAttentionHead, FeatureExtractor, TemporalAttention
from training.dataset import make_splits


def test_attention_weights_sum_to_one():
    attn = TemporalAttention(32)
    ctx, w = attn(torch.randn(3, 10, 32))
    assert ctx.shape == (3, 32) and w.shape == (3, 10)
    assert torch.allclose(w.sum(dim=1), torch.ones(3), atol=1e-5)


def test_head_forward_shapes():
    head = BiLSTMAttentionHead(input_dim=20, num_classes=7, hidden=8, dense_units=8)
    logits, w = head(torch.randn(2, 5, 20), return_attention=True)
    assert logits.shape == (2, 7) and w.shape == (2, 5)
    assert torch.allclose(head.predict_proba(torch.randn(2, 5, 20)).sum(1), torch.ones(2), atol=1e-5)


def test_head_without_attention_runs():
    head = BiLSTMAttentionHead(20, 4, hidden=8, dense_units=8, use_attention=False)
    assert head(torch.randn(2, 6, 20)).shape == (2, 4)


def test_mobilenet_feature_extractor_shape():
    cnn = FeatureExtractor("mobilenet_v2", pretrained=False)
    feats = cnn.extract(torch.randn(5, 3, 64, 64))
    assert feats.shape == (5, 1280)
    assert not any(p.requires_grad for p in cnn.parameters())        # frozen


def _df(signers=True):
    rows = [{"video_path": f"v{i}", "label": f"c{i % 4}", **({"signer": f"s{i % 5}"} if signers else {})}
            for i in range(40)]
    return pd.DataFrame(rows)


def test_signer_independent_split_has_no_overlap():
    df = _df(True)
    splits, info = make_splits(df, 0.2, 0.2, seed=1)
    assert info["mode"] == "signer_independent"
    tr = set(df.iloc[splits["train"]].signer)
    assert not tr & set(df.iloc[splits["test"]].signer)
    assert not tr & set(df.iloc[splits["val"]].signer)


def test_split_without_signers_warns():
    splits, info = make_splits(_df(False), 0.2, 0.2, seed=1)
    assert any("signer" in w.lower() for w in info["warnings"])
    assert sum(len(v) for v in splits.values()) == 40

"""
Test fixtures.  Everything here is SAMPLE DATA generated on the fly (tiny random-noise
videos) - it is only used to check that the code works, never for real results.
"""
import os

import cv2
import numpy as np
import pandas as pd
import pytest

SENTENCES = ["hello how are you", "thank you", "good morning", "what is your name"]
SIGNERS = ["s1", "s2", "s3", "s4"]


def _write_video(path, seed, frames=16, size=(64, 48)):
    rng = np.random.RandomState(seed)
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10, size)
    base = rng.randint(0, 255, 3)
    for _ in range(frames):
        img = np.clip(base + rng.randint(-20, 20, (size[1], size[0], 3)), 0, 255).astype(np.uint8)
        w.write(img)
    w.release()


@pytest.fixture(scope="session")
def sample_dataset(tmp_path_factory):
    """4 sentences x 4 signers = 16 tiny sample videos + CSV (with a Signer column)."""
    root = tmp_path_factory.mktemp("sample_isl") / "ISL_SAMPLE"
    (root / "videos").mkdir(parents=True)
    rows = []
    for si, sent in enumerate(SENTENCES):
        for gi, signer in enumerate(SIGNERS):
            name = f"s{si}_{signer}.mp4"
            _write_video(root / "videos" / name, seed=si * 10 + gi, frames=12 + gi * 3)
            rows.append({"Sentences": sent, "Filename": name, "Signer": signer})
    rows.append({"Sentences": "ghost sentence", "Filename": "missing.mp4", "Signer": "s1"})
    pd.DataFrame(rows).to_csv(root / "labels.csv", index=False)
    return root


@pytest.fixture()
def settings_env(sample_dataset, tmp_path, monkeypatch):
    """Point the app at the sample dataset and a temporary models folder."""
    monkeypatch.setenv("SIGNBRIDGE_DATASET_PATH", str(sample_dataset))
    monkeypatch.setenv("SIGNBRIDGE_MODEL_PATH", str(tmp_path / "models" / "head.pt"))
    monkeypatch.setenv("SIGNBRIDGE_FEATURE_CACHE_DIR", str(tmp_path / "models" / "cache"))
    monkeypatch.setenv("SIGNBRIDGE_CNN_PRETRAINED", "false")     # no download needed in tests
    monkeypatch.setenv("SIGNBRIDGE_IMAGE_SIZE", "64")
    monkeypatch.setenv("SIGNBRIDGE_NUM_FRAMES", "8")
    monkeypatch.setenv("SIGNBRIDGE_EPOCHS", "3")
    monkeypatch.setenv("SIGNBRIDGE_BATCH_SIZE", "4")
    monkeypatch.setenv("SIGNBRIDGE_LSTM_HIDDEN", "16")
    monkeypatch.setenv("SIGNBRIDGE_DENSE_UNITS", "16")
    from app.config import reload_settings
    from app.services import dataset_service, model_service
    dataset_service.reset_cache()
    model_service._rec = None
    yield reload_settings()
    dataset_service.reset_cache()
    model_service._rec = None

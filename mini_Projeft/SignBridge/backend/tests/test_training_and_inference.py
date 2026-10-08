"""Feature extraction -> training -> saved model -> loading -> prediction (all on SAMPLE data)."""
import json

import pytest

from app.inference.recognizer import ModelNotTrainedError, Recognizer
from app.preprocessing.dataset_index import build_index
from training.train import run_training


@pytest.fixture()
def trained(settings_env):
    idx = build_index(settings_env)
    out = run_training(settings_env, idx, epochs=3)
    return settings_env, idx, out


def test_untrained_model_is_reported_not_faked(settings_env):
    with pytest.raises(ModelNotTrainedError) as e:
        Recognizer(settings_env.model_path)
    assert "Model not trained yet. Run the training command." in str(e.value)


def test_training_saves_all_artifacts(trained):
    s, idx, out = trained
    for name in ("head.pt", "classes.json", "history.json", "splits.json", "metrics.json",
                 "confusion_matrix.csv"):
        assert (s.models_dir / name).is_file(), name
    assert json.loads((s.models_dir / "classes.json").read_text()) == idx.classes
    assert len(out["history"]) >= 1
    assert out["split_info"]["mode"] == "signer_independent"     # sample CSV has signer IDs


def test_metrics_are_real_and_consistent(trained):
    s, _, out = trained
    m = json.loads((s.models_dir / "metrics.json").read_text())
    for k in ("accuracy", "precision_macro", "recall_macro", "f1_macro"):
        assert 0.0 <= m[k] <= 1.0
    assert m["num_samples"] == len(json.loads((s.models_dir / "splits.json").read_text())["splits"]["test"]["rows"])


def test_model_loads_and_predicts(trained):
    s, idx, _ = trained
    rec = Recognizer(s.model_path)
    r = rec.predict_video(idx.df.iloc[0]["video_path"])
    assert r["label"] in idx.classes
    assert 0.0 <= r["confidence"] <= 1.0
    assert abs(sum(a for a in r["attention"]) - 1.0) < 1e-3
    assert r["frames_used"] == s.NUM_FRAMES
    assert r["latency_ms"]["total_ms"] > 0                        # measured, not invented

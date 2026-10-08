"""Dataset loading and label mapping."""
import json

from app.preprocessing.dataset_index import build_index
from app.config import reload_settings
from tests.conftest import SENTENCES


def test_dataset_loads_dynamically(settings_env):
    idx = build_index(settings_env)
    assert idx.available
    assert idx.video_count == 16
    assert idx.num_classes == 4
    assert idx.video_column == "Filename" and idx.label_column == "Sentences"
    assert idx.signer_column == "Signer"                      # detected automatically
    assert len(idx.df) == 16


def test_missing_video_references_reported(settings_env):
    idx = build_index(settings_env)
    assert idx.missing_refs == ["missing.mp4"]
    assert any("do not exist" in w for w in idx.warnings)


def test_label_mapping_is_stable(settings_env):
    idx = build_index(settings_env)
    assert idx.classes == sorted(SENTENCES)
    l2i = idx.label_to_id()
    assert [idx.classes[l2i[c]] for c in idx.classes] == idx.classes
    assert json.loads(json.dumps(idx.classes)) == idx.classes  # classes.json round trip


def test_dataset_missing_does_not_crash(monkeypatch, tmp_path):
    monkeypatch.setenv("SIGNBRIDGE_DATASET_PATH", str(tmp_path / "nope"))
    idx = build_index(reload_settings())
    assert not idx.available
    assert "dataset/README.md" in idx.message

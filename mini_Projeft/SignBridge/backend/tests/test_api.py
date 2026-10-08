"""API endpoints (FastAPI TestClient) - consistent with the frontend's URLs."""
import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.preprocessing.dataset_index import build_index
from training.train import run_training


@pytest.fixture()
def client(settings_env):
    return TestClient(app)


def test_health_and_root(client):
    assert client.get("/api/health").json() == {"status": "ok"}
    assert client.get("/").json()["docs"] == "/docs"


def test_status_before_training(client):
    j = client.get("/api/status").json()
    assert j["dataset"]["available"] and j["dataset"]["classes"] == 4
    assert j["model"]["trained"] is False
    assert j["model"]["message"] == "Model not trained yet. Run the training command."


def test_metrics_before_training_are_not_faked(client):
    j = client.get("/api/metrics").json()
    assert j["trained"] is False and j["metrics"] is None
    assert j["message"] == "Model not trained yet. Run the training command."


def test_predict_before_training_returns_503(client):
    files = [("frames", (f"f{i}.jpg", cv2.imencode(".jpg", np.zeros((48, 64, 3), np.uint8))[1].tobytes(), "image/jpeg"))
             for i in range(4)]
    r = client.post("/api/predict/frames", files=files)
    assert r.status_code == 503 and r.json()["status"] == "model_not_trained"


def test_config_and_vocabulary(client):
    cfg = client.get("/api/config").json()
    assert cfg["num_frames"] == 8 and "confidence_threshold" in cfg
    v = client.get("/api/vocabulary").json()
    assert v["available"] and len(v["items"]) == 4 and all(i["has_video"] for i in v["items"])


def test_speech_match_endpoint(client):
    ok = client.post("/api/speech/match", json={"text": "Thank you!"}).json()
    assert ok["status"] == "matched" and ok["match"]["label"] == "thank you"
    assert client.get(ok["match"]["video_url"]).status_code == 200
    fuzzy = client.post("/api/speech/match", json={"text": "good mornin"}).json()
    assert fuzzy["status"] == "matched" and fuzzy["match"]["method"] == "fuzzy"
    none = client.post("/api/speech/match", json={"text": "purple monkey dishwasher"}).json()
    assert none["status"] == "no_match" and none["match"] is None
    assert none["message"] == "No matching ISL sign available."


def test_reference_video_404(client):
    assert client.get("/api/reference-video/999").status_code == 404


def test_predict_after_training(settings_env, client):
    run_training(settings_env, build_index(settings_env), epochs=2)
    j = client.get("/api/status").json()
    assert j["model"]["trained"] is True
    imgs = []
    for i in range(6):
        frame = np.random.randint(0, 255, (48, 64, 3), np.uint8)
        imgs.append(("frames", (f"f{i}.jpg", cv2.imencode(".jpg", frame)[1].tobytes(), "image/jpeg")))
    r = client.post("/api/predict/frames", files=imgs)
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok" and "latency_ms" in body and body["label"]
    assert isinstance(body["accepted"], bool)
    m = client.get("/api/metrics").json()
    assert m["trained"] and 0 <= m["metrics"]["accuracy"] <= 1


def test_predict_video_upload(settings_env, client):
    idx = build_index(settings_env)
    run_training(settings_env, idx, epochs=2)
    with open(idx.df.iloc[0]["video_path"], "rb") as fh:
        r = client.post("/api/predict/video", files={"file": ("clip.mp4", fh.read(), "video/mp4")})
    assert r.status_code == 200 and r.json()["label"] in idx.classes


def test_app_starts_without_dataset(monkeypatch, tmp_path):
    monkeypatch.setenv("SIGNBRIDGE_DATASET_PATH", str(tmp_path / "missing"))
    monkeypatch.setenv("SIGNBRIDGE_MODEL_PATH", str(tmp_path / "m.pt"))
    from app.config import reload_settings
    from app.services import dataset_service
    reload_settings(); dataset_service.reset_cache()
    c = TestClient(app)
    st = c.get("/api/status").json()
    assert st["dataset"]["available"] is False and "dataset/README.md" in st["dataset"]["message"]
    assert c.post("/api/speech/match", json={"text": "hello"}).json()["status"] == "dataset_missing"
    assert c.get("/api/vocabulary").json()["available"] is False
    dataset_service.reset_cache()

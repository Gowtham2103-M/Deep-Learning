"""Health, status, configuration, vocabulary and metrics endpoints."""
from fastapi import APIRouter

from app.config import get_settings
from app.inference.recognizer import NOT_TRAINED_MESSAGE
from app.services import dataset_service as ds
from app.services.model_service import model_status, read_json_if_exists

router = APIRouter(prefix="/api")


@router.get("/health")
def health():
    return {"status": "ok"}


@router.get("/config")
def config():
    s = get_settings()
    return {"num_frames": s.NUM_FRAMES, "capture_fps": s.CAPTURE_FPS, "window_seconds": s.WINDOW_SECONDS,
            "confidence_threshold": s.CONFIDENCE_THRESHOLD, "stable_count": s.STABLE_COUNT,
            "speak_cooldown_seconds": s.SPEAK_COOLDOWN_SECONDS, "use_mediapipe": s.USE_MEDIAPIPE,
            "match_threshold": s.MATCH_THRESHOLD}


@router.get("/status")
def status():
    idx = ds.get_index()
    return {"dataset": idx.summary(), "model": model_status(), "config": config()}


@router.post("/dataset/refresh")
def refresh_dataset():
    return ds.get_index(refresh=True).summary()


@router.get("/vocabulary")
def vocabulary():
    idx = ds.get_index()
    if not idx.available:
        return {"available": False, "message": idx.message, "items": []}
    return {"available": True, "message": "",
            "items": [{"class_id": i, "label": l, "has_video": ds.reference_video(i) is not None}
                      for i, l in enumerate(idx.classes)]}


@router.get("/metrics")
def metrics():
    st = model_status()
    if not st["trained"]:
        return {"trained": False, "message": NOT_TRAINED_MESSAGE, "metrics": None, "latency": None, "history": None}
    m = read_json_if_exists("metrics.json")
    return {"trained": True,
            "message": "" if m else "Model is trained but evaluation has not been run. Run: python -m training.evaluate",
            "metrics": m, "latency": read_json_if_exists("latency.json"),
            "history": read_json_if_exists("history.json")}

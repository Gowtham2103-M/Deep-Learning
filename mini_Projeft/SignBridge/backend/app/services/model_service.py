"""Loads the trained recognizer once and reloads it automatically after re-training."""
from __future__ import annotations

import json
from typing import Optional

from app.config import get_settings
from app.inference.recognizer import NOT_TRAINED_MESSAGE, ModelNotTrainedError, Recognizer

_rec: Optional[Recognizer] = None
_mtime: float = 0.0


def get_recognizer() -> Recognizer:
    """Return the trained Recognizer or raise ModelNotTrainedError."""
    global _rec, _mtime
    path = get_settings().model_path
    if not path.is_file():
        _rec = None
        raise ModelNotTrainedError()
    m = path.stat().st_mtime
    if _rec is None or m != _mtime:
        _rec = Recognizer(path)
        _mtime = m
    return _rec


def model_status() -> dict:
    s = get_settings()
    if not s.model_path.is_file():
        return {"trained": False, "message": NOT_TRAINED_MESSAGE}
    try:
        rec = get_recognizer()
        meta = rec.meta
        return {"trained": True, "message": "Model loaded.", "trained_at": meta.get("trained_at"),
                "classes": len(rec.classes), "num_frames": meta["num_frames"], "backbone": meta["backbone"],
                "use_attention": meta["use_attention"], "use_mediapipe": meta["use_mediapipe"],
                "split_mode": meta.get("split_mode"), "device": str(rec.pipeline.device)}
    except Exception as exc:  # noqa
        return {"trained": False, "message": f"Model file exists but could not be loaded: {exc}"}


def read_json_if_exists(name: str):
    f = get_settings().models_dir / name
    if f.is_file():
        try:
            return json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            return None
    return None

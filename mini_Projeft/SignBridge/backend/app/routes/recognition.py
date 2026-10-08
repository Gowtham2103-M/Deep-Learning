"""ISL -> text: prediction endpoints."""
import os
import tempfile
from typing import List

import cv2
import numpy as np
from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from app.config import get_settings
from app.inference.recognizer import NOT_TRAINED_MESSAGE, ModelNotTrainedError
from app.services.model_service import get_recognizer

router = APIRouter(prefix="/api/predict")
MAX_FRAMES = 120


def _not_trained():
    return JSONResponse(status_code=503, content={"status": "model_not_trained", "message": NOT_TRAINED_MESSAGE})


def _finish(result: dict) -> dict:
    thr = get_settings().CONFIDENCE_THRESHOLD
    result["accepted"] = result["confidence"] >= thr
    result["threshold"] = thr
    result["status"] = "ok"
    result["text"] = result["label"] if result["accepted"] else ""
    return result


@router.post("/frames")
async def predict_frames(frames: List[UploadFile] = File(...)):
    """Receives the last few seconds of camera frames (JPEG images) and predicts the sentence."""
    try:
        rec = get_recognizer()
    except ModelNotTrainedError:
        return _not_trained()
    if len(frames) < 2:
        raise HTTPException(400, "Send at least 2 frames.")
    if len(frames) > MAX_FRAMES:
        raise HTTPException(400, f"Send at most {MAX_FRAMES} frames.")
    decoded = []
    for f in frames:
        img = cv2.imdecode(np.frombuffer(await f.read(), np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise HTTPException(400, f"Could not decode image '{f.filename}'.")
        decoded.append(img)
    return _finish(rec.predict_frames(decoded))


@router.post("/video")
async def predict_video(file: UploadFile = File(...)):
    """Predict from an uploaded video file (useful for testing with dataset videos)."""
    try:
        rec = get_recognizer()
    except ModelNotTrainedError:
        return _not_trained()
    suffix = os.path.splitext(file.filename or "")[1] or ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        path = tmp.name
    try:
        return _finish(rec.predict_video(path))
    except (IOError, ValueError) as exc:
        raise HTTPException(400, str(exc))
    finally:
        os.unlink(path)

"""Speech -> text -> ISL: sentence matching and reference video streaming."""
import mimetypes

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from app.services import dataset_service as ds
from app.utils.text import normalize_text

router = APIRouter(prefix="/api")
NO_MATCH_MESSAGE = "No matching ISL sign available."


class SpeechText(BaseModel):
    text: str


@router.post("/speech/match")
def match_speech(body: SpeechText):
    """The browser converts speech to text (Web Speech API); this endpoint maps the text to a sign."""
    idx = ds.get_index()
    normalized = normalize_text(body.text)
    if not idx.available:
        return {"status": "dataset_missing", "input": body.text, "normalized": normalized,
                "match": None, "message": idx.message}
    matcher = ds.get_matcher()
    m = matcher.match(body.text) if matcher else None
    if m is None or ds.reference_video(m.class_id) is None:
        cand, score = matcher.best_candidate(body.text) if matcher else (None, 0.0)
        return {"status": "no_match", "input": body.text, "normalized": normalized, "match": None,
                "closest_score": round(score, 3), "message": NO_MATCH_MESSAGE}
    return {"status": "matched", "input": body.text, "normalized": normalized, "message": "",
            "match": {"class_id": m.class_id, "label": m.label, "score": m.score, "method": m.method,
                      "video_url": f"/api/reference-video/{m.class_id}"}}


@router.get("/reference-video/{class_id}")
def reference_video(class_id: int):
    path = ds.reference_video(class_id)
    if not path:
        raise HTTPException(404, "No reference video for this sentence.")
    mime = mimetypes.guess_type(path)[0] or "video/mp4"
    return FileResponse(path, media_type=mime)

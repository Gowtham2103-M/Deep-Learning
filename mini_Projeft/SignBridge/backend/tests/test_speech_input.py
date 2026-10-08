"""Speech-to-text input handling.

Speech is converted to text by the browser (Web Speech API). What the backend must handle
correctly is the TEXT that comes back: noisy, differently-cased, with punctuation or filler.
"""
import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.mark.parametrize("spoken", ["THANK YOU", "thank you.", "  Thank   you  ", "Thank you!!!"])
def test_typical_recognizer_output_matches(settings_env, spoken):
    r = TestClient(app).post("/api/speech/match", json={"text": spoken}).json()
    assert r["status"] == "matched" and r["match"]["label"] == "thank you"


def test_empty_speech_gives_no_match(settings_env):
    r = TestClient(app).post("/api/speech/match", json={"text": "   "}).json()
    assert r["status"] == "no_match"
    assert r["message"] == "No matching ISL sign available."


def test_bad_request_body_is_rejected(settings_env):
    assert TestClient(app).post("/api/speech/match", json={}).status_code == 422

"""MediaPipe is OPTIONAL. The system must work without it and fail with a clear message if it is unusable."""
import numpy as np
import pytest

from app.preprocessing.landmarks import LandmarkExtractor, MediaPipeUnavailable, landmark_dim


def test_landmark_dimensions():
    assert landmark_dim(["hands"]) == 126
    assert landmark_dim(["hands", "pose"]) == 258
    assert landmark_dim(["hands", "pose", "face"]) == 1662
    with pytest.raises(ValueError):
        landmark_dim(["feet"])


def test_extractor_works_or_fails_clearly():
    try:
        ext = LandmarkExtractor(["hands", "pose"])
    except MediaPipeUnavailable as exc:
        assert "USE_MEDIAPIPE=false" in str(exc)          # clear instruction for the student
        return
    out = ext.extract([np.zeros((120, 160, 3), np.uint8)] * 2)
    assert out.shape == (2, 258)                          # zeros when nothing is detected
    ext.close()


def test_pipeline_without_mediapipe_is_default(settings_env):
    assert settings_env.USE_MEDIAPIPE is False

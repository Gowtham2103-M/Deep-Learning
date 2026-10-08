"""Frame extraction and preprocessing."""
import numpy as np
import pytest

from app.preprocessing.frames import (load_sampled_frames, preprocess_frames, read_video_frames,
                                      sample_frames, uniform_indices)


def test_uniform_indices_length_and_range():
    idx = uniform_indices(100, 30)
    assert len(idx) == 30 and idx[0] == 0 and idx[-1] == 99


def test_short_video_repeats_frames():
    idx = uniform_indices(5, 30)
    assert len(idx) == 30 and idx.max() == 4


def test_zero_frames_raises():
    with pytest.raises(ValueError):
        uniform_indices(0, 30)


def test_read_and_sample_video(sample_dataset):
    path = str(next((sample_dataset / "videos").glob("*.mp4")))
    assert len(read_video_frames(path)) >= 12
    frames = load_sampled_frames(path, 8)
    assert len(frames) == 8 and frames[0].ndim == 3


def test_preprocess_shape_and_type():
    frames = [np.zeros((48, 64, 3), np.uint8)] * 5
    out = preprocess_frames(frames, 32)
    assert out.shape == (5, 3, 32, 32) and out.dtype == np.float32


def test_num_frames_is_configurable():
    frames = [np.zeros((10, 10, 3), np.uint8)] * 40
    assert len(sample_frames(frames, 12)) == 12

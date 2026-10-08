"""
Turning a video into a fixed number of frames the CNN can read.

  video  ->  read frames  ->  pick NUM_FRAMES evenly spaced ones  ->  resize + normalise
"""
from __future__ import annotations

from typing import List

import cv2
import numpy as np

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
MAX_STORED_FRAMES = 600      # safety limit for very long videos
MAX_SIDE = 640               # frames wider than this are shrunk to save memory


def uniform_indices(total: int, num: int) -> np.ndarray:
    """Pick `num` evenly spaced indices from `total` frames.

    If the video has fewer than `num` frames some indices repeat, so every video
    always gives exactly `num` frames.
    """
    if total <= 0:
        raise ValueError("Video has no frames.")
    if num <= 0:
        raise ValueError("num must be positive.")
    return np.linspace(0, total - 1, num).round().astype(int)


def sample_frames(frames: List[np.ndarray], num: int) -> List[np.ndarray]:
    """Evenly sample `num` frames from a list of frames."""
    idx = uniform_indices(len(frames), num)
    return [frames[i] for i in idx]


def read_video_frames(path: str) -> List[np.ndarray]:
    """Read all frames of a video (BGR). Very large frames are shrunk to save memory."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    stride = max(1, int(np.ceil(total / MAX_STORED_FRAMES))) if total > 0 else 1
    frames, i = [], 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i % stride == 0:
            h, w = frame.shape[:2]
            if max(h, w) > MAX_SIDE:
                scale = MAX_SIDE / max(h, w)
                frame = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
            frames.append(frame)
        i += 1
    cap.release()
    if not frames:
        raise IOError(f"No frames could be decoded from: {path}")
    return frames


def load_sampled_frames(path: str, num_frames: int) -> List[np.ndarray]:
    return sample_frames(read_video_frames(path), num_frames)


def preprocess_frames(frames_bgr: List[np.ndarray], image_size: int) -> np.ndarray:
    """BGR frames -> float array (T, 3, H, W): RGB, resized, ImageNet-normalised."""
    out = []
    for f in frames_bgr:
        rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (image_size, image_size), interpolation=cv2.INTER_AREA)
        arr = (rgb.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
        out.append(arr.transpose(2, 0, 1))
    return np.stack(out).astype(np.float32)

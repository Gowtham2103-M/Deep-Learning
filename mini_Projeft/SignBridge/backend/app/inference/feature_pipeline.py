"""
FeaturePipeline: frames -> (CNN [+ MediaPipe]) -> feature sequence  (T, D)

The SAME code is used to build the training features and to run live inference,
so training and prediction always see identical preprocessing.
"""
from __future__ import annotations

import time
from typing import Dict, List, Optional

import numpy as np
import torch

from app.config import Settings
from app.models.signbridge_model import FeatureExtractor
from app.preprocessing.frames import preprocess_frames
from app.preprocessing.landmarks import LandmarkExtractor, landmark_dim


def pick_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class FeaturePipeline:
    def __init__(self, backbone: str, pretrained: bool, image_size: int,
                 use_mediapipe: bool = False, mediapipe_parts: Optional[List[str]] = None):
        self.backbone_name = backbone
        self.pretrained = pretrained
        self.image_size = image_size
        self.use_mediapipe = use_mediapipe
        self.parts = mediapipe_parts or []
        self.device = pick_device()
        self.cnn = FeatureExtractor(backbone, pretrained).to(self.device)
        self.landmarks = LandmarkExtractor(self.parts) if use_mediapipe else None
        self.feature_dim = self.cnn.feature_dim + (landmark_dim(self.parts) if use_mediapipe else 0)

    @classmethod
    def from_settings(cls, s: Settings) -> "FeaturePipeline":
        return cls(s.CNN_BACKBONE, s.CNN_PRETRAINED, s.IMAGE_SIZE, s.USE_MEDIAPIPE, s.mediapipe_parts)

    def signature(self, num_frames: int) -> str:
        """Used to name the feature-cache folder, so different settings never get mixed."""
        mp = "mp-" + "+".join(self.parts) if self.use_mediapipe else "nomp"
        pre = "pre" if self.pretrained else "rand"
        return f"{self.backbone_name}_{pre}_{num_frames}f_{self.image_size}px_{mp}"

    def from_frames(self, frames_bgr: List[np.ndarray], timings: Optional[Dict[str, float]] = None) -> np.ndarray:
        """frames_bgr: already-sampled frames (list of length T). Returns (T, D) float32."""
        t0 = time.perf_counter()
        arr = preprocess_frames(frames_bgr, self.image_size)
        tensor = torch.from_numpy(arr).to(self.device)
        t1 = time.perf_counter()
        feats = self.cnn.extract(tensor).cpu().numpy()
        t2 = time.perf_counter()
        if self.landmarks is not None:
            feats = np.concatenate([feats, self.landmarks.extract(frames_bgr)], axis=1)
        t3 = time.perf_counter()
        if timings is not None:
            timings["preprocess_ms"] = (t1 - t0) * 1000
            timings["cnn_ms"] = (t2 - t1) * 1000
            timings["landmarks_ms"] = (t3 - t2) * 1000
        return feats.astype(np.float32)

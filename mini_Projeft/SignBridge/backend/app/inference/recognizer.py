"""
Recognizer: loads the trained model and predicts a sentence class from frames.

If no trained model exists it raises ModelNotTrainedError - it NEVER invents a prediction.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

from app.inference.feature_pipeline import FeaturePipeline
from app.models.signbridge_model import head_from_meta
from app.preprocessing.frames import read_video_frames, sample_frames

NOT_TRAINED_MESSAGE = "Model not trained yet. Run the training command."


class ModelNotTrainedError(RuntimeError):
    def __init__(self, message: str = NOT_TRAINED_MESSAGE):
        super().__init__(message)


class Recognizer:
    def __init__(self, model_path: Path):
        model_path = Path(model_path)
        if not model_path.is_file():
            raise ModelNotTrainedError()
        ckpt = torch.load(model_path, map_location="cpu", weights_only=True)
        self.meta: dict = ckpt["meta"]
        self.classes: List[str] = self.meta["classes"]
        self.num_frames: int = self.meta["num_frames"]
        self.pipeline = FeaturePipeline(self.meta["backbone"], self.meta["cnn_pretrained"],
                                        self.meta["image_size"], self.meta["use_mediapipe"],
                                        self.meta["mediapipe_parts"])
        if self.pipeline.feature_dim != self.meta["input_dim"]:
            raise RuntimeError("Checkpoint feature size does not match the current CNN/MediaPipe settings.")
        self.head = head_from_meta(self.meta)
        self.head.load_state_dict(ckpt["state_dict"])
        self.head.to(self.pipeline.device).eval()

    @torch.no_grad()
    def predict_frames(self, frames_bgr: List[np.ndarray], top_k: int = 3) -> Dict:
        if len(frames_bgr) < 2:
            raise ValueError("At least 2 frames are needed for recognition.")
        t_start = time.perf_counter()
        timings: Dict[str, float] = {}
        sampled = sample_frames(frames_bgr, self.num_frames)
        feats = self.pipeline.from_frames(sampled, timings)
        t_head = time.perf_counter()
        x = torch.from_numpy(feats).unsqueeze(0).to(self.pipeline.device)      # (1, T, D)
        logits, attn = self.head(x, return_attention=True)
        probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()
        timings["head_ms"] = (time.perf_counter() - t_head) * 1000
        timings["total_ms"] = (time.perf_counter() - t_start) * 1000

        order = np.argsort(-probs)[:top_k]
        top = [{"class_id": int(i), "label": self.classes[i], "confidence": float(probs[i])} for i in order]
        return {
            "class_id": top[0]["class_id"], "label": top[0]["label"], "confidence": top[0]["confidence"],
            "top_k": top, "frames_received": len(frames_bgr), "frames_used": self.num_frames,
            "attention": [float(a) for a in attn[0].cpu().numpy()],
            "latency_ms": {k: round(v, 2) for k, v in timings.items()},
            "device": str(self.pipeline.device),
        }

    def predict_video(self, path: str, top_k: int = 3) -> Dict:
        return self.predict_frames(read_video_frames(path), top_k)

"""Experiment 4 robust landmark normalization and temporal motion features.

Frame layout: pose (33 * xyz+visibility), left hand (21 * xyz), right hand
(21 * xyz), two hand-presence flags, followed by frame-to-frame xyz velocity
for the 33 pose and 42 hand landmarks. Total: 260 base + 225 velocity = 485.

Coordinates are hip-midpoint centered and shoulder-distance scaled. Shoulder
scales below 0.10 normalized image units reuse the previous good scale, or a
0.25 fallback. Centers use the current hip midpoint when possible, otherwise
the previous center or image center. All normalized xyz are clipped to [-8, 8].
Velocity is zero at frame zero and unless the landmark group is present in both
adjacent frames. No temporal pooling or left/right hand swapping occurs.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

import cv2
import numpy as np

from app.preprocessing.landmark_sequence import (
    HAND_DIM,
    HAND_LANDMARKS,
    POSE_DIM,
    POSE_LANDMARKS,
    MediaPipeHolisticSequenceExtractor,
)

NUM_FRAMES = 30
POSE_XYZ_DIM = POSE_LANDMARKS * 3
HAND_XYZ_DIM = HAND_LANDMARKS * 3
BASE_DIM = POSE_DIM + 2 * HAND_DIM + 2
MOTION_DIM = POSE_XYZ_DIM + 2 * HAND_XYZ_DIM
FEATURE_DIM = BASE_DIM + MOTION_DIM
MIN_SHOULDER_SCALE = 0.10
FALLBACK_SHOULDER_SCALE = 0.25
COORDINATE_CLIP = 8.0
MIN_VISIBLE = 0.4


@dataclass
class MotionDetectionSummary:
    frames: int = 0
    pose_frames: int = 0
    left_hand_frames: int = 0
    right_hand_frames: int = 0
    fallback_center_frames: int = 0
    fallback_scale_frames: int = 0
    clipped_coordinate_values: int = 0
    normalized_coordinate_values: int = 0
    left_hand_present_frames: int = 0
    right_hand_present_frames: int = 0
    both_hand_frames: int = 0

    def add(self, other: "MotionDetectionSummary") -> None:
        for name in self.__dataclass_fields__:
            setattr(self, name, getattr(self, name) + getattr(other, name))

    def to_dict(self) -> dict:
        n = self.frames
        return {
            "frames": n,
            "pose_detection_percent": 100.0 * self.pose_frames / n if n else 0.0,
            "left_hand_detection_percent": 100.0 * self.left_hand_frames / n if n else 0.0,
            "right_hand_detection_percent": 100.0 * self.right_hand_frames / n if n else 0.0,
            "left_hand_present_frames": self.left_hand_present_frames,
            "right_hand_present_frames": self.right_hand_present_frames,
            "both_hand_frames": self.both_hand_frames,
            "fallback_center_frames": self.fallback_center_frames,
            "fallback_scale_frames": self.fallback_scale_frames,
            "clipped_coordinate_values": self.clipped_coordinate_values,
            "normalized_coordinate_values": self.normalized_coordinate_values,
            "coordinate_clip_percent": 100.0 * self.clipped_coordinate_values / self.normalized_coordinate_values
                if self.normalized_coordinate_values else 0.0,
        }


class MediaPipeMotionSequenceExtractor:
    def __init__(self, model_complexity: int = 1, min_detection_confidence: float = 0.5,
                 min_tracking_confidence: float = 0.5):
        try:
            import mediapipe as mp
            self._holistic = mp.solutions.holistic.Holistic(
                static_image_mode=False,
                model_complexity=model_complexity,
                smooth_landmarks=True,
                enable_segmentation=False,
                refine_face_landmarks=False,
                min_detection_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence,
            )
        except Exception as exc:  # pragma: no cover - environment-dependent
            raise RuntimeError(f"Experiment 4 requires the classic MediaPipe Holistic API: {exc}") from exc
        self._previous_center: Optional[np.ndarray] = None
        self._previous_scale: Optional[float] = None

    def reset_sequence(self) -> None:
        """Clear temporal normalization state before processing another video."""
        self._previous_center = None
        self._previous_scale = None

    @staticmethod
    def _points(landmarks: Any, count: int, visibility: bool = False) -> Optional[np.ndarray]:
        # Reuse Experiment 3's verified MediaPipe output parser.
        return MediaPipeHolisticSequenceExtractor._landmark_array(landmarks, count, visibility)

    @staticmethod
    def _visible(pose: np.ndarray, index: int) -> bool:
        return float(pose[index, 3]) >= MIN_VISIBLE and np.isfinite(pose[index, :3]).all()

    def _reference(self, pose: Optional[np.ndarray], stats: MotionDetectionSummary) -> Tuple[np.ndarray, float]:
        center_valid = pose is not None and self._visible(pose, 23) and self._visible(pose, 24)
        if center_valid:
            self._previous_center = ((pose[23, :3] + pose[24, :3]) * 0.5).astype(np.float32)
        else:
            stats.fallback_center_frames += 1
        center = self._previous_center if self._previous_center is not None else np.asarray([0.5, 0.5, 0.0], np.float32)

        scale_valid = pose is not None and self._visible(pose, 11) and self._visible(pose, 12)
        scale = float(np.linalg.norm(pose[11, :2] - pose[12, :2])) if scale_valid else 0.0
        if not np.isfinite(scale) or scale < MIN_SHOULDER_SCALE:
            stats.fallback_scale_frames += 1
            scale = self._previous_scale if self._previous_scale is not None else FALLBACK_SHOULDER_SCALE
        else:
            self._previous_scale = scale
        return np.asarray(center, dtype=np.float32), float(scale)

    @staticmethod
    def _normalize_xyz(points: np.ndarray, center: np.ndarray, scale: float,
                       stats: MotionDetectionSummary) -> np.ndarray:
        xyz = (points[:, :3] - center[None, :]) / scale
        stats.normalized_coordinate_values += int(xyz.size)
        stats.clipped_coordinate_values += int(np.count_nonzero((xyz < -COORDINATE_CLIP) | (xyz > COORDINATE_CLIP)))
        return np.clip(xyz, -COORDINATE_CLIP, COORDINATE_CLIP).astype(np.float32)

    def extract(self, frames_bgr: List[np.ndarray]) -> Tuple[np.ndarray, MotionDetectionSummary]:
        if len(frames_bgr) != NUM_FRAMES:
            raise ValueError(f"Experiment 4 expects exactly {NUM_FRAMES} sampled frames; received {len(frames_bgr)}.")
        base = np.zeros((NUM_FRAMES, BASE_DIM), dtype=np.float32)
        pose_xyz = np.zeros((NUM_FRAMES, POSE_XYZ_DIM), dtype=np.float32)
        left_xyz = np.zeros((NUM_FRAMES, HAND_XYZ_DIM), dtype=np.float32)
        right_xyz = np.zeros((NUM_FRAMES, HAND_XYZ_DIM), dtype=np.float32)
        pose_present = np.zeros(NUM_FRAMES, dtype=bool)
        left_present = np.zeros(NUM_FRAMES, dtype=bool)
        right_present = np.zeros(NUM_FRAMES, dtype=bool)
        stats = MotionDetectionSummary()

        for t, frame in enumerate(frames_bgr):
            result = self._holistic.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            pose = self._points(result.pose_landmarks, POSE_LANDMARKS, visibility=True)
            left = self._points(result.left_hand_landmarks, HAND_LANDMARKS)
            right = self._points(result.right_hand_landmarks, HAND_LANDMARKS)
            center, scale = self._reference(pose, stats)
            cursor = 0
            if pose is not None:
                normalized_pose = pose.copy()
                normalized_pose[:, :3] = self._normalize_xyz(pose, center, scale, stats)
                base[t, cursor:cursor + POSE_DIM] = normalized_pose.reshape(-1)
                pose_xyz[t] = normalized_pose[:, :3].reshape(-1)
                pose_present[t] = True
                stats.pose_frames += 1
            cursor += POSE_DIM
            if left is not None:
                left_xyz[t] = self._normalize_xyz(left, center, scale, stats).reshape(-1)
                base[t, cursor:cursor + HAND_DIM] = left_xyz[t]
                left_present[t] = True
                stats.left_hand_frames += 1
            cursor += HAND_DIM
            if right is not None:
                right_xyz[t] = self._normalize_xyz(right, center, scale, stats).reshape(-1)
                base[t, cursor:cursor + HAND_DIM] = right_xyz[t]
                right_present[t] = True
                stats.right_hand_frames += 1
            cursor += HAND_DIM
            base[t, cursor] = float(left is not None)
            base[t, cursor + 1] = float(right is not None)
            stats.left_hand_present_frames += int(left is not None)
            stats.right_hand_present_frames += int(right is not None)
            stats.both_hand_frames += int(left is not None and right is not None)
            stats.frames += 1

        velocity = np.zeros((NUM_FRAMES, MOTION_DIM), dtype=np.float32)
        for t in range(1, NUM_FRAMES):
            if pose_present[t] and pose_present[t - 1]:
                velocity[t, :POSE_XYZ_DIM] = pose_xyz[t] - pose_xyz[t - 1]
            left_start = POSE_XYZ_DIM
            right_start = POSE_XYZ_DIM + HAND_XYZ_DIM
            if left_present[t] and left_present[t - 1]:
                velocity[t, left_start:left_start + HAND_XYZ_DIM] = left_xyz[t] - left_xyz[t - 1]
            if right_present[t] and right_present[t - 1]:
                velocity[t, right_start:right_start + HAND_XYZ_DIM] = right_xyz[t] - right_xyz[t - 1]
        sequence = np.concatenate([base, velocity], axis=1).astype(np.float32, copy=False)
        if sequence.shape != (NUM_FRAMES, FEATURE_DIM) or not np.isfinite(sequence).all():
            raise RuntimeError(f"Invalid Experiment 4 sequence: shape={sequence.shape}, finite={np.isfinite(sequence).all()}")
        return sequence, stats

    def close(self) -> None:
        self._holistic.close()

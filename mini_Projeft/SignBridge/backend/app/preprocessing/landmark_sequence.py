"""MediaPipe Holistic extraction for temporally ordered ISL pose/hand features.

Each frame contains pose (33 x xyz+visibility), left hand (21 x xyz), and
right hand (21 x xyz), concatenated in that order for 258 values. Coordinates
are centered on the pose hip midpoint and scaled by shoulder distance. Visibility
is retained unscaled. Per-frame parameters update over time, so temporal movement
is preserved; if current pose references are unavailable the prior valid body
reference is reused, then a (0.5, 0.5, 0), scale-1 fallback is used initially.
Missing landmark groups are all zeros.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

POSE_LANDMARKS = 33
HAND_LANDMARKS = 21
POSE_DIM = POSE_LANDMARKS * 4
HAND_DIM = HAND_LANDMARKS * 3
FRAME_DIM = POSE_DIM + 2 * HAND_DIM


@dataclass
class LandmarkDetectionSummary:
    total_frames: int = 0
    pose_detected_frames: int = 0
    left_hand_detected_frames: int = 0
    right_hand_detected_frames: int = 0
    detected_landmarks_total: int = 0

    def add(self, other: "LandmarkDetectionSummary") -> None:
        self.total_frames += other.total_frames
        self.pose_detected_frames += other.pose_detected_frames
        self.left_hand_detected_frames += other.left_hand_detected_frames
        self.right_hand_detected_frames += other.right_hand_detected_frames
        self.detected_landmarks_total += other.detected_landmarks_total

    def to_dict(self) -> dict:
        frames = self.total_frames
        return {
            "total_frames": frames,
            "pose_detected_frames": self.pose_detected_frames,
            "left_hand_detected_frames": self.left_hand_detected_frames,
            "right_hand_detected_frames": self.right_hand_detected_frames,
            "pose_detection_percent": 100.0 * self.pose_detected_frames / frames if frames else 0.0,
            "left_hand_detection_percent": 100.0 * self.left_hand_detected_frames / frames if frames else 0.0,
            "right_hand_detection_percent": 100.0 * self.right_hand_detected_frames / frames if frames else 0.0,
            "detected_landmarks_total": self.detected_landmarks_total,
            "average_detected_landmarks_per_frame": self.detected_landmarks_total / frames if frames else 0.0,
        }


class MediaPipeHolisticSequenceExtractor:
    """Track pose and both hands on frames and normalize each frame consistently."""

    def __init__(self, model_complexity: int = 1, min_detection_confidence: float = 0.5,
                 min_tracking_confidence: float = 0.5):
        try:
            import mediapipe as mp
            self._holistic_module = mp.solutions.holistic
        except Exception as exc:  # pragma: no cover - environment-dependent
            raise RuntimeError(
                "MediaPipe 0.10.14 with the classic solutions API is required. "
                "Install backend/requirements-optional.txt in backend\\.venv. "
                f"Details: {exc}"
            ) from exc
        self._holistic = self._holistic_module.Holistic(
            static_image_mode=False,
            model_complexity=model_complexity,
            smooth_landmarks=True,
            enable_segmentation=False,
            refine_face_landmarks=False,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._previous_reference: Optional[Tuple[np.ndarray, float]] = None

    @staticmethod
    def _landmark_array(landmarks: Any, count: int, with_visibility: bool) -> Optional[np.ndarray]:
        if landmarks is None:
            return None
        values = []
        for point in landmarks.landmark:
            row = [point.x, point.y, point.z]
            if with_visibility:
                row.append(point.visibility)
            values.append(row)
        arr = np.asarray(values, dtype=np.float32)
        expected = 4 if with_visibility else 3
        if arr.shape != (count, expected) or not np.isfinite(arr).all():
            return None
        return arr

    @staticmethod
    def _visible(pose: np.ndarray, index: int) -> bool:
        return float(pose[index, 3]) >= 0.4 and np.isfinite(pose[index, :3]).all()

    def _body_reference(self, pose: Optional[np.ndarray]) -> Tuple[np.ndarray, float]:
        if pose is not None and all(self._visible(pose, i) for i in (11, 12, 23, 24)):
            center = (pose[23, :3] + pose[24, :3]) * 0.5
            shoulder_delta = pose[11, :2] - pose[12, :2]
            scale = float(np.linalg.norm(shoulder_delta))
            if scale > 1e-4 and np.isfinite(scale) and np.isfinite(center).all():
                self._previous_reference = (center.astype(np.float32), scale)
        if self._previous_reference is not None:
            return self._previous_reference
        return np.asarray([0.5, 0.5, 0.0], dtype=np.float32), 1.0

    def extract(self, frames_bgr: List[np.ndarray]) -> Tuple[np.ndarray, LandmarkDetectionSummary]:
        if not frames_bgr:
            raise ValueError("Cannot extract landmarks from an empty frame sequence.")
        output = np.zeros((len(frames_bgr), FRAME_DIM), dtype=np.float32)
        summary = LandmarkDetectionSummary()
        for frame_id, frame in enumerate(frames_bgr):
            result = self._holistic.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            pose = self._landmark_array(result.pose_landmarks, POSE_LANDMARKS, with_visibility=True)
            left = self._landmark_array(result.left_hand_landmarks, HAND_LANDMARKS, with_visibility=False)
            right = self._landmark_array(result.right_hand_landmarks, HAND_LANDMARKS, with_visibility=False)
            center, scale = self._body_reference(pose)
            cursor = 0
            if pose is not None:
                normalized_pose = pose.copy()
                normalized_pose[:, :3] = (normalized_pose[:, :3] - center[None, :]) / scale
                output[frame_id, cursor:cursor + POSE_DIM] = normalized_pose.reshape(-1)
                summary.pose_detected_frames += 1
                summary.detected_landmarks_total += POSE_LANDMARKS
            cursor += POSE_DIM
            for hand in (left, right):
                if hand is not None:
                    normalized_hand = (hand - center[None, :]) / scale
                    output[frame_id, cursor:cursor + HAND_DIM] = normalized_hand.reshape(-1)
                    summary.detected_landmarks_total += HAND_LANDMARKS
                    if cursor == POSE_DIM:
                        summary.left_hand_detected_frames += 1
                    else:
                        summary.right_hand_detected_frames += 1
                cursor += HAND_DIM
            summary.total_frames += 1
        if output.shape != (len(frames_bgr), FRAME_DIM) or not np.isfinite(output).all():
            raise RuntimeError("Landmark extraction produced an invalid feature sequence.")
        return output, summary

    def close(self) -> None:
        self._holistic.close()

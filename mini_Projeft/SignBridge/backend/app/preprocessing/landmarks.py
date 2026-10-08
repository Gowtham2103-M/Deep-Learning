"""
OPTIONAL MediaPipe landmark extraction (hands + pose + face).

Only used when USE_MEDIAPIPE = true.  If the dataset videos do not clearly show hands / body /
face, keep it OFF - the main model works without it.

Per frame we produce one vector:
    hands : 2 hands x 21 points x (x, y, z)        = 126   (zeros if a hand is not found)
    pose  : 33 points x (x, y, z, visibility)      = 132
    face  : 468 points x (x, y, z)                 = 1404
Only the parts listed in MEDIAPIPE_PARTS are included.

Needs the classic MediaPipe "solutions" API (mediapipe==0.10.14 in requirements.txt).
Newer MediaPipe releases removed it; in that case a clear error is raised.
"""
from __future__ import annotations

from typing import List

import numpy as np

PART_DIMS = {"hands": 126, "pose": 132, "face": 1404}


class MediaPipeUnavailable(RuntimeError):
    pass


def landmark_dim(parts: List[str]) -> int:
    unknown = [p for p in parts if p not in PART_DIMS]
    if unknown:
        raise ValueError(f"Unknown MEDIAPIPE_PARTS {unknown}. Use hands, pose, face.")
    return sum(PART_DIMS[p] for p in parts)


class LandmarkExtractor:
    def __init__(self, parts: List[str]):
        self.parts = parts
        self.dim = landmark_dim(parts)
        try:
            import mediapipe as mp
            self._holistic_mod = mp.solutions.holistic
        except Exception as exc:  # noqa
            raise MediaPipeUnavailable(
                "MediaPipe 'solutions' API is not available. Install the pinned version with "
                "'pip install mediapipe==0.10.14' or set USE_MEDIAPIPE=false. "
                f"(details: {exc})")
        self._holistic = self._holistic_mod.Holistic(static_image_mode=True, model_complexity=1,
                                                     refine_face_landmarks=False)

    @staticmethod
    def _xyz(lms, n, with_vis=False):
        k = 4 if with_vis else 3
        if lms is None:
            return np.zeros(n * k, dtype=np.float32)
        rows = [[p.x, p.y, p.z] + ([p.visibility] if with_vis else []) for p in lms.landmark]
        return np.asarray(rows, dtype=np.float32).flatten()

    def extract(self, frames_bgr: List[np.ndarray]) -> np.ndarray:
        import cv2
        out = []
        for f in frames_bgr:
            res = self._holistic.process(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
            vec = []
            if "hands" in self.parts:
                vec += [self._xyz(res.left_hand_landmarks, 21), self._xyz(res.right_hand_landmarks, 21)]
            if "pose" in self.parts:
                vec.append(self._xyz(res.pose_landmarks, 33, with_vis=True))
            if "face" in self.parts:
                vec.append(self._xyz(res.face_landmarks, 468))
            out.append(np.concatenate(vec))
        return np.stack(out).astype(np.float32)

    def close(self):
        self._holistic.close()

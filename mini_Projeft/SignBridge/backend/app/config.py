"""
Central configuration for SignBridge.

How settings are chosen (later steps override earlier ones):
  1. The defaults written below.
  2. An optional file  config.json  in the project root (copy config.example.json).
  3. Environment variables named SIGNBRIDGE_<SETTING>, e.g. SIGNBRIDGE_NUM_FRAMES=24

All paths are stored RELATIVE to the project root, so the project runs from any folder.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict, fields
from pathlib import Path
from typing import List, Optional

# backend/app/config.py  ->  parents[2] is the project root (the folder holding README.md)
PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class Settings:
    # ---- dataset ----
    DATASET_PATH: str = "dataset/ISL_CSLRT_Corpus"   # where you unzip ISL-CSLTR
    NUM_CLASSES: Optional[int] = None                 # None = detect from the CSV automatically
    CSV_FILE: Optional[str] = None                    # optional: force a CSV (path relative to dataset)
    VIDEO_COLUMN: Optional[str] = None                # optional: force the CSV column with video names
    LABEL_COLUMN: Optional[str] = None                # optional: force the CSV column with sentences
    SIGNER_COLUMN: Optional[str] = None               # optional: force the signer ID column
    # ---- frames / input ----
    NUM_FRAMES: int = 30                              # frames sampled per video (configurable!)
    IMAGE_SIZE: int = 224
    # ---- CNN ----
    CNN_BACKBONE: str = "mobilenet_v2"                # mobilenet_v2 | resnet18 | efficientnet_b0
    CNN_PRETRAINED: bool = True                       # ImageNet weights (downloaded once, ~14 MB)
    # ---- optional MediaPipe fusion ----
    USE_MEDIAPIPE: bool = False
    MEDIAPIPE_PARTS: str = "hands,pose,face"          # comma separated subset
    # ---- temporal head ----
    LSTM_HIDDEN: int = 128
    LSTM_LAYERS: int = 1
    DENSE_UNITS: int = 128
    DROPOUT: float = 0.5
    USE_ATTENTION: bool = True                        # False = plain average (ablation)
    # ---- training ----
    BATCH_SIZE: int = 16
    EPOCHS: int = 60
    LEARNING_RATE: float = 1e-3
    WEIGHT_DECAY: float = 1e-4
    EARLY_STOP_PATIENCE: int = 10
    VAL_FRACTION: float = 0.15
    TEST_FRACTION: float = 0.20
    SEED: int = 42
    # ---- files ----
    MODEL_PATH: str = "models/signbridge_head.pt"
    FEATURE_CACHE_DIR: str = "models/feature_cache"
    # ---- inference / UI ----
    CONFIDENCE_THRESHOLD: float = 0.60                # below this a prediction is NOT accepted
    MATCH_THRESHOLD: float = 0.75                     # fuzzy speech->sentence match needed to show a sign
    CAPTURE_FPS: int = 10                             # browser camera frames per second kept in buffer
    WINDOW_SECONDS: float = 3.0                       # length of the sliding window sent for recognition
    STABLE_COUNT: int = 2                             # same accepted prediction N times in a row
    SPEAK_COOLDOWN_SECONDS: float = 4.0               # do not auto-speak faster than this

    # -------- helpers --------
    def path(self, value: str) -> Path:
        """Turn a setting like 'models/x.pt' into an absolute path under the project root."""
        p = Path(value)
        return p if p.is_absolute() else PROJECT_ROOT / p

    @property
    def dataset_dir(self) -> Path:
        return self.path(self.DATASET_PATH)

    @property
    def model_path(self) -> Path:
        return self.path(self.MODEL_PATH)

    @property
    def cache_dir(self) -> Path:
        return self.path(self.FEATURE_CACHE_DIR)

    @property
    def models_dir(self) -> Path:
        return self.model_path.parent

    @property
    def mediapipe_parts(self) -> List[str]:
        return [p.strip() for p in self.MEDIAPIPE_PARTS.split(",") if p.strip()]

    def to_dict(self) -> dict:
        return asdict(self)


def _cast(value: str, default):
    if isinstance(default, bool):
        return value.strip().lower() in ("1", "true", "yes", "on")
    if isinstance(default, int):
        return int(value)
    if isinstance(default, float):
        return float(value)
    if value.strip().lower() in ("", "none", "null"):
        return None
    return value


def load_settings() -> Settings:
    s = Settings()
    cfg_file = Path(os.environ.get("SIGNBRIDGE_CONFIG", PROJECT_ROOT / "config.json"))
    if cfg_file.is_file():
        try:
            data = json.loads(cfg_file.read_text(encoding="utf-8"))
            for k, v in data.items():
                if k.startswith("_"):
                    continue          # allow comment keys such as "_comment"
                if hasattr(s, k):
                    setattr(s, k, v)
                else:
                    print(f"[config] Unknown setting '{k}' in {cfg_file.name} was ignored.")
        except Exception as exc:  # noqa
            print(f"[config] Could not read {cfg_file}: {exc}. Using defaults.")
    for f in fields(Settings):
        env = os.environ.get(f"SIGNBRIDGE_{f.name}")
        if env is None:
            continue
        default = getattr(s, f.name)
        if f.name == "NUM_CLASSES":
            setattr(s, f.name, int(env) if env.strip().isdigit() else None)
        elif default is None:            # Optional[str] settings that default to None
            setattr(s, f.name, _cast(env, ""))
        else:
            setattr(s, f.name, _cast(env, default))
    return s


_cached: Optional[Settings] = None


def get_settings() -> Settings:
    global _cached
    if _cached is None:
        _cached = load_settings()
    return _cached


def reload_settings() -> Settings:
    global _cached
    _cached = load_settings()
    return _cached

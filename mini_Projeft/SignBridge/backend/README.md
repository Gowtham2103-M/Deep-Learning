# SignBridge backend

Python 3.10-3.12, FastAPI, PyTorch, OpenCV. Full instructions are in the main `../README.md`.

Quick reference (run from this folder, virtual environment active):

| Task | Command |
|---|---|
| Install | `pip install -r requirements.txt` |
| Audit dataset (read-only) | `python scripts\audit_dataset.py` |
| Extract CNN features | `python -m training.extract_features` |
| Train (includes extraction + test evaluation) | `python -m training.train` |
| Re-evaluate | `python -m training.evaluate` |
| Measure latency | `python -m training.benchmark_latency --videos 20` |
| Start API | `python -m uvicorn app.main:app --reload --port 8000` |
| Tests | `python -m pytest -q` |

## API endpoints (used by the frontend)

| Method | URL | Purpose |
|---|---|---|
| GET | `/api/health` | alive check |
| GET | `/api/status` | dataset + model status + config |
| GET | `/api/config` | settings the UI needs (window length, thresholds ...) |
| GET | `/api/vocabulary` | sentences available (from the dataset CSV) |
| GET | `/api/metrics` | real metrics/latency, or "Model not trained yet." |
| POST | `/api/predict/frames` | multipart `frames` (JPEG images) -> prediction (HTTP 503 if not trained) |
| POST | `/api/predict/video` | multipart `file` (video) -> prediction |
| POST | `/api/speech/match` | JSON `{"text": "..."}` -> matched sentence + video URL, or "No matching ISL sign available." |
| GET | `/api/reference-video/{class_id}` | streams the reference ISL video |
| POST | `/api/dataset/refresh` | re-read the dataset folder |

## What each main file does
* `app/config.py` - every setting (`DATASET_PATH`, `NUM_FRAMES`, `IMAGE_SIZE`, `BATCH_SIZE`, `EPOCHS`, `LEARNING_RATE`,
  `CNN_BACKBONE`, `USE_MEDIAPIPE`, `MODEL_PATH`, `CONFIDENCE_THRESHOLD`, ...).
* `app/preprocessing/dataset_index.py` - finds videos + CSV, guesses columns, builds the label table.
* `app/preprocessing/frames.py` - reads videos, samples N evenly spaced frames, resizes/normalises.
* `app/models/signbridge_model.py` - `FeatureExtractor` (frozen CNN), `TemporalAttention`, `BiLSTMAttentionHead`.
* `app/inference/feature_pipeline.py` - frames -> features (shared by training and live use).
* `app/inference/recognizer.py` - loads the trained model, predicts, measures latency.
* `app/services/matcher.py` - speech text -> exact / normalised / fuzzy sentence match.
* `training/train.py` - splits, trains with early stopping + checkpoint, evaluates.

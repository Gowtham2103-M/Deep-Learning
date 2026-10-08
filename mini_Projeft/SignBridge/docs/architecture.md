# SignBridge - Architecture

## 1. System overview

```
DIRECTION 1  (Person B signs)                         DIRECTION 2  (Person A speaks)

 Camera (browser getUserMedia)                         Microphone (browser Web Speech API)
   -> JPEG frames, sliding window (WINDOW_SECONDS)       -> Speech-to-Text (browser)
   -> POST /api/predict/frames                           -> POST /api/speech/match
 Backend                                               Backend
   -> sample NUM_FRAMES evenly                           -> text cleaning (lowercase, punctuation)
   -> resize + normalise                                 -> exact / normalised / fuzzy match
   -> CNN (MobileNetV2, frozen)  -> (T, 1280)            -> reference ISL video (dataset)
   -> [optional MediaPipe landmarks, fused]              -> below MATCH_THRESHOLD:
   -> BiLSTM -> Attention -> Dense -> Softmax               "No matching ISL sign available."
   -> class + confidence + measured latency            Frontend
 Frontend                                                 -> video player  (Play ISL Sign)
   -> accept if confidence >= CONFIDENCE_THRESHOLD
   -> stable in STABLE_COUNT windows -> text
   -> Speak button / debounced auto-speak (speechSynthesis)
```

## 2. Model

| Stage | Role | Shape (B batch, T frames, D features, C classes) |
|---|---|---|
| Frames | sampled video frames | (B, T, 3, H, W) e.g. (B, 30, 3, 224, 224) |
| CNN (MobileNetV2, ImageNet, frozen) | spatial features of each frame | (B, T, 1280) |
| [optional MediaPipe] | hand/pose/face landmarks concatenated per frame | (B, T, 1280 + L) |
| LayerNorm + Dropout | stabilise inputs | same |
| BiLSTM | temporal relations, both directions | (B, T, 2 x hidden) |
| Temporal Attention | weight per frame; weighted sum | weights (B, T) -> context (B, 2 x hidden) |
| Dense (ReLU + Dropout) | combine learned features | (B, dense_units) |
| Linear + Softmax | class probabilities | (B, C) |

Attention: `score_t = v^T tanh(W h_t)`, `weights = softmax(score)`, `context = sum_t weights_t * h_t`.
`USE_ATTENTION=false` replaces it by a plain average over time (for an ablation study).

The CNN is frozen (no fine-tuning) because the dataset is small; features are computed once and cached, so
training the head takes minutes. Training and live inference use the same `FeaturePipeline` code.

**Research position:** CNN + BiLSTM + Attention is an existing combination. It is our *selected* architecture; the
contribution is the integrated, evaluated two-way system (see `methodology.md`).

## 3. Modules requested in the project brief -> files

| # | Module | Where |
|---|---|---|
| 1 | User Interface | `frontend/src/App.tsx`, `components/` |
| 2 | Camera Input | `frontend/src/hooks/useRecognition.ts` |
| 3 | Video/Frame Processing | `backend/app/preprocessing/frames.py` |
| 4 | Hand/Pose Feature Extraction | `backend/app/preprocessing/landmarks.py` (optional), CNN features |
| 5 | Deep Learning Recognition | `backend/app/models/signbridge_model.py`, `inference/recognizer.py` |
| 6 | Temporal Sequence Processing | `BiLSTMAttentionHead` (BiLSTM + Attention) |
| 7 | Sign Classification | Dense + Softmax in `BiLSTMAttentionHead` |
| 8 | Sign-to-Text | `routes/recognition.py` (class id -> sentence from `classes`) |
| 9 | Text-to-Speech | `frontend/src/hooks/useSpeak.ts` |
| 10 | Speech-to-Text | `frontend/src/hooks/useSpeechRecognition.ts` |
| 11 | Text-to-ISL Mapping | `backend/app/services/matcher.py`, `routes/speech.py` |
| 12 | ISL Video Display | `frontend/src/components/SpeechPanel.tsx`, `GET /api/reference-video/{id}` |
| 13 | Evaluation & Monitoring | `training/evaluate.py`, `benchmark_latency.py`, `MetricsPanel.tsx` |

## 4. Design decisions
* **Browser does camera, microphone, speech-to-text and text-to-speech**: free, no paid API, works on Windows
  with Chrome/Edge. The Python backend does the deep learning and matching.
* **Dataset is read dynamically** (`dataset_index.py`): nothing about ISL-CSLTR is hard-coded.
* **Nothing is faked**: before training the API returns HTTP 503 `model_not_trained` and metrics are `null`.
* **Signer-independent split** when signer IDs exist; otherwise a warning is printed and stored with the metrics.

# SignBridge - Two-Way Indian Sign Language Communication Assistant

A college Deep Learning project that connects two people:

* **Person B (signs in ISL)**: camera -> frames -> CNN -> BiLSTM -> Attention -> Dense -> Softmax -> text -> speech
* **Person A (speaks)**: microphone -> speech-to-text -> text cleaning -> sentence matching -> ISL reference video

**Scope (please state this in your report):** the first system is a **100-class, sentence-level, closed-set ISL
classification** system trained on ISL-CSLTR. It is **not** unrestricted continuous ISL translation.
CNN + BiLSTM + Attention is an existing, well-known combination; our contribution is the integration and
evaluation of the whole two-way pipeline (see `docs/methodology.md`).

> **No results are included.** No trained model, accuracy, F1, confusion matrix or latency figure is shipped.
> They appear only after *you* train the model on the real dataset. Before that the app says
> "Model not trained yet. Run the training command."

---------------------------------------------------------------------

## 1. What is inside

```
SignBridge/
  README.md                 <- you are here
  config.example.json       <- copy to config.json to change settings (optional)
  run_backend.bat / run_frontend.bat / run_tests.bat   <- Windows shortcuts
  backend/                  <- Python (FastAPI + PyTorch)
    app/
      main.py               starts the API
      config.py             ALL settings in one place (frames, image size, paths, thresholds ...)
      routes/               API endpoints: system.py, recognition.py, speech.py
      services/             dataset_service.py, model_service.py, matcher.py (speech -> sentence)
      models/               signbridge_model.py  (CNN feature extractor, BiLSTM, Attention, Dense)
      inference/            feature_pipeline.py (frames -> features), recognizer.py (trained model -> prediction)
      preprocessing/        dataset_index.py (reads CSV + videos), frames.py, landmarks.py (optional MediaPipe)
      utils/text.py         text normalisation
    training/               dataset.py (splits), extract_features.py, train.py, evaluate.py, benchmark_latency.py
    scripts/audit_dataset.py   read-only dataset inspector
    tests/                  automated tests (use tiny generated SAMPLE videos)
    requirements.txt, requirements-optional.txt, pytest.ini
  frontend/                 <- React + TypeScript + Vite web page
    src/App.tsx, api.ts, components/, hooks/, styles.css
  dataset/README.md         <- where to put ISL-CSLTR
  models/README.md          <- trained files appear here
  docs/                     architecture.md, dataset.md, methodology.md
```

## 2. Installation (Windows, PowerShell)

**a) Install Python 3.11 (64-bit)** from https://www.python.org/downloads/ - tick **"Add python.exe to PATH"**
on the first screen. (3.10 - 3.12 also work.) Check:
```powershell
python --version
```
**b) Install Node.js LTS** from https://nodejs.org/ . Check:
```powershell
node --version
npm --version
```
**c) Create a virtual environment and install the Python packages** (the first install downloads PyTorch, which is large):
```powershell
cd path\to\SignBridge\backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```
If PowerShell says *"running scripts is disabled"*, run this once in the same window and activate again:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```
You should see `(.venv)` at the start of the prompt. **Every new PowerShell window needs the activate line again.**

**d) Install the web page packages:**
```powershell
cd ..\frontend
npm install
```

## 3. Get the dataset

Follow `dataset/README.md` (download ISL-CSLTR, read its licence, unzip into `dataset\ISL_CSLRT_Corpus`).
The app starts even without it and shows a message telling you where to put it.

## 4. Audit the dataset (do this before training)

```powershell
cd path\to\SignBridge\backend
.\.venv\Scripts\Activate.ps1
python scripts\audit_dataset.py
```
It prints the number of videos, classes, CSV columns, resolution, FPS, duration, frame counts, missing files, class
distribution and whether signer IDs exist, and saves a copy to `reports\dataset_audit_report.txt`.
Look at these lines carefully:
* **30-frame sampling** - are the shortest videos long enough for 30 frames? If not, lower `NUM_FRAMES` in `config.json`.
* **Signer information** - if it says "No signer column", the train/test split may contain the same signer
  in both, and accuracy will be optimistic. Say so in your report.
* **Missing files** - CSV rows that point to videos that do not exist are skipped and reported.

## 5. Settings (optional)

All settings are in `backend/app/config.py`. To change them without editing code, copy `config.example.json` to
`config.json` (in the project folder) and edit it. Environment variables `SIGNBRIDGE_<NAME>` also work.
Important ones: `NUM_FRAMES`, `IMAGE_SIZE`, `BATCH_SIZE`, `EPOCHS`, `LEARNING_RATE`, `CNN_BACKBONE`,
`USE_MEDIAPIPE`, `CONFIDENCE_THRESHOLD`, `DATASET_PATH`, `MODEL_PATH`.

## 6. Run the tests

```powershell
cd path\to\SignBridge\backend
.\.venv\Scripts\Activate.ps1
python -m pytest -q
```
The tests generate tiny random sample videos in a temporary folder; they do **not** need your dataset and
their numbers mean nothing scientifically - they only check that the code works.
For the frontend: `cd frontend` then `npm run build` (checks TypeScript and builds).

## 7. Train the model

From the `backend` folder with the virtual environment active:
```powershell
python -m training.train
```
This one command (1) extracts CNN features from every video (MobileNetV2, ImageNet weights are downloaded
automatically the first time; internet needed once), (2) splits the data, (3) trains BiLSTM + Attention + Dense +
Softmax with early stopping, and (4) evaluates on the held-out test videos.
Feature extraction can be run separately (and resumed if you stop it):
```powershell
python -m training.extract_features
python -m training.train --skip-extract
```
Quick smoke test: `python -m training.extract_features --limit 10`.

Afterwards:
```powershell
python -m training.evaluate            # recompute test metrics (writes models\metrics.json, confusion matrix)
python -m training.benchmark_latency   # measure REAL inference latency (writes models\latency.json)
```
Read the **WARNING** lines printed by training: they tell you if the split was signer-independent or not.

## 8. Run the app

Open **two** PowerShell windows.

**Window 1 - backend:**
```powershell
cd path\to\SignBridge\backend
.\.venv\Scripts\Activate.ps1
python -m uvicorn app.main:app --reload --port 8000
```
Check http://127.0.0.1:8000/docs (interactive API page) and http://127.0.0.1:8000/api/status .

**Window 2 - frontend:**
```powershell
cd path\to\SignBridge\frontend
npm run dev
```
Open **http://localhost:5173** in **Chrome or Edge**.
(`run_backend.bat` and `run_frontend.bat` do the same when double-clicked.)

## 9. Demo steps

1. Start both windows; open the page. Check the top banners (dataset found? model trained?).
2. **Demo 1 - Sign to speech:** click **Start recognition**, allow the camera. Person B performs a sign from the
   dataset vocabulary. The page shows the prediction, confidence, **measured latency**, and the recognized text
   (a sentence is accepted only when it is above the confidence threshold and stable in 2 consecutive windows).
   Click **Speak** (or tick **Auto-speak**, which will not repeat the same sentence during a cooldown).
3. **Demo 2 - Speech to sign:** click **Start Microphone**, say one of the dataset sentences. The page shows the
   recognized speech, the matched sentence and plays the reference ISL video. Say something else to see
   *"No matching ISL sign available."* (typing a sentence works too).
4. Scroll to **Evaluation & Performance** to show the real metrics and latency from training.
5. A fair way to demonstrate recognition when a signer is not available: play a dataset video to the camera
   (this works, but explain that it is not a live signer).

## 10. Troubleshooting

| Problem | Fix |
|---|---|
| `python` is not recognised | Reinstall Python and tick "Add to PATH", or use `py` instead of `python`. |
| Activate.ps1 "running scripts is disabled" | `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` then activate again. |
| `pip install` very slow / fails on torch | Retry; check disk space (PyTorch needs several GB). Use Python 3.10-3.12 64-bit. |
| Banner "Dataset not found" | Put the dataset as described in `dataset/README.md`, or set `DATASET_PATH` in `config.json`, then click **Refresh status**. |
| Audit says "No CSV column matched" | Set `CSV_FILE`, `VIDEO_COLUMN`, `LABEL_COLUMN` in `config.json`. |
| Banner "Model not trained yet" | Run `python -m training.train` (section 7), then click **Refresh status**. |
| Training stops with "split is empty" | The dataset is too small or not found; run the audit first. |
| Training says checkpoint/feature mismatch | Delete `models\feature_cache` and `models\signbridge_head.pt`, then train again (settings changed). |
| Page says "Cannot reach the backend" | Start the backend (Window 1); check port 8000 is free. |
| Camera does not start | Allow the camera in the browser address bar; close Zoom/Teams/other apps using it. Use `localhost`, not an IP address. |
| Microphone button disabled | Speech recognition needs Chrome or Edge (Firefox does not support it) and internet (Chrome sends audio to Google). Type the sentence instead. |
| No voice on **Speak** | Check Windows sound settings/volume; the browser needs an installed voice. |
| Reference video is black / does not play | The browser may not support the video's codec (Chrome/Edge play H.264 MP4). Check with the audit script which format the videos use. |
| `USE_MEDIAPIPE=true` fails | Install the optional package: `pip install -r requirements-optional.txt` (needs the classic MediaPipe `solutions` API). |
| `npm run build` shows TypeScript errors | Run `npm install` again; make sure you did not edit files by mistake. |
| Port 5173 / 8000 already in use | Close old windows, or change the port (`--port 8001` and update `vite.config.ts`). |

## 11. Known limitations (be honest about these in your report)

* **Closed set:** it can only recognise the sentences in the dataset CSV (about 100 if the dataset matches its
  description). It has no "no sign" class, so it *always* produces a best guess; the confidence threshold and the
  two-in-a-row rule reduce, but do not remove, wrong outputs.
* **Sentence-level, not continuous translation.** Unseen sentences or signs outside the vocabulary are not handled.
  True continuous recognition would need a sequence decoder (e.g. CTC or a transformer) and much more data.
* **Small dataset:** few videos per sentence and few signers (to be confirmed by the audit). Expect overfitting
  and weak generalisation to new signers, backgrounds and cameras. If the CSV has no signer column, the test
  score can be optimistic.
* **Domain shift:** the model is trained on dataset videos, but live webcam video differs (lighting, camera,
  distance). Live accuracy can be much lower than the test accuracy.
* **Timing:** each prediction looks at the last `WINDOW_SECONDS` of video; the sign must be performed inside that
  window. Latency is measured and displayed; whether it feels real-time depends on your laptop (CPU/GPU).
* **Non-manual cues:** the frozen CNN sees the whole frame, so facial expression may be captured implicitly, but
  it is not modelled explicitly. Optional MediaPipe fusion is provided but **off by default and untested by us
  on real data**; enable it only if the audit/visual check shows hands, body and face are clearly visible.
* **Speech -> ISL** works only for the dataset's sentences and plays a recorded reference video (not an
  animation). Which reference video is used = the first video of that sentence in the dataset.
* **Speech and voice** use the browser (Web Speech API and speechSynthesis): Chrome/Edge, internet for speech input.
* **Dataset licence** must be checked by you on the Mendeley page (we could not verify it).
* The `.bat` helper scripts and PowerShell commands were written for Windows but were **not run on Windows by the
  author of this package** (development/testing was done on Linux). Report any path problem.

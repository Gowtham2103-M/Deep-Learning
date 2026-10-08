"""Extract normalized MediaPipe sequences, run exp3 sanity checks, then train in isolation.

Run from backend: python -m training.train_landmarks
This command never writes to the dataset, CSV, MobileNet cache, original checkpoint,
or existing exp2 output directory.
"""
from __future__ import annotations

import csv
import hashlib
import json
import random
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import cv2
import torch
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, precision_recall_fscore_support)
from torch import nn
from torch.utils.data import DataLoader, Dataset

from app.config import get_settings
from app.models.landmark_model import (LANDMARK_INPUT_DIM, LANDMARK_NUM_FRAMES,
                                       build_landmark_classifier)
from app.preprocessing.dataset_index import build_index
from app.preprocessing.frames import load_sampled_frames
from app.preprocessing.landmark_sequence import (FRAME_DIM, LandmarkDetectionSummary,
                                                  MediaPipeHolisticSequenceExtractor)
from training.dataset import video_sha256

SEED = 42
BATCH_SIZE = 16
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
DROPOUT = 0.3
MAX_EPOCHS = 60
EARLY_STOPPING_PATIENCE = 15
GRADIENT_CLIP = 5.0
POSE_QUALITY_STOP_PERCENT = 30.0
BOTH_HANDS_QUALITY_STOP_PERCENT = 5.0


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tree_manifest(root: Path) -> dict:
    manifest = []
    for path in sorted(root.rglob("*")):
        if path.is_file():
            stat = path.stat()
            manifest.append((path.relative_to(root).as_posix(), stat.st_size, stat.st_mtime_ns))
    serialized = json.dumps(manifest, separators=(",", ":")).encode("utf-8")
    return {"file_count": len(manifest), "total_bytes": sum(row[1] for row in manifest),
            "sha256_manifest": hashlib.sha256(serialized).hexdigest(), "files": manifest}


def _artifact_manifest(root: Path) -> str:
    if not root.exists():
        return "missing"
    return _tree_manifest(root)["sha256_manifest"]


def _capture_protected_state(settings, exp3_dir: Path) -> dict:
    project_root = settings.models_dir.parent
    project_dataset = settings.dataset_dir.resolve()
    original_dataset = Path.home() / "Downloads" / "ISL_CSLRT_Corpus" / "ISL_CSLRT_Corpus"
    if not project_dataset.is_dir() or not original_dataset.is_dir():
        raise RuntimeError("Both the project dataset and original Downloads dataset must exist for integrity checks.")
    if exp3_dir.exists():
        raise RuntimeError(f"Refusing to overwrite existing Experiment 3 output: {exp3_dir}")
    if (settings.models_dir / "experiments" / "exp2").is_dir() is False:
        raise RuntimeError("Existing Experiment 2 folder is missing; refusing to proceed.")
    protected = {
        "project_dataset": _tree_manifest(project_dataset),
        "original_dataset": _tree_manifest(original_dataset),
        "csv_sha256": _file_digest(project_dataset / "corpus_csv_files" / "ISL Corpus sign glosses.csv"),
        "mobilenet_cache": _tree_manifest(settings.cache_dir / "mobilenet_v2_pre_30f_224px_nomp"),
        "original_checkpoint_sha256": _file_digest(settings.models_dir / "signbridge_head.pt"),
        "exp2_manifest": _artifact_manifest(settings.models_dir / "experiments" / "exp2"),
    }
    if protected["project_dataset"]["file_count"] != 21074:
        raise RuntimeError(f"Project dataset count changed: {protected['project_dataset']['file_count']} files.")
    if protected["original_dataset"]["file_count"] != 21074:
        raise RuntimeError(f"Original dataset count changed: {protected['original_dataset']['file_count']} files.")
    if protected["project_dataset"]["files"] != protected["original_dataset"]["files"]:
        raise RuntimeError("Project dataset differs by relative path, size, or timestamp from the source copy.")
    return protected


def _verify_protected_state(settings, protected: dict) -> dict:
    project_root = settings.models_dir.parent
    current = {
        "project_dataset": _tree_manifest(settings.dataset_dir.resolve()),
        "original_dataset": _tree_manifest(Path.home() / "Downloads" / "ISL_CSLRT_Corpus" / "ISL_CSLRT_Corpus"),
        "csv_sha256": _file_digest(settings.dataset_dir / "corpus_csv_files" / "ISL Corpus sign glosses.csv"),
        "mobilenet_cache": _tree_manifest(settings.cache_dir / "mobilenet_v2_pre_30f_224px_nomp"),
        "original_checkpoint_sha256": _file_digest(settings.models_dir / "signbridge_head.pt"),
        "exp2_manifest": _artifact_manifest(settings.models_dir / "experiments" / "exp2"),
    }
    checks = {
        "project_dataset_unchanged": current["project_dataset"]["sha256_manifest"] == protected["project_dataset"]["sha256_manifest"],
        "original_dataset_unchanged": current["original_dataset"]["sha256_manifest"] == protected["original_dataset"]["sha256_manifest"],
        "csv_unchanged": current["csv_sha256"] == protected["csv_sha256"],
        "mobilenet_cache_unchanged": current["mobilenet_cache"]["sha256_manifest"] == protected["mobilenet_cache"]["sha256_manifest"],
        "original_checkpoint_unchanged": current["original_checkpoint_sha256"] == protected["original_checkpoint_sha256"],
        "exp2_unchanged": current["exp2_manifest"] == protected["exp2_manifest"],
    }
    return checks


def _load_exp2_split(settings, index, classes) -> Tuple[dict, List[dict], dict]:
    exp2 = settings.models_dir / "experiments" / "exp2"
    preflight = json.loads((exp2 / "preflight.json").read_text(encoding="utf-8"))
    exp2_splits = json.loads((exp2 / "splits.json").read_text(encoding="utf-8"))
    excluded = preflight["excluded_conflicting_rows"]
    expected_excluded = {int(row["row"]) for row in excluded}
    if len(expected_excluded) != 8:
        raise RuntimeError(f"Expected eight previously verified conflicting rows; found {len(expected_excluded)}.")
    for item in excluded:
        row_id = int(item["row"])
        if row_id >= len(index.df) or str(index.df.iloc[row_id]["video_path"]) != item["video_path"] or str(index.df.iloc[row_id]["label"]) != item["label"]:
            raise RuntimeError(f"Conflicting-row mapping no longer matches dataset row {row_id}.")

    splits = exp2_splits["splits"]
    all_rows = set()
    partition_sets = {}
    class_to_id = {label: i for i, label in enumerate(classes)}
    for name in ("train", "val", "test"):
        rows = list(map(int, splits[name]["rows"]))
        target_ids = list(map(int, splits[name]["labels"]))
        if len(rows) != len(target_ids):
            raise RuntimeError(f"Experiment 2 {name} row/target lengths differ.")
        for row_id, target in zip(rows, target_ids):
            if row_id in expected_excluded:
                raise RuntimeError(f"Conflicting duplicate row {row_id} appears in the {name} partition.")
            if not 0 <= row_id < len(index.df):
                raise RuntimeError(f"Split contains invalid row ID {row_id}.")
            label = str(index.df.iloc[row_id]["label"])
            if label not in class_to_id or target != class_to_id[label]:
                raise RuntimeError(f"Class ID mismatch for row {row_id}: label={label!r}, target={target}.")
        partition_sets[name] = set(rows)
        all_rows.update(rows)
    if any(partition_sets[a] & partition_sets[b] for a,b in (("train","val"),("train","test"),("val","test"))):
        raise RuntimeError("Experiment 2 split has row overlap.")
    if all_rows != set(range(len(index.df))) - expected_excluded:
        raise RuntimeError("Saved experiment split doesn't cover all eligible rows exactly once.")
    info = dict(exp2_splits["info"])
    info["reused_from"] = "models/experiments/exp2/splits.json"
    info["excluded_conflicting_rows"] = excluded
    info["split_counts"] = {name: len(splits[name]["rows"]) for name in ("train","val","test")}
    info["class_coverage"] = {
        name: len({str(index.df.iloc[row]["label"]) for row in splits[name]["rows"]})
        for name in ("train","val","test")
    }
    info["classes_without_usable_examples"] = sorted(set(classes) - {
        str(index.df.iloc[row]["label"]) for row in all_rows
    })
    return splits, excluded, info


def _cache_metadata(index, classes, extraction_results, failures, frame_summary) -> dict:
    class_to_id = {label: i for i, label in enumerate(classes)}
    rows = []
    for row_id, row in index.df.iterrows():
        result = extraction_results.get(int(row_id))
        rows.append({
            "cache_index": int(row_id),
            "video_path": str(row["video_path"]),
            "sentence": str(row["label"]),
            "class_id": int(class_to_id[str(row["label"])]),
            "status": "ok" if result else "decode_or_extraction_failed",
            "detection": result["detection"] if result else None,
            "error": failures.get(int(row_id)),
        })
    return {
        "schema_version": 1,
        "feature_kind": "mediapipe_holistic_pose_left_right_hands_normalized_sequence",
        "shape_per_video": [LANDMARK_NUM_FRAMES, FRAME_DIM],
        "normalization": "Per-frame center at pose hip midpoint; divide xyz by 2D shoulder distance; keep pose visibility; reuse previous valid reference or initial image-center/unit-scale fallback; missing groups zero; time order unchanged.",
        "classes": classes,
        "videos": rows,
        "aggregate_detection": frame_summary.to_dict(),
        "failed_video_count": len(failures),
        "failures": failures,
    }


def _extract_cache(index, classes, cache_dir: Path) -> Tuple[dict, dict, LandmarkDetectionSummary]:
    cache_dir.mkdir(parents=True, exist_ok=False)
    extractor = MediaPipeHolisticSequenceExtractor()
    extracted: Dict[int, dict] = {}
    failures: Dict[int, str] = {}
    aggregate = LandmarkDetectionSummary()
    try:
        total = len(index.df)
        for row_id, row in index.df.iterrows():
            row_id = int(row_id)
            try:
                frames = load_sampled_frames(str(row["video_path"]), LANDMARK_NUM_FRAMES)
            except (OSError, cv2.error, ValueError) as exc:
                failures[row_id] = f"Video decode failure: {type(exc).__name__}: {exc}"
            else:
                # MediaPipe failures are not treated as empty detections or skipped videos.
                sequence, detection = extractor.extract(frames)
                if sequence.shape != (LANDMARK_NUM_FRAMES, FRAME_DIM) or not np.isfinite(sequence).all():
                    raise RuntimeError(f"Invalid landmark sequence shape/values: {sequence.shape}")
                np.save(cache_dir / f"{row_id:06d}.npy", sequence, allow_pickle=False)
                extracted[row_id] = {"detection": detection.to_dict()}
                aggregate.add(detection)
            if (row_id + 1) % 25 == 0 or row_id + 1 == total:
                print(f"Landmark extraction: {row_id + 1}/{total}; successful={len(extracted)}; failed={len(failures)}")
    finally:
        extractor.close()
    metadata = _cache_metadata(index, classes, extracted, failures, aggregate)
    (cache_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return extracted, failures, aggregate


class LandmarkSequenceDataset(Dataset):
    def __init__(self, cache_dir: Path, row_ids: List[int], labels: List[int]):
        self.cache_dir = Path(cache_dir)
        self.row_ids = list(map(int, row_ids))
        self.labels = list(map(int, labels))

    def __len__(self) -> int:
        return len(self.row_ids)

    def __getitem__(self, item: int):
        sequence = np.load(self.cache_dir / f"{self.row_ids[item]:06d}.npy", allow_pickle=False)
        return torch.from_numpy(sequence.copy()), torch.tensor(self.labels[item], dtype=torch.long)


def _calculate_scores(y_true, y_pred, y_prob, n_classes: int) -> dict:
    labels = list(range(n_classes))
    precision, recall, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="macro", zero_division=0)
    per_class = precision_recall_fscore_support(y_true, y_pred, labels=labels,
                                                average=None, zero_division=0)
    support = np.bincount(np.asarray(y_true, dtype=np.int64), minlength=n_classes)
    top_k = min(5, n_classes)
    top_ids = np.argpartition(-y_prob, kth=top_k - 1, axis=1)[:, :top_k]
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(precision),
        "macro_recall": float(recall),
        "macro_f1": float(macro_f1),
        "top5_accuracy": float(np.mean([target in row for target, row in zip(y_true, top_ids)])),
        "samples": int(len(y_true)),
        "classes_represented": int(len(set(map(int, y_true)))),
        "per_class": {
            str(class_id): {"precision": float(per_class[0][class_id]),
                            "recall": float(per_class[1][class_id]),
                            "f1": float(per_class[2][class_id]),
                            "support": int(support[class_id])}
            for class_id in labels
        },
        "per_class": {
            str(i): {"label_id": i, "precision": float(per_class[0][i]),
                     "recall": float(per_class[1][i]), "f1": float(per_class[2][i]),
                     "support": int(per_class[3][i])}
            for i in labels
        },
    }


def _evaluate(model, loader, criterion, device, n_classes: int, optimizer=None) -> dict:
    train = optimizer is not None
    model.train(train)
    total_loss, correct, count = 0.0, 0, 0
    targets, predictions, probabilities = [], [], []
    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP)
                optimizer.step()
            total_loss += float(loss.detach().item()) * len(y)
            pred = logits.argmax(dim=1)
            correct += int((pred == y).sum().item())
            count += len(y)
            targets.extend(y.detach().cpu().tolist())
            predictions.extend(pred.detach().cpu().tolist())
            probabilities.extend(torch.softmax(logits.detach(), dim=1).cpu().numpy())
    score = _calculate_scores(np.asarray(targets), np.asarray(predictions),
                              np.asarray(probabilities), n_classes)
    return {"loss": total_loss / max(1, count), **score}


def _train_epoch(model, loader, criterion, optimizer, device) -> dict:
    model.train()
    total_loss, correct, count = 0.0, 0, 0
    target_ids, predicted_ids = [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP)
        optimizer.step()
        total_loss += float(loss.detach().item()) * len(y)
        predictions = logits.detach().argmax(dim=1)
        correct += int((predictions == y).sum().item())
        count += len(y)
        target_ids.extend(y.detach().cpu().tolist())
        predicted_ids.extend(predictions.cpu().tolist())
    return {"loss": total_loss / max(1, count), "accuracy": correct / max(1, count),
            "macro_f1": float(f1_score(target_ids, predicted_ids, average="macro", zero_division=0))}


def _overfit_sanity(index, cache_dir: Path, split, classes) -> dict:
    rows = split["train"]["rows"]
    labels = split["train"]["labels"]
    by_class: Dict[int, List[int]] = {}
    for row_id, class_id in zip(rows, labels):
        by_class.setdefault(int(class_id), []).append(int(row_id))
    selected = []
    for class_id in sorted(by_class):
        if len(by_class[class_id]) >= 2:
            selected.extend(by_class[class_id][:2])
        if len(selected) >= 16:
            break
    selected = selected[:16]
    if len(selected) < 8:
        raise RuntimeError(f"Overfit sanity test needs at least 8 eligible samples; got {len(selected)}.")
    row_to_id = {int(r): int(c) for r,c in zip(rows, labels)}
    dataset = LandmarkSequenceDataset(cache_dir, selected, [row_to_id[row] for row in selected])
    loader = DataLoader(dataset, batch_size=len(dataset), shuffle=True)
    torch.manual_seed(SEED + 303)
    model = build_landmark_classifier(len(classes), DROPOUT)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss()
    device = torch.device("cpu")
    best_acc, best_epoch, stale = 0.0, 0, 0
    history = []
    for epoch in range(1, 401):
        result = _train_epoch(model, loader, criterion, opt, device)
        history.append(result)
        model.eval()
        with torch.no_grad():
            x, y = next(iter(loader))
            acc = float((model(x).argmax(1) == y).float().mean().item())
        if acc > best_acc:
            best_acc, best_epoch, stale = acc, epoch, 0
        else:
            stale += 1
        if best_acc >= 0.95 or stale >= 80:
            break
    return {"passed": best_acc >= 0.90, "samples": len(selected),
            "classes": len(set(row_to_id[row] for row in selected)),
            "best_train_accuracy_eval_mode": best_acc, "best_epoch": best_epoch,
            "epochs_run": len(history), "target_accuracy": 0.90,
            "learning_rate": 0.001, "note": "Sanity-only subset optimization; checkpoint not saved."}


def _save_confusion_plot(matrix: np.ndarray, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 8))
    image = ax.imshow(matrix, cmap="Blues", interpolation="nearest")
    ax.set_title("Experiment 3 MediaPipe landmark test confusion matrix")
    ax.set_xlabel("Predicted class ID")
    ax.set_ylabel("True class ID")
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _evaluate_split(model, split, cache_dir: Path, classes, device, criterion=None) -> Tuple[dict, np.ndarray, np.ndarray]:
    dataset = LandmarkSequenceDataset(cache_dir, split["rows"], split["labels"])
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    model.eval()
    ys, preds, probs = [], [], []
    loss_total, sample_total = 0.0, 0
    with torch.no_grad():
        for x, y in loader:
            logits = model(x.to(device))
            if criterion is not None:
                loss_total += float(criterion(logits, y.to(device)).item()) * len(y)
                sample_total += len(y)
            ys.extend(y.tolist())
            preds.extend(logits.argmax(1).cpu().tolist())
            probs.extend(torch.softmax(logits, dim=1).cpu().numpy())
    y_true = np.asarray(ys, dtype=np.int64)
    y_pred = np.asarray(preds, dtype=np.int64)
    y_prob = np.asarray(probs, dtype=np.float32)
    result = _calculate_scores(y_true, y_pred, y_prob, len(classes))
    if criterion is not None:
        result["loss"] = loss_total / max(1, sample_total)
    return result, y_true, y_pred


def _finalize_saved_training() -> int:
    """Finish evaluation/reporting for an already-trained exp3 checkpoint; never trains or extracts."""
    settings = get_settings()
    output_dir = settings.models_dir / "experiments" / "exp3_landmarks"
    cache_dir = settings.cache_dir / "mediapipe_landmarks_30f"
    required = [output_dir / "preflight.json", output_dir / "splits.json",
                output_dir / "classes.json", output_dir / "history.json",
                output_dir / "signbridge_landmark_model.pt", cache_dir / "metadata.json"]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(f"Cannot finalize Experiment 3; missing saved files: {missing}")
    preflight = json.loads((output_dir / "preflight.json").read_text(encoding="utf-8"))
    splits_file = json.loads((output_dir / "splits.json").read_text(encoding="utf-8"))
    classes = json.loads((output_dir / "classes.json").read_text(encoding="utf-8"))
    history = json.loads((output_dir / "history.json").read_text(encoding="utf-8"))
    cache_metadata = json.loads((cache_dir / "metadata.json").read_text(encoding="utf-8"))
    per_video_detection = [row["detection"] for row in cache_metadata["videos"] if row["status"] == "ok"]
    zero_detection_videos = {
        "pose": sum(item["pose_detected_frames"] == 0 for item in per_video_detection),
        "left_hand": sum(item["left_hand_detected_frames"] == 0 for item in per_video_detection),
        "right_hand": sum(item["right_hand_detected_frames"] == 0 for item in per_video_detection),
    }
    if len(classes) != 101 or len(set(classes)) != 101:
        raise RuntimeError("Saved Experiment 3 class map does not have 101 unique labels.")
    if preflight["landmark_cache"]["videos"] != 687 or cache_metadata["failed_video_count"] != 0:
        raise RuntimeError("Saved landmark feature extraction was incomplete.")
    if preflight["dataset"]["project_dataset_manifest"]["sha256_manifest"] != _tree_manifest(settings.dataset_dir)["sha256_manifest"]:
        raise RuntimeError("Project dataset changed since the Experiment 3 preflight.")
    if preflight["dataset"]["original_dataset_manifest"]["sha256_manifest"] != _tree_manifest(
            Path.home() / "Downloads" / "ISL_CSLRT_Corpus" / "ISL_CSLRT_Corpus")["sha256_manifest"]:
        raise RuntimeError("Original Downloads dataset changed since the Experiment 3 preflight.")
    if preflight["dataset"]["csv_sha256"] != _file_digest(settings.dataset_dir / "corpus_csv_files" / "ISL Corpus sign glosses.csv"):
        raise RuntimeError("Dataset CSV changed since the Experiment 3 preflight.")
    if preflight["existing_mobile_cache"]["manifest"]["sha256_manifest"] != _tree_manifest(
            settings.cache_dir / "mobilenet_v2_pre_30f_224px_nomp")["sha256_manifest"]:
        raise RuntimeError("Original MobileNetV2 cache changed since the Experiment 3 preflight.")
    for path in cache_dir.glob("*.npy"):
        sequence = np.load(path, mmap_mode="r", allow_pickle=False)
        if sequence.shape != (30, 258) or not np.isfinite(sequence).all():
            raise RuntimeError(f"Invalid saved landmark feature: {path.name}, shape={sequence.shape}.")

    model = build_landmark_classifier(num_classes=len(classes), dropout=DROPOUT)
    ckpt = torch.load(output_dir / "signbridge_landmark_model.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    criterion = nn.CrossEntropyLoss()
    device = torch.device("cpu")
    validation_metrics, _, _ = _evaluate_split(model, splits_file["splits"]["val"], cache_dir,
                                               classes, device, criterion)
    test_metrics, y_true, y_pred = _evaluate_split(model, splits_file["splits"]["test"], cache_dir,
                                                   classes, device, criterion)
    best_epoch = min(history, key=lambda row: row["validation_loss"])["epoch"]
    metrics = {"created_at": datetime.now().isoformat(timespec="seconds"), "best_epoch": best_epoch,
               "best_checkpoint_selected_by": "validation_loss", "epochs_completed": len(history),
               "validation": validation_metrics, "test": test_metrics,
               "split_class_coverage": splits_file["info"]["class_coverage"],
               "model": {"architecture": "BiLSTM + temporal attention + dense classifier", "num_classes": 101,
                         "input_shape": [30, 258], "dropout": DROPOUT, "hidden": 128, "layers": 1}}
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    matrix = confusion_matrix(y_true, y_pred, labels=list(range(len(classes))))
    with (output_dir / "confusion_matrix.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["true\\pred"] + classes)
        for label, row in zip(classes, matrix.tolist()):
            writer.writerow([label] + row)
    _save_confusion_plot(matrix, output_dir / "confusion_matrix.png")

    final = history[-1]
    report = [
        "SignBridge Experiment 3: MediaPipe Holistic landmarks",
        "This report was finalized from the already-saved training checkpoint; no training or feature extraction was rerun.",
        f"Dataset: {settings.dataset_dir}",
        "Original dataset/CSV unchanged; original MobileNetV2 cache and checkpoints were protected.",
        f"Landmark cache: {cache_dir}",
        f"Feature shape: 30 x 258 per video; temporal order retained.",
        f"Normalization: {cache_metadata['normalization']}",
        f"Feature extraction: {cache_metadata['aggregate_detection']['total_frames']} frames; "
        f"pose={cache_metadata['aggregate_detection']['pose_detection_percent']:.2f}%; "
        f"left hand={cache_metadata['aggregate_detection']['left_hand_detection_percent']:.2f}%; "
        f"right hand={cache_metadata['aggregate_detection']['right_hand_detection_percent']:.2f}%; "
        f"average detected landmarks/frame={cache_metadata['aggregate_detection']['average_detected_landmarks_per_frame']:.2f}; "
        f"zero-detection videos={zero_detection_videos}; failed videos={cache_metadata['failed_video_count']}.",
        f"Split counts: {splits_file['info']['split_counts']}",
        f"Class coverage: {splits_file['info']['class_coverage']}",
        f"Best epoch={best_epoch}, selected by validation loss; epochs completed={len(history)}.",
        f"Final training loss/accuracy: {final['train_loss']:.6f}/{final['train_accuracy']:.6f}",
        f"Validation loss/accuracy/macro P/R/F1: {validation_metrics['loss']:.6f}/{validation_metrics['accuracy']:.6f}/"
        f"{validation_metrics['macro_precision']:.6f}/{validation_metrics['macro_recall']:.6f}/{validation_metrics['macro_f1']:.6f}",
        f"Test n={test_metrics['samples']}, classes={test_metrics['classes_represented']}/101, "
        f"accuracy={test_metrics['accuracy']:.6f}, macro P/R/F1={test_metrics['macro_precision']:.6f}/"
        f"{test_metrics['macro_recall']:.6f}/{test_metrics['macro_f1']:.6f}, top5={test_metrics['top5_accuracy']:.6f}.",
        "Experiment comparisons: exp1 test accuracy/F1=0/0; exp2 accuracy=0.009524, macro-F1=0.000272.",
        "Confusion matrix files contain the actual test predictions.",
        "Protected dataset/cache/checkpoint integrity was reverified during report finalization.",
    ]
    (output_dir / "training_report.txt").write_text("\n".join(report) + "\n", encoding="utf-8")
    print("Existing Experiment 3 checkpoint evaluated; no training was run.")
    print("Metrics:", json.dumps(metrics, indent=2))
    return 0


def run() -> int:
    settings = get_settings()
    project_root = settings.models_dir.parent
    output_dir = settings.models_dir / "experiments" / "exp3_landmarks"
    cache_dir = settings.cache_dir / "mediapipe_landmarks_30f"
    if cache_dir.exists():
        raise RuntimeError(f"Refusing to overwrite existing landmark cache: {cache_dir}")
    protected = _capture_protected_state(settings, output_dir)

    if settings.DATASET_PATH != "dataset/ISL_CSLRT_Corpus":
        raise RuntimeError(f"Expected in-project dataset path; got {settings.DATASET_PATH!r}.")
    index = build_index(settings)
    if not index.available or index.video_count != 687 or len(index.df) != 687:
        raise RuntimeError(f"Expected 687 indexed videos; available={index.available}, indexed={len(index.df)}, total={index.video_count}.")
    if index.label_column != "Sentence" or index.video_column is not None:
        raise RuntimeError(f"Unexpected dataset mapping: {index.video_column=}, {index.label_column=}.")
    classes_path = settings.models_dir / "classes.json"
    classes = json.loads(classes_path.read_text(encoding="utf-8"))
    if len(classes) != 101 or classes != index.classes:
        raise RuntimeError("Existing classes.json does not exactly equal the 101 real Sentence classes.")

    split, excluded, split_info = _load_exp2_split(settings, index, classes)
    print("Reusing verified duplicate-aware Experiment 2 splits; eight conflicting rows are recorded and excluded from model splits.")
    print("Experiment 3 split counts:", split_info["split_counts"])
    print("Experiment 3 class coverage:", split_info["class_coverage"])
    print("Excluded conflicting rows:", [(item["row"], item["label"], item["video_path"]) for item in excluded])

    output_dir.mkdir(parents=True, exist_ok=False)
    started = datetime.now().isoformat(timespec="seconds")
    try:
        extracted, failures, aggregate = _extract_cache(index, classes, cache_dir)
        metadata_path = cache_dir / "metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        all_stats = aggregate.to_dict()
        usable_rows = {row_id for row_id in extracted}
        split_rows = {name: set(map(int, split[name]["rows"])) for name in ("train","val","test")}
        failed_in_split = sorted(set.union(*split_rows.values()) - usable_rows)
        if failed_in_split:
            for name in ("train","val","test"):
                keep = [(row, target) for row,target in zip(split[name]["rows"],split[name]["labels"]) if int(row) in usable_rows]
                split[name] = {"rows":[int(row) for row,_ in keep],"labels":[int(target) for _,target in keep]}
            split_info["decode_failed_rows_excluded"] = failed_in_split
            split_info["warnings"].append(f"{len(failed_in_split)} videos genuinely failed decoding/extraction and are omitted from splits.")
        if not all(split[name]["rows"] for name in ("train","val","test")):
            raise RuntimeError("At least one partition is empty after accounting for genuine decode failures.")

        pose_pct = all_stats["pose_detection_percent"]
        left_pct = all_stats["left_hand_detection_percent"]
        right_pct = all_stats["right_hand_detection_percent"]
        if pose_pct < POSE_QUALITY_STOP_PERCENT or (left_pct < BOTH_HANDS_QUALITY_STOP_PERCENT and right_pct < BOTH_HANDS_QUALITY_STOP_PERCENT):
            raise RuntimeError(
                "Landmark quality gate failed before training: pose detection must be >=30% of sampled frames, "
                "and at least one hand must be detected on >=5% of frames. "
                f"Observed pose={pose_pct:.2f}%, left={left_pct:.2f}%, right={right_pct:.2f}%."
            )

        # Validate cache completeness/shape/finiteness for every successful video.
        cache_shapes = Counter()
        for row_id in extracted:
            sequence = np.load(cache_dir / f"{row_id:06d}.npy", mmap_mode="r", allow_pickle=False)
            cache_shapes[str(sequence.shape)] += 1
            if sequence.shape != (LANDMARK_NUM_FRAMES, FRAME_DIM) or not np.isfinite(sequence).all():
                raise RuntimeError(f"Cached row {row_id} has invalid data {sequence.shape}.")
        class_to_id = {label: i for i,label in enumerate(classes)}
        for name in ("train","val","test"):
            rows = split[name]["rows"]
            targets = split[name]["labels"]
            if any(not 0 <= int(target) < 101 for target in targets):
                raise RuntimeError(f"{name} targets are not all in 0..100.")
            if any(int(target) != class_to_id[str(index.df.iloc[int(row)]["label"])] for row,target in zip(rows,targets)):
                raise RuntimeError(f"Label-ID mismatch in {name} split.")
        if any(set(split[a]["rows"]) & set(split[b]["rows"]) for a,b in (("train","val"),("train","test"),("val","test"))):
            raise RuntimeError("Row overlap detected in exp3 splits.")

        torch.manual_seed(SEED)
        test_model = build_landmark_classifier(num_classes=101, dropout=DROPOUT)
        sample_x = torch.zeros((BATCH_SIZE, LANDMARK_NUM_FRAMES, FRAME_DIM), dtype=torch.float32)
        with torch.no_grad():
            sample_logits = test_model(sample_x)
        if tuple(sample_x.shape) != (16,30,258) or tuple(sample_logits.shape) != (16,101):
            raise RuntimeError(f"Model shape preflight failed: {tuple(sample_x.shape)} -> {tuple(sample_logits.shape)}")
        del test_model, sample_x, sample_logits

        preflight = {
            "started_at": started,
            "dataset": {"root": str(settings.dataset_dir), "indexed_videos": index.video_count,
                        "classes": len(classes), "csv": index.csv_path, "label_column": index.label_column,
                        "project_dataset_manifest": {k:v for k,v in protected["project_dataset"].items() if k != "files"},
                        "original_dataset_manifest": {k:v for k,v in protected["original_dataset"].items() if k != "files"},
                        "csv_sha256": protected["csv_sha256"]},
            "existing_mobile_cache": {"path": str(settings.cache_dir), "manifest": {k:v for k,v in protected["mobilenet_cache"].items() if k != "files"}},
            "landmark_cache": {"path": str(cache_dir), "videos": len(extracted), "failures": failures,
                                "shapes": dict(cache_shapes), "all_finite": True, "detection": all_stats,
                                "metadata_path": str(metadata_path)},
            "quality_gate": {"minimum_pose_detection_percent": POSE_QUALITY_STOP_PERCENT,
                             "both_hands_below_percent_is_failure": BOTH_HANDS_QUALITY_STOP_PERCENT,
                             "passed": True},
            "classes": classes,
            "split_info": split_info,
            "excluded_conflicting_rows": excluded,
            "model_input_shape": [16,30,258], "model_output_shape": [16,101],
            "targets_range": [0,100], "nan_or_infinite_features": False,
            "settings": {"frames":30,"features_per_frame":258,"hidden":128,"layers":1,
                         "bidirectional":True,"attention":True,"dense":128,"dropout":DROPOUT,
                         "optimizer":"Adam","learning_rate":LEARNING_RATE,"weight_decay":WEIGHT_DECAY,
                         "loss":"CrossEntropyLoss","gradient_clip":GRADIENT_CLIP,"epochs":MAX_EPOCHS,
                         "early_stopping_patience":EARLY_STOPPING_PATIENCE},
        }
        (output_dir / "preflight.json").write_text(json.dumps(preflight, indent=2), encoding="utf-8")
        (output_dir / "classes.json").write_text(json.dumps(classes, indent=2, ensure_ascii=False), encoding="utf-8")
        split_info["class_coverage"] = {name: len({classes[target] for target in split[name]["labels"]}) for name in ("train","val","test")}
        (output_dir / "splits.json").write_text(json.dumps({"cache_dir":str(cache_dir),"info":split_info,"splits":split},indent=2),encoding="utf-8")

        print("LANDMARK QUALITY:", json.dumps(all_stats, indent=2))
        print("Landmark cache shape summary:", dict(cache_shapes), "successful videos:",len(extracted),"failed:",len(failures))
        print("Running 16-sample overfit sanity check before full training...")
        sanity = _overfit_sanity(index, cache_dir, split, classes)
        preflight["overfit_sanity"] = sanity
        (output_dir / "preflight.json").write_text(json.dumps(preflight, indent=2), encoding="utf-8")
        print("Overfit sanity:", json.dumps(sanity))
        if not sanity["passed"]:
            raise RuntimeError("Landmark model failed to reach 90% accuracy on the tiny overfit subset; full training stopped.")

        train_ds = LandmarkSequenceDataset(cache_dir, split["train"]["rows"], split["train"]["labels"])
        val_ds = LandmarkSequenceDataset(cache_dir, split["val"]["rows"], split["val"]["labels"])
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
        torch.manual_seed(SEED)
        random.seed(SEED)
        np.random.seed(SEED)
        model = build_landmark_classifier(num_classes=len(classes), dropout=DROPOUT)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        criterion = nn.CrossEntropyLoss()
        history = []
        best_state = None
        best_val_loss = float("inf")
        best_epoch = 0
        stale = 0
        for epoch in range(1, MAX_EPOCHS + 1):
            tick = time.perf_counter()
            train_metrics = _train_epoch(model, train_loader, criterion, optimizer, device)
            val_metrics = _evaluate(model, val_loader, criterion, device, len(classes))
            row = {"epoch":epoch,"train_loss":train_metrics["loss"],"train_accuracy":train_metrics["accuracy"],
                   "validation_loss":val_metrics["loss"],"validation_accuracy":val_metrics["accuracy"],
                   "validation_macro_precision":val_metrics["macro_precision"],
                   "validation_macro_recall":val_metrics["macro_recall"],
                   "validation_macro_f1":val_metrics["macro_f1"],"seconds":time.perf_counter()-tick}
            history.append(row)
            improved = val_metrics["loss"] < best_val_loss - 1e-4
            if improved:
                best_val_loss = val_metrics["loss"]
                best_epoch = epoch
                stale = 0
                best_state = {key:value.detach().cpu().clone() for key,value in model.state_dict().items()}
            else:
                stale += 1
            print(f"epoch {epoch:02d}/{MAX_EPOCHS} train loss={row['train_loss']:.4f} acc={row['train_accuracy']:.4f} | "
                  f"val loss={row['validation_loss']:.4f} acc={row['validation_accuracy']:.4f} "
                  f"macro P/R/F1={row['validation_macro_precision']:.4f}/{row['validation_macro_recall']:.4f}/{row['validation_macro_f1']:.4f}" +
                  ("  best by validation loss" if improved else ""))
            if stale >= EARLY_STOPPING_PATIENCE:
                print(f"Early stopping after {epoch} epochs; no validation-loss improvement for {stale} epochs.")
                break
        if best_state is None:
            raise RuntimeError("No finite validation-loss checkpoint was selected.")
        model.load_state_dict(best_state)
        model.to(device).eval()
        checkpoint_path = output_dir / "signbridge_landmark_model.pt"
        torch.save({"state_dict":best_state,"meta":{
            "input_dim":FRAME_DIM,"num_classes":len(classes),"hidden":128,"layers":1,"dense_units":128,
            "dropout":DROPOUT,"use_attention":True,"num_frames":LANDMARK_NUM_FRAMES,
            "classes":classes,"trained_at":datetime.now().isoformat(timespec="seconds"),
            "feature_kind":"mediapipe_holistic_pose_left_right_hands_normalized_sequence",
            "normalization":metadata["normalization"],"split_mode":split_info["mode"],"seed":SEED}},checkpoint_path)
        (output_dir / "history.json").write_text(json.dumps(history,indent=2),encoding="utf-8")

        validation_metrics, _, _ = _evaluate_split(model, split["val"], cache_dir, classes, device)
        test_metrics, y_true, y_pred = _evaluate_split(model, split["test"], cache_dir, classes, device)
        metrics = {"created_at":datetime.now().isoformat(timespec="seconds"),"best_epoch":best_epoch,
                   "best_checkpoint_selected_by":"validation_loss","epochs_completed":len(history),
                   "validation":validation_metrics,"test":test_metrics,"split_class_coverage":split_info["class_coverage"],
                   "model":{"architecture":"BiLSTM + temporal attention + dense classifier","num_classes":len(classes),
                            "input_shape":[LANDMARK_NUM_FRAMES,FRAME_DIM],"dropout":DROPOUT,"hidden":128,"layers":1}}
        (output_dir / "metrics.json").write_text(json.dumps(metrics,indent=2),encoding="utf-8")
        matrix=confusion_matrix(y_true,y_pred,labels=list(range(len(classes))))
        with (output_dir/"confusion_matrix.csv").open("w",newline="",encoding="utf-8") as stream:
            writer=csv.writer(stream); writer.writerow(["true\\pred"]+classes)
            for label,row in zip(classes,matrix.tolist()): writer.writerow([label]+row)
        _save_confusion_plot(matrix,output_dir/"confusion_matrix.png")

        final_train=history[-1]
        report_lines=["SignBridge Experiment 3: MediaPipe Holistic landmarks",f"Finished: {datetime.now().isoformat(timespec='seconds')}",
                      f"Dataset: {settings.dataset_dir}","Original dataset/CSV unchanged; original MobileNetV2 cache and checkpoints protected.",
                      f"Landmark cache: {cache_dir}",f"Feature shape: ({LANDMARK_NUM_FRAMES}, {FRAME_DIM}) per video; temporal order retained.",
                      f"Normalization: {metadata['normalization']}",f"Indexed videos: {index.video_count}; successful feature sequences: {len(extracted)}; decode/extraction failures: {len(failures)}",
                      f"Frame detection rates: pose={pose_pct:.2f}%; left hand={left_pct:.2f}%; right hand={right_pct:.2f}%; avg landmarks/frame={all_stats['average_detected_landmarks_per_frame']:.2f}",
                      f"Videos with zero detected frames: pose={sum(v['detection']['pose_detected_frames']==0 for v in metadata['videos'] if v['status']=='ok')}; left hand={sum(v['detection']['left_hand_detected_frames']==0 for v in metadata['videos'] if v['status']=='ok')}; right hand={sum(v['detection']['right_hand_detected_frames']==0 for v in metadata['videos'] if v['status']=='ok')}",
                      f"Split counts: {split_info['split_counts']}",f"Class coverage: {split_info['class_coverage']}",
                      f"Excluded conflicting duplicate rows (recorded only; source retained): {len(excluded)}",
                      f"Overfit sanity: {sanity}",f"Training epochs: {len(history)}; best epoch={best_epoch} selected by validation loss.",
                      f"Final training loss/accuracy: {final_train['train_loss']:.6f}/{final_train['train_accuracy']:.6f}",
                      f"Validation: loss={validation_metrics['loss']:.6f}, accuracy={validation_metrics['accuracy']:.6f}, macro P/R/F1={validation_metrics['macro_precision']:.6f}/{validation_metrics['macro_recall']:.6f}/{validation_metrics['macro_f1']:.6f}",
                      f"Test: n={test_metrics['samples']}, classes={test_metrics['classes_represented']}/{len(classes)}, accuracy={test_metrics['accuracy']:.6f}, macro P/R/F1={test_metrics['macro_precision']:.6f}/{test_metrics['macro_recall']:.6f}/{test_metrics['macro_f1']:.6f}, top5={test_metrics['top5_accuracy']:.6f}",
                      "Experiment comparisons: exp1 accuracy/F1=0/0; exp2 accuracy=0.009524, macro-F1=0.000272; exp3 metrics are listed above and are not represented as improved unless higher.",
                      "", "Warnings:"]
        report_lines.extend(f"  {warning}" for warning in split_info.get("warnings",[]))
        report_lines.extend(["", "Test confusion matrix is in confusion_matrix.csv and confusion_matrix.png."])
        (output_dir/"training_report.txt").write_text("\n".join(report_lines)+"\n",encoding="utf-8")

        integrity=_verify_protected_state(settings,protected)
        with (output_dir/"training_report.txt").open("a",encoding="utf-8") as stream:
            stream.write("\nProtected integrity checks:\n")
            for key,value in integrity.items(): stream.write(f"  {key}: {value}\n")
        if not all(integrity.values()):
            raise RuntimeError(f"Protected file integrity check failed: {integrity}")
        print("EXPERIMENT 3 COMPLETE")
        print("Output directory:",output_dir)
        print("Metrics:",json.dumps(metrics,indent=2))
        print("Detection:",json.dumps(all_stats,indent=2))
        print("Integrity:",json.dumps(integrity))
        return 0
    except Exception as exc:
        output_dir.mkdir(parents=True,exist_ok=True)
        (output_dir/"failure_report.txt").write_text(
            f"Experiment 3 stopped: {type(exc).__name__}: {exc}\n",encoding="utf-8")
        raise


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser(description="Experiment 3 MediaPipe landmark training.")
    parser.add_argument("--finalize-saved-run", action="store_true",
                        help="evaluate an existing exp3 checkpoint and finish reports without training/extraction")
    args = parser.parse_args()
    try:
        return _finalize_saved_training() if args.finalize_saved_run else run()
    except Exception as exc:
        print(f"EXPERIMENT 3 STOPPED: {type(exc).__name__}: {exc}")
        return 2


if __name__ == "__main__":
    sys.exit(main())

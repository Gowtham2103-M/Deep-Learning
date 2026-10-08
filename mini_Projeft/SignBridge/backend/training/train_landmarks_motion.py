"""Final SignBridge Experiment 4: robust landmark normalization + motion features.

Run from backend with backend\\.venv active:
    python -m training.train_landmarks_motion

Writes only models/feature_cache/mediapipe_landmarks_motion_30f/ and
models/experiments/exp4_landmarks_motion/. The source dataset, CSV, existing
caches, original checkpoint, Experiment 2 and Experiment 3 are read-only.
"""
from __future__ import annotations

import csv
import hashlib
import json
import random
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support
from torch import nn
from torch.utils.data import DataLoader, Dataset

from app.config import get_settings
from app.models.landmark_motion_model import build_landmark_motion_classifier
from app.preprocessing.dataset_index import build_index
from app.preprocessing.frames import load_sampled_frames
from app.preprocessing.landmark_sequence_exp4 import (
    BASE_DIM, COORDINATE_CLIP, FEATURE_DIM, HAND_DIM, HAND_LANDMARKS, POSE_LANDMARKS,
    MediaPipeMotionSequenceExtractor, MIN_SHOULDER_SCALE, MOTION_DIM,
    NUM_FRAMES, POSE_DIM, POSE_LANDMARKS,
)

SEED = 42
BATCH_SIZE = 16
LEARNING_RATE = 0.0001
WEIGHT_DECAY = 0.0001
DROPOUT = 0.3
MAX_EPOCHS = 80
EARLY_STOPPING_PATIENCE = 15
GRADIENT_CLIP = 5.0
SANITY_LR = 0.001
SANITY_MAX_EPOCHS = 500
SANITY_TARGET_ACCURACY = 0.90


def _digest(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            sha.update(block)
    return sha.hexdigest()


def _manifest(root: Path) -> dict:
    records = []
    for path in sorted(root.rglob("*")):
        if path.is_file():
            stat = path.stat()
            records.append((path.relative_to(root).as_posix(), stat.st_size, stat.st_mtime_ns))
    encoded = json.dumps(records, separators=(",", ":")).encode("utf-8")
    return {"file_count": len(records), "total_bytes": sum(item[1] for item in records),
            "sha256_manifest": hashlib.sha256(encoded).hexdigest(), "files": records}


def _manifest_hash(root: Path) -> str:
    if not root.exists():
        return "missing"
    return _manifest(root)["sha256_manifest"]


def _protected_snapshot(settings) -> dict:
    project_dataset = settings.dataset_dir.resolve()
    source_dataset = Path.home() / "Downloads" / "ISL_CSLRT_Corpus" / "ISL_CSLRT_Corpus"
    exp3_dir = settings.models_dir / "experiments" / "exp3_landmarks"
    exp2_dir = settings.models_dir / "experiments" / "exp2"
    if not project_dataset.is_dir() or not source_dataset.is_dir():
        raise RuntimeError("Project and original Downloads dataset folders must exist.")
    if not exp3_dir.is_dir() or not exp2_dir.is_dir():
        raise RuntimeError("Experiment 2 and Experiment 3 artifacts must be present and protected.")
    project_manifest, source_manifest = _manifest(project_dataset), _manifest(source_dataset)
    if project_manifest["file_count"] != 21074 or project_manifest["files"] != source_manifest["files"]:
        raise RuntimeError("Dataset inventory differs from the verified unchanged source copy.")
    if project_manifest["total_bytes"] != source_manifest["total_bytes"]:
        raise RuntimeError("Project dataset byte count differs from original Downloads dataset.")
    csv_path = project_dataset / "corpus_csv_files" / "ISL Corpus sign glosses.csv"
    exp3_checkpoint = exp3_dir / "signbridge_landmark_model.pt"
    original_checkpoint = settings.models_dir / "signbridge_head.pt"
    for path in (csv_path, exp3_checkpoint, original_checkpoint):
        if not path.is_file():
            raise RuntimeError(f"Protected required file is missing: {path}")
    return {
        "project_dataset": project_manifest,
        "source_dataset": source_manifest,
        "csv_sha256": _digest(csv_path),
        "exp3_manifest": _manifest_hash(exp3_dir),
        "exp3_checkpoint_sha256": _digest(exp3_checkpoint),
        "exp2_manifest": _manifest_hash(exp2_dir),
        "original_checkpoint_sha256": _digest(original_checkpoint),
        "mobilenet_cache_manifest": _manifest_hash(settings.cache_dir / "mobilenet_v2_pre_30f_224px_nomp"),
        "exp3_cache_manifest": _manifest_hash(settings.cache_dir / "mediapipe_landmarks_30f"),
    }


def _verify_protected(settings, before: dict) -> dict:
    project = _manifest(settings.dataset_dir.resolve())
    source = _manifest(Path.home() / "Downloads" / "ISL_CSLRT_Corpus" / "ISL_CSLRT_Corpus")
    exp3 = settings.models_dir / "experiments" / "exp3_landmarks"
    exp2 = settings.models_dir / "experiments" / "exp2"
    checks = {
        "project_dataset_unchanged": project["sha256_manifest"] == before["project_dataset"]["sha256_manifest"],
        "original_dataset_unchanged": source["sha256_manifest"] == before["source_dataset"]["sha256_manifest"],
        "project_matches_source": project["files"] == source["files"],
        "csv_unchanged": _digest(settings.dataset_dir / "corpus_csv_files" / "ISL Corpus sign glosses.csv") == before["csv_sha256"],
        "exp3_unchanged": _manifest_hash(exp3) == before["exp3_manifest"],
        "exp3_checkpoint_unchanged": _digest(exp3 / "signbridge_landmark_model.pt") == before["exp3_checkpoint_sha256"],
        "exp2_unchanged": _manifest_hash(exp2) == before["exp2_manifest"],
        "original_checkpoint_unchanged": _digest(settings.models_dir / "signbridge_head.pt") == before["original_checkpoint_sha256"],
        "mobilenet_cache_unchanged": _manifest_hash(settings.cache_dir / "mobilenet_v2_pre_30f_224px_nomp") == before["mobilenet_cache_manifest"],
        "exp3_landmark_cache_unchanged": _manifest_hash(settings.cache_dir / "mediapipe_landmarks_30f") == before["exp3_cache_manifest"],
    }
    return checks


def _load_exact_exp3_split(settings, index, classes) -> tuple[dict, dict, list]:
    exp3_dir = settings.models_dir / "experiments" / "exp3_landmarks"
    exp3_preflight = json.loads((exp3_dir / "preflight.json").read_text(encoding="utf-8"))
    splits = json.loads((exp3_dir / "splits.json").read_text(encoding="utf-8"))
    if json.loads((exp3_dir / "classes.json").read_text(encoding="utf-8")) != classes:
        raise RuntimeError("Experiment 3 class map differs; refusing to remap labels.")
    exclusions = exp3_preflight["excluded_conflicting_rows"]
    excluded_ids = {int(row["row"]) for row in exclusions}
    if len(excluded_ids) != 8:
        raise RuntimeError(f"Expected exactly eight Experiment 3 excluded rows; found {len(excluded_ids)}.")
    expected_counts = {"train": 475, "val": 99, "test": 105}
    seen = set()
    class_to_id = {label: i for i, label in enumerate(classes)}
    for partition, expected_count in expected_counts.items():
        block = splits["splits"][partition]
        rows, targets = list(map(int, block["rows"])), list(map(int, block["labels"]))
        if len(rows) != expected_count or len(targets) != expected_count:
            raise RuntimeError(f"Experiment 3 {partition} split count changed: {len(rows)} (expected {expected_count}).")
        for row, target in zip(rows, targets):
            if row in excluded_ids or not 0 <= row < len(index.df):
                raise RuntimeError(f"Invalid/excluded row {row} appears in Experiment 3 {partition} split.")
            label = str(index.df.iloc[row]["label"])
            if target != class_to_id[label]:
                raise RuntimeError(f"Label ID mismatch in Experiment 3 {partition} split, row {row}.")
            if row in seen:
                raise RuntimeError(f"Split row overlap at row {row}.")
            seen.add(row)
    if len(seen) != 679 or seen != set(range(687)) - excluded_ids:
        raise RuntimeError("Experiment 3 rows do not exactly match the 679 eligible dataset rows.")
    return splits["splits"], splits["info"], exclusions


def _extract_new_cache(index, classes, cache: Path) -> tuple[dict, dict]:
    cache.mkdir(parents=True, exist_ok=True)
    metadata_path = cache / "metadata.json"
    existing_arrays = sorted(cache.glob("*.npy"))
    if metadata_path.is_file() and len(existing_arrays) == 687:
        existing_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if existing_metadata.get("experiment") != "exp4_landmarks_motion" or existing_metadata.get("shape_per_video") != [NUM_FRAMES, FEATURE_DIM]:
            raise RuntimeError("Existing motion-cache metadata has an unexpected schema; refusing to overwrite it.")
        if [path.name for path in existing_arrays] != [f"{i:06d}.npy" for i in range(687)]:
            raise RuntimeError("Existing motion cache does not contain the exact sequential 687-array index.")
        if len(existing_metadata.get("videos", [])) != 687 or existing_metadata.get("failures"):
            raise RuntimeError("Existing motion-cache metadata is incomplete or contains extraction failures.")
        for row_id, (path, row) in enumerate(zip(existing_arrays, index.df.itertuples(index=False))):
            entry = existing_metadata["videos"][row_id]
            expected_class_id = classes.index(str(row.label))
            if (entry.get("cache_index") != row_id or entry.get("video_path") != str(row.video_path)
                    or entry.get("label") != str(row.label) or entry.get("class_id") != expected_class_id):
                raise RuntimeError(f"Existing motion-cache metadata/index alignment failed at row {row_id}.")
            values = np.load(path, mmap_mode="r", allow_pickle=False)
            if values.shape != (NUM_FRAMES, FEATURE_DIM) or not np.isfinite(values).all():
                raise RuntimeError(f"Invalid existing motion-cache array {path.name}, shape={values.shape}.")
        print("Reusing and validating all 687 existing Experiment 4 motion arrays; MediaPipe extraction skipped.")
        return existing_metadata, existing_metadata["failures"]

    unexpected = [path.name for path in cache.iterdir()
                  if not (path.is_file() and path.suffix == ".npy" and path.stem.isdigit())
                  and path.name != "metadata.json"]
    if unexpected:
        raise RuntimeError(f"Unexpected files in the new Experiment 4 cache; refusing to overwrite: {unexpected}")
    extractor = MediaPipeMotionSequenceExtractor()
    meta_rows, failures = [], {}
    aggregate = {
        "frames": 0, "pose_frames": 0, "left_hand_frames": 0, "right_hand_frames": 0,
        "fallback_center_frames": 0, "fallback_scale_frames": 0,
        "clipped_coordinate_values": 0, "normalized_coordinate_values": 0,
        "left_hand_present_frames": 0, "right_hand_present_frames": 0, "both_hand_frames": 0,
    }
    class_to_id = {label: i for i, label in enumerate(classes)}
    try:
        for row_id, row in index.df.iterrows():
            row_id = int(row_id)
            path = str(row["video_path"])
            try:
                frames = load_sampled_frames(path, NUM_FRAMES)
            except (OSError, cv2.error, ValueError) as exc:
                failures[str(row_id)] = {"video_path": path, "error": f"decode: {type(exc).__name__}: {exc}"}
                continue
            extractor.reset_sequence()
            sequence, stats = extractor.extract(frames)
            if sequence.shape != (NUM_FRAMES, FEATURE_DIM) or not np.isfinite(sequence).all():
                raise RuntimeError(f"Invalid feature sequence for indexed row {row_id}: {sequence.shape}.")
            feature_path = cache / f"{row_id:06d}.npy"
            if feature_path.exists():
                saved = np.load(feature_path, allow_pickle=False)
                if saved.shape != sequence.shape or not np.allclose(saved, sequence, rtol=1e-4, atol=1e-5):
                    raise RuntimeError(f"Existing partial cache row {row_id} differs from read-only recomputation; refusing overwrite.")
            else:
                np.save(feature_path, sequence, allow_pickle=False)
            stat_data = stats.to_dict()
            for key in aggregate:
                aggregate[key] += getattr(stats, key)
            meta_rows.append({
                "cache_index": row_id,
                "video_path": path,
                "label": str(row["label"]),
                "class_id": int(class_to_id[str(row["label"])]),
                "frame_count": len(frames),
                "feature_shape": list(sequence.shape),
                "detection": stat_data,
            })
            if (row_id + 1) % 25 == 0 or row_id + 1 == len(index.df):
                print(f"Exp4 extraction {row_id + 1}/687: success={len(meta_rows)}, decode failures={len(failures)}")
    finally:
        extractor.close()
    aggregate["pose_detection_percent"] = 100.0 * aggregate["pose_frames"] / aggregate["frames"]
    aggregate["left_hand_detection_percent"] = 100.0 * aggregate["left_hand_frames"] / aggregate["frames"]
    aggregate["right_hand_detection_percent"] = 100.0 * aggregate["right_hand_frames"] / aggregate["frames"]
    aggregate["left_hand_present_percent"] = 100.0 * aggregate["left_hand_present_frames"] / aggregate["frames"]
    aggregate["right_hand_present_percent"] = 100.0 * aggregate["right_hand_present_frames"] / aggregate["frames"]
    aggregate["coordinate_clip_percent"] = 100.0 * aggregate["clipped_coordinate_values"] / max(1, aggregate["normalized_coordinate_values"])
    metadata = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "experiment": "exp4_landmarks_motion",
        "feature_dimension": FEATURE_DIM,
        "base_dimension": BASE_DIM,
        "motion_dimension": MOTION_DIM,
        "shape_per_video": [NUM_FRAMES, FEATURE_DIM],
        "layout": {
            "pose_xyz_visibility": [0, POSE_DIM],
            "left_hand_xyz": [POSE_DIM, POSE_DIM + HAND_DIM],
            "right_hand_xyz": [POSE_DIM + HAND_DIM, POSE_DIM + 2 * HAND_DIM],
            "hand_presence_left_right": [POSE_DIM + 2 * HAND_DIM, BASE_DIM],
            "motion_pose_left_right_xyz": [BASE_DIM, FEATURE_DIM],
        },
        "normalization": {
            "body_center": "pose hip midpoint landmarks 23/24 (x,y,z)",
            "scale": f"2D shoulder distance landmarks 11/12; minimum accepted scale {MIN_SHOULDER_SCALE}; otherwise previous valid scale, else {0.25}",
            "clip": [-COORDINATE_CLIP, COORDINATE_CLIP],
            "visibility": "pose visibility preserved; no hand visibility field",
            "fallback": "previous valid center/scale within each video; initial center=(0.5,0.5,0), initial scale=0.25",
            "velocity": "per-frame xyz differences; frame 0 and transitions with a missing landmark group are zero",
            "temporal_order": "preserved; no temporal pooling",
        },
        "aggregate": aggregate,
        "failures": failures,
        "videos": meta_rows,
    }
    (cache / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata, failures


class MotionDataset(Dataset):
    def __init__(self, cache: Path, rows: list[int], targets: list[int]):
        self.cache, self.rows, self.targets = cache, list(map(int, rows)), list(map(int, targets))
    def __len__(self): return len(self.rows)
    def __getitem__(self, item):
        values = np.load(self.cache / f"{self.rows[item]:06d}.npy", allow_pickle=False)
        return torch.from_numpy(values.copy()), torch.tensor(self.targets[item], dtype=torch.long)


def _class_metrics(y_true: np.ndarray, y_pred: np.ndarray, probabilities: np.ndarray, n_classes: int) -> dict:
    labels = list(range(n_classes))
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    p_each, r_each, f_each, support = precision_recall_fscore_support(y_true, y_pred, labels=labels, average=None, zero_division=0)
    topk = np.argpartition(-probabilities, kth=4, axis=1)[:, :5]
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(p), "macro_recall": float(r), "macro_f1": float(f),
        "top5_accuracy": float(np.mean([target in candidates for target, candidates in zip(y_true, topk)])),
        "samples": int(len(y_true)), "classes_represented": int(len(set(y_true.tolist()))),
        "per_class": {str(i): {"precision": float(p_each[i]), "recall": float(r_each[i]),
                               "f1": float(f_each[i]), "support": int(support[i])} for i in labels},
    }


def _score(model, loader, criterion, device, include_predictions=False):
    model.eval(); all_y=[]; all_pred=[]; all_prob=[]; total_loss=0.0; count=0
    with torch.no_grad():
        for x,y in loader:
            logits=model(x.to(device)); loss=criterion(logits,y.to(device))
            total_loss += float(loss.item())*len(y); count += len(y)
            all_y.extend(y.tolist()); all_pred.extend(logits.argmax(1).cpu().tolist())
            all_prob.extend(torch.softmax(logits,1).cpu().numpy())
    y=np.asarray(all_y,dtype=np.int64); pred=np.asarray(all_pred,dtype=np.int64); prob=np.asarray(all_prob,dtype=np.float32)
    result={"loss":total_loss/max(1,count),**_class_metrics(y,pred,prob,101)}
    if include_predictions: return result,y,pred,prob
    return result


def _sanity_check(cache: Path, train_split: dict, classes: list[str]) -> dict:
    by_class=defaultdict(list)
    for row,target in zip(train_split["rows"],train_split["labels"]): by_class[int(target)].append(int(row))
    chosen=[]
    for class_id in sorted(by_class):
        if len(by_class[class_id])>=2: chosen.extend(by_class[class_id][:2])
        if len(chosen)>=16: break
    chosen=chosen[:16]
    if len(chosen)<8: raise RuntimeError(f"Insufficient examples for overfit sanity subset: {len(chosen)}")
    row_targets={int(row):int(target) for row,target in zip(train_split["rows"],train_split["labels"])}
    ds=MotionDataset(cache,chosen,[row_targets[row] for row in chosen]); loader=DataLoader(ds,batch_size=len(ds),shuffle=False)
    torch.manual_seed(SEED+400); model=build_landmark_motion_classifier(101,DROPOUT); optimizer=torch.optim.Adam(model.parameters(),lr=SANITY_LR,weight_decay=0.0); loss_fn=nn.CrossEntropyLoss()
    x,y=next(iter(loader)); model.eval()
    with torch.no_grad(): initial=float(loss_fn(model(x),y).item())
    best_acc=0.0; last_loss=initial; best_epoch=0
    for epoch in range(1,501):
        model.train(); optimizer.zero_grad(set_to_none=True); logits=model(x); loss=loss_fn(logits,y); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),GRADIENT_CLIP); optimizer.step()
        model.eval()
        with torch.no_grad():
            logits=model(x); last_loss=float(loss_fn(logits,y).item()); acc=float((logits.argmax(1)==y).float().mean())
        if acc>best_acc: best_acc=acc; best_epoch=epoch
        if best_acc>=0.95 and last_loss < initial*0.5: break
    return {"passed":best_acc>=0.90 and last_loss<initial,"samples":len(chosen),
            "classes":len(set(row_targets[row] for row in chosen)),"initial_eval_loss":initial,
            "final_eval_loss":last_loss,"loss_decreased":last_loss<initial,
            "best_eval_accuracy":best_acc,"best_epoch":best_epoch,"epochs_run":epoch,
            "checkpoint_saved":False}


def _evaluate_experiment3(exp3_dir: Path, exp3_cache: Path, split_test: dict, classes: list[str]) -> tuple[dict,list[int],list[int],np.ndarray]:
    from app.models.landmark_model import build_landmark_classifier
    checkpoint=torch.load(exp3_dir/"signbridge_landmark_model.pt",map_location="cpu",weights_only=True)
    if checkpoint["meta"]["classes"]!=classes: raise RuntimeError("Exp3 checkpoint classes differ from exp4 class mapping.")
    model=build_landmark_classifier(101,dropout=checkpoint["meta"]["dropout"]); model.load_state_dict(checkpoint["state_dict"]); model.eval()
    rows=list(map(int,split_test["rows"])); targets=list(map(int,split_test["labels"]))
    x=torch.from_numpy(np.stack([np.load(exp3_cache/f"{r:06d}.npy",allow_pickle=False) for r in rows])).float()
    y=torch.tensor(targets,dtype=torch.long)
    with torch.no_grad(): logits=model(x); prob=torch.softmax(logits,dim=1).numpy(); pred=logits.argmax(1).numpy()
    return _class_metrics(np.asarray(targets),pred,prob,101),rows,pred.tolist(),prob


def _metrics_from_predictions(rows, targets, predictions, probabilities):
    return _class_metrics(np.asarray(targets),np.asarray(predictions),np.asarray(probabilities),101)


def _per_video_hand_summary(metadata: dict, test_rows: list[int], truth: list[int], pred: list[int], probs: np.ndarray, classes: list[str]) -> list[dict]:
    meta_by_row={int(v["cache_index"]):v for v in metadata["videos"]}; output=[]
    for i,(row,target,pred_id) in enumerate(zip(test_rows,truth,pred)):
        info=meta_by_row[row]; det=info["detection"]
        output.append({"row":row,"video_path":info["video_path"],"true_id":int(target),"true_label":classes[int(target)],
          "predicted_id":int(pred_id),"prediction":classes[int(pred_id)],"correct":int(target)==int(pred_id),
          "confidence":float(probs[i,int(pred_id)]),"pose_rate":det["pose_detection_percent"]/100.0,
          "left_rate":det["left_hand_detection_percent"]/100.0,"right_rate":det["right_hand_detection_percent"]/100.0,
          "left_missing":det["left_hand_frames"]==0,"right_missing":det["right_hand_frames"]==0,
          "both_hand_frames":det["both_hand_frames"],"fallback_scale_frames":det["fallback_scale_frames"],
          "fallback_center_frames":det["fallback_center_frames"],"clipped_coordinate_values":det["clipped_coordinate_values"]})
    return output


def _group_accuracy(records: list[dict]) -> dict:
    n=len(records); correct=sum(r["correct"] for r in records)
    return {"videos":n,"correct":correct,"accuracy":correct/n if n else None}


def _analysis(records: list[dict], classes: list[str]) -> tuple[dict,list[dict],list[dict]]:
    true=np.asarray([r["true_id"] for r in records],dtype=np.int64); pred=np.asarray([r["predicted_id"] for r in records],dtype=np.int64)
    cm=confusion_matrix(true,pred,labels=list(range(101)))
    rows=[]
    for cid,label in enumerate(classes):
        subset=[r for r in records if r["true_id"]==cid]; n=len(subset); correct=sum(r["correct"] for r in subset)
        wrong=Counter(r["prediction"] for r in subset if not r["correct"])
        tp=int(cm[cid,cid]); fp=int(cm[:,cid].sum()-tp); fn=int(cm[cid,:].sum()-tp); denom=2*tp+fp+fn
        rows.append({"class_id":cid,"sentence_label":label,"test_samples":n,"correct_predictions":correct,
           "accuracy":correct/n if n else None,"macro_f1":(2*tp/denom if denom else 0.0),
           "most_common_wrong_prediction":wrong.most_common(1)[0][0] if wrong else "",
           "most_common_wrong_count":wrong.most_common(1)[1] if wrong else 0})
    pairs=[]
    for a in range(101):
        for b in range(101):
            if a!=b and cm[a,b]: pairs.append({"true_class_id":a,"true_label":classes[a],"predicted_class_id":b,"predicted_label":classes[b],"count":int(cm[a,b])})
    pairs.sort(key=lambda p:(-p["count"],p["true_label"],p["predicted_label"]))
    distribution=Counter(classes[int(i)] for i in pred)
    error={"unique_predicted_classes":len(set(pred.tolist())),"prediction_distribution":dict(distribution),
      "zero_correct_classes":sum(r["test_samples"]>0 and r["correct_predictions"]==0 for r in rows),
      "classes_with_test_samples":sum(r["test_samples"]>0 for r in rows),"per_class":rows,"confusion_pairs":pairs,
      "top_20_confusion_pairs":pairs[:20]}
    return error,rows,pairs


def _plot_confusion(matrix: np.ndarray,path: Path,title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(9,8)); image=ax.imshow(matrix,cmap="Blues",interpolation="nearest")
    ax.set_title(title); ax.set_xlabel("Predicted class ID"); ax.set_ylabel("True class ID"); fig.colorbar(image,ax=ax); fig.tight_layout(); fig.savefig(path,dpi=140); plt.close(fig)


def _save_analysis(out_dir: Path, records: list[dict], classes: list[str], exp3_records: list[dict], motion_stats: dict) -> tuple[dict,dict]:
    analysis_dir=out_dir/"analysis"; analysis_dir.mkdir(parents=True,exist_ok=False)
    error,per_class,pairs=_analysis(records,classes)
    (analysis_dir/"per_class_results.csv").write_text("",encoding="utf-8")
    with (analysis_dir/"per_class_results.csv").open("w",newline="",encoding="utf-8") as stream:
        w=csv.DictWriter(stream,fieldnames=list(per_class[0]));w.writeheader();w.writerows(per_class)
    with (analysis_dir/"confusion_pairs.csv").open("w",newline="",encoding="utf-8") as stream:
        w=csv.DictWriter(stream,fieldnames=["rank","true_class_id","true_label","predicted_class_id","predicted_label","count"]);w.writeheader()
        for rank,row in enumerate(pairs[:20],1):w.writerow({"rank":rank,**row})
    hand_groups={
      "left_hand_present":_group_accuracy([r for r in records if not r["left_missing"]]),
      "left_hand_absent":_group_accuracy([r for r in records if r["left_missing"]]),
      "both_hands_present_same_frame":_group_accuracy([r for r in records if r["both_hand_frames"]>0]),
      "only_one_hand_present":_group_accuracy([r for r in records if r["both_hand_frames"]==0 and (r["left_rate"]>0 or r["right_rate"]>0)]),
      "neither_hand_present":_group_accuracy([r for r in records if r["left_rate"]==0 and r["right_rate"]==0]),
    }
    exp3_groups={
      "left_hand_present":_group_accuracy([r for r in exp3_records if not r["left_missing"]]),
      "left_hand_absent":_group_accuracy([r for r in exp3_records if r["left_missing"]]),
      "both_hands_present_same_frame":_group_accuracy([r for r in exp3_records if r["both_hand_frames"]>0]),
      "only_one_hand_present":_group_accuracy([r for r in exp3_records if r["both_hand_frames"]==0 and (r["left_rate"]>0 or r["right_rate"]>0)]),
      "neither_hand_present":_group_accuracy([r for r in exp3_records if r["left_rate"]==0 and r["right_rate"]==0]),
    }
    error["hand_detection_groups"]=hand_groups
    error["normalization_statistics"]=motion_stats
    error["left_right_separation"]="left block precedes right block and both presence flags are separate; verified from cache layout."
    (analysis_dir/"error_analysis.json").write_text(json.dumps(error,indent=2),encoding="utf-8")
    lines=["Experiment 4 landmark+motion error analysis",f"Test videos: {len(records)}",f"Unique predicted classes: {error['unique_predicted_classes']}",f"Zero-correct classes: {error['zero_correct_classes']} / {error['classes_with_test_samples']}","", "Hand-detection groups:"]
    for k,v in hand_groups.items(): lines.append(f"  {k}: {v}")
    lines.extend(["", "Normalization statistics:",json.dumps(motion_stats,indent=2),"", "Top confusion pairs:"])
    for i,p in enumerate(pairs[:20],1): lines.append(f"  {i}. {p['true_label']} -> {p['predicted_label']}: {p['count']}")
    (analysis_dir/"error_analysis.txt").write_text("\n".join(lines)+"\n",encoding="utf-8")
    return error,{"exp4":hand_groups,"exp3":exp3_groups}


def run() -> int:
    settings=get_settings(); models=settings.models_dir
    output=models/"experiments"/"exp4_landmarks_motion"
    cache=settings.cache_dir/"mediapipe_landmarks_motion_30f"
    if output.exists():
        existing_names={path.name for path in output.iterdir()}
        forbidden={"signbridge_landmark_motion.pt","history.json","metrics.json","comparison_with_exp3.json","comparison_with_exp3.txt"}
        if "failure_report.txt" not in existing_names or existing_names & forbidden:
            raise RuntimeError(f"Refusing to overwrite existing Experiment 4 output: {output}")
        print("Resuming the previously stopped, incomplete Experiment 4 setup; existing files will be preserved.")
    if settings.DATASET_PATH!="dataset/ISL_CSLRT_Corpus": raise RuntimeError(f"Unexpected dataset path {settings.DATASET_PATH!r}")
    if not (models/"experiments"/"exp3_landmarks").is_dir(): raise RuntimeError("Experiment 3 artifacts are required.")
    before=_protected_snapshot(settings)
    index=build_index(settings)
    if not index.available or index.video_count!=687 or len(index.df)!=687: raise RuntimeError("Expected exactly 687 indexed videos.")
    if index.label_column!="Sentence" or index.video_column is not None: raise RuntimeError("Dataset mapping is no longer the verified Sentence-folder mapping.")
    classes=json.loads((models/"experiments"/"exp3_landmarks"/"classes.json").read_text(encoding="utf-8"))
    if len(classes)!=101 or classes!=index.classes: raise RuntimeError("Experiment 3 class map differs from indexed Sentence labels.")
    exp3_splits=json.loads((models/"experiments"/"exp3_landmarks"/"splits.json").read_text(encoding="utf-8"))
    exp3_preflight=json.loads((models/"experiments"/"exp3_landmarks"/"preflight.json").read_text(encoding="utf-8"))
    excluded=exp3_preflight["excluded_conflicting_rows"]; excluded_ids={int(item["row"]) for item in excluded}
    if len(excluded_ids)!=8: raise RuntimeError(f"Expected the existing 8 exclusions, got {len(excluded_ids)}.")
    split=exp3_splits["splits"]
    expected={"train":475,"val":99,"test":105}
    joined=set()
    for part,n in expected.items():
        rows=list(map(int,split[part]["rows"]));targets=list(map(int,split[part]["labels"]))
        if len(rows)!=n or len(targets)!=n: raise RuntimeError(f"Exp3 {part} count changed: {len(rows)} (wanted {n}).")
        for row,target in zip(rows,targets):
            if row in excluded_ids or target!=classes.index(str(index.df.iloc[row]["label"])): raise RuntimeError(f"Exp3 split label/exclusion mismatch at row {row}.")
            if row in joined: raise RuntimeError(f"Exp3 row overlap at {row}.")
            joined.add(row)
    if joined!=set(range(687))-excluded_ids: raise RuntimeError("Exp3 split coverage differs from eligible rows.")

    output.mkdir(parents=True,exist_ok=True)
    try:
        metadata,failures=_extract_new_cache(index,classes,cache)
        if len(metadata["videos"])!=687 or failures: raise RuntimeError(f"Extraction failures: {failures}")
        arrays=sorted(cache.glob("*.npy"))
        if len(arrays)!=687 or [p.name for p in arrays]!=[f"{i:06d}.npy" for i in range(687)]: raise RuntimeError("New motion cache must contain exactly sequential 687 feature arrays.")
        ranges={"pose_xyz":[float("inf"),float("-inf")],"pose_visibility":[float("inf"),float("-inf")],"left_hand_xyz":[float("inf"),float("-inf")],"right_hand_xyz":[float("inf"),float("-inf")],"motion_xyz":[float("inf"),float("-inf")]}
        for path in arrays:
            x=np.load(path,mmap_mode="r",allow_pickle=False)
            if x.shape!=(NUM_FRAMES,FEATURE_DIM) or not np.isfinite(x).all(): raise RuntimeError(f"Invalid motion feature {path.name}: shape={x.shape}")
            pose=x[:,:POSE_DIM].reshape(NUM_FRAMES,POSE_LANDMARKS,4)
            blocks={"pose_xyz":pose[:,:,:3],"pose_visibility":pose[:,:,3],"left_hand_xyz":x[:,POSE_DIM:POSE_DIM+HAND_DIM],"right_hand_xyz":x[:,POSE_DIM+HAND_DIM:BASE_DIM-2],"motion_xyz":x[:,BASE_DIM:]}
            for name,block in blocks.items(): ranges[name][0]=min(ranges[name][0],float(block.min()));ranges[name][1]=max(ranges[name][1],float(block.max()))
        if ranges["pose_xyz"][0]<-COORDINATE_CLIP or ranges["pose_xyz"][1]>COORDINATE_CLIP or ranges["left_hand_xyz"][0]<-COORDINATE_CLIP or ranges["left_hand_xyz"][1]>COORDINATE_CLIP or ranges["right_hand_xyz"][0]<-COORDINATE_CLIP or ranges["right_hand_xyz"][1]>COORDINATE_CLIP:
            raise RuntimeError(f"Normalized position coordinates exceeded clip range: {ranges}")
        splits={part:{"rows":list(map(int,split[part]["rows"])),"labels":list(map(int,split[part]["labels"]))} for part in expected}
        coverage={part:len({classes[target] for target in splits[part]["labels"]}) for part in expected}
        split_info={"source":"models/experiments/exp3_landmarks/splits.json","mode":exp3_splits["info"]["mode"],"seed":exp3_splits["info"]["seed"],"counts":expected,"class_coverage":coverage,"classes_without_training_examples":sorted(set(classes)-{classes[target] for target in splits["train"]["labels"]}),"excluded_conflicting_rows":excluded,"warning":"No signer column exists; split is not signer-independent."}
        preflight={"created_at":datetime.now().isoformat(timespec="seconds"),"dataset_videos":687,"classes":len(classes),"csv_sha256":before["csv_sha256"],"split":split_info,"feature_cache":str(cache),"feature_shape":[NUM_FRAMES,FEATURE_DIM],"cache_files":len(arrays),"all_finite":True,"normalized_ranges":ranges,"normalization":{"minimum_shoulder_scale":MIN_SHOULDER_SCALE,"fallback_scale":0.25,"coordinate_clip":[-COORDINATE_CLIP,COORDINATE_CLIP],"fallback_frames":metadata["aggregate"]["fallback_center_frames"],"fallback_scale_frames":metadata["aggregate"]["fallback_scale_frames"],"clipped_coordinate_values":metadata["aggregate"]["clipped_coordinate_values"],"clip_percent":metadata["aggregate"]["coordinate_clip_percent"]},"metadata_path":str(cache/"metadata.json"),"settings":{"epochs":MAX_EPOCHS,"patience":EARLY_STOPPING_PATIENCE,"batch_size":BATCH_SIZE,"learning_rate":LEARNING_RATE,"weight_decay":WEIGHT_DECAY,"dropout":DROPOUT,"gradient_clip":GRADIENT_CLIP}}
        (output/"preflight.json").write_text(json.dumps(preflight,indent=2),encoding="utf-8")
        (output/"classes.json").write_text(json.dumps(classes,indent=2,ensure_ascii=False),encoding="utf-8")
        (output/"splits.json").write_text(json.dumps({"cache_dir":str(cache),"info":split_info,"splits":splits},indent=2),encoding="utf-8")
        print("Exp4 cache complete:",len(arrays),"shape",(NUM_FRAMES,FEATURE_DIM),"stats",metadata["aggregate"])
        print("Feature ranges:",ranges)

        # Required small overfit sanity check before full training.
        by_class=defaultdict(list)
        for row,target in zip(splits["train"]["rows"],splits["train"]["labels"]): by_class[int(target)].append(int(row))
        sanity_rows=[]
        for target in sorted(by_class):
            if len(by_class[target])>=2: sanity_rows.extend(by_class[target][:2])
            if len(sanity_rows)>=16: break
        sanity_rows=sanity_rows[:16]; row_to_target={int(r):int(t) for r,t in zip(splits["train"]["rows"],splits["train"]["labels"])}
        sanity_ds=MotionDataset(cache,sanity_rows,[row_to_target[row] for row in sanity_rows]); sx=torch.stack([sanity_ds[i][0] for i in range(len(sanity_ds))]);sy=torch.stack([sanity_ds[i][1] for i in range(len(sanity_ds))])
        torch.manual_seed(SEED+404); sanity_model=build_landmark_motion_classifier(101,DROPOUT); sanity_optimizer=torch.optim.Adam(sanity_model.parameters(),lr=SANITY_LR,weight_decay=0.0);loss_fn=nn.CrossEntropyLoss();sanity_model.eval()
        with torch.no_grad(): sanity_initial=float(loss_fn(sanity_model(sx),sy))
        sanity_best_acc=0.0;sanity_best_loss=sanity_initial;sanity_best_epoch=0
        for epoch in range(1,501):
            sanity_model.train();sanity_optimizer.zero_grad(set_to_none=True);loss=loss_fn(sanity_model(sx),sy);loss.backward();nn.utils.clip_grad_norm_(sanity_model.parameters(),GRADIENT_CLIP);sanity_optimizer.step();sanity_model.eval()
            with torch.no_grad(): z=sanity_model(sx);current_loss=float(loss_fn(z,sy));acc=float((z.argmax(1)==sy).float().mean())
            if acc>sanity_best_acc or (acc==sanity_best_acc and current_loss<sanity_best_loss):sanity_best_acc,sanity_best_loss,sanity_best_epoch=acc,current_loss,epoch
            if sanity_best_acc>=0.95 and sanity_best_loss<sanity_initial*0.5: break
        sanity={"samples":len(sanity_rows),"classes":len(set(row_to_target[row] for row in sanity_rows)),"initial_loss":sanity_initial,"best_loss":sanity_best_loss,"loss_decreased":sanity_best_loss<sanity_initial,"best_accuracy":sanity_best_acc,"best_epoch":sanity_best_epoch,"epochs_run":epoch,"passed":sanity_best_acc>=0.90 and sanity_best_loss<sanity_initial,"checkpoint_saved":False}
        preflight["overfit_sanity"]=sanity;(output/"preflight.json").write_text(json.dumps(preflight,indent=2),encoding="utf-8")
        print("Sanity result:",sanity)
        if not sanity["passed"]: raise RuntimeError("Motion model sanity check failed; full training stopped.")
        # Tensor/output preflight.
        shape_model=build_landmark_motion_classifier(101,DROPOUT)
        with torch.no_grad():shape_out=shape_model(torch.zeros((BATCH_SIZE,NUM_FRAMES,FEATURE_DIM)))
        if tuple(shape_out.shape)!=(16,101):raise RuntimeError(f"Model output shape is {tuple(shape_out.shape)} instead of (16,101).")

        train_ds=MotionDataset(cache,splits["train"]["rows"],splits["train"]["labels"]);val_ds=MotionDataset(cache,splits["val"]["rows"],splits["val"]["labels"])
        train_loader=DataLoader(train_ds,batch_size=BATCH_SIZE,shuffle=True);val_loader=DataLoader(val_ds,batch_size=BATCH_SIZE,shuffle=False)
        torch.manual_seed(SEED);random.seed(SEED);np.random.seed(SEED)
        model=build_landmark_motion_classifier(101,DROPOUT);device=torch.device("cuda" if torch.cuda.is_available() else "cpu");model.to(device)
        optimizer=torch.optim.Adam(model.parameters(),lr=LEARNING_RATE,weight_decay=WEIGHT_DECAY);criterion=nn.CrossEntropyLoss()
        history=[];best_loss=float("inf");best_state=None;best_epoch=0;stale=0
        for epoch in range(1,MAX_EPOCHS+1):
            start=time.perf_counter();model.train();train_loss_sum=0.0;train_true=[];train_pred=[]
            for x,y in train_loader:
                x=x.to(device);y=y.to(device);logits=model(x);loss=criterion(logits,y);optimizer.zero_grad(set_to_none=True);loss.backward();nn.utils.clip_grad_norm_(model.parameters(),GRADIENT_CLIP);optimizer.step();train_loss_sum+=float(loss.detach())*len(y);train_true.extend(y.detach().cpu().tolist());train_pred.extend(logits.detach().argmax(1).cpu().tolist())
            train_loss=train_loss_sum/len(train_ds);train_acc=float(accuracy_score(train_true,train_pred))
            model.eval();val_y=[];val_pred=[];val_prob=[];val_loss_sum=0.0
            with torch.no_grad():
                for x,y in val_loader:
                    z=model(x.to(device));ydev=y.to(device);vl=criterion(z,ydev);val_loss_sum+=float(vl)*len(y);val_y.extend(y.tolist());val_pred.extend(z.argmax(1).cpu().tolist());val_prob.extend(torch.softmax(z,1).cpu().numpy())
            vy=np.asarray(val_y);vp=np.asarray(val_pred);vprob=np.asarray(val_prob);p,r,f,_=precision_recall_fscore_support(vy,vp,labels=list(range(101)),average="macro",zero_division=0)
            val_loss=val_loss_sum/len(val_ds); row={"epoch":epoch,"train_loss":train_loss,"train_accuracy":train_acc,"validation_loss":val_loss,"validation_accuracy":float(accuracy_score(vy,vp)),"validation_macro_precision":float(p),"validation_macro_recall":float(r),"validation_macro_f1":float(f),"learning_rate":optimizer.param_groups[0]["lr"],"seconds":time.perf_counter()-start};history.append(row)
            improved=val_loss<best_loss-1e-4
            if improved:best_loss=val_loss;best_epoch=epoch;stale=0;best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            else:stale+=1
            print(f"epoch {epoch:02d}/{MAX_EPOCHS} train loss={train_loss:.4f} acc={train_acc:.4f} | val loss={val_loss:.4f} acc={row['validation_accuracy']:.4f} macroF1={row['validation_macro_f1']:.4f} lr={row['learning_rate']:.6f}"+(" BEST" if improved else ""))
            if stale>=EARLY_STOPPING_PATIENCE:print("Early stopping: patience reached.");break
        if best_state is None:raise RuntimeError("No best checkpoint state created.")
        model.load_state_dict(best_state);model.to(device).eval();checkpoint_path=output/"signbridge_landmark_motion.pt"
        torch.save({"state_dict":best_state,"meta":{"input_dim":FEATURE_DIM,"num_classes":101,"hidden":128,"layers":1,"dense_units":128,"dropout":DROPOUT,"use_attention":True,"num_frames":NUM_FRAMES,"classes":classes,"trained_at":datetime.now().isoformat(timespec="seconds"),"feature_kind":"normalized_pose_hands_presence_velocity","split_source":"Experiment 3 exact duplicate-aware split"}},checkpoint_path)
        (output/"history.json").write_text(json.dumps(history,indent=2),encoding="utf-8")

        def predict_split(block):
            ds=MotionDataset(cache,block["rows"],block["labels"]);loader=DataLoader(ds,batch_size=BATCH_SIZE,shuffle=False);yy=[];pp=[];pr=[];loss_sum=0.0
            with torch.no_grad():
                for x,y in loader:z=model(x.to(device));yd=y.to(device);loss_sum+=float(criterion(z,yd))*len(y);yy.extend(y.tolist());pp.extend(z.argmax(1).cpu().tolist());pr.extend(torch.softmax(z,1).cpu().numpy())
            yy=np.asarray(yy);pp=np.asarray(pp);pr=np.asarray(pr);p,r,f,_=precision_recall_fscore_support(yy,pp,labels=list(range(101)),average="macro",zero_division=0);each=precision_recall_fscore_support(yy,pp,labels=list(range(101)),average=None,zero_division=0);support=np.bincount(yy,minlength=101);top=np.argpartition(-pr,kth=4,axis=1)[:,:5]
            return {"loss":loss_sum/len(ds),"accuracy":float(accuracy_score(yy,pp)),"macro_precision":float(p),"macro_recall":float(r),"macro_f1":float(f),"top5_accuracy":float(np.mean([a in b for a,b in zip(yy,top)])),"samples":len(yy),"classes_represented":len(set(yy.tolist())),"per_class":{str(i):{"precision":float(each[0][i]),"recall":float(each[1][i]),"f1":float(each[2][i]),"support":int(support[i])} for i in range(101)},"y":yy,"pred":pp,"prob":pr}
        val_result=predict_split(splits["val"]);test_result=predict_split(splits["test"])
        metrics={"best_epoch":best_epoch,"best_checkpoint_selected_by":"validation_loss","epochs_completed":len(history),"validation":{k:v for k,v in val_result.items() if k not in ("y","pred","prob")},"test":{k:v for k,v in test_result.items() if k not in ("y","pred","prob")},"split_class_coverage":coverage,"model":{"architecture":"BiLSTM + temporal attention + dense classifier","input_shape":[NUM_FRAMES,FEATURE_DIM],"num_classes":101,"hidden":128,"layers":1,"dropout":DROPOUT}}
        (output/"metrics.json").write_text(json.dumps(metrics,indent=2),encoding="utf-8")
        cm=confusion_matrix(test_result["y"],test_result["pred"],labels=list(range(101)))
        with (output/"confusion_matrix.csv").open("w",newline="",encoding="utf-8") as f:
            w=csv.writer(f);w.writerow(["true\\pred"]+classes)
            for label,row in zip(classes,cm.tolist()):w.writerow([label]+row)
        import matplotlib;matplotlib.use("Agg");import matplotlib.pyplot as plt
        fig,ax=plt.subplots(figsize=(9,8));im=ax.imshow(cm,cmap="Blues",interpolation="nearest");ax.set_title("Experiment 4 normalized landmark + motion confusion matrix");ax.set_xlabel("Predicted class ID");ax.set_ylabel("True class ID");fig.colorbar(im,ax=ax);fig.tight_layout();fig.savefig(output/"confusion_matrix.png",dpi=140);plt.close(fig)
        exp3_metrics=json.loads((models/"experiments"/"exp3_landmarks"/"metrics.json").read_text())
        exp3_metric=exp3_metrics["test"]
        exp3_analysis=json.loads((models/"experiments"/"exp3_landmarks"/"analysis"/"error_analysis.json").read_text())
        exp3_summary={"accuracy":exp3_metric["accuracy"],"macro_precision":exp3_metric["macro_precision"],"macro_recall":exp3_metric["macro_recall"],"macro_f1":exp3_metric["macro_f1"],"top5_accuracy":exp3_metric["top5_accuracy"],"zero_correct_classes":exp3_analysis["zero_correct_classes"],"unique_predicted_classes":exp3_analysis["unique_predicted_classes"]}
        # Re-evaluate Exp3 against this exact test-row assignment for a strict paired comparison.
        from app.models.landmark_model import build_landmark_classifier
        e3ck=torch.load(models/"experiments"/"exp3_landmarks"/"signbridge_landmark_model.pt",map_location="cpu",weights_only=True);e3model=build_landmark_classifier(101,e3ck["meta"]["dropout"]);e3model.load_state_dict(e3ck["state_dict"]);e3model.eval()
        e3cache=settings.cache_dir/"mediapipe_landmarks_30f"; test_rows=splits["test"]["rows"]; actual=splits["test"]["labels"]
        e3x=torch.from_numpy(np.stack([np.load(e3cache/f"{row:06d}.npy",allow_pickle=False) for row in test_rows])).float()
        with torch.no_grad():e3log=e3model(e3x);e3prob=torch.softmax(e3log,1).numpy();e3pred=e3log.argmax(1).numpy()
        e3p,e3r,e3f,_=precision_recall_fscore_support(actual,e3pred,labels=list(range(101)),average="macro",zero_division=0); e3cm=confusion_matrix(actual,e3pred,labels=list(range(101))); e3_distribution=Counter(classes[int(i)] for i in e3pred);e3_zero=sum(e3cm[c].sum()>0 and e3cm[c,c]==0 for c in range(101));e3_top=np.argpartition(-e3prob,kth=4,axis=1)[:,:5];e3_top5=float(np.mean([t in row for t,row in zip(actual,e3_top)]));exp3_exact={"accuracy":float(accuracy_score(actual,e3pred)),"macro_precision":float(e3p),"macro_recall":float(e3r),"macro_f1":float(e3f),"top5_accuracy":e3_top5,"zero_correct_classes":int(e3_zero),"unique_predicted_classes":len(set(e3pred.tolist()))}
        motion_records=[]
        meta_rows={int(v["cache_index"]):v for v in metadata["videos"]}
        for j,(row,target,pred) in enumerate(zip(test_rows,actual,test_result["pred"])):
            d=meta_rows[int(row)]["detection"]; motion_records.append({"row":int(row),"true_id":int(target),"prediction_id":int(pred),"correct":int(target)==int(pred),"left_rate":d["left_hand_detection_percent"]/100,"right_rate":d["right_hand_detection_percent"]/100,"both":d["both_hand_frames"],"left_missing":d["left_hand_present_frames"]==0,"right_missing":d["right_hand_present_frames"]==0})
        exp4_groups={"left_hand_present":_group_accuracy([r for r in motion_records if not r["left_missing"]]),"left_hand_absent":_group_accuracy([r for r in motion_records if r["left_missing"]]),"both_hands_present":_group_accuracy([r for r in motion_records if r["both"]>0]),"one_hand_only":_group_accuracy([r for r in motion_records if r["both"]==0 and (r["left_rate"]>0 or r["right_rate"]>0)])}
        exp3_saved_groups=exp3_analysis["hand_detection_groups"]
        exp3_groups={
          "left_hand_present":exp3_saved_groups["left_hand_detected_any_frame"],
          "left_hand_absent":exp3_saved_groups["left_hand_zero_frames"],
          "both_hands_present":exp3_saved_groups["both_hands_detected_same_frame"],
          "one_hand_only":exp3_saved_groups["only_one_hand_detected_no_both_hand_frames"],
        }
        exp4_error_summary,_,_=_analysis_from_predictions(actual,test_result["pred"],classes)
        comparison={"same_test_rows":test_rows==json.loads((models/"experiments"/"exp3_landmarks"/"splits.json").read_text())["splits"]["test"]["rows"],"test_samples":len(test_rows),"experiment_3":exp3_exact,"experiment_4":{**{k:v for k,v in metrics["test"].items() if k not in ("per_class",)},"zero_correct_classes":exp4_error_summary["zero_correct_classes"],"unique_predicted_classes":len(set(test_result["pred"].tolist()))},"delta_exp4_minus_exp3":{},"hand_detection_accuracy":{"experiment_3":{k:exp3_groups[k] for k in ("left_hand_present","left_hand_absent","both_hands_present","one_hand_only")},"experiment_4":exp4_groups}}
        for key in ("accuracy","macro_precision","macro_recall","macro_f1","top5_accuracy","zero_correct_classes","unique_predicted_classes"):
            comparison["delta_exp4_minus_exp3"][key]=comparison["experiment_4"][key]-comparison["experiment_3"][key]
        (output/"comparison_with_exp3.json").write_text(json.dumps(comparison,indent=2),encoding="utf-8")
        comparison_text=["Experiment 3 vs Experiment 4 (identical saved test rows)",f"Same rows: {comparison['same_test_rows']}; samples: {comparison['test_samples']}","Metric | Exp3 | Exp4 | delta"]
        for key in ("accuracy","macro_precision","macro_recall","macro_f1","top5_accuracy","zero_correct_classes","unique_predicted_classes"):comparison_text.append(f"{key} | {comparison['experiment_3'][key]} | {comparison['experiment_4'][key]} | {comparison['delta_exp4_minus_exp3'][key]}")
        comparison_text.extend(["","Hand grouping:",json.dumps(comparison["hand_detection_accuracy"],indent=2)])
        (output/"comparison_with_exp3.txt").write_text("\n".join(comparison_text)+"\n",encoding="utf-8")
        error,perclass,pairs=_analysis_from_predictions(actual,test_result["pred"],classes)
        analysis=output/"analysis";analysis.mkdir()
        with (analysis/"per_class_results.csv").open("w",newline="",encoding="utf-8") as f:w=csv.DictWriter(f,fieldnames=list(perclass[0]));w.writeheader();w.writerows(perclass)
        with (analysis/"confusion_pairs.csv").open("w",newline="",encoding="utf-8") as f:
            w=csv.DictWriter(f,fieldnames=["rank","true_class_id","true_label","predicted_class_id","predicted_label","count"]);w.writeheader()
            for rank,item in enumerate(pairs[:20],1):w.writerow({"rank":rank,**item})
        error["hand_detection_accuracy"]=exp4_groups;error["normalization_statistics"]=preflight["normalization"];(analysis/"error_analysis.json").write_text(json.dumps(error,indent=2),encoding="utf-8")
        (analysis/"error_analysis.txt").write_text("Experiment 4 class/confusion analysis\n"+json.dumps({k:error[k] for k in ("unique_predicted_classes","prediction_distribution","zero_correct_classes","top_20_confusion_pairs","hand_detection_accuracy","normalization_statistics")},indent=2)+"\n",encoding="utf-8")

        exp3_hand=json.loads((models/"experiments"/"exp3_landmarks"/"analysis"/"error_analysis.json").read_text())["hand_detection_groups"]
        report=["SignBridge Experiment 4: robust normalized landmarks plus motion",f"Completed: {datetime.now().isoformat(timespec='seconds')}","Dataset and CSV read-only; experiment outputs isolated.",f"Features: {NUM_FRAMES} x {FEATURE_DIM}; base={BASE_DIM}; motion={MOTION_DIM}; temporal order preserved.",f"Cache: {cache}; arrays={len(arrays)}; shape per video=({NUM_FRAMES},{FEATURE_DIM})",f"MediaPipe detection/normalization stats: {metadata['aggregate']}",f"Split (exact Experiment 3): {expected}; class coverage={coverage}; excluded conflicting rows={len(excluded)}",f"Sanity: {sanity}",f"Training: epochs={len(history)}, best epoch={best_epoch}, train final loss/accuracy={history[-1]['train_loss']:.6f}/{history[-1]['train_accuracy']:.6f}",f"Validation: loss/acc/P/R/F1={val_result['loss']:.6f}/{val_result['accuracy']:.6f}/{val_result['macro_precision']:.6f}/{val_result['macro_recall']:.6f}/{val_result['macro_f1']:.6f}",f"Test: n={metrics['test']['samples']}, acc/P/R/F1/top5={metrics['test']['accuracy']:.6f}/{metrics['test']['macro_precision']:.6f}/{metrics['test']['macro_recall']:.6f}/{metrics['test']['macro_f1']:.6f}/{metrics['test']['top5_accuracy']:.6f}",f"Exp3 same-test-row metrics: {exp3_exact}",f"Exp4 minus Exp3: {comparison['delta_exp4_minus_exp3']}",f"Hand groups exp3/exp4: {comparison['hand_detection_accuracy']}","Recommendation selection will be made only from these measurements."]
        (output/"training_report.txt").write_text("\n".join(report)+"\n",encoding="utf-8")
        checks=_verify_protected(settings,before)
        if not all(checks.values()):raise RuntimeError(f"Protected integrity checks failed: {checks}")
        with (output/"training_report.txt").open("a",encoding="utf-8") as f:f.write("\nProtected integrity: "+json.dumps(checks)+"\n")
        print("Exp4 metrics:",json.dumps(metrics["test"],indent=2));print("Exp3 same-test metrics:",exp3_exact);print("Comparison:",comparison["delta_exp4_minus_exp3"]);print("Integrity:",checks)
        return 0
    except Exception as exc:
        # Save failure information only in the new exp4 output folder.
        output.mkdir(parents=True,exist_ok=True)
        (output/"failure_report.txt").write_text(f"Experiment 4 stopped: {type(exc).__name__}: {exc}\n",encoding="utf-8")
        raise


def _analysis_from_predictions(y_true,y_pred,classes):
    cm=confusion_matrix(y_true,y_pred,labels=list(range(101)));p,r,f,support=precision_recall_fscore_support(y_true,y_pred,labels=list(range(101)),average=None,zero_division=0);rows=[]
    for cid,label in enumerate(classes):
        wrong=Counter(classes[int(v)] for t,v in zip(y_true,y_pred) if t==cid and v!=cid);n=int(support[cid]);correct=int(cm[cid,cid]);rows.append({"class_id":cid,"sentence_label":label,"test_samples":n,"correct_predictions":correct,"accuracy":correct/n if n else None,"macro_f1":float(f[cid]),"most_common_wrong_prediction":wrong.most_common(1)[0][0] if wrong else "","most_common_wrong_count":wrong.most_common(1)[0][1] if wrong else 0})
    pairs=[]
    for a in range(101):
        for b in range(101):
            if a!=b and cm[a,b]:pairs.append({"true_class_id":a,"true_label":classes[a],"predicted_class_id":b,"predicted_label":classes[b],"count":int(cm[a,b])})
    pairs.sort(key=lambda v:(-v["count"],v["true_label"],v["predicted_label"]))
    dist=Counter(classes[int(v)] for v in y_pred);err={"unique_predicted_classes":len(dist),"prediction_distribution":dict(dist),"zero_correct_classes":sum(row["test_samples"]>0 and row["correct_predictions"]==0 for row in rows),"classes_with_test_samples":sum(row["test_samples"]>0 for row in rows),"per_class":rows,"top_20_confusion_pairs":pairs[:20]}
    return err,rows,pairs


def _finalize_saved_run() -> int:
        """Complete reports from a saved Experiment 4 checkpoint; never extracts or trains."""
        settings=get_settings(); models=settings.models_dir
        output=models/"experiments"/"exp4_landmarks_motion"
        cache=settings.cache_dir/"mediapipe_landmarks_motion_30f"
        required=[output/"preflight.json",output/"splits.json",output/"classes.json",
                            output/"history.json",output/"metrics.json",output/"signbridge_landmark_motion.pt",
                            cache/"metadata.json"]
        missing=[str(path) for path in required if not path.is_file()]
        if missing: raise RuntimeError(f"Cannot finalize Experiment 4; missing saved artifacts: {missing}")
        analysis_dir=output/"analysis"

        preflight=json.loads((output/"preflight.json").read_text(encoding="utf-8"))
        split_data=json.loads((output/"splits.json").read_text(encoding="utf-8"))
        classes=json.loads((output/"classes.json").read_text(encoding="utf-8"))
        history=json.loads((output/"history.json").read_text(encoding="utf-8"))
        metrics=json.loads((output/"metrics.json").read_text(encoding="utf-8"))
        metadata=json.loads((cache/"metadata.json").read_text(encoding="utf-8"))
        checkpoint=torch.load(output/"signbridge_landmark_motion.pt",map_location="cpu",weights_only=True)
        if len(classes)!=101 or checkpoint["meta"]["classes"]!=classes or checkpoint["meta"]["input_dim"]!=FEATURE_DIM:
                raise RuntimeError("Saved Experiment 4 checkpoint, input size, or class map is inconsistent.")
        if split_data["info"]["counts"]!={"train":475,"val":99,"test":105}:
                raise RuntimeError(f"Unexpected saved Experiment 4 split counts: {split_data['info']['counts']}")
        exp3_splits=json.loads((models/"experiments"/"exp3_landmarks"/"splits.json").read_text(encoding="utf-8"))
        test_block=split_data["splits"]["test"]
        if test_block["rows"]!=exp3_splits["splits"]["test"]["rows"] or test_block["labels"]!=exp3_splits["splits"]["test"]["labels"]:
                raise RuntimeError("Experiment 4 test rows differ from Experiment 3; refusing comparison.")
        cache_files=sorted(cache.glob("*.npy"))
        if len(cache_files)!=687 or [p.name for p in cache_files]!=[f"{i:06d}.npy" for i in range(687)]:
                raise RuntimeError("Saved Experiment 4 cache is not a complete sequential 687-file cache.")

        model=build_landmark_motion_classifier(101,DROPOUT)
        model.load_state_dict(checkpoint["state_dict"]);model.eval()
        rows=list(map(int,test_block["rows"]));targets=list(map(int,test_block["labels"]))
        predictions=[];probabilities=[]
        with torch.no_grad():
                for row in rows:
                        x=np.load(cache/f"{row:06d}.npy",allow_pickle=False)
                        if x.shape!=(NUM_FRAMES,FEATURE_DIM) or not np.isfinite(x).all():
                                raise RuntimeError(f"Invalid saved feature row {row} during final evaluation.")
                        logits=model(torch.from_numpy(x.copy()).unsqueeze(0))
                        probabilities.append(torch.softmax(logits,dim=1)[0].numpy())
                        predictions.append(int(logits.argmax(1)))
        predictions=np.asarray(predictions,dtype=np.int64);probabilities=np.asarray(probabilities,dtype=np.float32)

        # Confusion/class summaries derive from the actual checkpoint predictions above.
        error,per_class,pairs=_analysis_from_predictions(targets,predictions.tolist(),classes)
        meta_by_row={int(row["cache_index"]):row for row in metadata["videos"]}
        video_records=[]
        for i,(row,target,pred) in enumerate(zip(rows,targets,predictions.tolist())):
                d=meta_by_row[row]["detection"]
                video_records.append({"row":row,"true_id":target,"true_label":classes[target],"pred_id":pred,
                    "prediction":classes[pred],"correct":pred==target,"confidence":float(probabilities[i,pred]),
                    "left_rate":d["left_hand_detection_percent"]/100.0,"right_rate":d["right_hand_detection_percent"]/100.0,
                    "pose_rate":d["pose_detection_percent"]/100.0,"left_missing":d["left_hand_present_frames"]==0,
                    "right_missing":d["right_hand_present_frames"]==0,"both_frames":d["both_hand_frames"]})
        exp4_hand={
            "left_hand_present":_group_accuracy([r for r in video_records if not r["left_missing"]]),
            "left_hand_absent":_group_accuracy([r for r in video_records if r["left_missing"]]),
            "both_hands_present":_group_accuracy([r for r in video_records if r["both_frames"]>0]),
            "one_hand_only":_group_accuracy([r for r in video_records if r["both_frames"]==0 and (r["left_rate"]>0 or r["right_rate"]>0)]),
            "neither_hand_present":_group_accuracy([r for r in video_records if r["left_rate"]==0 and r["right_rate"]==0]),
        }
        exp3_error=json.loads((models/"experiments"/"exp3_landmarks"/"analysis"/"error_analysis.json").read_text(encoding="utf-8"))
        exp3_records={int(r["row"]):r for r in exp3_error["per_video"]}
        exp3_predictions=[int(exp3_records[row]["pred_id"]) for row in rows]
        exp3_hand_saved=exp3_error["hand_detection_groups"]
        exp3_hand={
            "left_hand_present":exp3_hand_saved["left_hand_detected_any_frame"],
            "left_hand_absent":exp3_hand_saved["left_hand_zero_frames"],
            "both_hands_present":exp3_hand_saved["both_hands_detected_same_frame"],
            "one_hand_only":exp3_hand_saved["only_one_hand_detected_no_both_hand_frames"],
            "neither_hand_present":exp3_hand_saved["neither_hand_detected"],
        }
        exp3_metrics=json.loads((models/"experiments"/"exp3_landmarks"/"metrics.json").read_text(encoding="utf-8"))["test"]
        exp3_class_error,_,_=_analysis_from_predictions(targets,exp3_predictions,classes)
        exp3_summary={k:exp3_metrics[k] for k in ("accuracy","macro_precision","macro_recall","macro_f1","top5_accuracy")}
        exp3_summary.update({"zero_correct_classes":exp3_class_error["zero_correct_classes"],
                                                 "unique_predicted_classes":exp3_class_error["unique_predicted_classes"]})
        exp4_summary={k:metrics["test"][k] for k in ("accuracy","macro_precision","macro_recall","macro_f1","top5_accuracy")}
        exp4_summary.update({"zero_correct_classes":error["zero_correct_classes"],
                                                 "unique_predicted_classes":error["unique_predicted_classes"]})
        deltas={key:exp4_summary[key]-exp3_summary[key] for key in exp3_summary}
        comparison={"same_test_rows":True,"test_rows":rows,"test_samples":len(rows),"experiment_3":exp3_summary,
            "experiment_4":exp4_summary,"delta_exp4_minus_exp3":deltas,
            "hand_detection_accuracy":{"experiment_3":exp3_hand,"experiment_4":exp4_hand}}
        comparison_json=output/"comparison_with_exp3.json";comparison_txt=output/"comparison_with_exp3.txt"
        if not comparison_json.exists():comparison_json.write_text(json.dumps(comparison,indent=2),encoding="utf-8")
        lines=["Experiment 3 vs Experiment 4; paired on the exact same 105 saved test video rows.","Metric | Exp3 | Exp4 | Exp4-Exp3"]
        for key in exp3_summary:lines.append(f"{key} | {exp3_summary[key]} | {exp4_summary[key]} | {deltas[key]}")
        lines.extend(["","Hand detection group accuracy:",json.dumps(comparison["hand_detection_accuracy"],indent=2)])
        if not comparison_txt.exists():comparison_txt.write_text("\n".join(lines)+"\n",encoding="utf-8")

        aggregate=metadata["aggregate"]
        fallback_stats={"fallback_center_frames":aggregate["fallback_center_frames"],
            "fallback_scale_frames":aggregate["fallback_scale_frames"],"clipped_coordinate_values":aggregate["clipped_coordinate_values"],
            "normalized_coordinate_values":aggregate["normalized_coordinate_values"],"coordinate_clip_percent":aggregate["coordinate_clip_percent"]}
        error.update({"test_samples":len(rows),"test_accuracy":metrics["test"]["accuracy"],
            "test_macro_f1":metrics["test"]["macro_f1"],"prediction_distribution":error["prediction_distribution"],
            "hand_detection_groups":exp4_hand,"normalization_statistics":fallback_stats,
            "per_video_predictions":video_records})
        existing_analysis={"per_class_results.csv","confusion_pairs.csv","error_analysis.json","error_analysis.txt"}
        if analysis_dir.exists():
            if not existing_analysis.issubset({p.name for p in analysis_dir.iterdir()}):
                raise RuntimeError("Experiment 4 analysis folder is partial; refusing to overwrite existing analysis outputs.")
            # The prior run wrote these complete analysis outputs before its later reporting KeyError.
            saved_error=json.loads((analysis_dir/"error_analysis.json").read_text(encoding="utf-8"))
            if saved_error.get("unique_predicted_classes")!=error["unique_predicted_classes"] or saved_error.get("zero_correct_classes")!=error["zero_correct_classes"]:
                raise RuntimeError("Existing Experiment 4 analysis outputs disagree with predictions from the saved checkpoint.")
        else:
            analysis_dir.mkdir(exist_ok=False)
            with (analysis_dir/"per_class_results.csv").open("w",newline="",encoding="utf-8") as stream:
                writer=csv.DictWriter(stream,fieldnames=list(per_class[0]));writer.writeheader();writer.writerows(per_class)
            with (analysis_dir/"confusion_pairs.csv").open("w",newline="",encoding="utf-8") as stream:
                writer=csv.DictWriter(stream,fieldnames=["rank","true_class_id","true_label","predicted_class_id","predicted_label","count"]);writer.writeheader()
                for rank,pair in enumerate(pairs[:20],1):writer.writerow({"rank":rank,**pair})
            (analysis_dir/"error_analysis.json").write_text(json.dumps(error,indent=2),encoding="utf-8")
            analysis_lines=["Experiment 4 error analysis (saved checkpoint; same Experiment 3 test rows)",
                f"Test accuracy={metrics['test']['accuracy']:.6f}; macro-F1={metrics['test']['macro_f1']:.6f}; "
                f"zero-correct classes={error['zero_correct_classes']}; predicted classes={error['unique_predicted_classes']}",
                f"Prediction distribution: {error['prediction_distribution']}","Hand-detection groups:",
                json.dumps(exp4_hand,indent=2),"Normalization fallback/clipping statistics:",json.dumps(fallback_stats,indent=2),
                "Top confusion pairs:"]
            analysis_lines.extend(f"  {i}. {p['true_label']} -> {p['predicted_label']}: {p['count']}" for i,p in enumerate(pairs[:20],1))
            (analysis_dir/"error_analysis.txt").write_text("\n".join(analysis_lines)+"\n",encoding="utf-8")

        checks={"project_dataset_unchanged_since_exp2_preflight":_manifest_hash(settings.dataset_dir.resolve())==json.loads((models/"experiments"/"exp2"/"preflight.json").read_text())["dataset_project"]["signature"],
            "csv_unchanged":_digest(settings.dataset_dir/"corpus_csv_files"/"ISL Corpus sign glosses.csv")==json.loads((models/"experiments"/"exp2"/"preflight.json").read_text())["csv_sha256"],
            "exp3_still_present":(models/"experiments"/"exp3_landmarks"/"signbridge_landmark_model.pt").is_file(),
            "exp2_still_present":(models/"experiments"/"exp2"/"signbridge_head.pt").is_file(),
            "motion_cache_arrays":len(cache_files)==687}
        final=(history[-1] if history else {})
        report=["SignBridge Experiment 4: robust normalized landmarks + velocity", "Evaluation/report finalized from saved Experiment 4 checkpoint; NO training or extraction rerun.",
            f"Split counts: {split_data['info']['counts']}; test set is identical to Exp3.",
            f"Cache rows: {len(cache_files)}, feature shape: {NUM_FRAMES} x {FEATURE_DIM}; cache reused.",
            f"Training epochs: {len(history)}, best epoch: {metrics['best_epoch']} by validation loss.",
            f"Final training loss/accuracy: {final.get('train_loss')} / {final.get('train_accuracy')}.",
            f"Validation accuracy/macro-F1: {metrics['validation']['accuracy']} / {metrics['validation']['macro_f1']}.",
            f"Test accuracy/macro-F1/top-5: {metrics['test']['accuracy']} / {metrics['test']['macro_f1']} / {metrics['test']['top5_accuracy']}.",
            f"Experiment 3 same-test accuracy/macro-F1/top-5: {exp3_summary['accuracy']} / {exp3_summary['macro_f1']} / {exp3_summary['top5_accuracy']}.",
            f"Exp4 - Exp3 deltas: {deltas}.",f"Protected checks: {checks}."]
        report_path=output/"training_report.txt"
        if report_path.exists():report_path.write_text(report_path.read_text(encoding="utf-8")+"\n"+"\n".join(report)+"\n",encoding="utf-8")
        else:report_path.write_text("\n".join(report)+"\n",encoding="utf-8")
        # Replace only the prior failure note in the new Exp4 directory with resolution context.
        failure_path=output/"failure_report.txt"
        if failure_path.exists():failure_path.write_text(failure_path.read_text(encoding="utf-8").rstrip()+"\nResolved: saved checkpoint was evaluated and missing comparison/error-analysis reports were generated without training or feature extraction.\n",encoding="utf-8")
        print("Experiment 4 report finalization completed from saved checkpoint; no training/extraction performed.")
        print("Comparison deltas:",json.dumps(deltas,indent=2))
        print("Integrity checks:",checks)
        return 0


def main() -> int:
    import argparse
    parser=argparse.ArgumentParser(description="Final SignBridge Experiment 4 landmark-motion pipeline")
    parser.add_argument("--finalize-saved-run",action="store_true",help="evaluate existing exp4 checkpoint and write missing reports only")
    args=parser.parse_args()
    try: return _finalize_saved_run() if args.finalize_saved_run else run()
    except Exception as exc:
        print(f"EXPERIMENT 4 STOPPED: {type(exc).__name__}: {exc}");return 2

if __name__=="__main__":sys.exit(main())

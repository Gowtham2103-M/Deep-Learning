"""
Step 2 of training:  cached CNN features -> BiLSTM -> Attention -> Dense -> Softmax

Run from the backend folder:
    python -m training.train
    python -m training.train --epochs 5          (quick test)
    python -m training.train --skip-extract      (features already extracted)

What it does:
  1. reads the dataset CSV and matches videos with sentence labels
  2. extracts (or reuses) CNN features
  3. splits into train / validation / test  (signer-independent if signer IDs exist)
  4. trains with early stopping, keeps the best checkpoint
  5. saves: model, class mapping, history, splits, and real test metrics
"""
from __future__ import annotations

import argparse
import copy
import datetime as dt
import hashlib
import json
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

from app.config import Settings, get_settings
from app.inference.feature_pipeline import FeaturePipeline, pick_device
from app.models.signbridge_model import BiLSTMAttentionHead
from app.preprocessing.dataset_index import DatasetIndex, build_index
from training.dataset import (FeatureDataset, find_conflicting_duplicate_rows, make_grouped_splits,
                              make_splits, video_sha256)
from training.extract_features import extract_all, feature_dir


def _run_epoch(model, loader, loss_fn, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss, correct, n = 0.0, 0, 0
    true_ids, predicted_ids = [], []
    with torch.set_grad_enabled(training):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = loss_fn(logits, y)
            if training:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
            total_loss += loss.item() * len(y)
            correct += (logits.argmax(1) == y).sum().item()
            n += len(y)
            true_ids.extend(y.detach().cpu().tolist())
            predicted_ids.extend(logits.argmax(1).detach().cpu().tolist())
    macro_f1 = f1_score(true_ids, predicted_ids, average="macro", zero_division=0) if n else 0.0
    return total_loss / max(n, 1), correct / max(n, 1), float(macro_f1)


def run_training(s: Settings, index: DatasetIndex, pipeline: Optional[FeaturePipeline] = None,
                 epochs: Optional[int] = None, skip_extract: bool = False,
                 artifact_dir: Optional[Path] = None, grouped_split: bool = False,
                 excluded_rows: Optional[list] = None) -> dict:
    torch.manual_seed(s.SEED)
    np.random.seed(s.SEED)
    pipeline = pipeline or FeaturePipeline.from_settings(s)
    if not skip_extract:
        print("Step 1/3: extracting CNN features (skips videos already done) ...")
        extract_all(s, index, pipeline)
    cache = feature_dir(s, pipeline)

    df = index.df
    have = np.array([(cache / f"{i:06d}.npy").exists() for i in df.index])
    if not have.any():
        raise RuntimeError(f"No feature files found in {cache}. Run: python -m training.extract_features")
    df = df[have].reset_index().rename(columns={"index": "row"})      # 'row' = feature file number
    if len(df) < len(index.df):
        print(f"WARNING: {len(index.df) - len(df)} videos have no features and are skipped.")

    label_to_id = index.label_to_id()
    y_all = df["label"].map(label_to_id).values

    print("Step 2/3: splitting the data ...")
    if grouped_split:
        splits, info = make_grouped_splits(df, s.VAL_FRACTION, s.TEST_FRACTION, s.SEED)
        info["split_counts"] = {name: int(len(rows)) for name, rows in splits.items()}
        info["classes_without_usable_examples"] = sorted(set(index.classes) - set(df["label"].unique()))
        info["excluded_conflicting_rows"] = excluded_rows or []
        info["warnings"] = [
            "The CSV has no signer IDs; this is not a signer-independent split.",
            "Classes with fewer than three unique content groups cannot appear in every partition.",
        ]
    else:
        splits, info = make_splits(df, s.VAL_FRACTION, s.TEST_FRACTION, s.SEED)
    print(f"  split mode: {info['mode']} | train {len(splits['train'])} | val {len(splits['val'])} "
          f"| test {len(splits['test'])}")
    for w in info["warnings"]:
        print("  WARNING:", w)

    def make_ds(name):
        pos = splits[name]
        return FeatureDataset(cache, df["row"].values[pos], y_all[pos])

    train_ds, val_ds, test_ds = make_ds("train"), make_ds("val"), make_ds("test")
    if len(train_ds) == 0 or len(val_ds) == 0 or len(test_ds) == 0:
        raise RuntimeError("Train, validation, or test split is empty - the dataset is too small.")
    train_loader = DataLoader(train_ds, batch_size=s.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=s.BATCH_SIZE)

    device = pick_device()
    input_dim = np.load(cache / f"{int(df['row'].iloc[0]):06d}.npy").shape[1]
    model = BiLSTMAttentionHead(input_dim, index.num_classes, s.LSTM_HIDDEN, s.LSTM_LAYERS,
                                s.DENSE_UNITS, s.DROPOUT, s.USE_ATTENTION).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Step 3/3: training on {device} | classes: {index.num_classes} | feature dim: {input_dim} "
          f"| trainable parameters: {n_params:,}")

    opt = torch.optim.Adam(model.parameters(), lr=s.LEARNING_RATE, weight_decay=s.WEIGHT_DECAY)
    loss_fn = nn.CrossEntropyLoss()
    max_epochs = epochs or s.EPOCHS
    history, best_loss, best_state, bad = [], float("inf"), None, 0
    for ep in range(1, max_epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc, _ = _run_epoch(model, train_loader, loss_fn, device, opt)
        va_loss, va_acc, va_f1 = _run_epoch(model, val_loader, loss_fn, device)
        history.append({"epoch": ep, "train_loss": tr_loss, "train_acc": tr_acc,
                "val_loss": va_loss, "val_acc": va_acc, "val_macro_f1": va_f1,
                "seconds": round(time.time() - t0, 2)})
        flag = ""
        if va_loss < best_loss - 1e-4:                                  # model checkpointing
            best_loss, bad = va_loss, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            flag = "  <- best so far (checkpoint kept)"
        else:
            bad += 1
        print(f"  epoch {ep:3d}/{max_epochs}  train loss {tr_loss:.3f} acc {tr_acc:.3f} | "
              f"val loss {va_loss:.3f} acc {va_acc:.3f} macro F1 {va_f1:.3f}{flag}")
        if bad >= s.EARLY_STOP_PATIENCE:                                # early stopping
            print(f"  early stopping: no improvement for {s.EARLY_STOP_PATIENCE} epochs.")
            break

    meta = {"input_dim": input_dim, "num_classes": index.num_classes, "hidden": s.LSTM_HIDDEN,
            "layers": s.LSTM_LAYERS, "dense_units": s.DENSE_UNITS, "dropout": s.DROPOUT,
            "use_attention": s.USE_ATTENTION, "num_frames": s.NUM_FRAMES, "image_size": s.IMAGE_SIZE,
            "backbone": s.CNN_BACKBONE, "cnn_pretrained": s.CNN_PRETRAINED,
            "use_mediapipe": s.USE_MEDIAPIPE, "mediapipe_parts": s.mediapipe_parts,
            "classes": index.classes, "trained_at": dt.datetime.now().isoformat(timespec="seconds"),
            "split_mode": info["mode"], "seed": s.SEED, "epochs_run": len(history)}
    out_dir = Path(artifact_dir) if artifact_dir is not None else s.models_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = out_dir / "signbridge_head.pt" if artifact_dir is not None else s.model_path
    torch.save({"state_dict": best_state, "meta": meta}, checkpoint_path)
    (out_dir / "classes.json").write_text(json.dumps(index.classes, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    splits_out = {name: {"rows": [int(df["row"].values[p]) for p in splits[name]],
                         "labels": [int(y_all[p]) for p in splits[name]]} for name in splits}
    (out_dir / "splits.json").write_text(json.dumps(
        {"cache_dir": str(cache), "info": info, "splits": splits_out}, indent=2), encoding="utf-8")
    print(f"\nSaved model to {checkpoint_path}")

    from training.evaluate import evaluate
    metrics = evaluate(s, "test", checkpoint_path=checkpoint_path, artifact_dir=out_dir) if len(test_ds) else None
    best_epoch = min(history, key=lambda row: row["val_loss"])["epoch"]
    result = {"history": history, "best_val_loss": best_loss, "best_epoch": best_epoch,
              "checkpoint_selection": "minimum validation loss", "metrics": metrics,
              "split_info": info, "checkpoint_path": str(checkpoint_path)}
    if artifact_dir is not None:
        (out_dir / "training_report.txt").write_text(
            _format_training_report(s, index, result, excluded_rows or []), encoding="utf-8")
    return result


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tree_signature(root: Path) -> dict:
    files = []
    total_bytes = 0
    for path in sorted(root.rglob("*")):
        if path.is_file():
            stat = path.stat()
            rel = path.relative_to(root).as_posix()
            files.append((rel, stat.st_size, stat.st_mtime_ns))
            total_bytes += stat.st_size
    signature = hashlib.sha256(json.dumps(files, separators=(",", ":")).encode("utf-8")).hexdigest()
    return {"files": len(files), "bytes": total_bytes, "signature": signature,
            "file_manifest": files}


def _format_training_report(s: Settings, index: DatasetIndex, result: dict, excluded_rows: list) -> str:
    metrics = result["metrics"] or {}
    lines = [
        "SignBridge controlled experiment exp2",
        f"Finished: {dt.datetime.now().isoformat(timespec='seconds')}",
        f"Dataset root: {index.path}",
        f"Training rows after duplicate-conflict exclusion: {len(index.df)}",
        f"Output classes retained in class map: {index.num_classes}",
        "CNN features reused from existing cache; feature extraction was not run.",
        f"Feature cache: {s.cache_dir}",
        f"Excluded conflicting duplicate rows: {len(excluded_rows)}",
    ]
    for item in excluded_rows:
        lines.append(f"  row {item['row']}: {item['video_path']} | label={item['label']} | "
                     f"sha256={item['sha256']} | reason={item['reason']}")
    lines.extend([
        f"Split mode: {result['split_info']['mode']}",
        f"Split counts: {result['split_info'].get('split_counts', {})}",
        f"Class coverage: {result['split_info'].get('class_coverage', {})}",
        f"Classes insufficient for all partitions: {result['split_info'].get('insufficient_samples_for_all_partitions', {})}",
        f"Classes with no remaining examples: {result['split_info'].get('classes_without_usable_examples', [])}",
        f"Best epoch: {result['best_epoch']} (selected by minimum validation loss)",
        f"Training settings: learning_rate={s.LEARNING_RATE}, dropout={s.DROPOUT}, epochs={s.EPOCHS}, "
        f"early_stopping_patience={s.EARLY_STOP_PATIENCE}, weight_decay={s.WEIGHT_DECAY}, "
        f"gradient_clip_norm=5, batch_size={s.BATCH_SIZE}",
    ])
    if result["history"]:
        final = result["history"][-1]
        best = min(result["history"], key=lambda row: row["val_loss"])
        lines.extend([
            f"Epochs completed: {len(result['history'])}",
            f"Final train loss/accuracy: {final['train_loss']:.6f} / {final['train_acc']:.6f}",
            f"Final validation loss/accuracy/macro-F1: {final['val_loss']:.6f} / "
            f"{final['val_acc']:.6f} / {final['val_macro_f1']:.6f}",
            f"Best validation loss: {best['val_loss']:.6f} at epoch {best['epoch']}",
        ])
    if metrics:
        lines.extend([
            f"Test samples: {metrics['num_samples']}",
            f"Test classes represented: {metrics['classes_present_in_split']} / {metrics['num_classes']}",
            f"Test accuracy: {metrics['accuracy']:.6f}",
            f"Test macro precision/recall/F1: {metrics['precision_macro']:.6f} / "
            f"{metrics['recall_macro']:.6f} / {metrics['f1_macro']:.6f}",
            f"Test top-5 accuracy: {metrics['top5_accuracy']:.6f}",
        ])
    lines.extend(["", "Warnings:"] + [f"  {warning}" for warning in result["split_info"].get("warnings", [])])
    return "\n".join(lines) + "\n"


def _exp2_preflight(s: Settings) -> tuple[DatasetIndex, list, dict, dict, Path]:
    """Validate all experiment safety conditions without writing to data or cache."""
    project_dataset = s.dataset_dir.resolve()
    original_dataset = Path.home() / "Downloads" / "ISL_CSLRT_Corpus" / "ISL_CSLRT_Corpus"
    expected_cache = s.cache_dir / "mobilenet_v2_pre_30f_224px_nomp"
    experiment_dir = s.models_dir / "experiments" / "exp2"
    if experiment_dir.exists():
        raise RuntimeError(f"Refusing to overwrite existing experiment directory: {experiment_dir}")
    if not project_dataset.is_dir() or not original_dataset.is_dir():
        raise RuntimeError(f"Dataset root missing: project={project_dataset}, original={original_dataset}")

    current_settings = (s.CNN_BACKBONE, s.CNN_PRETRAINED, s.NUM_FRAMES, s.BATCH_SIZE,
                        s.LEARNING_RATE, s.DROPOUT, s.EPOCHS, s.EARLY_STOP_PATIENCE, s.WEIGHT_DECAY)
    expected_settings = ("mobilenet_v2", True, 30, 16, 0.0001, 0.3, 60, 15, 0.0001)
    if current_settings != expected_settings:
        raise RuntimeError(f"Unexpected base training settings: {current_settings!r}")

    dataset_signature = _tree_signature(project_dataset)
    source_signature = _tree_signature(original_dataset)
    if dataset_signature["files"] != 21074 or source_signature["files"] != 21074:
        raise RuntimeError("Dataset file count no longer matches the verified 21,074-file copy.")
    if dataset_signature["files"] != source_signature["files"] or dataset_signature["bytes"] != source_signature["bytes"]:
        raise RuntimeError("Project dataset file count/size differs from the original Downloads dataset.")
    project_files = [(name, size) for name, size, _ in dataset_signature["file_manifest"]]
    source_files = [(name, size) for name, size, _ in source_signature["file_manifest"]]
    if project_files != source_files:
        raise RuntimeError("Project dataset file paths/sizes differ from original Downloads dataset.")

    csv_relative = Path("corpus_csv_files") / "ISL Corpus sign glosses.csv"
    project_csv, source_csv = project_dataset / csv_relative, original_dataset / csv_relative
    if not project_csv.is_file() or not source_csv.is_file():
        raise RuntimeError("Expected real dataset CSV is missing.")
    project_csv_hash, source_csv_hash = _sha256_file(project_csv), _sha256_file(source_csv)
    if project_csv_hash != source_csv_hash:
        raise RuntimeError("Project CSV differs from the original Downloads CSV.")

    index = build_index(s)
    if not index.available or index.video_count != 687 or index.num_classes != 101:
        raise RuntimeError(f"Dataset index mismatch: available={index.available}, videos={index.video_count}, "
                           f"classes={index.num_classes}, message={index.message}")
    if index.label_column != "Sentence" or index.video_column is not None:
        raise RuntimeError(f"Unexpected dataset mapping columns: video={index.video_column}, label={index.label_column}")
    classes_file = s.models_dir / "classes.json"
    if not classes_file.is_file():
        raise RuntimeError("Existing 101-class class map is missing.")
    classes = json.loads(classes_file.read_text(encoding="utf-8"))
    if len(classes) != 101 or len(set(classes)) != 101 or classes != index.classes:
        raise RuntimeError("Existing classes.json does not exactly match the 101 real Sentence labels.")
    label_to_id = {label: class_id for class_id, label in enumerate(classes)}
    if any(label not in label_to_id for label in index.df["label"]):
        raise RuntimeError("Dataset contains a label with no class ID.")

    if not expected_cache.is_dir():
        raise RuntimeError(f"Expected feature cache is missing: {expected_cache}")
    cache_files = sorted(expected_cache.glob("*.npy"))
    expected_names = [f"{row:06d}.npy" for row in range(687)]
    if [path.name for path in cache_files] != expected_names:
        raise RuntimeError(f"Expected 687 sequential cache files; found {len(cache_files)} or unexpected names.")
    for cache_file in cache_files:
        features = np.load(cache_file, mmap_mode="r")
        if features.shape != (30, 1280) or not np.isfinite(features).all():
            raise RuntimeError(f"Invalid cached feature array: {cache_file.name}, shape={features.shape}")
    if len(index.df) != 687 or index.df.index.tolist() != list(range(687)):
        raise RuntimeError("Dataset index row order no longer matches cache row IDs 000000..000686.")

    conflicts, hash_groups = find_conflicting_duplicate_rows(index.df)
    excluded_ids = {row["row"] for conflict in conflicts for row in conflict["rows"]}
    excluded_rows = []
    for conflict in conflicts:
        for row in conflict["rows"]:
            excluded_rows.append({**row, "sha256": conflict["sha256"],
                                  "reason": "byte-identical video content is assigned a different Sentence label"})
    if len(conflicts) != 4 or len(excluded_ids) != 8:
        raise RuntimeError(f"Expected the four diagnosed conflicting duplicate pairs/eight rows; found "
                           f"{len(conflicts)} pairs/{len(excluded_ids)} rows.")

    filtered = index.df.drop(index=sorted(excluded_ids)).copy()
    filtered["__video_sha256"] = [
        next(digest for digest, rows in hash_groups.items() if int(row_id) in rows)
        for row_id in filtered.index
    ]
    filtered_index = copy.copy(index)
    filtered_index.df = filtered
    filtered_index.video_count = len(filtered)
    splits, split_info = make_grouped_splits(filtered, s.VAL_FRACTION, s.TEST_FRACTION, s.SEED)
    split_sets = {name: set(map(int, rows)) for name, rows in splits.items()}
    if any(split_sets[a] & split_sets[b] for a, b in (("train", "val"), ("train", "test"), ("val", "test"))):
        raise RuntimeError("Safety check failed: split rows overlap.")
    if len(set().union(*split_sets.values())) != len(filtered):
        raise RuntimeError("Safety check failed: filtered rows are missing or duplicated in the splits.")
    if set(split_info["classes_by_split"]["train"]) != set(filtered["label"].unique()):
        raise RuntimeError("Safety check failed: a usable label is not represented in training.")

    unavailable = sorted(set(classes) - set(filtered["label"].unique()))
    split_info["classes_without_usable_examples"] = unavailable
    split_info["split_counts"] = {name: int(len(rows)) for name, rows in splits.items()}
    split_info["excluded_conflicting_rows"] = excluded_rows
    split_info["warnings"] = [
        "The CSV has no signer IDs; this is not a signer-independent split.",
        "Classes with fewer than three unique content groups cannot appear in every partition.",
    ]

    # Capture output-baseline hashes so exp2 cannot overwrite the first run.
    old_artifacts = {}
    for name in ("signbridge_head.pt", "classes.json", "history.json", "splits.json", "metrics.json",
                 "confusion_matrix.csv", "confusion_matrix.png"):
        path = s.models_dir / name
        if path.is_file():
            old_artifacts[name] = _sha256_file(path)

    print("SAFETY PREFLIGHT: PASS")
    print(f"Dataset roots: project={project_dataset}; original={original_dataset}")
    print(f"Dataset inventory: {dataset_signature['files']} files; {dataset_signature['bytes']} bytes; "
          "project/original relative paths and sizes match.")
    print(f"CSV: SHA-256 identical in project and Downloads ({project_csv_hash}); 101 rows, "
          "columns Sentence / SIGN GLOSSES.")
    print(f"Feature cache: {len(cache_files)} arrays, all finite and shape (30, 1280); will be reused, "
          "not regenerated.")
    print(f"Class map: {len(classes)} unique real Sentence labels; every indexed label has a valid ID.")
    print(f"Index: {index.video_count} videos / {index.num_classes} classes; filtered experiment rows: {len(filtered)}.")
    print(f"Split checks: counts={split_info['split_counts']}; no row overlap; duplicate-content groups stay together.")
    print(f"Excluded {len(excluded_rows)} rows from {len(conflicts)} conflicting identical-content pairs:")
    for row in excluded_rows:
        print(f"  row {row['row']}: {row['video_path']} | Sentence={row['label']!r} | "
              f"sha256={row['sha256']} | reason={row['reason']}")
    print(f"Classes with no remaining examples after required exclusion: {unavailable}")
    print(f"Classes with insufficient examples for all partitions: {split_info['insufficient_samples_for_all_partitions']}")
    print(f"Old experiment artifacts preserved; new output target is {experiment_dir}.")

    baseline = {"project_dataset": dataset_signature, "original_dataset": source_signature,
                "csv_sha256": project_csv_hash, "old_artifacts": old_artifacts}
    return filtered_index, excluded_rows, split_info, baseline, expected_cache


def _run_exp2(s: Settings, requested_epochs: Optional[int] = None) -> dict:
    if requested_epochs is not None:
        raise ValueError("exp2 uses the approved fixed maximum of 60 epochs; do not pass --epochs.")
    if s.cache_dir.resolve() != (s.models_dir / "feature_cache").resolve():
        raise RuntimeError("Unexpected feature cache root; refusing to train.")
    settings = replace(s, LEARNING_RATE=0.0001, DROPOUT=0.3, EPOCHS=60,
                       EARLY_STOP_PATIENCE=15, WEIGHT_DECAY=0.0001)
    index, excluded, split_info, baseline, cache_dir = _exp2_preflight(settings)

    # Prevent torchvision from reaching the network during this experiment.
    from torchvision.models import MobileNet_V2_Weights
    weight_file = Path(torch.hub.get_dir()) / "checkpoints" / Path(
        MobileNet_V2_Weights.IMAGENET1K_V1.url).name
    if settings.CNN_PRETRAINED and not weight_file.is_file():
        raise RuntimeError(f"Cached MobileNetV2 weights are missing; refusing network access: {weight_file}")
    pipeline = FeaturePipeline.from_settings(settings)
    actual_cache = feature_dir(settings, pipeline).resolve()
    if actual_cache != cache_dir.resolve():
        raise RuntimeError(f"Feature cache signature mismatch: expected {cache_dir}, got {actual_cache}")

    experiment_dir = settings.models_dir / "experiments" / "exp2"
    if experiment_dir.exists():
        raise RuntimeError(f"Refusing to overwrite experiment output: {experiment_dir}")
    experiment_dir.mkdir(parents=True, exist_ok=False)
    (experiment_dir / "preflight.json").write_text(json.dumps({
        "dataset_project": {k: v for k, v in baseline["project_dataset"].items() if k != "file_manifest"},
        "dataset_original": {k: v for k, v in baseline["original_dataset"].items() if k != "file_manifest"},
        "csv_sha256": baseline["csv_sha256"],
        "cache_dir": str(cache_dir),
        "cache_arrays": 687,
        "cache_shape": [30, 1280],
        "excluded_conflicting_rows": excluded,
        "split_info": split_info,
        "settings": {"cnn": settings.CNN_BACKBONE, "cnn_pretrained": settings.CNN_PRETRAINED,
                     "num_frames": settings.NUM_FRAMES, "batch_size": settings.BATCH_SIZE,
                     "learning_rate": settings.LEARNING_RATE, "dropout": settings.DROPOUT,
                     "epochs": settings.EPOCHS, "early_stopping_patience": settings.EARLY_STOP_PATIENCE,
                     "weight_decay": settings.WEIGHT_DECAY},
    }, indent=2), encoding="utf-8")

    result = run_training(settings, index, pipeline=pipeline, epochs=60, skip_extract=True,
                          artifact_dir=experiment_dir, grouped_split=True, excluded_rows=excluded)

    project_after = _tree_signature(settings.dataset_dir.resolve())
    original_after = _tree_signature(Path.home() / "Downloads" / "ISL_CSLRT_Corpus" / "ISL_CSLRT_Corpus")
    csv_after = _sha256_file(settings.dataset_dir / "corpus_csv_files" / "ISL Corpus sign glosses.csv")
    integrity = {
        "project_dataset_unchanged": project_after["signature"] == baseline["project_dataset"]["signature"],
        "original_dataset_unchanged": original_after["signature"] == baseline["original_dataset"]["signature"],
        "csv_unchanged": csv_after == baseline["csv_sha256"],
        "old_artifacts_unchanged": all(
            _sha256_file(settings.models_dir / name) == digest
            for name, digest in baseline["old_artifacts"].items()
        ),
    }
    (experiment_dir / "training_report.txt").write_text(
        _format_training_report(settings, index, result, excluded) +
        "\nPost-training integrity checks:\n" +
        "\n".join(f"  {name}: {value}" for name, value in integrity.items()) + "\n",
        encoding="utf-8")
    if not all(integrity.values()):
        raise RuntimeError(f"Post-training integrity verification failed: {integrity}")
    print("POST-TRAINING INTEGRITY: PASS", integrity)
    return result


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Train the SignBridge BiLSTM + Attention head.")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--skip-extract", action="store_true", help="do not (re)run feature extraction")
    ap.add_argument("--experiment", choices=("exp2",), help="run the controlled, isolated exp2 protocol")
    ap.add_argument("--preflight-only", action="store_true", help="validate exp2 safety without training or writing files")
    args = ap.parse_args(argv)
    s = get_settings()
    if args.experiment == "exp2":
        try:
            if args.preflight_only:
                _exp2_preflight(replace(s, LEARNING_RATE=0.0001, DROPOUT=0.3, EPOCHS=60,
                                        EARLY_STOP_PATIENCE=15, WEIGHT_DECAY=0.0001))
                print("PRE-FLIGHT ONLY: no experiment artifacts created and no training started.")
                return 0
            out = _run_exp2(s, requested_epochs=args.epochs)
        except Exception as exc:
            print(f"EXP2 STOPPED: {type(exc).__name__}: {exc}")
            return 2
        m = out["metrics"]
        if m:
            print("\n=== EXP2 TEST RESULTS ===")
            print(f"Best epoch {out['best_epoch']} (selected by validation loss)")
            print(f"Accuracy {m['accuracy']:.4f} | macro precision {m['precision_macro']:.4f} | "
                  f"macro recall {m['recall_macro']:.4f} | macro F1 {m['f1_macro']:.4f} | "
                  f"top-5 accuracy {m['top5_accuracy']:.4f}")
            print(f"Test samples {m['num_samples']} | classes represented "
                  f"{m['classes_present_in_split']}/{m['num_classes']}")
        return 0

    if args.preflight_only:
        ap.error("--preflight-only requires --experiment exp2")

    index = build_index(s)
    if not index.available:
        print("\nDATASET PROBLEM:", index.message)
        return 2
    for w in index.warnings:
        print("WARNING:", w)
    out = run_training(s, index, epochs=args.epochs, skip_extract=args.skip_extract)
    m = out["metrics"]
    if m:
        print("\n=== REAL TEST RESULTS (computed just now) ===")
        print(f"Accuracy {m['accuracy']:.4f} | macro precision {m['precision_macro']:.4f} | "
              f"macro recall {m['recall_macro']:.4f} | macro F1 {m['f1_macro']:.4f}")
        for w in m["split_warnings"]:
            print("WARNING:", w)
    return 0


if __name__ == "__main__":
    sys.exit(main())

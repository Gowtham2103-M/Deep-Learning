"""
Evaluation of a TRAINED model on the held-out test split.

Only real numbers are ever written: accuracy, precision, recall, F1 and the confusion matrix
are computed from the model's actual predictions.  If the model has not been trained, this
script simply says so.

Run from the backend folder:
    python -m training.evaluate
"""
from __future__ import annotations

import csv
import datetime as dt
import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             precision_recall_fscore_support)

from app.config import Settings, get_settings
from app.models.signbridge_model import head_from_meta
from training.dataset import FeatureDataset


def load_checkpoint(model_path: Path):
    ckpt = torch.load(model_path, map_location="cpu", weights_only=True)
    head = head_from_meta(ckpt["meta"])
    head.load_state_dict(ckpt["state_dict"])
    head.eval()
    return head, ckpt["meta"]


@torch.no_grad()
def predict_dataset(head, ds: FeatureDataset, batch_size: int = 32, return_top5: bool = False):
    ys, ps, top5 = [], [], []
    for i in range(0, len(ds), batch_size):
        batch = [ds[j] for j in range(i, min(i + batch_size, len(ds)))]
        x = torch.stack([b[0] for b in batch])
        ys += [int(b[1]) for b in batch]
        logits = head(x)
        ps += logits.argmax(dim=1).tolist()
        if return_top5:
            top5.extend(logits.topk(min(5, logits.shape[1]), dim=1).indices.tolist())
    if return_top5:
        return np.array(ys), np.array(ps), np.array(top5)
    return np.array(ys), np.array(ps)


def save_confusion_png(cm: np.ndarray, classes, path: Path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return False
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xlabel("Predicted class id")
    ax.set_ylabel("True class id")
    ax.set_title("Confusion matrix (test split)")
    fig.colorbar(im, ax=ax)
    if len(classes) <= 25:
        ax.set_xticks(range(len(classes)))
        ax.set_yticks(range(len(classes)))
        ax.set_xticklabels([str(c)[:14] for c in classes], rotation=90, fontsize=6)
        ax.set_yticklabels([str(c)[:14] for c in classes], fontsize=6)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return True


def evaluate(s: Optional[Settings] = None, split: str = "test",
             checkpoint_path: Optional[Path] = None, artifact_dir: Optional[Path] = None) -> dict:
    s = s or get_settings()
    model_file = Path(checkpoint_path) if checkpoint_path is not None else s.model_path
    out = Path(artifact_dir) if artifact_dir is not None else s.models_dir
    splits_file = out / "splits.json"
    if not model_file.is_file():
        raise FileNotFoundError("Model not trained yet. Run the training command.")
    if not splits_file.is_file():
        raise FileNotFoundError("splits.json is missing - run the training command first.")

    head, meta = load_checkpoint(model_file)
    sp = json.loads(splits_file.read_text(encoding="utf-8"))
    part = sp["splits"][split]
    ds = FeatureDataset(Path(sp["cache_dir"]), np.array(part["rows"]), np.array(part["labels"]))
    y_true, y_pred, top5 = predict_dataset(head, ds, return_top5=True)
    classes = meta["classes"]
    labels = list(range(len(classes)))

    p_m, r_m, f_m, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    p_w, r_w, f_w, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    report = classification_report(y_true, y_pred, labels=labels, target_names=classes,
                                   output_dict=True, zero_division=0)
    metrics = {
        "evaluated_at": dt.datetime.now().isoformat(timespec="seconds"),
        "split": split, "num_samples": int(len(y_true)),
        "split_mode": sp["info"]["mode"], "split_warnings": sp["info"]["warnings"],
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "top5_accuracy": float(np.mean([int(y) in candidates for y, candidates in zip(y_true, top5)])),
        "precision_macro": float(p_m), "recall_macro": float(r_m), "f1_macro": float(f_m),
        "precision_weighted": float(p_w), "recall_weighted": float(r_w), "f1_weighted": float(f_w),
        "num_classes": len(classes), "classes_present_in_split": int(len(set(y_true.tolist()))),
        "model": {k: meta[k] for k in ("backbone", "num_frames", "image_size", "use_attention",
                                       "use_mediapipe", "hidden", "layers", "dense_units")},
        "per_class": {c: report[c] for c in classes},
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    with open(out / "confusion_matrix.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["true\\pred"] + classes)
        for name, row in zip(classes, cm.tolist()):
            w.writerow([name] + row)
    save_confusion_png(cm, classes, out / "confusion_matrix.png")
    return metrics


def main() -> int:
    try:
        m = evaluate()
    except FileNotFoundError as exc:
        print(exc)
        return 2
    print(f"Test samples : {m['num_samples']}   (split mode: {m['split_mode']})")
    print(f"Accuracy     : {m['accuracy']:.4f}")
    print(f"Precision    : {m['precision_macro']:.4f} (macro)   Recall: {m['recall_macro']:.4f} (macro)   "
          f"F1: {m['f1_macro']:.4f} (macro)")
    for w in m["split_warnings"]:
        print("WARNING:", w)
    print("Saved metrics.json, confusion_matrix.csv and confusion_matrix.png in the models folder.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

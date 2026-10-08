"""
Measures REAL end-to-end inference latency of the trained model on real dataset videos.

Run from the backend folder (after training):
    python -m training.benchmark_latency --videos 20

Saves models/latency.json.  Nothing is estimated: every number comes from a timed run.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import statistics
import sys

from app.config import get_settings
from app.inference.recognizer import ModelNotTrainedError, Recognizer
from app.preprocessing.dataset_index import build_index


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", type=int, default=20)
    args = ap.parse_args()
    s = get_settings()
    try:
        rec = Recognizer(s.model_path)
    except ModelNotTrainedError as exc:
        print(exc)
        return 2
    index = build_index(s)
    if not index.available:
        print("DATASET PROBLEM:", index.message)
        return 2
    step = max(1, len(index.df) // args.videos)
    paths = list(index.df["video_path"].iloc[::step][:args.videos])
    rec.predict_video(paths[0])                     # warm-up run (not counted)
    totals, cnn, head = [], [], []
    for p in paths:
        r = rec.predict_video(p)["latency_ms"]
        totals.append(r["total_ms"]); cnn.append(r["cnn_ms"]); head.append(r["head_ms"])
    totals_sorted = sorted(totals)
    result = {
        "measured_at": dt.datetime.now().isoformat(timespec="seconds"),
        "videos": len(totals), "device": str(rec.pipeline.device), "frames_per_clip": rec.num_frames,
        "note": "Inference only (video file decoding is not included). Excludes browser and network time.",
        "total_ms": {"mean": statistics.mean(totals), "median": statistics.median(totals),
                     "p95": totals_sorted[min(len(totals_sorted) - 1, int(0.95 * len(totals_sorted)))],
                     "min": min(totals), "max": max(totals)},
        "cnn_ms_mean": statistics.mean(cnn), "head_ms_mean": statistics.mean(head),
    }
    (s.models_dir / "latency.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())

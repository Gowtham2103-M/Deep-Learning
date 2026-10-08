"""
Step 1 of training:  video -> 30 frames -> CNN (+ optional MediaPipe) -> feature file (.npy)

Because the CNN is frozen, features only need to be computed ONCE. Training the BiLSTM head
afterwards takes minutes instead of hours.

Run from the backend folder:
    python -m training.extract_features
    python -m training.extract_features --limit 10      (quick smoke test)
It is safe to stop and run again: videos that already have a feature file are skipped.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from app.config import Settings, get_settings
from app.inference.feature_pipeline import FeaturePipeline
from app.preprocessing.dataset_index import DatasetIndex, build_index
from app.preprocessing.frames import load_sampled_frames


def feature_dir(s: Settings, pipeline: FeaturePipeline) -> Path:
    return s.cache_dir / pipeline.signature(s.NUM_FRAMES)


def extract_all(s: Settings, index: DatasetIndex, pipeline: FeaturePipeline, limit: int = 0) -> dict:
    out_dir = feature_dir(s, pipeline)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = index.df if not limit else index.df.head(limit)
    failed, done, skipped = [], 0, 0
    t0 = time.time()
    for i, row in df.iterrows():
        f = out_dir / f"{i:06d}.npy"
        if f.exists():
            skipped += 1
            continue
        try:
            frames = load_sampled_frames(row["video_path"], s.NUM_FRAMES)
            np.save(f, pipeline.from_frames(frames))
            done += 1
        except Exception as exc:  # noqa
            failed.append({"row": int(i), "video": row["video_path"], "error": str(exc)})
        if (done + skipped + len(failed)) % 25 == 0:
            print(f"  processed {done + skipped + len(failed)}/{len(df)} videos "
                  f"({time.time() - t0:.0f}s elapsed)")
    report = {"cache_dir": str(out_dir), "extracted": done, "skipped_existing": skipped,
              "failed": failed, "signature": pipeline.signature(s.NUM_FRAMES)}
    (out_dir / "extract_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Extract CNN features for every dataset video.")
    ap.add_argument("--limit", type=int, default=0, help="only process the first N videos (test run)")
    args = ap.parse_args(argv)

    s = get_settings()
    index = build_index(s)
    if not index.available:
        print("\nDATASET PROBLEM:", index.message)
        return 2
    print(f"Dataset: {len(index.df)} labelled videos, {index.num_classes} classes "
          f"(CSV: {index.csv_path}, video column: {index.video_column}, label column: {index.label_column})")
    print(f"Loading CNN '{s.CNN_BACKBONE}' (pretrained={s.CNN_PRETRAINED}) ...")
    pipeline = FeaturePipeline.from_settings(s)
    print(f"Device: {pipeline.device} | frames per video: {s.NUM_FRAMES} | feature size: {pipeline.feature_dim}")
    report = extract_all(s, index, pipeline, args.limit)
    print(f"\nDone. New: {report['extracted']}, already cached: {report['skipped_existing']}, "
          f"failed: {len(report['failed'])}")
    for f in report["failed"][:5]:
        print("  FAILED:", f["video"], "->", f["error"])
    print("Features saved in:", report["cache_dir"])
    return 0


if __name__ == "__main__":
    sys.exit(main())

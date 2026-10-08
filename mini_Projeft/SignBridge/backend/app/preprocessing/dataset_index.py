"""
Reads the ISL-CSLTR folder DYNAMICALLY (nothing about the dataset is hard-coded).

It finds every video, finds the CSV that maps videos to sentence labels, guesses which
column is which (you can override this in config.json), and returns a clean table:

    video_path | label | signer (only if a signer-like column exists)

Everything here is READ-ONLY.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import pandas as pd

from app.config import Settings, get_settings

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".webm", ".mpg", ".mpeg", ".m4v"}
LABEL_HINTS = ("sentence", "label", "text", "class", "gloss", "caption", "transcript", "word")
SIGNER_HINTS = ("signer", "subject", "person", "participant", "speaker", "user", "actor", "volunteer")


@dataclass
class DatasetIndex:
    available: bool
    path: str
    message: str = ""
    df: pd.DataFrame = field(default_factory=lambda: pd.DataFrame(columns=["video_path", "label"]))
    classes: List[str] = field(default_factory=list)
    csv_path: Optional[str] = None
    video_column: Optional[str] = None
    label_column: Optional[str] = None
    signer_column: Optional[str] = None
    video_count: int = 0
    csv_count: int = 0
    missing_refs: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    @property
    def num_classes(self) -> int:
        return len(self.classes)

    def label_to_id(self) -> dict:
        return {c: i for i, c in enumerate(self.classes)}

    def summary(self) -> dict:
        return dict(available=self.available, path=self.path, message=self.message,
                    videos=self.video_count, csv_files=self.csv_count, csv=self.csv_path,
                    classes=self.num_classes, labelled_videos=int(len(self.df)),
                    video_column=self.video_column, label_column=self.label_column,
                    signer_column=self.signer_column, missing_refs=len(self.missing_refs),
                    warnings=self.warnings)


def find_files(root: Path):
    videos, csvs = [], []
    for folder, _, files in os.walk(root):
        for f in files:
            ext = os.path.splitext(f)[1].lower()
            full = os.path.join(folder, f)
            if ext in VIDEO_EXTS:
                videos.append(full)
            elif ext == ".csv":
                csvs.append(full)
    return sorted(videos), sorted(csvs)


def read_csv_safely(path: str):
    last = None
    for enc in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception as exc:  # noqa
            last = exc
    raise last  # type: ignore


def _base(v: str) -> str:
    return os.path.basename(str(v).replace("\\", "/")).strip().lower()


def _stem(v: str) -> str:
    return os.path.splitext(_base(v))[0]


def _folder_key(value: str) -> str:
    """Normalize sentence-folder labels conservatively (case and whitespace only)."""
    return re.sub(r"\s+", "", str(value)).casefold()


def _count_hits(df: pd.DataFrame, col, by_name: dict, by_stem: dict) -> int:
    vals = df[col].dropna().astype(str)
    return sum(1 for v in vals if _base(v) in by_name or _stem(v) in by_stem)


def _match_column(df: pd.DataFrame, by_name: dict, by_stem: dict):
    """Find the column whose values look like video file names. Returns (column, hits)."""
    best, best_hits = None, 0
    for col in df.columns:
        hits = _count_hits(df, col, by_name, by_stem)
        if hits > best_hits:
            best, best_hits = col, hits
    return best, best_hits


def build_index(settings: Optional[Settings] = None) -> DatasetIndex:
    s = settings or get_settings()
    root = s.dataset_dir
    if not root.is_dir():
        return DatasetIndex(False, str(root), message=(
            f"Dataset not found. Download ISL-CSLTR and place it in: {root}  "
            "(see dataset/README.md)."))

    videos, csvs = find_files(root)
    idx = DatasetIndex(True, str(root), video_count=len(videos), csv_count=len(csvs))
    if not videos:
        idx.available = False
        idx.message = f"No video files were found under {root}. Check dataset/README.md."
        return idx
    if not csvs:
        idx.available = False
        idx.message = "Videos were found but no CSV file with sentence labels was found."
        return idx

    by_name, by_stem = {}, {}
    for v in videos:
        by_name.setdefault(_base(v), []).append(v)
        by_stem.setdefault(_stem(v), []).append(v)
    dup = [k for k, v in by_name.items() if len(v) > 1]
    if dup:
        idx.warnings.append(f"{len(dup)} video file names exist in more than one folder; the first one is used.")

    # choose the CSV that references the most videos (or the one forced in config)
    candidates = csvs
    if s.CSV_FILE:
        forced = root / s.CSV_FILE
        if forced.is_file():
            candidates = [str(forced)]
        else:
            idx.warnings.append(f"CSV_FILE '{s.CSV_FILE}' not found; auto-detecting instead.")
    best = None
    for c in candidates:
        try:
            df = read_csv_safely(c)
        except Exception:
            continue
        if s.VIDEO_COLUMN and s.VIDEO_COLUMN in df.columns:
            col = s.VIDEO_COLUMN
            hits = _count_hits(df, col, by_name, by_stem)
        else:
            col, hits = _match_column(df, by_name, by_stem)
        if col and (best is None or hits > best[0]):
            best = (hits, c, df, col)

    # Some ISL-CSLRT releases have no per-video filename column: videos are
    # grouped in sentence-named folders and the CSV carries the sentence list.
    # Use that layout only when ordinary filename matching found no mapping.
    if not s.VIDEO_COLUMN:
        folder_dirs: dict[str, set[str]] = {}
        folder_videos: dict[str, list[str]] = {}
        for video in videos:
            folder = str(Path(video).parent)
            key = _folder_key(Path(video).parent.name)
            folder_dirs.setdefault(key, set()).add(folder)
            folder_videos.setdefault(key, []).append(video)
        ambiguous_folders = {key for key, paths in folder_dirs.items() if len(paths) > 1}

        folder_best = None
        for c in candidates:
            try:
                df = read_csv_safely(c)
            except Exception:
                continue
            lcol = s.LABEL_COLUMN if s.LABEL_COLUMN in df.columns else None
            if lcol is None:
                lcol = next((col for col in df.columns
                             if any(h in str(col).lower() for h in LABEL_HINTS)), None)
            if lcol is None:
                continue
            labels = df[lcol].dropna().astype(str)
            hits = sum(_folder_key(label) in folder_videos and
                       _folder_key(label) not in ambiguous_folders for label in labels)
            if hits and (folder_best is None or hits > folder_best[0]):
                folder_best = (hits, c, df, lcol, folder_videos, ambiguous_folders)

        if folder_best is not None and (best is None or folder_best[0] > best[0]):
            _, csv_path, df, lcol, folder_videos, ambiguous_folders = folder_best
            idx.csv_path = os.path.relpath(csv_path, root)
            idx.label_column = lcol
            idx.video_column = None
            scol = s.SIGNER_COLUMN if s.SIGNER_COLUMN in df.columns else None
            if scol is None:
                scol = next((col for col in df.columns if col != lcol and
                             any(h in str(col).lower() for h in SIGNER_HINTS)), None)
            idx.signer_column = scol

            rows = []
            matched_keys = set()
            for _, r in df.iterrows():
                if pd.isna(r[lcol]):
                    continue
                label = str(r[lcol]).strip()
                key = _folder_key(label)
                matches = folder_videos.get(key, []) if key not in ambiguous_folders else []
                if not matches:
                    idx.missing_refs.append(label)
                    continue
                matched_keys.add(key)
                for video in matches:
                    row = {"video_path": video, "label": label}
                    if scol and not pd.isna(r[scol]):
                        row["signer"] = str(r[scol]).strip()
                    rows.append(row)

            if idx.missing_refs:
                idx.warnings.append(f"{len(idx.missing_refs)} CSV sentence labels have no unambiguous video folder.")
            unmatched = [key for key in folder_videos
                         if key not in matched_keys and key not in ambiguous_folders]
            if unmatched:
                idx.warnings.append(f"{sum(len(folder_videos[key]) for key in unmatched)} videos are in folders not listed in the CSV.")
            if ambiguous_folders:
                idx.warnings.append(f"{len(ambiguous_folders)} sentence folder names are ambiguous; their videos were skipped.")

            out = pd.DataFrame(rows)
            if out.empty:
                idx.available = False
                idx.message = "The CSV was found but none of its sentence labels matched a video folder."
                return idx
            out = out.drop_duplicates(subset=["video_path"]).sort_values("video_path").reset_index(drop=True)
            idx.df = out
            idx.classes = sorted(out["label"].unique())
            if s.NUM_CLASSES and s.NUM_CLASSES != idx.num_classes:
                idx.warnings.append(f"NUM_CLASSES={s.NUM_CLASSES} in config but the CSV has {idx.num_classes}; "
                                    "the value found in the data is used.")
            idx.message = "Dataset loaded from sentence-named video folders."
            return idx

    if best is None:
        idx.available = False
        idx.message = ("No CSV column matched video file names or sentence folders. "
                       "Set CSV_FILE / VIDEO_COLUMN / LABEL_COLUMN in config.json.")
        return idx

    _, csv_path, df, vcol = best
    idx.csv_path = os.path.relpath(csv_path, root)
    idx.video_column = vcol

    others = [c for c in df.columns if c != vcol]
    lcol = s.LABEL_COLUMN if (s.LABEL_COLUMN in df.columns) else None
    if lcol is None:
        lcol = next((c for c in others if any(h in str(c).lower() for h in LABEL_HINTS)), None)
    if lcol is None:
        obj = [c for c in others if df[c].dtype == object and 1 < df[c].nunique() < 0.9 * len(df)]
        lcol = max(obj, key=lambda c: df[c].nunique()) if obj else None
    if lcol is None:
        idx.available = False
        idx.message = "Could not find the sentence-label column. Set LABEL_COLUMN in config.json."
        return idx
    idx.label_column = lcol

    scol = s.SIGNER_COLUMN if (s.SIGNER_COLUMN in df.columns) else None
    if scol is None:
        scol = next((c for c in others if c != lcol and any(h in str(c).lower() for h in SIGNER_HINTS)), None)
    idx.signer_column = scol

    rows = []
    for _, r in df.iterrows():
        ref = r[vcol]
        if pd.isna(ref) or pd.isna(r[lcol]):
            continue
        ref = str(ref)
        path = None
        direct = root / ref.replace("\\", "/")
        if direct.is_file():
            path = str(direct)
        elif _base(ref) in by_name:
            path = by_name[_base(ref)][0]
        elif _stem(ref) in by_stem:
            path = by_stem[_stem(ref)][0]
        if path is None:
            idx.missing_refs.append(ref)
            continue
        row = {"video_path": path, "label": str(r[lcol]).strip()}
        if scol:
            row["signer"] = str(r[scol]).strip()
        rows.append(row)

    out = pd.DataFrame(rows)
    if out.empty:
        idx.available = False
        idx.message = "The CSV was found but none of its rows could be matched to a video file."
        return idx
    out = out.drop_duplicates(subset=["video_path"]).sort_values("video_path").reset_index(drop=True)
    idx.df = out
    idx.classes = sorted(out["label"].unique())
    if s.NUM_CLASSES and s.NUM_CLASSES != idx.num_classes:
        idx.warnings.append(f"NUM_CLASSES={s.NUM_CLASSES} in config but the CSV has {idx.num_classes}; "
                            "the value found in the data is used.")
    if idx.missing_refs:
        idx.warnings.append(f"{len(idx.missing_refs)} CSV rows point to videos that do not exist.")
    idx.message = "Dataset loaded."
    return idx

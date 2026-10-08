"""
Training data helpers:
  * make_splits  - train / validation / test split (signer-independent when signer IDs exist)
  * FeatureDataset - loads the cached CNN feature sequences (.npy) for a list of rows
"""
from __future__ import annotations

import hashlib
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset


def video_sha256(path: str | Path) -> str:
    """Return a content hash for a video without modifying it."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as video:
        for chunk in iter(lambda: video.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def find_conflicting_duplicate_rows(df: pd.DataFrame) -> Tuple[List[dict], Dict[str, List[int]]]:
    """Find identical video files assigned different labels; report original row indices."""
    groups: Dict[str, List[int]] = {}
    for row_id, row in df.iterrows():
        groups.setdefault(video_sha256(row["video_path"]), []).append(int(row_id))

    conflicts = []
    for digest, row_ids in groups.items():
        if len({str(df.loc[row_id, "label"]) for row_id in row_ids}) > 1:
            conflicts.append({
                "sha256": digest,
                "rows": [
                    {"row": row_id, "video_path": str(df.loc[row_id, "video_path"]),
                     "label": str(df.loc[row_id, "label"])}
                    for row_id in row_ids
                ],
            })
    return conflicts, groups


def make_grouped_splits(df: pd.DataFrame, val_fraction: float, test_fraction: float,
                        seed: int) -> Tuple[Dict[str, np.ndarray], Dict]:
    """Split whole video-content groups, stratifying by label where counts allow."""
    if not 0 < val_fraction < 1 or not 0 < test_fraction < 1 or val_fraction + test_fraction >= 1:
        raise ValueError("Validation and test fractions must be positive and sum to less than one.")

    content_groups: Dict[str, List[int]] = {}
    group_labels: Dict[str, str] = {}
    for position, (_, row) in enumerate(df.iterrows()):
        digest = str(row["__video_sha256"]) if "__video_sha256" in df.columns else video_sha256(row["video_path"])
        label = str(row["label"])
        previous = group_labels.setdefault(digest, label)
        if previous != label:
            raise ValueError(f"Conflicting duplicate video reached split: {digest} ({previous!r}, {label!r}).")
        content_groups.setdefault(digest, []).append(position)

    by_label: Dict[str, List[List[int]]] = {}
    for digest, positions in content_groups.items():
        by_label.setdefault(group_labels[digest], []).append(positions)

    rng = random.Random(seed)
    split_groups: Dict[str, List[List[int]]] = {"train": [], "val": [], "test": []}
    insufficient: Dict[str, dict] = {}
    for label in sorted(by_label):
        groups = by_label[label]
        rng.shuffle(groups)
        count = len(groups)
        if count >= 3:
            n_test = max(1, round(count * test_fraction))
            n_val = max(1, round(count * val_fraction))
            while n_test + n_val >= count:
                if n_val > 1:
                    n_val -= 1
                else:
                    n_test -= 1
            split_groups["test"].extend(groups[:n_test])
            split_groups["val"].extend(groups[n_test:n_test + n_val])
            split_groups["train"].extend(groups[n_test + n_val:])
        elif count == 2:
            split_groups["train"].append(groups[0])
            split_groups["val"].append(groups[1])
            insufficient[label] = {"content_groups": count, "missing_partitions": ["test"]}
        else:
            split_groups["train"].append(groups[0])
            insufficient[label] = {"content_groups": count, "missing_partitions": ["val", "test"]}

    splits = {
        name: np.array(sorted(position for group in groups for position in group), dtype=int)
        for name, groups in split_groups.items()
    }
    all_positions = [int(position) for values in splits.values() for position in values]
    if len(all_positions) != len(df) or len(set(all_positions)) != len(df):
        raise RuntimeError("Grouped split did not assign every training row exactly once.")

    content_to_split = {}
    for name, groups in split_groups.items():
        for group in groups:
            row = df.iloc[group[0]]
            digest = str(row["__video_sha256"]) if "__video_sha256" in df.columns else video_sha256(row["video_path"])
            content_to_split.setdefault(digest, set()).add(name)
    if any(len(names) != 1 for names in content_to_split.values()):
        raise RuntimeError("Identical video content was assigned to more than one split.")

    class_coverage = {
        name: sorted({str(df.iloc[position]["label"]) for position in positions})
        for name, positions in splits.items()
    }
    usable_labels = set(map(str, df["label"].unique()))
    if set(class_coverage["train"]) != usable_labels:
        raise RuntimeError("At least one usable class is missing from the training partition.")

    info = {
        "mode": "deterministic_stratified_content_grouped",
        "seed": seed,
        "warnings": [],
        "class_coverage": {name: len(labels) for name, labels in class_coverage.items()},
        "classes_by_split": class_coverage,
        "insufficient_samples_for_all_partitions": insufficient,
        "content_groups": len(content_groups),
    }
    return splits, info


def make_splits(df: pd.DataFrame, val_fraction: float, test_fraction: float, seed: int
                ) -> Tuple[Dict[str, np.ndarray], Dict]:
    """Return ({'train': idx, 'val': idx, 'test': idx}, info).  idx = row numbers of df."""
    warnings: List[str] = []
    n = len(df)
    rows = np.arange(n)
    have_signers = "signer" in df.columns and df["signer"].nunique() >= 3

    if have_signers:
        signers = sorted(df["signer"].unique())
        rng = np.random.RandomState(seed)
        rng.shuffle(signers)
        n_test = max(1, int(round(len(signers) * test_fraction)))
        n_val = max(1, int(round(len(signers) * val_fraction)))
        test_s = set(signers[:n_test])
        val_s = set(signers[n_test:n_test + n_val])
        train_s = set(signers[n_test + n_val:])
        if not train_s:
            have_signers = False
            warnings.append("Too few signers for a signer-independent split; using a random split.")
        else:
            splits = {"train": rows[df["signer"].isin(train_s).values],
                      "val": rows[df["signer"].isin(val_s).values],
                      "test": rows[df["signer"].isin(test_s).values]}
            train_labels = set(df.iloc[splits["train"]]["label"])
            for name in ("val", "test"):
                miss = set(df.iloc[splits[name]]["label"]) - train_labels
                if miss:
                    warnings.append(f"{len(miss)} classes in the {name} split never appear in training.")
            info = {"mode": "signer_independent", "warnings": warnings,
                    "test_signers": sorted(test_s), "val_signers": sorted(val_s), "train_signers": sorted(train_s)}
            return splits, info

    # ---- random (stratified when possible) split ----
    warnings.append("No usable signer ID column: the split may contain the SAME signer in train and test, "
                    "so accuracy can be optimistic. Do not present it as signer-independent.")
    labels = df["label"].values
    counts = pd.Series(labels).value_counts()
    n_classes = counts.size

    def _split(idx, size, stratify):
        return train_test_split(idx, test_size=size, random_state=seed,
                                stratify=labels[idx] if stratify else None)

    can_strat = counts.min() >= 3 and int(round(n * test_fraction)) >= n_classes
    try:
        rest, test = _split(rows, test_fraction, can_strat)
        val_size = val_fraction / (1.0 - test_fraction)
        can_strat2 = pd.Series(labels[rest]).value_counts().min() >= 2 and int(round(len(rest) * val_size)) >= n_classes
        train, val = _split(rest, val_size, can_strat2)
        if not (can_strat and can_strat2):
            warnings.append("Some classes have very few videos; the split is not perfectly stratified.")
    except ValueError:
        rest, test = _split(rows, test_fraction, False)
        train, val = _split(rest, val_fraction / (1.0 - test_fraction), False)
        warnings.append("Stratified splitting was not possible; used a plain random split.")
    splits = {"train": np.sort(train), "val": np.sort(val), "test": np.sort(test)}
    return splits, {"mode": "random_stratified" if can_strat else "random", "warnings": warnings}


class FeatureDataset(Dataset):
    """Loads (T, D) feature sequences saved by extract_features.py."""

    def __init__(self, cache_dir: Path, row_ids: np.ndarray, label_ids: np.ndarray):
        self.cache_dir = Path(cache_dir)
        self.row_ids = list(map(int, row_ids))
        self.labels = list(map(int, label_ids))

    def __len__(self):
        return len(self.row_ids)

    def __getitem__(self, i):
        x = np.load(self.cache_dir / f"{self.row_ids[i]:06d}.npy")
        return torch.from_numpy(x), torch.tensor(self.labels[i], dtype=torch.long)

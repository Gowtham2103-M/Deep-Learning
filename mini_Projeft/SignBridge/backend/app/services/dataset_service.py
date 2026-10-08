"""Cached access to the dataset index, reference videos and the sentence matcher."""
from __future__ import annotations

from typing import Dict, Optional

from app.config import get_settings
from app.preprocessing.dataset_index import DatasetIndex, build_index
from app.services.matcher import SentenceMatcher

_index: Optional[DatasetIndex] = None
_matcher: Optional[SentenceMatcher] = None
_ref_videos: Dict[int, str] = {}


def get_index(refresh: bool = False) -> DatasetIndex:
    global _index, _matcher, _ref_videos
    if _index is None or refresh:
        _index = build_index(get_settings())
        _matcher, _ref_videos = None, {}
        if _index.available:
            _matcher = SentenceMatcher(_index.classes, get_settings().MATCH_THRESHOLD)
            l2i = _index.label_to_id()
            for _, row in _index.df.iterrows():          # first video (sorted by path) per sentence
                _ref_videos.setdefault(l2i[row["label"]], row["video_path"])
    return _index


def get_matcher() -> Optional[SentenceMatcher]:
    get_index()
    return _matcher


def reference_video(class_id: int) -> Optional[str]:
    get_index()
    return _ref_videos.get(class_id)


def reset_cache() -> None:
    global _index, _matcher, _ref_videos
    _index, _matcher, _ref_videos = None, None, {}

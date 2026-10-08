"""
Speech text -> known ISL sentence.

Order of attempts:  1) exact match   2) normalised match   3) fuzzy match
If nothing is similar enough we return None - we NEVER show a wrong sign.
"""
from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import List, Optional

from app.utils.text import normalize_text


@dataclass
class Match:
    class_id: int
    label: str
    score: float
    method: str          # exact | normalized | fuzzy


class SentenceMatcher:
    def __init__(self, labels: List[str], threshold: float = 0.75):
        self.labels = labels
        self.threshold = threshold
        self.norm = [normalize_text(l) for l in labels]

    @staticmethod
    def _similarity(a: str, b: str) -> float:
        """0..1 similarity. Character-level ratio tolerates small speech-recognition typos
        ('mornin' vs 'morning'); the sorted-words ratio tolerates a different word order
        (slightly penalised, because word order can change the meaning)."""
        if not a or not b:
            return 0.0
        chars = SequenceMatcher(None, a, b).ratio()
        a_sorted, b_sorted = " ".join(sorted(a.split())), " ".join(sorted(b.split()))
        words = 0.95 * SequenceMatcher(None, a_sorted, b_sorted).ratio()
        return max(chars, words)

    def best_candidate(self, text: str):
        """(class_id, score) of the closest sentence, even if below the threshold."""
        q = normalize_text(text)
        if not q or not self.labels:
            return None, 0.0
        scores = [self._similarity(q, n) for n in self.norm]
        i = max(range(len(scores)), key=scores.__getitem__)
        return i, scores[i]

    def match(self, text: str) -> Optional[Match]:
        if not text or not text.strip() or not self.labels:
            return None
        raw = text.strip()
        for i, l in enumerate(self.labels):
            if raw == l:
                return Match(i, l, 1.0, "exact")
        q = normalize_text(raw)
        for i, n in enumerate(self.norm):
            if q and q == n:
                return Match(i, self.labels[i], 1.0, "normalized")
        i, score = self.best_candidate(raw)
        if i is not None and score >= self.threshold:
            return Match(i, self.labels[i], round(score, 3), "fuzzy")
        return None

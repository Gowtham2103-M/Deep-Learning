"""Sentence matching for speech -> ISL."""
from app.services.matcher import SentenceMatcher
from app.utils.text import normalize_text

LABELS = ["Hello how are you", "Thank you", "Good morning", "What is your name"]


def test_normalize_text():
    assert normalize_text("  Thank   YOU!! ") == "thank you"
    assert normalize_text("What's your name?") == "whats your name"


def test_exact_normalized_and_fuzzy():
    m = SentenceMatcher(LABELS, threshold=0.7)
    assert m.match("Thank you").method == "exact"
    r = m.match("thank you!")
    assert r.method == "normalized" and r.label == "Thank you"
    r = m.match("what is you name")
    assert r.method == "fuzzy" and r.label == "What is your name"


def test_no_match_returns_none():
    m = SentenceMatcher(LABELS, threshold=0.75)
    assert m.match("the elephant flew over the moon") is None
    assert m.match("") is None
    assert m.match("   ") is None

"""Small text helpers shared by the sentence matcher and the API."""
import re
import unicodedata


def normalize_text(text: str) -> str:
    """lower-case, remove punctuation, collapse spaces.  'Thank  you!' -> 'thank you'"""
    text = unicodedata.normalize("NFKC", str(text)).lower()
    text = text.replace("'", "").replace("\u2019", "")
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    text = text.replace("_", " ")
    return re.sub(r"\s+", " ", text).strip()

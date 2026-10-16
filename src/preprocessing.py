"""Text preprocessing for the Dholuo topic classifier.

This module reproduces, rule for rule, the preprocessing defined in the training
notebook (`PREPROC_SRC`, section 5 of notebooks/dholuo_topic_classification_final.ipynb)
and exported with the model as `models/final/preprocess.py`. The deployed model must
receive text processed exactly as it was during training, so any change here must be
mirrored in the notebook. `tests/test_preprocessing.py` checks parity against the
notebook source.
"""

import re
import unicodedata

# Transcription markers such as [cs], [CS], [pause], [?] (up to 20 characters inside brackets).
TAG_RE = re.compile(r"\[\s*([^\]\[]{0,20}?)\s*\]")

# Curly and other apostrophe-like characters are mapped to a straight apostrophe.
APOS_MAP = str.maketrans({"’": "'", "‘": "'", "`": "'", "´": "'", "ʼ": "'"})

_WS = re.compile(r"\s+")
_SPACE_BEFORE_PUNCT = re.compile(r"\s+([,.;:!?])")
_NO_SPACE_AFTER_PUNCT = re.compile(r"([,.;:!?])(?=[^\s\d,.;:!?<'\"])")


def canonical_tag(inner: str) -> str:
    """Map a transcription marker to one canonical token; unknown markers become <tag>."""
    t = inner.strip().lower().replace(" ", "")
    if t in ("cs", "codeswitch", "code-switch"):
        return " <cs> "
    if t in ("pause", "pauses"):
        return " <pause> "
    if t in ("?", "??", "unclear", "inaudible"):
        return " <unclear> "
    return " <tag> "


def normalize_minimal(text: str) -> str:
    """Unicode NFC and whitespace normalisation only (the 'minimal' variant)."""
    t = unicodedata.normalize("NFC", str(text))
    return _WS.sub(" ", t).strip()


def normalize_full(text: str, keep_tags: bool = True) -> str:
    """NFC, straight apostrophes, canonical markers, lower case and consistent punctuation spacing."""
    t = unicodedata.normalize("NFC", str(text)).translate(APOS_MAP)
    t = TAG_RE.sub(lambda m: canonical_tag(m.group(1)) if keep_tags else " ", t)
    t = t.lower()
    t = _SPACE_BEFORE_PUNCT.sub(r"\1", t)
    t = _NO_SPACE_AFTER_PUNCT.sub(r"\1 ", t)
    return _WS.sub(" ", t).strip()


TEXT_VARIANTS = {
    "minimal": normalize_minimal,
    "normalized": lambda t: normalize_full(t, keep_tags=True),
    "normalized_notags": lambda t: normalize_full(t, keep_tags=False),
}

TOKEN_RE = re.compile(r"<[a-z_]+>|\[[^\]]*\]|[\w']+|[^\w\s]")
WORD_PATTERN = r"<[a-z_]+>|\[[^\]]*\]|[\w']+"


def tokenize(text: str) -> list:
    """Split normalised text into word, marker and punctuation tokens."""
    return TOKEN_RE.findall(text)


def word_tokens(text: str) -> list:
    """Word and marker tokens only (the units the TF-IDF word model was trained on)."""
    return re.findall(WORD_PATTERN, text)


def get_normalizer(variant: str):
    """Return the normalisation function for a text variant name stored in label_map.json."""
    try:
        return TEXT_VARIANTS[variant]
    except KeyError as exc:
        raise ValueError(f"Unknown text variant '{variant}'. Expected one of {sorted(TEXT_VARIANTS)}.") from exc

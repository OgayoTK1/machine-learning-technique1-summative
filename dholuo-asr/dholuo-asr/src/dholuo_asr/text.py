"""Transcript normalisation for Dholuo.

Two transcript "views" are used in this project (Experiment B):

* ``minimal`` – Unicode NFC + whitespace only. Keeps case and punctuation, i.e. the
  orthographic transcript as distributed.
* ``normalised`` – lower-case, apostrophe variants unified, punctuation removed,
  bracketed non-speech tags removed.

Evaluation ALWAYS applies :func:`eval_normalise` to both reference and hypothesis,
so models trained on different transcript views are scored on identical terms.

Dholuo-specific decision: the apostrophe is NOT punctuation. In Dholuo orthography
``ng'`` (as in *ng'ato*, "someone") is a distinct consonant from ``ng``. Deleting the
apostrophe would merge two phonemes, so it is kept and only its typographic variants
(’ ‘ ` ´ ʼ) are unified to ASCII ``'``.
"""
from __future__ import annotations

import re
import unicodedata

APOSTROPHE_VARIANTS = "’‘`´ʼʻ′"
_APOS_TABLE = str.maketrans({c: "'" for c in APOSTROPHE_VARIANTS})

# Non-speech / annotation tags such as [noise], <unk>, (laughs), {breath}
TAG_RE = re.compile(r"\[[^\]]*\]|<[^>]*>|\{[^}]*\}|\([^)]*\)")
# Anything that is not a letter, digit, apostrophe, hyphen or whitespace
PUNCT_RE = re.compile(r"[^\w\s'\-]", flags=re.UNICODE)
WS_RE = re.compile(r"\s+")


def nfc(text: str) -> str:
    return unicodedata.normalize("NFC", text)


def minimal_normalise(text: str) -> str:
    """NFC + collapse whitespace. Nothing linguistic is changed."""
    if text is None:
        return ""
    return WS_RE.sub(" ", nfc(str(text))).strip()


def normalise(text: str, *, remove_tags: bool = True, keep_hyphen: bool = False) -> str:
    """Training-side normalisation (the ``normalised`` transcript view)."""
    t = minimal_normalise(text)
    t = t.translate(_APOS_TABLE)
    if remove_tags:
        t = TAG_RE.sub(" ", t)
    t = t.lower()
    t = PUNCT_RE.sub(" ", t)
    t = t.replace("_", " ")
    if not keep_hyphen:
        t = re.sub(r"-", " ", t)
    # apostrophes that are not attached to a letter on the left are stray quotes
    t = re.sub(r"(^|\s)'+", r"\1", t)
    t = re.sub(r"'{2,}", "'", t)
    return WS_RE.sub(" ", t).strip()


def eval_normalise(text: str) -> str:
    """Normaliser applied to BOTH reference and hypothesis before scoring."""
    return normalise(text, remove_tags=True, keep_hyphen=False)


VIEWS = {"minimal": minimal_normalise, "normalised": normalise}


def apply_view(text: str, view: str) -> str:
    if view not in VIEWS:
        raise ValueError(f"unknown transcript view {view!r}; choose from {list(VIEWS)}")
    return VIEWS[view](text)


def words(text: str) -> list[str]:
    return [w for w in text.split(" ") if w]


# --------------------------------------------------------------------------------------
# Audit helpers
# --------------------------------------------------------------------------------------
def char_categories(text: str) -> dict[str, int]:
    """Count characters by Unicode general category group (for the audit)."""
    out = {"letter": 0, "digit": 0, "space": 0, "apostrophe": 0, "punct": 0, "other": 0}
    for ch in text:
        if ch == "'" or ch in APOSTROPHE_VARIANTS:
            out["apostrophe"] += 1
        elif ch.isalpha():
            out["letter"] += 1
        elif ch.isdigit():
            out["digit"] += 1
        elif ch.isspace():
            out["space"] += 1
        elif unicodedata.category(ch).startswith("P"):
            out["punct"] += 1
        else:
            out["other"] += 1
    return out


def has_tags(text: str) -> bool:
    return bool(TAG_RE.search(text or ""))


def non_ascii_letters(text: str) -> set[str]:
    return {c for c in nfc(text or "") if c.isalpha() and not c.isascii()}

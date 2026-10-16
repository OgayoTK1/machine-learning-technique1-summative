
import re, unicodedata

TAG_RE = re.compile(r"\[\s*([^\]\[]{0,20}?)\s*\]")
APOS_MAP = str.maketrans({"’": "'", "‘": "'", "`": "'", "´": "'", "ʼ": "'"})
_WS = re.compile(r"\s+")
_SPACE_BEFORE_PUNCT = re.compile(r"\s+([,.;:!?])")
_NO_SPACE_AFTER_PUNCT = re.compile(r"([,.;:!?])(?=[^\s\d,.;:!?<'\"])")

def canonical_tag(inner):
    """Map transcription markers to one canonical token each; unknown markers become <tag>."""
    t = inner.strip().lower().replace(" ", "")
    if t in ("cs", "codeswitch", "code-switch"):
        return " <cs> "
    if t in ("pause", "pauses"):
        return " <pause> "
    if t in ("?", "??", "unclear", "inaudible"):
        return " <unclear> "
    return " <tag> "

def normalize_minimal(text):
    """Control condition for Experiment B: Unicode NFC + whitespace only."""
    t = unicodedata.normalize("NFC", str(text))
    return _WS.sub(" ", t).strip()

def normalize_full(text, keep_tags=True):
    """NFC, straight apostrophes, canonical markers, lower-case, consistent punctuation spacing."""
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

def tokenize(text):
    return TOKEN_RE.findall(text)

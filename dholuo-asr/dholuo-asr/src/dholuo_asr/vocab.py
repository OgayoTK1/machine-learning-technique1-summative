"""Character vocabulary for CTC models (built from TRAINING transcripts only)."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

PAD, UNK, WORD_DELIM = "[PAD]", "[UNK]", "|"


def build_char_vocab(texts, min_count: int = 1) -> dict[str, int]:
    """PAD=0 (also the CTC blank for HF wav2vec2 models), UNK=1, then '|' and chars."""
    counts = Counter(ch for t in texts for ch in t.replace(" ", WORD_DELIM))
    chars = sorted(c for c, n in counts.items() if n >= min_count and c != WORD_DELIM)
    vocab = {PAD: 0, UNK: 1, WORD_DELIM: 2}
    for c in chars:
        vocab[c] = len(vocab)
    return vocab


def oov_chars(texts, vocab: dict[str, int]) -> Counter:
    return Counter(ch for t in texts for ch in t.replace(" ", WORD_DELIM) if ch not in vocab)


def save_vocab(vocab: dict[str, int], path: str | Path) -> Path:
    path = Path(path)
    path.write_text(json.dumps(vocab, ensure_ascii=False, indent=1), encoding="utf-8")
    return path


class CharTokenizer:
    """Minimal tokenizer used by the from-scratch BiLSTM-CTC model."""

    def __init__(self, vocab: dict[str, int]):
        self.vocab = vocab
        self.inv = {i: c for c, i in vocab.items()}
        self.blank_id = vocab[PAD]

    def encode(self, text: str) -> list[int]:
        unk = self.vocab[UNK]
        return [self.vocab.get(c, unk) for c in text.replace(" ", WORD_DELIM)]

    def decode_ids(self, ids) -> str:
        """Greedy CTC collapse: merge repeats, drop blanks."""
        out, prev = [], None
        for i in ids:
            i = int(i)
            if i != prev and i != self.blank_id:
                out.append(self.inv.get(i, ""))
            prev = i
        s = "".join(c for c in out if c not in (PAD, UNK)).replace(WORD_DELIM, " ")
        return " ".join(s.split())

    def __len__(self):
        return len(self.vocab)

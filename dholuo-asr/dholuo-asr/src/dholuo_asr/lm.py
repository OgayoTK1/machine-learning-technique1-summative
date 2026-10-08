"""Word n-gram language model written directly as an ARPA file.

Why not KenLM's ``lmplz``? It must be compiled from C++ source, which is fragile on
Kaggle. This module builds an interpolated absolute-discounting back-off n-gram model in
pure Python and serialises it as a standard ARPA file. KenLM's Python bindings (or
pyctcdecode) can then load the ARPA file for beam-search decoding.

Model (order k >= 2, history h, word w):
    p(w | h) = max(c(hw) - D_k, 0) / c(h) + D_k * N1+(h) / c(h) * p(w | h')
where h' drops the oldest word. Unigrams use add-delta smoothing over V ∪ {<unk>}.
Back-off weights are computed exactly so every conditional distribution sums to 1
(this is verified by the unit tests).

LEAKAGE WARNING: fit the LM on TRAINING transcripts only. If test prompts also appear in
the training transcripts (shared read prompts), an LM will partly memorise them; report
LM results on the unseen-prompt test subset as the primary number.
"""
from __future__ import annotations

import math
from collections import Counter, defaultdict
from pathlib import Path

BOS, EOS, UNK = "<s>", "</s>", "<unk>"


class NGramLM:
    def __init__(self, order: int = 3, delta: float = 0.01):
        self.order = order
        self.delta = delta
        self.logp: list[dict[tuple, float]] = []  # per order (index 0 = unigrams)
        self.bow: list[dict[tuple, float]] = []

    # ------------------------------------------------------------------ fitting
    def fit(self, sentences: list[list[str]]) -> "NGramLM":
        N = self.order
        counts = [Counter() for _ in range(N)]
        for s in sentences:
            toks = [BOS] + list(s) + [EOS]
            for k in range(1, N + 1):
                for i in range(len(toks) - k + 1):
                    ng = tuple(toks[i:i + k])
                    if k == 1 and ng[0] == BOS:
                        continue
                    counts[k - 1][ng] += 1
        # discounts per order (Ney et al. estimate, clamped)
        self.discounts = []
        for k in range(N):
            cc = Counter(counts[k].values())
            n1, n2 = cc.get(1, 0), cc.get(2, 0)
            D = n1 / (n1 + 2 * n2) if (n1 + 2 * n2) else 0.5
            self.discounts.append(min(max(D, 0.1), 0.9))

        # unigrams: add-delta over vocabulary incl. </s> and <unk>
        vocab = {ng[0] for ng in counts[0]} | {EOS, UNK}
        self.vocab = sorted(vocab)
        total = sum(counts[0].values())
        denom = total + self.delta * len(vocab)
        p1 = {(w,): (counts[0].get((w,), 0) + self.delta) / denom for w in vocab}
        self.logp = [{k: math.log10(v) for k, v in p1.items()}]
        self.logp[0][(BOS,)] = -99.0
        self.bow = [dict() for _ in range(N)]

        for k in range(2, N + 1):
            D = self.discounts[k - 1]
            by_hist: dict[tuple, dict[str, int]] = defaultdict(dict)
            for ng, c in counts[k - 1].items():
                by_hist[ng[:-1]][ng[-1]] = c
            lp_k = {}
            for h, nexts in by_hist.items():
                ch = sum(nexts.values())
                lam = D * len(nexts) / ch
                seen_p, seen_lower = 0.0, 0.0
                for w, c in nexts.items():
                    lower = 10 ** self.score(w, h[1:])
                    p = max(c - D, 0) / ch + lam * lower
                    lp_k[h + (w,)] = math.log10(p)
                    seen_p += p
                    seen_lower += lower
                num, den = max(1.0 - seen_p, 1e-12), max(1.0 - seen_lower, 1e-12)
                self.bow[k - 2][h] = math.log10(num / den)
            self.logp.append(lp_k)
        return self

    # ------------------------------------------------------------------ scoring
    def score(self, w: str, h: tuple) -> float:
        """log10 p(w | h) using ARPA back-off semantics."""
        if (w,) not in self.logp[0] or w == BOS:
            w = UNK
        h = tuple(x if (x,) in self.logp[0] else UNK for x in h)[-(self.order - 1):] if self.order > 1 else ()
        ng = h + (w,)
        k = len(ng)
        if k == 1:
            return self.logp[0][ng]
        if ng in self.logp[k - 1]:
            return self.logp[k - 1][ng]
        return self.bow[k - 2].get(h, 0.0) + self.score(w, h[1:])

    def sentence_logprob(self, words: list[str]) -> float:
        toks = [BOS] + list(words) + [EOS]
        lp = 0.0
        for i in range(1, len(toks)):
            lp += self.score(toks[i], tuple(toks[max(0, i - self.order + 1):i]))
        return lp

    def perplexity(self, sentences: list[list[str]]) -> float:
        lp, n = 0.0, 0
        for s in sentences:
            lp += self.sentence_logprob(s)
            n += len(s) + 1
        return 10 ** (-lp / max(n, 1))

    # ------------------------------------------------------------------ export
    def write_arpa(self, path: str | Path) -> Path:
        path = Path(path)
        with open(path, "w", encoding="utf-8") as f:
            f.write("\\data\\\n")
            for k in range(1, self.order + 1):
                f.write(f"ngram {k}={len(self.logp[k - 1])}\n")
            for k in range(1, self.order + 1):
                f.write(f"\n\\{k}-grams:\n")
                for ng in sorted(self.logp[k - 1]):
                    line = f"{self.logp[k - 1][ng]:.6f}\t{' '.join(ng)}"
                    if k < self.order:
                        b = self.bow[k - 1].get(ng)
                        if b is not None:
                            line += f"\t{b:.6f}"
                    f.write(line + "\n")
            f.write("\n\\end\\\n")
        return path

    def unigrams(self) -> list[str]:
        return [w for w in self.vocab if w not in (BOS, EOS, UNK)]

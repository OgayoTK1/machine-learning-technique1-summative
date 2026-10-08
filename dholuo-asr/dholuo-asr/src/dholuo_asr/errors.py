"""Systematic ASR error analysis built on word/char alignments."""
from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd

from .metrics import align, char_align
from .text import eval_normalise

FREQ_BUCKETS = [(-1, 0, "unseen (OOV)"), (1, 2, "1-2"), (3, 10, "3-10"), (11, 100, "11-100"),
                (101, 10**12, ">100")]


def freq_bucket(n: int) -> str:
    for lo, hi, name in FREQ_BUCKETS:
        if lo <= n <= hi:
            return name
    return "?"


def ref_word_outcomes(preds: pd.DataFrame, train_counts: Counter) -> pd.DataFrame:
    """One row per REFERENCE word: was it recognised, and how often was it seen in training.

    A reference word counts as correct only if aligned to an identical hypothesis word.
    """
    rows = []
    for _, r in preds.iterrows():
        ref, hyp = eval_normalise(r["ref"]).split(), eval_normalise(r["hyp"]).split()
        for op, rw, hw in align(ref, hyp).ops:
            if rw is None:
                continue
            rows.append({"utt_id": r["utt_id"], "word": rw, "op": op, "hyp_word": hw,
                         "correct": op == "=", "train_count": train_counts.get(rw, 0),
                         "has_apostrophe": "'" in rw, "word_len": len(rw)})
    out = pd.DataFrame(rows)
    out["freq_bucket"] = out["train_count"].map(freq_bucket)
    return out


def segmentation_errors(preds: pd.DataFrame) -> pd.DataFrame:
    """Detect word-boundary errors: a reference word pair written as one hypothesis word
    (merge) or one reference word split into two (split). Also flags utterances whose
    only difference is spacing."""
    rows = []
    for _, r in preds.iterrows():
        ref, hyp = eval_normalise(r["ref"]), eval_normalise(r["hyp"])
        rw, hw = ref.split(), hyp.split()
        merges = sum(1 for i in range(len(rw) - 1) if rw[i] + rw[i + 1] in set(hw))
        splits = sum(1 for i in range(len(hw) - 1) if hw[i] + hw[i + 1] in set(rw))
        rows.append({"utt_id": r["utt_id"], "merges": merges, "splits": splits,
                     "spacing_only": ref != hyp and ref.replace(" ", "") == hyp.replace(" ", "")})
    return pd.DataFrame(rows)


def char_confusions(preds: pd.DataFrame, top: int = 25) -> pd.DataFrame:
    c = Counter()
    for ref, hyp in zip(preds["ref"], preds["hyp"]):
        for op, a, b in char_align(ref, hyp, ignore_spaces=True).ops:
            if op == "S":
                c[(a, b)] += 1
            elif op == "D":
                c[(a, "∅")] += 1
            elif op == "I":
                c[("∅", b)] += 1
    return pd.DataFrame([{"ref_char": a, "hyp_char": b, "count": n} for (a, b), n in c.most_common(top)])


def word_substitutions(outcomes: pd.DataFrame, top: int = 30) -> pd.DataFrame:
    s = outcomes[outcomes["op"] == "S"]
    return (s.groupby(["word", "hyp_word"]).size().sort_values(ascending=False)
            .head(top).rename("count").reset_index())


def duration_buckets(scored: pd.DataFrame, edges=(0, 2, 4, 6, 10, 15, 1e9)) -> pd.Series:
    labels = [f"{a:g}-{b:g}s" if b < 1e8 else f">{a:g}s" for a, b in zip(edges[:-1], edges[1:])]
    return pd.cut(scored["duration_s"], bins=list(edges), labels=labels, right=False)


def qualitative_sample(scored: pd.DataFrame, by: str = "dialect", per_group: int = 3,
                       seed: int = 0) -> pd.DataFrame:
    """Deterministic, NON-cherry-picked sample: for each group take utterances at the
    best, median and worst WER quantiles plus random picks, so successes and failures
    are both represented."""
    rng = np.random.default_rng(seed)
    picks = []
    for g, d in scored.groupby(by, dropna=False):
        d = d.sort_values(["utt_wer", "utt_id"]).reset_index(drop=True)
        if d.empty:
            continue
        for label, q in [("best", 0.0), ("median", 0.5), ("worst", 1.0)]:
            row = d.iloc[min(int(q * (len(d) - 1)), len(d) - 1)].copy()
            row["sample_reason"] = label
            picks.append(row)
        for i in rng.choice(len(d), size=min(per_group, len(d)), replace=False):
            row = d.iloc[int(i)].copy()
            row["sample_reason"] = "random"
            picks.append(row)
    out = pd.DataFrame(picks).drop_duplicates("utt_id")
    out["annotator_notes"] = ""
    return out

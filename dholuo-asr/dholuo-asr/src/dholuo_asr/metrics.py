"""WER / CER with alignments, subgroup metrics and speaker-level bootstrap CIs.

Implemented from scratch (and unit-tested against ``jiwer``) so that alignments are
available for error analysis. Corpus-level rates are computed as
``total edits / total reference units`` (not the mean of per-utterance rates).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .text import eval_normalise


@dataclass
class Alignment:
    sub: int
    ins: int
    dele: int
    ref_len: int
    ops: list  # list of (op, ref_token|None, hyp_token|None); op in {=,S,I,D}

    @property
    def errors(self) -> int:
        return self.sub + self.ins + self.dele


def align(ref: list, hyp: list) -> Alignment:
    """Levenshtein alignment with backtrace (unit costs)."""
    n, m = len(ref), len(hyp)
    # plain Python lists: much faster than numpy scalar indexing in a double loop
    d = [list(range(m + 1))] + [[i] + [0] * m for i in range(1, n + 1)]
    for i in range(1, n + 1):
        ri, prev, cur = ref[i - 1], d[i - 1], d[i]
        for j in range(1, m + 1):
            sub = prev[j - 1] + (ri != hyp[j - 1])
            dl = prev[j] + 1
            il = cur[j - 1] + 1
            cur[j] = sub if sub <= dl and sub <= il else (dl if dl <= il else il)
    i, j, ops = n, m, []
    s = ins = de = 0
    while i > 0 or j > 0:
        if i > 0 and j > 0 and d[i][j] == d[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1]):
            if ref[i - 1] == hyp[j - 1]:
                ops.append(("=", ref[i - 1], hyp[j - 1]))
            else:
                ops.append(("S", ref[i - 1], hyp[j - 1]))
                s += 1
            i, j = i - 1, j - 1
        elif i > 0 and d[i][j] == d[i - 1][j] + 1:
            ops.append(("D", ref[i - 1], None))
            de += 1
            i -= 1
        else:
            ops.append(("I", None, hyp[j - 1]))
            ins += 1
            j -= 1
    return Alignment(s, ins, de, n, ops[::-1])


def word_align(ref: str, hyp: str, normalise: bool = True) -> Alignment:
    if normalise:
        ref, hyp = eval_normalise(ref), eval_normalise(hyp)
    return align(ref.split(), hyp.split())


def char_align(ref: str, hyp: str, normalise: bool = True, ignore_spaces: bool = False) -> Alignment:
    if normalise:
        ref, hyp = eval_normalise(ref), eval_normalise(hyp)
    if ignore_spaces:
        ref, hyp = ref.replace(" ", ""), hyp.replace(" ", "")
    return align(list(ref), list(hyp))


def score_frame(df: pd.DataFrame, ref_col: str = "ref", hyp_col: str = "hyp") -> pd.DataFrame:
    """Add per-utterance word/char edit counts to a predictions frame."""
    rows = []
    for ref, hyp in zip(df[ref_col].fillna(""), df[hyp_col].fillna("")):
        w = word_align(ref, hyp)
        c = char_align(ref, hyp)
        rows.append({"w_err": w.errors, "w_sub": w.sub, "w_ins": w.ins, "w_del": w.dele,
                     "w_ref": w.ref_len, "c_err": c.errors, "c_ref": c.ref_len})
    out = pd.concat([df.reset_index(drop=True), pd.DataFrame(rows)], axis=1)
    out["utt_wer"] = out["w_err"] / out["w_ref"].clip(lower=1)
    out["utt_cer"] = out["c_err"] / out["c_ref"].clip(lower=1)
    return out


def corpus_rates(scored: pd.DataFrame) -> dict:
    w_ref, c_ref = scored["w_ref"].sum(), scored["c_ref"].sum()
    return {
        "wer": scored["w_err"].sum() / max(w_ref, 1),
        "cer": scored["c_err"].sum() / max(c_ref, 1),
        "sub_rate": scored["w_sub"].sum() / max(w_ref, 1),
        "ins_rate": scored["w_ins"].sum() / max(w_ref, 1),
        "del_rate": scored["w_del"].sum() / max(w_ref, 1),
        "sent_err_rate": float((scored["w_err"] > 0).mean()),
        "n_utts": int(len(scored)),
        "n_ref_words": int(w_ref),
    }


def group_rates(scored: pd.DataFrame, by: str) -> pd.DataFrame:
    rows = []
    for key, g in scored.groupby(by, dropna=False):
        r = corpus_rates(g)
        r[by] = key
        if "speaker" in g:
            r["n_speakers"] = g["speaker"].nunique()
        rows.append(r)
    return pd.DataFrame(rows).set_index(by)


def speaker_bootstrap(scored: pd.DataFrame, n_boot: int = 1000, seed: int = 0,
                      group_col: str = "speaker", alpha: float = 0.05) -> dict:
    """95% CI for corpus WER/CER by resampling SPEAKERS with replacement.

    Utterances from one speaker are correlated, so resampling utterances would give
    over-confident intervals. With few test speakers the intervals will be wide; that
    is an honest reflection of the evaluation's uncertainty.
    """
    rng = np.random.default_rng(seed)
    g = scored.groupby(group_col)[["w_err", "w_ref", "c_err", "c_ref"]].sum()
    arr = g.to_numpy(dtype=float)
    k = len(arr)
    idx = rng.integers(0, k, size=(n_boot, k))
    sums = arr[idx].sum(axis=1)
    wer = sums[:, 0] / np.maximum(sums[:, 1], 1)
    cer = sums[:, 2] / np.maximum(sums[:, 3], 1)
    lo, hi = 100 * alpha / 2, 100 * (1 - alpha / 2)
    return {"wer_ci": (float(np.percentile(wer, lo)), float(np.percentile(wer, hi))),
            "cer_ci": (float(np.percentile(cer, lo)), float(np.percentile(cer, hi))),
            "n_groups": int(k), "n_boot": n_boot}


def paired_speaker_bootstrap(a: pd.DataFrame, b: pd.DataFrame, n_boot: int = 2000,
                             seed: int = 0, group_col: str = "speaker", key: str = "utt_id") -> dict:
    """CI for WER(a) - WER(b) on the SAME utterances, resampling speakers.
    A CI that excludes 0 indicates a difference robust to speaker sampling."""
    m = a[[key, group_col, "w_err", "w_ref"]].merge(
        b[[key, "w_err"]], on=key, suffixes=("_a", "_b"))
    if len(m) != len(a) or len(m) != len(b):
        raise ValueError("paired bootstrap requires identical utterance sets")
    g = m.groupby(group_col)[["w_err_a", "w_err_b", "w_ref"]].sum().to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(g), size=(n_boot, len(g)))
    s = g[idx].sum(axis=1)
    diff = (s[:, 0] - s[:, 1]) / np.maximum(s[:, 2], 1)
    point = (g[:, 0].sum() - g[:, 1].sum()) / max(g[:, 2].sum(), 1)
    return {"wer_diff": float(point),
            "ci": (float(np.percentile(diff, 2.5)), float(np.percentile(diff, 97.5))),
            "p_a_not_better": float(np.mean(diff >= 0))}

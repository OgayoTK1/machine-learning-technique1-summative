"""Speaker-independent splitting, leakage checks and training-data subsampling."""
from __future__ import annotations

import numpy as np
import pandas as pd

SPLITS = ("train", "val", "test")


def _score_assignment(spk: pd.DataFrame, assign: np.ndarray, targets: dict[str, float],
                      balance_cols: list[str]) -> float:
    """Lower is better. Penalise deviation from target utterance shares, overall and
    within each value of every balance column (e.g. dialect, gender)."""
    total = spk["n_utt"].sum()
    score = 0.0
    for s, tgt in targets.items():
        share = spk.loc[assign == s, "n_utt"].sum() / total
        score += 4.0 * abs(share - tgt)
    for col in balance_cols:
        for _, grp in spk.groupby(col, dropna=False):
            idx = grp.index.to_numpy()
            g_total = grp["n_utt"].sum()
            if g_total == 0:
                continue
            for s, tgt in targets.items():
                share = grp.loc[assign[idx] == s, "n_utt"].sum() / g_total
                score += abs(share - tgt)
    return score


def speaker_split(df: pd.DataFrame, *, speaker_col: str = "speaker",
                  stratify_col: str | None = "dialect", balance_cols: tuple[str, ...] = ("gender",),
                  ratios: tuple[float, float, float] = (0.70, 0.15, 0.15),
                  min_speakers: tuple[int, int, int] = (1, 1, 1),
                  n_trials: int = 3000, seed: int = 42) -> pd.Series:
    """Assign every speaker to exactly one of train/val/test.

    Within each ``stratify_col`` stratum (dialect) speakers are shuffled ``n_trials`` times
    with a seeded RNG and cut so each split receives roughly the target number of
    speakers; the permutation whose utterance shares best match ``ratios`` (overall and
    per ``balance_cols`` value) wins. Fully deterministic for a given seed and input.

    Returns a Series mapping speaker -> split.
    """
    if df[speaker_col].isna().any():
        raise ValueError("speaker ids must be non-null before splitting")
    agg = {"n_utt": (speaker_col, "size")}
    cols = [c for c in [stratify_col, *balance_cols] if c]
    for c in cols:
        # a speaker's dialect/gender is taken as their most frequent value
        agg[c] = (c, lambda x: x.mode(dropna=False).iloc[0] if len(x.mode(dropna=False)) else pd.NA)
    spk = df.groupby(speaker_col).agg(**agg).reset_index()
    spk = spk.sort_values(speaker_col).reset_index(drop=True)  # input-order independent
    targets = dict(zip(SPLITS, ratios))
    rng = np.random.default_rng(seed)
    strata = [g.index.to_numpy() for _, g in spk.groupby(stratify_col, dropna=False)] \
        if stratify_col else [spk.index.to_numpy()]

    best, best_score = None, np.inf
    for _ in range(n_trials):
        assign = np.empty(len(spk), dtype=object)
        for idx in strata:
            perm = rng.permutation(idx)
            n = len(perm)
            n_val = max(min_speakers[1], int(round(n * ratios[1])))
            n_test = max(min_speakers[2], int(round(n * ratios[2])))
            if n - n_val - n_test < min_speakers[0]:  # tiny stratum: everything to train
                n_val = n_test = 0
            assign[perm[:n_test]] = "test"
            assign[perm[n_test:n_test + n_val]] = "val"
            assign[perm[n_test + n_val:]] = "train"
        sc = _score_assignment(spk, assign, targets, list(balance_cols))
        if sc < best_score:
            best, best_score = assign.copy(), sc
    return pd.Series(best, index=spk[speaker_col].to_numpy(), name="split")


def random_split(df: pd.DataFrame, sizes: dict[str, int], seed: int = 42) -> pd.Series:
    """Row-level random split with exactly the given number of rows per split.
    Used ONLY for Experiment F (leakage demonstration)."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(df.index.to_numpy())
    out = pd.Series(index=df.index, dtype=object)
    start = 0
    for s in ("test", "val", "train"):
        n = sizes[s]
        out.loc[perm[start:start + n]] = s
        start += n
    out.loc[perm[start:]] = "train"
    return out


def leakage_report(df: pd.DataFrame, split_col: str = "split", speaker_col: str = "speaker",
                   text_col: str = "text_norm") -> dict:
    """Quantify speaker and prompt overlap between splits."""
    rep: dict = {}
    spk = {s: set(df.loc[df[split_col] == s, speaker_col].dropna()) for s in SPLITS}
    txt = {s: set(df.loc[df[split_col] == s, text_col]) for s in SPLITS}
    for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
        rep[f"speaker_overlap_{a}_{b}"] = len(spk[a] & spk[b])
    for s in ("val", "test"):
        sub = df[df[split_col] == s]
        seen = sub[text_col].isin(txt["train"])
        rep[f"{s}_utts"] = int(len(sub))
        rep[f"{s}_utts_prompt_in_train"] = int(seen.sum())
        rep[f"{s}_frac_prompt_in_train"] = float(seen.mean()) if len(sub) else float("nan")
        rep[f"{s}_unique_prompts"] = len(txt[s])
        rep[f"{s}_unique_prompts_in_train"] = len(txt[s] & txt["train"])
    return rep


def subsample_train_speakers(train: pd.DataFrame, frac: float, *, speaker_col: str = "speaker",
                             stratify_col: str | None = "dialect", seed: int = 42) -> pd.DataFrame:
    """Nested speaker-level subsample for the data-size experiment (A).

    Speakers are ranked once per stratum by a seeded permutation; the first
    ceil(frac * n) speakers are kept. Because the ranking does not depend on ``frac``,
    the 10% subset is contained in the 25% subset, which is contained in 50%, etc.
    """
    if frac >= 1.0:
        return train
    rng = np.random.default_rng(seed)
    keep: list = []
    groups = train.groupby(stratify_col, dropna=False) if stratify_col else [(None, train)]
    for _, g in groups:
        spk = np.array(sorted(g[speaker_col].unique()))
        order = spk[rng.permutation(len(spk))]
        k = max(1, int(np.ceil(frac * len(spk))))
        keep.extend(order[:k])
    return train[train[speaker_col].isin(keep)]

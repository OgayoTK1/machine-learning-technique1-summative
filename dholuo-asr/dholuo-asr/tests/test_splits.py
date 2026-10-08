import numpy as np
import pandas as pd
from dholuo_asr.splits import speaker_split, subsample_train_speakers, leakage_report, random_split


def make_df(seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for s in range(40):
        dialect = "Milambo" if s < 24 else "Nyanduat"
        gender = "female" if s % 2 else "male"
        for u in range(rng.integers(20, 80)):
            rows.append({"utt_id": f"{s}_{u}", "speaker": f"spk{s:02d}", "dialect": dialect,
                         "gender": gender, "text_norm": f"prompt {u % 30}"})
    return pd.DataFrame(rows)


def test_no_speaker_overlap_and_all_dialects_in_test():
    df = make_df()
    m = speaker_split(df, n_trials=300)
    df["split"] = df["speaker"].map(m)
    rep = leakage_report(df)
    assert rep["speaker_overlap_train_test"] == 0
    assert rep["speaker_overlap_train_val"] == 0 and rep["speaker_overlap_val_test"] == 0
    assert set(df.loc[df.split == "test", "dialect"]) == {"Milambo", "Nyanduat"}
    share = df.split.value_counts(normalize=True)
    assert 0.6 < share["train"] < 0.8


def test_split_is_deterministic_and_order_independent():
    df = make_df()
    a = speaker_split(df, n_trials=200, seed=7)
    b = speaker_split(df.sample(frac=1, random_state=3), n_trials=200, seed=7)
    assert a.sort_index().equals(b.sort_index())


def test_subsamples_are_nested():
    df = make_df()
    sets = [set(subsample_train_speakers(df, f, seed=1).speaker) for f in (0.1, 0.25, 0.5, 1.0)]
    assert all(a <= b for a, b in zip(sets, sets[1:]))
    assert sets[0] and len(sets[0]) < len(sets[-1])


def test_random_split_sizes():
    df = make_df()
    r = random_split(df, {"train": 100, "val": 50, "test": 60}, seed=1)
    assert (r == "test").sum() == 60 and (r == "val").sum() == 50

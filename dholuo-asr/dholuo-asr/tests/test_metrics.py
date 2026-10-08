import random
import jiwer
import pandas as pd
from dholuo_asr.metrics import align, score_frame, corpus_rates, speaker_bootstrap, paired_speaker_bootstrap


def test_align_counts_match_jiwer():
    rng = random.Random(0)
    vocab = "an in ber ng'ato dala nyithindo chiemo".split()
    refs, hyps = [], []
    for _ in range(200):
        r = [rng.choice(vocab) for _ in range(rng.randint(1, 9))]
        h = [w if rng.random() > 0.3 else rng.choice(vocab) for w in r]
        if rng.random() < 0.3:
            h.insert(rng.randint(0, len(h)), rng.choice(vocab))
        if rng.random() < 0.3 and len(h) > 1:
            h.pop(rng.randrange(len(h)))
        refs.append(" ".join(r)); hyps.append(" ".join(h))
    df = pd.DataFrame({"utt_id": range(200), "speaker": [i % 7 for i in range(200)], "ref": refs, "hyp": hyps})
    s = score_frame(df)
    assert abs(corpus_rates(s)["wer"] - jiwer.wer(refs, hyps)) < 1e-9
    assert abs(corpus_rates(s)["cer"] - jiwer.cer(refs, hyps)) < 1e-9
    for r, h in zip(refs[:50], hyps[:50]):
        a = align(r.split(), h.split())
        assert a.errors == jiwer.process_words(r, h).substitutions + jiwer.process_words(r, h).insertions + jiwer.process_words(r, h).deletions


def test_bootstrap_ci_contains_point_and_paired_zero():
    df = pd.DataFrame({"utt_id": range(60), "speaker": [i % 6 for i in range(60)],
                       "ref": ["a b c d"] * 60, "hyp": ["a b x d" if i % 3 else "a b c d" for i in range(60)]})
    s = score_frame(df)
    ci = speaker_bootstrap(s, n_boot=500)
    w = corpus_rates(s)["wer"]
    assert ci["wer_ci"][0] <= w <= ci["wer_ci"][1]
    p = paired_speaker_bootstrap(s, s, n_boot=200)
    assert p["wer_diff"] == 0 and p["ci"] == (0.0, 0.0)

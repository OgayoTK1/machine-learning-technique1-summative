import math
import random
from dholuo_asr.lm import NGramLM, EOS, UNK


def corpus():
    rng = random.Random(0)
    v = "an ok ber dala nyithindo chiemo ng'ato kod gi e".split()
    return [[rng.choice(v) for _ in range(rng.randint(1, 8))] for _ in range(300)]


def test_conditional_distributions_sum_to_one():
    lm = NGramLM(order=3).fit(corpus())
    targets = [w for w in lm.vocab if w != "<s>"]
    for h in [(), ("an",), ("<s>",), ("<s>", "an"), ("ber", "dala"), ("zzz", "an"), ("unseen", "words")]:
        total = sum(10 ** lm.score(w, h) for w in targets if w != UNK) + 10 ** lm.score(UNK, h)
        assert abs(total - 1.0) < 1e-6, (h, total)


def test_arpa_roundtrip_and_perplexity(tmp_path):
    lm = NGramLM(order=3).fit(corpus())
    p = lm.write_arpa(tmp_path / "lm.arpa")
    txt = p.read_text()
    assert txt.startswith("\\data\\") and txt.rstrip().endswith("\\end\\")
    assert f"ngram 3={len(lm.logp[2])}" in txt
    assert lm.perplexity(corpus()[:50]) < len(lm.vocab)

from dholuo_asr.vocab import build_char_vocab, CharTokenizer


def test_vocab_roundtrip():
    texts = ["ng'ato ber", "dala"]
    v = build_char_vocab(texts)
    assert v["[PAD]"] == 0 and "'" in v and " " not in v
    tok = CharTokenizer(v)
    ids = tok.encode("ng'ato ber")
    # insert blanks and repeats as CTC would produce
    noisy = []
    for i in ids:
        noisy += [i, i, 0]
    assert tok.decode_ids(noisy) == "ng'ato ber"

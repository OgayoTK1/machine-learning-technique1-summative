from dholuo_asr.text import normalise, minimal_normalise, eval_normalise, has_tags


def test_apostrophe_kept_and_unified():
    # ng' is a distinct Dholuo consonant: the apostrophe must survive normalisation
    assert normalise("Ng’ato") == "ng'ato"
    assert normalise("ng`eyo ng´ama") == "ng'eyo ng'ama"


def test_punctuation_case_and_tags():
    assert normalise("Amosi, ERE?  [noise] Ber!") == "amosi ere ber"
    assert has_tags("hello [laugh]")


def test_stray_quotes_removed_but_inner_apostrophe_kept():
    # a leading quote is stray; an apostrophe attached to a word (ng'ato) is linguistic
    assert normalise("'ber") == "ber"
    assert normalise("'Nyithindo ng'ato") == "nyithindo ng'ato"


def test_minimal_keeps_orthography():
    assert minimal_normalise("  Amosi,\tERE? ") == "Amosi, ERE?"


def test_eval_normalise_idempotent():
    s = "Ng’ato, ere?"
    assert eval_normalise(eval_normalise(s)) == eval_normalise(s)

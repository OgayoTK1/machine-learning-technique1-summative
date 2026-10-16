"""Preprocessing must match the training notebook exactly."""

import json
from pathlib import Path

import pytest

from src.preprocessing import TEXT_VARIANTS, canonical_tag, get_normalizer, tokenize, word_tokens

NOTEBOOK = Path(__file__).resolve().parent.parent / "notebooks" / "dholuo_topic_classification_final.ipynb"

# Inputs and outputs printed by the executed notebook (section 5).
NOTEBOOK_EXAMPLES = [
    ("Nyithindo ipuonjo yore mag pur ,koro [CS]net[cs] ok?", {
        "minimal": "Nyithindo ipuonjo yore mag pur ,koro [CS]net[cs] ok?",
        "normalized": "nyithindo ipuonjo yore mag pur, koro <cs> net <cs> ok?",
        "normalized_notags": "nyithindo ipuonjo yore mag pur, koro net ok?"}),
    ("Jopur  gi’ jopith [pause] maber [?] ahinya.", {
        "minimal": "Jopur gi’ jopith [pause] maber [?] ahinya.",
        "normalized": "jopur gi' jopith <pause> maber <unclear> ahinya.",
        "normalized_notags": "jopur gi' jopith maber ahinya."}),
    ("Ng'ado cham 2.5 kg [cs]fertilizer[cs]", {
        "minimal": "Ng'ado cham 2.5 kg [cs]fertilizer[cs]",
        "normalized": "ng'ado cham 2.5 kg <cs> fertilizer <cs>",
        "normalized_notags": "ng'ado cham 2.5 kg fertilizer"}),
]

EXTRA_INPUTS = [
    "", "   ", "Gweno\tgwen\n\nrech", "[[cs]kuku[cs]", "[saa] mokuongo", "[Pause] ok [ unclear ]",
    "jopur,koro;ka:ni!to?", "kik ,ok .", "‘quoted’ `tick` ´acute´ ʼmodʼ",
    "Café NFC check", "[cs[ broken tag", "123,456 and 7.5", "<cs> already canonical",
]


@pytest.mark.parametrize("text,expected", NOTEBOOK_EXAMPLES)
def test_matches_notebook_printed_examples(text, expected):
    for variant, want in expected.items():
        assert TEXT_VARIANTS[variant](text) == want


def _notebook_preprocessing_namespace():
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    for cell in nb["cells"]:
        src = "".join(cell["source"]) if isinstance(cell["source"], list) else cell["source"]
        if cell["cell_type"] == "code" and src.startswith("PREPROC_SRC = r'''"):
            code = src.split("r'''", 1)[1].split("'''", 1)[0]
            ns = {}
            exec(code, ns)  # noqa: S102 - trusted project notebook
            return ns
    raise AssertionError("PREPROC_SRC cell not found in the notebook")


@pytest.mark.skipif(not NOTEBOOK.exists(), reason="training notebook not present")
def test_parity_with_training_notebook_source():
    ns = _notebook_preprocessing_namespace()
    inputs = [t for t, _ in NOTEBOOK_EXAMPLES] + EXTRA_INPUTS
    for variant in TEXT_VARIANTS:
        for text in inputs:
            assert TEXT_VARIANTS[variant](text) == ns["TEXT_VARIANTS"][variant](text), (variant, text)
    for text in inputs:
        normalized = TEXT_VARIANTS["normalized"](text)
        assert tokenize(normalized) == ns["tokenize"](normalized)
    assert ns["WORD_PATTERN"] == __import__("src.preprocessing", fromlist=["WORD_PATTERN"]).WORD_PATTERN


def test_canonical_tags():
    assert canonical_tag("CS") == " <cs> "
    assert canonical_tag(" pause ") == " <pause> "
    assert canonical_tag("?") == " <unclear> "
    assert canonical_tag("saa") == " <tag> "


def test_word_tokens_keep_apostrophes_and_markers():
    assert word_tokens("ng'ado <cs> net <cs>, ok.") == ["ng'ado", "<cs>", "net", "<cs>", "ok"]


def test_unknown_variant_raises():
    with pytest.raises(ValueError):
        get_normalizer("stemmed")

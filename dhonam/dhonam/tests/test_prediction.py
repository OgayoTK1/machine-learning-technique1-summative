"""Prediction logic, tested with a small stand-in model (see conftest.py)."""

import json

import numpy as np
import pytest

from src.prediction import InputValidationError, ModelArtifactError, TopicClassifier


def test_loads_and_predicts_expected_topic(fixture_model_dir):
    clf = TopicClassifier(fixture_model_dir)
    assert clf.predict("Jopith rech tiyo gi yawo mondo rech odongi maber").label == "Fish farming"
    assert clf.predict("Gweno gi gwen ma tuo ochanj").label == "Poultry farming"


def test_ranking_covers_all_labels_sorted_and_normalised(fixture_model_dir):
    clf = TopicClassifier(fixture_model_dir)
    pred = clf.predict("olo pii e puoth gi soko")
    scores = [s for _, s in pred.ranking]
    assert {t for t, _ in pred.ranking} == set(clf.labels)
    assert scores == sorted(scores, reverse=True)
    assert np.isclose(sum(scores), 1.0)
    assert len(pred.top(3)) == 3


def test_uncalibrated_by_default(fixture_model_dir):
    clf = TopicClassifier(fixture_model_dir)
    assert clf.calibrated is False
    assert clf.predict("rech e yawo").calibrated is False


def test_validation_temperature_is_applied(make_model_dir):
    base = TopicClassifier(make_model_dir())
    text = "rech ngege e yawo"
    p_raw = base.predict(text).ranking[0][1]
    sharp = TopicClassifier(make_model_dir(temperature=0.25, temperature_fitted_on="validation"))
    pred = sharp.predict(text)
    assert sharp.calibrated and pred.calibrated
    assert pred.label == base.predict(text).label        
    assert pred.ranking[0][1] > p_raw                       


def test_preprocessing_is_applied_before_the_model(fixture_model_dir):
    clf = TopicClassifier(fixture_model_dir)
    pred = clf.predict("REch  [CS]net[cs] ,yawo")
    assert pred.normalized_text == "rech <cs> net <cs>, yawo"


@pytest.mark.parametrize("bad", ["", "    ", "\n\t", "...", "123 456"])
def test_empty_or_wordless_input_rejected(fixture_model_dir, bad):
    with pytest.raises(InputValidationError):
        TopicClassifier(fixture_model_dir).predict(bad)


def test_non_text_and_oversized_input_rejected(fixture_model_dir):
    clf = TopicClassifier(fixture_model_dir)
    with pytest.raises(InputValidationError):
        clf.predict(None)
    with pytest.raises(InputValidationError):
        clf.predict("rech " * 3000)


def test_short_and_unknown_word_warnings(fixture_model_dir):
    clf = TopicClassifier(fixture_model_dir)
    short = clf.predict("rech yawo")
    assert any("only 2 words" in w for w in short.warnings)
    foreign = clf.predict("the quick brown fox jumps over the lazy dog near the river bank today")
    assert foreign.vocab_coverage == 0.0
    assert any("known to the model" in w for w in foreign.warnings)


def test_missing_model_directory(tmp_path):
    with pytest.raises(ModelArtifactError, match="models/README.md"):
        TopicClassifier(tmp_path / "does-not-exist")


def test_wrong_kind_rejected(make_model_dir):
    with pytest.raises(ModelArtifactError, match="kind"):
        TopicClassifier(make_model_dir(kind="transformer"))


def test_corrupt_label_map_rejected(make_model_dir):
    d = make_model_dir()
    (d / "label_map.json").write_text("{not json", encoding="utf-8")
    with pytest.raises(ModelArtifactError):
        TopicClassifier(d)


def test_label_count_mismatch_rejected(make_model_dir):
    d = make_model_dir()
    meta = json.loads((d / "label_map.json").read_text())
    meta["labels"] = meta["labels"][:2]
    (d / "label_map.json").write_text(json.dumps(meta))
    with pytest.raises(ModelArtifactError, match="labels"):
        TopicClassifier(d)


def test_model_dir_from_environment(fixture_model_dir, monkeypatch):
    monkeypatch.setenv("DHOLUO_MODEL_DIR", str(fixture_model_dir))
    assert TopicClassifier().info()["run_id"] == "test-fixture"

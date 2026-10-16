"""Shared test fixtures.

The real model is trained on gated data and is not part of the repository, so the tests
build a small stand-in model with the same structure as the notebook export (a scikit-learn
Pipeline with steps 'tfidf' and 'clf', plus label_map.json). It is trained on a handful of
synthetic sentences and exists only to exercise loading, preprocessing and prediction code.
It says nothing about real model quality.
"""

import json
import sys
from pathlib import Path

import joblib
import pytest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.preprocessing import WORD_PATTERN, get_normalizer  # noqa: E402

FIXTURE_LABELS = ["Fish farming", "Poultry farming", "Water Usage and Irrigation Methods"]
FIXTURE_TEXTS = {
    0: ["jopith rech e yawo", "rech ngege mbuta yawo", "chiemo mar rech e yawo", "pidho rech ngege"],
    1: ["gweno gwen tong", "pidho gweno e od", "tuo mar gwen chanjo", "gwende tong gweno"],
    2: ["olo pii e puoth", "pii mar soko", "kunyo soko pii", "tuomo pii e puothe"],
}


def build_fixture_model(model_dir: Path, svm: bool = False, **meta_overrides) -> Path:
    model_dir.mkdir(parents=True, exist_ok=True)
    norm = get_normalizer("normalized")
    texts, y = [], []
    for label_id, rows in FIXTURE_TEXTS.items():
        for r in rows:
            texts.append(norm(r)); y.append(label_id)
    word = TfidfVectorizer(analyzer="word", token_pattern=WORD_PATTERN, lowercase=False,
                           ngram_range=(1, 2) if svm else (1, 1), min_df=1, sublinear_tf=True)
    if svm:   # same structure as the run-2 export: FeatureUnion(word, char_wb) + LinearSVC
        pipe = Pipeline([
            ("tfidf", FeatureUnion([("w", word), ("c", TfidfVectorizer(analyzer="char_wb", lowercase=False,
                                                                         ngram_range=(2, 5), sublinear_tf=True))])),
            ("clf", LinearSVC(C=0.1, class_weight="balanced", random_state=0)),
        ]).fit(texts, y)
    else:
        pipe = Pipeline([
            ("tfidf", word),
            ("clf", LogisticRegression(C=1.0, max_iter=1000, class_weight="balanced")),
        ]).fit(texts, y)
    joblib.dump(pipe, model_dir / "model.joblib")
    meta = {"labels": FIXTURE_LABELS, "label2id": {l: i for i, l in enumerate(FIXTURE_LABELS)},
            "text_variant": "normalized", "kind": "sklearn", "run_id": "test-fixture", "temperature": 1.0}
    meta.update(meta_overrides)
    (model_dir / "label_map.json").write_text(json.dumps(meta), encoding="utf-8")
    return model_dir


@pytest.fixture(scope="session")
def fixture_model_dir(tmp_path_factory):
    return build_fixture_model(tmp_path_factory.mktemp("model"))


@pytest.fixture()
def make_model_dir(tmp_path):
    def _make(svm=False, **meta_overrides):
        return build_fixture_model(tmp_path / "model", svm=svm, **meta_overrides)
    return _make

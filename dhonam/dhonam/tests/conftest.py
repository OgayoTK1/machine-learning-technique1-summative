"""Shared test fixtures.

"""

import json
import sys
from pathlib import Path

import joblib
import pytest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.preprocessing import WORD_PATTERN, get_normalizer  # noqa: E402

FIXTURE_LABELS = ["Fish farming", "Poultry farming", "Water Usage and Irrigation Methods"]
FIXTURE_TEXTS = {
    0: ["jopith rech e yawo", "rech ngege mbuta yawo", "chiemo mar rech e yawo", "pidho rech ngege"],
    1: ["gweno gwen tong", "pidho gweno e od", "tuo mar gwen chanjo", "gwende tong gweno"],
    2: ["olo pii e puoth", "pii mar soko", "kunyo soko pii", "tuomo pii e puothe"],
}


def build_fixture_model(model_dir: Path, **meta_overrides) -> Path:
    model_dir.mkdir(parents=True, exist_ok=True)
    norm = get_normalizer("normalized")
    texts, y = [], []
    for label_id, rows in FIXTURE_TEXTS.items():
        for r in rows:
            texts.append(norm(r)); y.append(label_id)
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(analyzer="word", token_pattern=WORD_PATTERN, lowercase=False,
                                  ngram_range=(1, 1), min_df=1, sublinear_tf=True)),
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
    def _make(**meta_overrides):
        return build_fixture_model(tmp_path / "model", **meta_overrides)
    return _make

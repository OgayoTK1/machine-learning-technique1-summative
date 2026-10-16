"""The calibration script, run offline against the stand-in model and synthetic rows."""

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import calibrate_temperature as cal  # noqa: E402

from src.prediction import TopicClassifier  # noqa: E402


def _write_data(tmp_path):
    rows = [
        ("v1", "rech ngege e yawo", "Fish farming", "val"), ("v2", "gweno gwen e od", "Poultry farming", "val"),
        ("v3", "olo pii e puoth", "Water Usage and Irrigation Methods", "val"),
        ("v4", "rech gweno pii", "Fish farming", "val"),
        ("t1", "pidho rech", "Fish farming", "test"), ("t2", "tuo mar gweno", "Poultry farming", "test"),
        ("t3", "soko pii", "Water Usage and Irrigation Methods", "test"), ("x1", "rech", "Fish farming", "train"),
    ]
    df = pd.DataFrame(rows, columns=["file_name", "transcript", "sub_domain", "split"])
    tx, sp = tmp_path / "transcripts.csv", tmp_path / "splits.csv"
    df.drop(columns=["split"]).to_csv(tx, index=False)
    df[["file_name", "split"]].to_csv(sp, index=False)
    return tx, sp


def test_fit_temperature_recovers_known_scale():
    import numpy as np
    rng = np.random.default_rng(0)
    y = rng.integers(0, 4, 2000)
    true_logits = rng.normal(size=(2000, 4)); true_logits[np.arange(2000), y] += 2.0
    y = np.array([rng.choice(4, p=cal.softmax(true_logits[i:i + 1])[0]) for i in range(2000)])
    t = cal.fit_temperature(true_logits / 3.0, y)   # logits shrunk 3x -> best temperature near 1/3
    assert 0.25 < t < 0.45


def test_script_writes_validation_temperature(make_model_dir, tmp_path):
    model_dir = make_model_dir()
    tx, sp = _write_data(tmp_path)
    report = cal.main(["--model-dir", str(model_dir), "--splits", str(sp), "--transcripts-csv", str(tx)])
    assert report["n_val"] == 4 and report["n_test"] == 3
    meta = json.loads((model_dir / "label_map.json").read_text())
    assert meta["temperature_fitted_on"] == "validation"
    assert abs(meta["temperature"] - report["temperature"]) < 1e-12
    assert TopicClassifier(model_dir).calibrated


def test_dry_run_leaves_files_unchanged(make_model_dir, tmp_path):
    model_dir = make_model_dir()
    before = (model_dir / "label_map.json").read_text()
    tx, sp = _write_data(tmp_path)
    cal.main(["--model-dir", str(model_dir), "--splits", str(sp), "--transcripts-csv", str(tx), "--dry-run"])
    assert (model_dir / "label_map.json").read_text() == before

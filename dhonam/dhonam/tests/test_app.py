"""End-to-end checks of the Streamlit app using Streamlit's testing harness."""

from pathlib import Path

import pytest

AppTest = pytest.importorskip("streamlit.testing.v1").AppTest
APP = str(Path(__file__).resolve().parent.parent / "app.py")


def _run(monkeypatch, model_dir):
    monkeypatch.setenv("DHOLUO_MODEL_DIR", str(model_dir))
    at = AppTest.from_file(APP, default_timeout=60)
    at.run()
    return at


def _all_text(at):
    parts = [m.value for m in at.markdown] + [s.value for s in at.subheader]
    parts += [w.value for w in at.warning] + [i.value for i in at.info] + [e.value for e in at.error]
    return "\n".join(str(p) for p in parts)


def test_app_classifies_text(monkeypatch, fixture_model_dir):
    at = _run(monkeypatch, fixture_model_dir)
    assert not at.exception
    at.text_area[0].input("Jopith rech tiyo gi yawo mondo rech odongi maber e pi")
    at.button[0].click()
    at.run()
    assert not at.exception
    text = _all_text(at)
    assert "Fish farming" in text
    assert "Confidence percentages are not shown" in text   # uncalibrated model: no percentages


def test_app_empty_input_message(monkeypatch, fixture_model_dir):
    at = _run(monkeypatch, fixture_model_dir)
    at.button[0].click()
    at.run()
    assert not at.exception
    assert any("Please enter a Dholuo transcript" in w.value for w in at.warning)


def test_app_shows_calibrated_confidences(monkeypatch, make_model_dir):
    d = make_model_dir(temperature=0.5, temperature_fitted_on="validation")
    at = _run(monkeypatch, d)
    at.text_area[0].input("gweno gwen tong e od")
    at.button[0].click()
    at.run()
    assert not at.exception
    assert any("fitted on the validation" in c.value for c in at.caption)
    conf = at.dataframe[0].value["Confidence"]
    assert conf.max() > 1.0 and conf.max() <= 100.0   # shown on a 0-100 scale


def test_app_reports_missing_model(monkeypatch, tmp_path):
    at = _run(monkeypatch, tmp_path / "missing")
    assert not at.exception
    assert any("could not be loaded" in e.value for e in at.error)


REAL_MODEL = Path(__file__).resolve().parent.parent / "models" / "final"


@pytest.mark.skipif(not (REAL_MODEL / "model.joblib").exists(),
                    reason="real exported model not present (it is not committed by default)")
def test_app_with_real_exported_model(monkeypatch):
    """Smoke test against the real model, run only where models/final/ has been placed locally."""
    at = _run(monkeypatch, REAL_MODEL)
    at.text_area[0].input("Jopith rech tiyo gi yawo mondo rech odongi maber kendo ging'iewo chiemb rech e chiro")
    at.button[0].click()
    at.run()
    assert not at.exception
    assert any("Fish farming" in str(m.value) for m in at.markdown)
    assert any("fitted on the validation" in c.value for c in at.caption)

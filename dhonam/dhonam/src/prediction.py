"""Inference for the Dholuo agricultural topic classifier.

Loads the model exported by the training notebook (section 17) from a directory that
contains at least:

    label_map.json   labels, text variant, model kind, run id (and optionally a temperature)
    model.joblib     the fitted scikit-learn pipeline (TF-IDF + classifier)

The Streamlit app and the tests use this module; it has no Streamlit dependency.

Security note: joblib files are Python pickles. Only load model files you produced
yourself (for example, downloaded from your own Kaggle notebook output).
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .preprocessing import get_normalizer, word_tokens

DEFAULT_MODEL_DIR = Path(__file__).resolve().parent.parent / "models" / "final"

# Input guards. Training transcripts had 16 to 158 words (notebook section 3).
MAX_CHARS = 10_000
SHORT_INPUT_WORDS = 10
LONG_INPUT_WORDS = 300
LOW_COVERAGE = 0.6


class ModelArtifactError(RuntimeError):
    """The model directory is missing, incomplete, or not compatible with this app."""


class InputValidationError(ValueError):
    """The user input cannot be classified (for example, empty text)."""


@dataclass
class Prediction:
    label: str
    ranking: list            # [(label, score), ...] sorted, highest first, all classes
    calibrated: bool         # True only if a temperature fitted on validation data was applied
    normalized_text: str
    n_words: int
    vocab_coverage: float | None
    warnings: list = field(default_factory=list)

    def top(self, k: int = 3) -> list:
        return self.ranking[:k]


def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


class TopicClassifier:
    """Wraps the exported scikit-learn pipeline with the exact training preprocessing."""

    REQUIRED_FILES = ("label_map.json", "model.joblib")

    def __init__(self, model_dir: str | os.PathLike | None = None):
        self.model_dir = Path(model_dir or os.environ.get("DHOLUO_MODEL_DIR", DEFAULT_MODEL_DIR))
        missing = [f for f in self.REQUIRED_FILES if not (self.model_dir / f).is_file()]
        if missing:
            raise ModelArtifactError(
                f"Model directory '{self.model_dir}' is missing {', '.join(missing)}. "
                "See models/README.md for how to obtain the exported model."
            )
        try:
            self.meta = json.loads((self.model_dir / "label_map.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ModelArtifactError(f"Could not read label_map.json: {exc}") from exc

        kind = self.meta.get("kind")
        if kind != "sklearn":
            raise ModelArtifactError(
                f"label_map.json declares model kind '{kind}'. This app serves the evaluated final "
                "model, a scikit-learn pipeline (kind 'sklearn')."
            )
        self.labels = list(self.meta.get("labels") or [])
        if not self.labels:
            raise ModelArtifactError("label_map.json has no 'labels'.")
        self.text_variant = self.meta.get("text_variant", "normalized")
        self.normalize = get_normalizer(self.text_variant)

        try:
            import joblib
            self.pipeline = joblib.load(self.model_dir / "model.joblib")
        except Exception as exc:  # unpickling errors vary by library version
            raise ModelArtifactError(
                f"Could not load model.joblib ({type(exc).__name__}: {exc}). The model was trained with "
                "scikit-learn 1.6.1; install the versions in requirements.txt."
            ) from exc

        steps = getattr(self.pipeline, "named_steps", {})
        if "clf" not in steps:
            raise ModelArtifactError("model.joblib is not the expected pipeline (no 'clf' step).")
        self.clf = steps["clf"]
        classes = np.asarray(self.clf.classes_)
        if classes.min() < 0 or classes.max() >= len(self.labels):
            raise ModelArtifactError("Classifier classes do not match the labels in label_map.json.")
        self.class_ids = classes.astype(int)

        # Vocabulary of the word-level TF-IDF step, used only for an input-coverage warning.
        vec = steps.get("tfidf")
        vocab = getattr(vec, "vocabulary_", None)
        self.vocabulary = set(vocab) if vocab is not None and getattr(vec, "analyzer", None) == "word" else None

        # Confidence handling. A temperature counts as calibration only if it was fitted on
        # validation data: explicitly recorded, or (for an SVM) fitted by the notebook itself.
        self.temperature = float(self.meta.get("temperature", 1.0) or 1.0)
        self.has_proba = hasattr(self.clf, "predict_proba")
        fitted_flag = self.meta.get("temperature_fitted_on") == "validation"
        svm_fitted = (not self.has_proba) and not math.isclose(self.temperature, 1.0)
        self.calibrated = bool(fitted_flag or svm_fitted)

    # ------------------------------------------------------------------ public API
    def info(self) -> dict:
        return {
            "model_dir": str(self.model_dir),
            "run_id": self.meta.get("run_id"),
            "text_variant": self.text_variant,
            "n_labels": len(self.labels),
            "calibrated": self.calibrated,
            "temperature": self.temperature,
        }

    def validate(self, text) -> str:
        if not isinstance(text, str):
            raise InputValidationError("Input must be text.")
        stripped = text.strip()
        if not stripped:
            raise InputValidationError("Please enter a Dholuo transcript to classify.")
        if len(stripped) > MAX_CHARS:
            raise InputValidationError(f"The text is too long ({len(stripped):,} characters; the limit is {MAX_CHARS:,}).")
        return stripped

    def predict(self, text: str) -> Prediction:
        raw = self.validate(text)
        normalized = self.normalize(raw)
        words = [w for w in word_tokens(normalized) if not w.startswith("<")]
        if not any(any(ch.isalpha() for ch in w) for w in words):
            raise InputValidationError("The text contains no words to classify.")

        scores = self._class_scores([normalized])[0]
        order = np.argsort(scores)[::-1]
        ranking = [(self.labels[i], float(scores[i])) for i in order]

        coverage = None
        if self.vocabulary is not None:
            coverage = float(np.mean([w in self.vocabulary for w in words]))

        warnings = []
        if len(words) < SHORT_INPUT_WORDS:
            warnings.append(
                f"The text has only {len(words)} words. Training answers had at least 16, so predictions "
                "on very short inputs are less reliable."
            )
        if len(words) > LONG_INPUT_WORDS:
            warnings.append(
                f"The text has {len(words)} words, longer than any training answer (158 words). Long texts "
                "often mix several topics."
            )
        if coverage is not None and coverage < LOW_COVERAGE:
            warnings.append(
                f"Only {coverage:.0%} of the words are known to the model. The text may not be Dholuo, or not "
                "about agriculture; the model will still choose one of its 23 topics."
            )
        return Prediction(label=ranking[0][0], ranking=ranking, calibrated=self.calibrated,
                          normalized_text=normalized, n_words=len(words), vocab_coverage=coverage,
                          warnings=warnings)

    # ------------------------------------------------------------------ internals
    def _class_scores(self, texts: list) -> np.ndarray:
        """Scores over all labels. Probabilities if calibrated, otherwise raw model outputs."""
        k = len(self.labels)
        if self.has_proba:
            p = self.pipeline.predict_proba(texts)
            if self.calibrated:
                p = _softmax(np.log(np.clip(p, 1e-12, 1.0)) / self.temperature)
        else:
            z = np.atleast_2d(self.pipeline.decision_function(texts))
            p = _softmax(z / self.temperature)
        out = np.zeros((len(texts), k))
        out[:, self.class_ids] = p
        return out

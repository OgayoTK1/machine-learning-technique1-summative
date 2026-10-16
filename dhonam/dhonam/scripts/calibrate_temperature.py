"""Fit a softmax temperature for an exported probabilistic model on the validation speakers.

When it is needed: only for a model with predict_proba, such as the TF-IDF + logistic-regression
export. In the first training run that model was the selected one and its raw probabilities were
badly under-confident on the test speakers. The current selected model, a linear SVM, already has
a temperature fitted on validation inside the notebook (section 9), so the script refuses it.
Temperature scaling (Guo et al., 2017) rescales the probabilities with one parameter fitted on
validation data. It never changes which topic is predicted.

The script rebuilds the validation and test sets from the dataset and the split file the
notebook exported (results/splits.csv), so it uses exactly the same speakers. It then writes
`temperature` and `temperature_fitted_on: "validation"` into label_map.json, which makes the
app show calibrated confidences.

Usage (requires access to the gated dataset; set HF_TOKEN in your environment):

    python scripts/calibrate_temperature.py --model-dir models/final --splits reports/results/splits.csv

For offline testing, --transcripts-csv can point to a local transcripts file instead.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.prediction import TopicClassifier  # noqa: E402

DATASET = "MCAA1-MSU/twb_luo_ke"
REVISION = "5cbcd61d844e29e80e5e574fa14a7c54603052c3"
TRANSCRIPTS = "utavu-freeform_prompts_data/Files/transcripts.csv"
GRID = np.geomspace(0.01, 10, 400)


def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def fit_temperature(logp: np.ndarray, y: np.ndarray) -> float:
    """Temperature minimising validation negative log-likelihood of softmax(logp / T)."""
    nll = [-np.mean(np.log(softmax(logp / t)[np.arange(len(y)), y] + 1e-12)) for t in GRID]
    return float(GRID[int(np.argmin(nll))])


def ece(proba: np.ndarray, y: np.ndarray, bins: int = 10) -> float:
    conf = proba.max(1)
    ok = proba.argmax(1) == y
    idx = np.minimum((conf * bins).astype(int), bins - 1)
    return float(sum(abs(ok[idx == b].mean() - conf[idx == b].mean()) * (idx == b).mean()
                     for b in range(bins) if (idx == b).any()))


def load_transcripts(local_csv: str | None) -> pd.DataFrame:
    if local_csv:
        return pd.read_csv(local_csv)
    from huggingface_hub import hf_hub_download  # imported here so tests do not need it
    return pd.read_csv(hf_hub_download(DATASET, TRANSCRIPTS, repo_type="dataset", revision=REVISION))


def raw_log_proba(clf: TopicClassifier, texts: list) -> np.ndarray:
    p = clf.pipeline.predict_proba([clf.normalize(t) for t in texts])
    out = np.full((len(texts), len(clf.labels)), 1e-12)
    out[:, clf.class_ids] = np.clip(p, 1e-12, 1.0)
    return np.log(out)


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model-dir", default=str(ROOT / "models" / "final"))
    ap.add_argument("--splits", required=True, help="results/splits.csv exported by the notebook")
    ap.add_argument("--transcripts-csv", default=None, help="local transcripts CSV (default: download)")
    ap.add_argument("--dry-run", action="store_true", help="report only, do not modify label_map.json")
    args = ap.parse_args(argv)

    clf = TopicClassifier(args.model_dir)
    if not clf.has_proba:
        raise SystemExit("This script calibrates probabilistic classifiers only. The SVM export is already "
                         "calibrated: its temperature was fitted on validation in the notebook (section 9).")

    splits = pd.read_csv(args.splits)
    df = load_transcripts(args.transcripts_csv)
    df.columns = [c.lower() for c in df.columns]
    df = df.merge(splits[["file_name", "split"]], on="file_name", how="inner")
    label2id = {lab: i for i, lab in enumerate(clf.labels)}
    df = df[df["sub_domain"].isin(label2id)]
    result = {}
    for part in ("val", "test"):
        sub = df[df["split"] == part]
        if sub.empty:
            raise SystemExit(f"No '{part}' rows after joining the split file; check --splits.")
        result[part] = (raw_log_proba(clf, sub["transcript"].tolist()), sub["sub_domain"].map(label2id).to_numpy())

    t = fit_temperature(*result["val"])
    report = {"temperature": t, "n_val": int(len(result["val"][1])), "n_test": int(len(result["test"][1]))}
    for part, (logp, y) in result.items():
        before, after = softmax(logp), softmax(logp / t)
        report[part] = {"accuracy": float((before.argmax(1) == y).mean()),
                        "mean_confidence_before": float(before.max(1).mean()), "ece_before": ece(before, y),
                        "mean_confidence_after": float(after.max(1).mean()), "ece_after": ece(after, y)}
    print(json.dumps(report, indent=2))

    if not args.dry_run:
        path = Path(args.model_dir) / "label_map.json"
        meta = json.loads(path.read_text(encoding="utf-8"))
        meta.update({"temperature": t, "temperature_fitted_on": "validation",
                     "calibration_report": {k: report[k] for k in ("val", "test")}})
        path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        print(f"Updated {path}")
    return report


if __name__ == "__main__":
    main()

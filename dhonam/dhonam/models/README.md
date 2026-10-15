# Model artifacts

The trained model is **not** stored in this repository. It was trained on the gated Utavu Foundation Dholuo Speech Dataset, and its TF-IDF vocabulary is derived directly from the dataset's transcripts, so treat it under the same access conditions as the data (see "Sharing" below).

## What the app expects

```
models/
└── final/
    ├── label_map.json      # labels, text variant, model kind, run id, optional temperature
    ├── model.joblib        # fitted scikit-learn Pipeline: 'tfidf' (TfidfVectorizer) + 'clf' (LogisticRegression)
    ├── preprocess.py       # exported copy of the training preprocessing (not imported by the app)
    ├── predict.py          # exported command-line predictor (not used by the app)
    └── MODEL_CARD.md       # exported model card
```

Only `label_map.json` and `model.joblib` are required. The app reads them from `models/final/` by default, or from the directory named in the `DHOLUO_MODEL_DIR` environment variable.

The evaluated final model is run `main|lr_word|normalized|0`: word-level TF-IDF (unigrams, `min_df=1`, sublinear tf) with class-weighted logistic regression (`C=1`), trained on the `normalized` text variant.

## How to obtain it

The notebook writes the model during section 17.

1. Open your Kaggle notebook (`notebooks/dholuo_topic_classification_final.ipynb`) after a completed run, or re-run it (about 171 minutes on a T4 GPU).
2. In the Kaggle output panel, open `/kaggle/working/dholuo_topic/models/final/` and download the whole folder.
3. Also download `/kaggle/working/dholuo_topic/results/splits.csv` if you plan to calibrate (next section).
4. Place the folder here so that `models/final/label_map.json` and `models/final/model.joblib` exist.
5. Install the pinned versions in `requirements.txt`. The pipeline was pickled with scikit-learn 1.6.1 and NumPy 2.1; other scikit-learn versions may fail to load it or behave differently.

Only load `model.joblib` files you produced yourself: joblib files are Python pickles and can execute code when loaded.

## Calibrating confidences (recommended)

On the test speakers the raw probabilities of this model are badly under-confident: mean top-class probability 0.110 against accuracy 0.424, expected calibration error 0.314 (notebook section 13). The app therefore hides confidence percentages until a temperature fitted on the **validation** speakers is recorded in `label_map.json`.

```bash
pip install -r requirements-dev.txt
export HF_TOKEN=...            # read token from the account that accepted the dataset terms
python scripts/calibrate_temperature.py --model-dir models/final --splits path/to/splits.csv
```

The script rebuilds the validation and test sets from `splits.csv`, fits one temperature on validation, prints the calibration error on validation and test before and after, and writes `temperature` and `temperature_fitted_on: "validation"` into `label_map.json`. Use `--dry-run` to see the report without changing any file. Calibration never changes which topic is predicted.

This step has been tested against a stand-in model and synthetic data (`tests/test_calibration_script.py`); it has not yet been run against the real model, because the dataset cannot be reached from the environment the code was written in.

## Sharing

Before making the model public (for example on Hugging Face or with a deployed app), check that the NOODL-1.0 licence and the dataset's access conditions allow redistributing a model whose vocabulary is drawn from the transcripts. If in doubt, keep the model private and share only the code and these instructions.

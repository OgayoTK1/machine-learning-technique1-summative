# Model artifacts

The trained model is **not committed by default** (`models/final/` is in `.gitignore`). It was trained on the gated Utavu Foundation Dholuo Speech Dataset, and its TF-IDF vocabulary is derived directly from the dataset's transcripts, so treat it under the same access conditions as the data (see "Sharing" below).

## What the app expects

```
models/
└── final/
    ├── label_map.json      # labels, text variant, model kind, run id, temperature
    ├── model.joblib        # fitted scikit-learn Pipeline: 'tfidf' + 'clf' (about 7 MB)
    ├── preprocess.py       # exported copy of the training preprocessing (not imported by the app)
    ├── predict.py          # exported command-line predictor (not used by the app)
    └── MODEL_CARD.md       # exported model card
```

Only `label_map.json` and `model.joblib` are required. The app reads them from `models/final/` by default, or from the directory named in the `DHOLUO_MODEL_DIR` environment variable.

The evaluated final model is run `main|svm_wordchar|normalized|0`:

* `tfidf`: a `FeatureUnion` of word TF-IDF (1-2-grams, `min_df=1`, sublinear tf) and character TF-IDF (`char_wb` 2-5-grams, `min_df=2`, sublinear tf), on the `normalized` text variant;
* `clf`: a class-weighted `LinearSVC` (`C=0.1`);
* `temperature` in `label_map.json` (0.0803): a softmax temperature fitted on the validation speakers in notebook section 9, which turns the SVM margins into confidences.

Test results (5 unseen speakers): macro-F1 0.413 (95% interval 0.345 to 0.459), accuracy 0.448, macro-F1 0.344 on answers to prompts not seen in training. Confidences: mean 0.50 against accuracy 0.45, expected calibration error 0.10.

## How to obtain it

The notebook writes the model during section 17.

1. Open the notebook (`notebooks/dholuo_topic_classification_final.ipynb`) after a completed run, or re-run it (about 174 minutes on a T4 GPU).
2. Download the output folder `dholuo_topic/models/final/` (on Kaggle from the output panel; on Colab from the files panel, or zip it to Google Drive as the last cell of the notebook does).
3. Place the folder here so that `models/final/label_map.json` and `models/final/model.joblib` exist.
4. Install the pinned versions in `requirements.txt`. The pipeline was pickled with scikit-learn 1.6.1 and NumPy 2.1; other scikit-learn versions may fail to load it or behave differently.
5. Check it: `python -m pytest tests/test_app.py` runs an extra smoke test against the real model when it is present.

Only load `model.joblib` files you produced yourself: joblib files are Python pickles and can execute code when loaded.

## Calibration

The SVM export needs no extra step: its temperature was fitted on the validation speakers inside the notebook, and the app shows confidence percentages for it.

`scripts/calibrate_temperature.py` is only for a probabilistic export (for example the logistic-regression model, if it were selected in a future run). For such a model it rebuilds the validation and test sets from `splits.csv`, fits one temperature on validation, and records `temperature` and `temperature_fitted_on: "validation"` in `label_map.json`:

```bash
pip install -r requirements-dev.txt
export HF_TOKEN=...            # read token from the account that accepted the dataset terms
python scripts/calibrate_temperature.py --model-dir models/final --splits reports/results/splits.csv
```

Run against the SVM, the script stops with a message saying the model is already calibrated. Calibration never changes which topic is predicted.

## Sharing

Before making the model public (for example in a public GitHub repository, on Hugging Face or with a deployed app), check that the NOODL-1.0 licence and the dataset's access conditions allow redistributing a model whose vocabulary is drawn from the transcripts. If in doubt, keep the repository private, or keep the model out of Git and supply it to the deployment separately.

To commit it deliberately despite `.gitignore`:

```bash
git add -f models/final/label_map.json models/final/model.joblib models/final/MODEL_CARD.md models/final/preprocess.py models/final/predict.py
```

# DhoNam: speaker-independent topic classification of spontaneous Dholuo

DhoNam assigns a transcribed, spontaneously spoken Dholuo answer about farming to one of 23 agricultural sub-domains (for example poultry farming, fish farming, or water usage and irrigation), and is evaluated on speakers the model never saw during training. The project covers the full path from a gated low-resource dataset to an audited, speaker-disjoint benchmark, ten controlled experiments, and a small Streamlit application around the selected model.

> **What the model predicts.** The labels in the dataset belong to the *question* each speaker was asked, not to an independent annotation of the *answer*. The model therefore predicts the agricultural sub-domain that the dataset associates with a transcript's question. It does not establish the objectively correct topic of what was said, and some answers legitimately touch other topics (see [Limitations](#13-limitations-ethics-and-responsible-use)).

The full technical report is in [`reports/final_report.md`](reports/final_report.md) (PDF: [`reports/final_report.pdf`](reports/final_report.pdf)); it is being updated to the latest training run after deployment testing. Every number in this README comes from the latest executed notebook [`notebooks/dholuo_topic_classification_final.ipynb`](notebooks/dholuo_topic_classification_final.ipynb).

## Project links

| | |
|---|---|
| Live application | [https://machine-learning-technique1-summative-6lhzhgn9tuwa7wuhtrdjcf.streamlit.app/](https://machine-learning-technique1-summative-6lhzhgn9tuwa7wuhtrdjcf.streamlit.app/) |
| GitHub repository | [https://github.com/OgayoTK1/machine-learning-technique1-summative](https://github.com/OgayoTK1/machine-learning-technique1-summative) |
| Technical report | [`reports/final_report.pdf`](reports/final_report.pdf) |
| Notebook (all experiments) | [`notebooks/dholuo_topic_classification_final.ipynb`](notebooks/dholuo_topic_classification_final.ipynb) |
| Demonstration video | [https://www.youtube.com/watch?v=Fs2sN2lSI-w](https://www.youtube.com/watch?v=Fs2sN2lSI-w) |

---

## 1. Overview and motivation

Dholuo (Luo) is a Western Nilotic language spoken mainly in western Kenya and northern Tanzania. It is used mostly in informal settings, while government, education and commerce run largely in English and Kiswahili (Mbogho et al., 2025). Farmers who are most comfortable in Dholuo are therefore poorly served by text-based agricultural information systems. A model that can route a Dholuo question or comment to the right agricultural topic is a small, useful building block for extension services, farmer call lines and voice services.

## 2. Problem definition and objectives

**Research question.** How accurately can classical, recurrent and pretrained multilingual models assign transcribed, spontaneously spoken Dholuo agricultural answers to their sub-domain when every test speaker is unseen during training, and how does that accuracy change with training-set size, normalisation, dialect, code-switching, fine-tuning strategy, prompt overlap, and compared with existing machine-translation tools?

| | |
|---|---|
| Input | Dholuo transcript of one spoken answer (10 to 60 s of speech; 16 to 158 words) |
| Output | One of 23 sub-domains, plus a ranking of the others |
| Primary metric | Macro-F1 on speaker-disjoint test data |
| Other metrics | Accuracy, macro precision and recall, weighted F1, per-class scores, confusion matrix, bootstrap 95% intervals, seed variation |

Objectives: audit the dataset and choose a defensible task; build leakage-controlled splits; compare baselines, a recurrent model and pretrained transformers; run controlled experiments; analyse errors; deploy the selected model in a simple, honest interface.

## 3. Dataset

| | |
|---|---|
| Name | Utavu Foundation Dholuo Speech Dataset |
| Source | Hugging Face `MCAA1-MSU/twb_luo_ke`, revision `5cbcd61d844e29e80e5e574fa14a7c54603052c3` |
| Creators | Utavu Foundation and CLEAR Global, collected on the TWB Voice platform; stewarded by the Maseno Centre for Applied AI |
| Licence and access | NOODL-1.0 (Nwulite Obodo Open Data License). Gated: users must accept conditions and share contact details. Voice is biometric data; contributors must not be re-identified. |
| Used here | Free-form subset: 1,359 approved transcripts, 23 speakers, 1,019 unique prompts, 30 sub-domains, 10.64 h of audio |
| Not used for modelling | Image-prompt subset (1,029 transcripts, no topic labels, same speakers), used only for vocabulary statistics |

The repository's CSV files live at `<subset>/Files/transcripts.csv` and `<subset>/Files/meta.csv`, which differs from the dataset README; the notebook discovers the paths from the repository listing. **No data is included in this repository.**

## 4. Task and label definition

The target is `sub_domain`, restricted to the **23 sub-domains with at least 25 recordings and at least 5 speakers** (1,248 transcripts). The other 7 sub-domains (111 transcripts) are too small to learn or to evaluate on unseen speakers. Among kept classes the largest-to-smallest ratio is 3.3 and no speaker contributes more than 25% of any class. Every prompt maps to exactly one sub-domain.

## 5. Preprocessing and methodology

* **Audit.** No missing values, duplicates or rejected transcriptions in the free-form subset; inconsistent markers (`[cs]`, `[CS]`, `[[cs]`, `[Pause]`), punctuation spacing and capitalisation; 225 transcripts (17%) contain code-switching. Speakers often repeat the question: the median answer contains 60% of its prompt's words.
* **Text variants.** `minimal` (Unicode NFC and whitespace), `normalized` (also straight apostrophes, lower case, punctuation spacing, canonical `<cs>`, `<pause>`, `<unclear>` tokens), `normalized_notags` (markers removed, code-switched words kept). No stemming, stop-word removal or spelling correction. Implemented in [`src/preprocessing.py`](src/preprocessing.py), tested for exact parity with the notebook.
* **Speaker-disjoint split.** A seeded search over 20,000 speaker groupings chose train/validation/test speakers matching the overall class distribution with every class and both dialects in each held-out split: **810 / 188 / 250 transcripts from 13 / 5 / 5 speakers**. No speaker overlaps. 101 of 250 test answers respond to a prompt also answered in training (studied in Experiment H).
* **Selection protocol.** Only validation scores were used for choices. Final-model rule, fixed in advance: among models within one standard deviation of the best mean validation macro-F1, choose the least complex.
* **Statistics.** Bootstrap 95% intervals over test rows and over test speakers; paired bootstrap for model comparisons; three training seeds (13, 42, 2026) for neural models.

## 6. Models and baselines evaluated

| Level | Model | Notes |
|---|---|---|
| Floors | Majority class, stratified random | |
| Classical | TF-IDF + logistic regression (word 1-2-grams; char 2-5-grams), TF-IDF + linear SVM | Grid search on validation; SVM margins temperature-scaled |
| Recurrent | BiLSTM with additive attention | Embeddings learned from scratch |
| Pretrained | `Davlan/afro-xlmr-large-114L`: full fine-tuning, LoRA (r=16), frozen encoder | Luo listed among its pretraining languages |
| Pretrained | `FacebookAI/xlm-roberta-large`: full fine-tuning | Same architecture and tokenizer, no African adaptation |
| Existing tools | NLLB-200 (600M distilled) to English, then BART-large-MNLI zero-shot, or TF-IDF + LR on translations | Experiment I |

## 7. Verified results

Speaker-disjoint test set (250 transcripts, 5 unseen speakers). Means over seeds where several were run; intervals are row-bootstrap 95% intervals for each model's best-validation run.

| Model | Runs | Val macro-F1 | Test macro-F1 | Test accuracy |
|---|---|---|---|---|
| **TF-IDF + SVM (word + char), selected** | 1 | 0.432 | **0.413** [0.345, 0.459] | 0.448 |
| LoRA AfroXLMR | 3 | 0.430 ± 0.018 | 0.416 ± 0.005 | 0.436 |
| TF-IDF + LR (word) | 1 | 0.427 | 0.396 [0.329, 0.442] | 0.424 |
| TF-IDF + LR (char) | 1 | 0.423 | 0.371 | 0.408 |
| Full fine-tuning AfroXLMR | 3 | 0.360 ± 0.014 | 0.404 ± 0.020 | 0.431 |
| BiLSTM + attention | 3 | 0.337 ± 0.027 | 0.307 ± 0.022 | 0.339 |
| Frozen AfroXLMR + LR | 1 | 0.290 | 0.227 | 0.256 |
| Full fine-tuning XLM-R | 3 | 0.031 ± 0.018 | 0.046 ± 0.042 | 0.089 |
| Majority class | 1 | 0.005 | 0.005 | 0.056 |

The selected SVM's interval over whole test speakers is [0.367, 0.425]; its validation-fitted confidences have an expected calibration error of 0.10 on the test speakers.

Key findings:

* **The SVM is selected; simple and pretrained models tie.** The SVM has the best validation score, so the selection rule chose it. On test it is statistically tied with LoRA AfroXLMR (difference -0.007, interval -0.074 to +0.056), word-level logistic regression (+0.017) and full AfroXLMR fine-tuning (-0.014). It beats character-level logistic regression (+0.042), the BiLSTM (+0.124), the frozen encoder (+0.186) and XLM-R (+0.323), all with intervals excluding zero.
* **African-language pretraining matters.** With identical architecture and tokenizer, AfroXLMR full fine-tuning reaches 0.404 while XLM-R averages 0.046 and never learns the task.
* **Prompt overlap inflates the TF-IDF scores.** The SVM scores 0.501 on answers to prompts seen in training and 0.344 on unseen prompts; removing shared prompts from training lowers logistic regression from 0.396 to 0.311. The realistic figure for new questions is about 0.31 to 0.34. LoRA AfroXLMR is the exception (0.360 seen vs 0.416 unseen, overlapping intervals).
* **Existing tools do not solve the task.** NLLB + zero-shot BART: 0.181; NLLB + TF-IDF trained on translations: 0.318; both significantly below the SVM's 0.413.
* **Speaker leakage is real but modest.** A random split overstates macro-F1 by 0.036 (logistic regression) and 0.020 (BiLSTM).
* **No dialect or code-switching gap** was detected (overlapping intervals), while per-speaker accuracy varies from 0.28 to 0.49.
* **Calibration.** SVM margins are turned into confidences with a softmax temperature (0.080) fitted on the validation speakers. On test, mean confidence is 0.50 against accuracy 0.45 (ECE 0.10): slightly over-confident but usable.

All result tables: [`reports/REPORT_TABLES.md`](reports/REPORT_TABLES.md); the underlying CSV and JSON files are in [`reports/results/`](reports/results/). Figures: [`reports/figures/`](reports/figures/).

## 8. Repository structure

```
dhonam/
├── app.py                         Streamlit interface
├── src/
│   ├── __init__.py
│   ├── preprocessing.py           training-identical text normalisation
│   └── prediction.py              model loading, validation, prediction, calibration handling
├── scripts/
│   └── calibrate_temperature.py   fits a validation temperature and records it in label_map.json
├── models/
│   ├── README.md                  how to obtain and place the trained model
│   └── final/                     exported SVM model used by the app
├── notebooks/
│   └── dholuo_topic_classification_final.ipynb   executed notebook (source of all results)
├── reports/
│   ├── final_report.md            technical report (Markdown source)
│   ├── final_report.pdf           technical report (PDF)
│   ├── REPORT_TABLES.md           result tables exported by the notebook
│   ├── results/                   result CSV/JSON files exported by the notebook (no transcript text)
│   └── figures/                   figures exported by the notebook
├── configs/
│   └── config_resolved.json       exact configuration of the reported run
├── data/
│   └── README.md                  how to obtain the gated dataset (no data included)
├── tests/                         pytest suite (preprocessing, prediction, app, calibration)
├── requirements.txt
├── requirements-dev.txt
├── pytest.ini
├── LICENSE                        MIT, for the code only
└── .gitignore
```

## 9. Prerequisites and installation

Python 3.10 to 3.13 (the model needs scikit-learn 1.6.1, which has no packages for Python 3.14). Developed with 3.13; deployed on Streamlit Community Cloud with Python 3.12.

```bash
git clone https://github.com/OgayoTK1/machine-learning-technique1-summative.git
cd machine-learning-technique1-summative
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt      # add requirements-dev.txt for tests and calibration
```

## 10. Running the Streamlit app

Place the trained model first (section 12), then:

```bash
streamlit run app.py
```

The app opens at http://localhost:8501. To load the model from another folder:

```bash
DHOLUO_MODEL_DIR=/path/to/final streamlit run app.py
```

The app accepts a Dholuo transcript (markers such as `[cs]...[cs]`, `[pause]` and `[?]` are understood), applies the training preprocessing, and shows the most likely sub-domain and the next two. Confidence percentages are shown only when the model files contain a temperature fitted on validation data; otherwise it shows the ranking and explains why. It warns about very short or very long inputs and about text with few words known to the model, and it handles empty input and missing model files without crashing.

## 11. Running the tests

```bash
pip install -r requirements-dev.txt
python -m pytest
```

The tests need no data or trained model: they build small stand-in pipelines with the same structure as the exported models (word TF-IDF + logistic regression, and word + character TF-IDF + linear SVM). They check preprocessing parity with the notebook, prediction and input validation, model-loading errors, the calibration script, and the Streamlit app through Streamlit's testing harness. One extra smoke test runs the app against the real model when `models/final/` is present and is skipped otherwise.

## 12. Model and dataset setup

* **Model:** follow [`models/README.md`](models/README.md) to place `models/final/` (exported by notebook section 17). The exported SVM already carries its validation-fitted temperature, so no calibration step is needed.
* **Dataset (only to reproduce training):** see [`data/README.md`](data/README.md).

### Reproducing the experiments

All preprocessing, training, evaluation and export run in one notebook, in order:

| Stage | Notebook sections | Output |
|---|---|---|
| Data acquisition and audit | 2, 3 | `results/audit.json`, `results/data_provenance.json` |
| Cleaning, normalisation, speaker-disjoint split | 5, 6 | `results/cleaning_log.csv`, `results/splits.csv` |
| Training: TF-IDF baselines, BiLSTM, transformers | 9, 10, 11 | `results/all_runs.csv` |
| Model selection and test evaluation | 12, 13 | `results/exp_C_model_family.csv`, `results/final_per_class.csv` |
| Experiments A to I and ablation | 14 | `results/exp_*.csv` |
| Qualitative and error analysis | 15, 16 | `results/error_flags.csv`, `results/confusion_pairs.csv` |
| Export of the deployed model | 17 | `models/final/` |

1. Open `notebooks/dholuo_topic_classification_final.ipynb` on Kaggle or Colab with a T4 GPU and internet enabled, and add the `HF_TOKEN` secret.
2. Keep `RUN_PROFILE = "full"` in section 1 (the settings are recorded in [`configs/config_resolved.json`](configs/config_resolved.json)); seeds are fixed for Python, NumPy and PyTorch.
3. Run all cells. The full run took about 166 minutes; `RUN_PROFILE = "fast"` gives a shorter single-seed check.
4. Copy `models/final/` from the notebook output into this repository to serve the newly trained model.

Inference outside the app:

```bash
python models/final/predict.py "Jopith rech tiyo gi yawo mondo rech odongi maber"
```

### Deployment

The app is deployed on Streamlit Community Cloud from this repository (branch `main`, main file `app.py`, Python 3.12). Streamlit installs `requirements.txt` and the app loads `models/final/` at start-up. The deployed model is the evaluated run `main|svm_wordchar|normalized|0`.

## 13. Limitations, ethics and responsible use

* **Labels follow the question, not the answer.** The model predicts the question-associated sub-domain. Qualitative analysis found answers that drift to neighbouring topics and labels that do not match what was said.
* **Prompt echo.** Part of the TF-IDF scores comes from recognising question wording; expect about 0.31 to 0.34 macro-F1 on genuinely new questions.
* **Small, narrow data.** 23 speakers (18 aged 18 to 29, one over 40), 13 of them in training; agriculture only; one transcription team with inconsistent conventions. Results describe these speakers, not Dholuo in general.
* **Dialect.** Only 9 Milambo speakers (3 in training). No gap was detected, but a small one cannot be ruled out, and the two dialect fields in the metadata disagree for 14 of 23 speakers.
* **Accuracy.** About 45% of test answers are classified correctly; 74 of 250 are misclassified by every model studied.
* **Closed label set.** Every input receives one of the 23 topics, including off-topic or non-Dholuo text.
* **Privacy and licence.** Only text and pseudonymous IDs are used; no transcripts or audio are published. Follow NOODL-1.0 and the dataset's access conditions, including before sharing the trained model.
* **Use.** A low-stakes routing or search aid with a person checking the result. Not for decisions about people, such as eligibility for subsidies, credit or programmes.

## 14. External resources and original work

**External resources used (not my work):**

| Resource | Use in this project |
|---|---|
| Utavu Foundation Dholuo Speech Dataset (`MCAA1-MSU/twb_luo_ke`, NOODL-1.0) | All transcripts, labels and speaker metadata |
| `Davlan/afro-xlmr-large-114L` (Alabi et al., 2022) | Pretrained encoder for full fine-tuning, LoRA and the frozen-encoder model |
| `FacebookAI/xlm-roberta-large` (Conneau et al., 2020) | Pretrained encoder for the comparison without African adaptation |
| NLLB-200 600M distilled (NLLB Team, 2022) | Dholuo-to-English translation in Experiment I and the qualitative section |
| BART-large-MNLI | English zero-shot classifier in Experiment I |
| LoRA method (Hu et al., 2022) via the `peft` library | Parameter-efficient fine-tuning |
| scikit-learn, PyTorch, Hugging Face `transformers`, `datasets` and `huggingface_hub`, pandas, NumPy, SciPy, matplotlib | Modelling, evaluation and plotting |
| Streamlit and Streamlit Community Cloud | Web application and hosting |
| Temperature scaling (Guo et al., 2017) | Calibrating the SVM's confidences |

**My original work:** the task definition and label-eligibility rule; the data audit and provenance record; the text normalisation; the constrained speaker-disjoint split search; the BiLSTM-with-attention implementation; the plain-PyTorch fine-tuning loop with collapse detection and out-of-memory retry; the evaluation code (bootstrap intervals over rows and speakers, paired bootstrap, Wilson intervals, error flags and relative risk, occlusion explanations); experiments A to I and the class-weighting ablation, including the prompt-overlap analysis; model selection; the inference package in `src/`; the Streamlit app; the tests; and this documentation.

## 15. Acknowledgements and references

The Dholuo speakers who recorded, transcribed and reviewed the data, and the Utavu Foundation, CLEAR Global and the Maseno Centre for Applied AI, who created and steward the dataset.

* Utavu Foundation and CLEAR Global (2026). *Utavu Foundation Dholuo Speech Dataset.* Hugging Face, `MCAA1-MSU/twb_luo_ke`.
* Mbogho, A., Awuor, Q., Kipkebut, A., Wanzare, L., and Oloo, V. (2025). Building Low-Resource African Language Corpora: A Case Study of Kidaw'ida, Kalenjin and Dholuo. arXiv:2501.11003.
* Alabi, J. O., Adelani, D. I., Mosbach, M., and Klakow, D. (2022). Adapting Pre-trained Language Models to African Languages via Multilingual Adaptive Fine-Tuning. COLING 2022.
* Conneau, A., et al. (2020). Unsupervised Cross-lingual Representation Learning at Scale. ACL 2020.
* Hu, E. J., et al. (2022). LoRA: Low-Rank Adaptation of Large Language Models. ICLR 2022.
* NLLB Team (2022). No Language Left Behind: Scaling Human-Centered Machine Translation. arXiv:2207.04672.
* Adelani, D. I., et al. (2024). SIB-200: A Simple, Inclusive, and Big Evaluation Dataset for Topic Classification in 200+ Languages and Dialects. EACL 2024.
* Adelani, D. I., et al. (2023). MasakhaNEWS: News Topic Classification for African Languages. IJCNLP-AACL 2023.
* Guo, C., Pleiss, G., Sun, Y., and Weinberger, K. Q. (2017). On Calibration of Modern Neural Networks. ICML 2017.

The complete reference list is in the report.

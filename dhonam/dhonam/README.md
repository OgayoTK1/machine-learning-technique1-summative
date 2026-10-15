# DhoNam: speaker-independent topic classification of spontaneous Dholuo

DhoNam assigns a transcribed, spontaneously spoken Dholuo answer about farming to one of 23 agricultural sub-domains (for example poultry farming, fish farming, or water usage and irrigation), and is evaluated on speakers the model never saw during training. The project covers the full path from a gated low-resource dataset to an audited, speaker-disjoint benchmark, ten controlled experiments, and a small Streamlit application around the selected model.

> **What the model predicts.** The labels in the dataset belong to the *question* each speaker was asked, not to an independent annotation of the *answer*. The model therefore predicts the agricultural sub-domain that the dataset associates with a transcript's question. It does not establish the objectively correct topic of what was said, and some answers legitimately touch other topics (see [Limitations](#13-limitations-ethics-and-responsible-use)).

The full technical report is in [`reports/final_report.md`](reports/final_report.md) (PDF: [`reports/final_report.pdf`](reports/final_report.pdf)). Every number in this README comes from the executed notebook [`notebooks/dholuo_topic_classification_final.ipynb`](notebooks/dholuo_topic_classification_final.ipynb).

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
| LoRA AfroXLMR | 3 | 0.440 ± 0.019 | 0.436 ± 0.060 | 0.452 |
| TF-IDF + SVM (word + char) | 1 | 0.432 | 0.413 | 0.448 |
| **TF-IDF + LR (word), selected** | 1 | 0.427 | **0.396** [0.329, 0.442] | 0.424 |
| TF-IDF + LR (char) | 1 | 0.423 | 0.371 | 0.408 |
| Full fine-tuning AfroXLMR | 3 | 0.386 ± 0.025 | 0.416 ± 0.025 | 0.435 |
| BiLSTM + attention | 3 | 0.337 ± 0.027 | 0.307 ± 0.022 | 0.339 |
| Frozen AfroXLMR + LR | 1 | 0.290 | 0.227 | 0.256 |
| Full fine-tuning XLM-R | 3 | 0.112 ± 0.142 | 0.101 ± 0.135 | 0.149 |
| Majority class | 1 | 0.005 | 0.005 | 0.056 |

Key findings:

* **Simple and pretrained models tie.** The five leading models are statistically indistinguishable (paired-bootstrap intervals for the differences include zero), so the selection rule chose word-level TF-IDF + logistic regression. It beats the BiLSTM (+0.107), the frozen encoder (+0.168) and XLM-R (+0.140), all with intervals excluding zero.
* **African-language pretraining matters.** With identical architecture and tokenizer, AfroXLMR full fine-tuning reaches 0.416 while XLM-R averages 0.101 and collapses on one seed.
* **Prompt overlap inflates every score.** Logistic regression scores 0.421 on answers to prompts seen in training and 0.361 on unseen prompts; removing shared prompts from training lowers it from 0.396 to 0.311. The realistic figure for new questions is about 0.31 to 0.36.
* **Existing tools do not solve the task.** NLLB + zero-shot BART: 0.181; NLLB + TF-IDF trained on translations: 0.318; both significantly below 0.396.
* **Speaker leakage is real but modest.** A random split overstates macro-F1 by 0.036 (logistic regression) and 0.020 (BiLSTM).
* **No dialect or code-switching gap** was detected (overlapping intervals), while per-speaker accuracy varies from 0.28 to 0.49.
* **Calibration.** The selected model's raw probabilities are under-confident (mean confidence 0.110 vs accuracy 0.424; ECE 0.314), so the app hides percentages until a validation-fitted temperature is added.

All result tables: [`reports/REPORT_TABLES.md`](reports/REPORT_TABLES.md). Figures: [`reports/figures/`](reports/figures/).

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
│   └── README.md                  how to obtain and place the trained model (not committed)
├── notebooks/
│   └── dholuo_topic_classification_final.ipynb   executed Kaggle notebook (source of all results)
├── reports/
│   ├── final_report.md            technical report (Markdown source)
│   ├── final_report.pdf           technical report (PDF)
│   ├── REPORT_TABLES.md           result tables exported by the notebook
│   └── figures/                   figures from the notebook outputs
├── tests/                         pytest suite (preprocessing, prediction, app, calibration)
├── requirements.txt
├── requirements-dev.txt
├── pytest.ini
└── .gitignore
```

## 9. Prerequisites and installation

Python 3.10 or newer (developed and tested with Python 3.13).

```bash
cd dhonam
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

The tests need no data or trained model: they build a small stand-in pipeline with the same structure as the exported model. They check preprocessing parity with the notebook, prediction and input validation, model-loading errors, the calibration script, and the Streamlit app through Streamlit's testing harness.

## 12. Model and dataset setup

* **Model:** follow [`models/README.md`](models/README.md) to download `models/final/` from your Kaggle notebook output and, optionally, calibrate it.
* **Dataset (only to reproduce training or calibrate):** request access to `MCAA1-MSU/twb_luo_ke` on Hugging Face, create a read token, and add it to Kaggle as the secret `HF_TOKEN`. Run the notebook on Kaggle with a GPU T4 and internet enabled; the full run took 171 minutes.

## 13. Limitations, ethics and responsible use

* **Labels follow the question, not the answer.** The model predicts the question-associated sub-domain. Qualitative analysis found answers that drift to neighbouring topics and labels that do not match what was said.
* **Prompt echo.** Part of every score comes from recognising question wording; expect about 0.31 to 0.36 macro-F1 on genuinely new questions.
* **Small, narrow data.** 23 speakers (18 aged 18 to 29, one over 40), 13 of them in training; agriculture only; one transcription team with inconsistent conventions. Results describe these speakers, not Dholuo in general.
* **Dialect.** Only 9 Milambo speakers (3 in training). No gap was detected, but a small one cannot be ruled out, and the two dialect fields in the metadata disagree for 14 of 23 speakers.
* **Accuracy.** About 4 in 10 test answers are classified correctly; 77 of 250 are misclassified by every model studied.
* **Closed label set.** Every input receives one of the 23 topics, including off-topic or non-Dholuo text.
* **Privacy and licence.** Only text and pseudonymous IDs are used; no transcripts or audio are published. Follow NOODL-1.0 and the dataset's access conditions, including before sharing the trained model.
* **Use.** A low-stakes routing or search aid with a person checking the result. Not for decisions about people, such as eligibility for subsidies, credit or programmes.

## 14. Acknowledgements and references

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

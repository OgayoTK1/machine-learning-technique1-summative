# Dholuo ASR under realistic low-resource conditions

Speaker-independent automatic speech recognition for **Dholuo** (Milambo and Nyanduat dialects) on the
**DhoNam: Dholuo Speech dataset**, comparing a from-scratch neural model, zero-shot multilingual ASR, and
several ways of adapting multilingual self-supervised speech models — with leakage-controlled evaluation,
speaker-level confidence intervals, systematic error analysis and a deployed demo.

> **Research question.** How accurately can multilingual self-supervised speech models transcribe Dholuo
> speech from *unseen speakers*, how does accuracy differ between Milambo and Nyanduat speakers, and which
> adaptation strategy gives the best accuracy per unit of compute on a single Kaggle GPU?

| | |
|---|---|
| Input | 16 kHz mono speech, one speaker, ≤ 15 s (training) / ≤ 30 s (app) |
| Output | Dholuo transcript (character-level CTC) |
| Primary metric | WER on the speaker-independent test set, after a fixed evaluation normaliser |
| Secondary | CER; WER per dialect, gender, age, length, prompt novelty; 95 % CIs by **speaker** bootstrap |
| Baselines | empty output (WER 100 %), zero-shot MMS (if a `luo` adapter exists), BiLSTM-CTC from scratch |
| Primary model | XLS-R 300M fine-tuned with CTC (final choice made on **validation** WER in notebook 06) |

Results are produced by the notebooks; none are quoted here until they have been run on the real data.

---

## 1. Dataset (external resource — not redistributed)

**DhoNam: Dholuo Speech dataset**, created by Dholuo community participants with researchers from the
Maseno Centre for Applied Artificial Intelligence (MCAAI), Maseno University, and the Centre for Intellectual
Property and Information Technology Law (CIPIT), Strathmore University; supported by Mozilla Foundation and
GIZ FAIR Forward; distributed through **Mozilla Data Collective**.

* Licence: **NOODL-1.0** (Nwulite Obodo Open Data License). Users agree **not to re-host or re-share** the data.
* Therefore this repository contains **no audio and no transcripts**. See [`data/README.md`](data/README.md)
  for how to obtain the data and attach it to Kaggle privately.
* Documented composition (provider datasheet): adult native speakers (19+) with informed consent;
  Milambo 36 speakers / 16,697 recordings, Nyanduat 23 speakers / 9,394 recordings. Notebook 01 re-derives
  these numbers from the files and reports any disagreement.

## 2. Repository layout

```
configs/experiments.yaml     every training run, as base config + named overrides
src/dholuo_asr/              all project code (original work)
  schema.py                  metadata discovery -> canonical manifest
  audio.py                   decoding, resampling, quality statistics
  text.py                    transcript views & evaluation normaliser (keeps the ' in ng')
  codeswitch.py              conservative code-switching *candidate* flags (needs human verification)
  splits.py                  speaker-independent split, leakage report, nested data-size subsets
  vocab.py / data.py         character vocabulary, datasets, duration-bucketed batching
  bilstm.py                  from-scratch log-mel -> BiLSTM -> CTC model
  training.py                model construction (full / frozen / LoRA / MMS adapters) + training loop
  runner.py                  config -> complete saved run
  metrics.py                 WER/CER with alignments, speaker bootstrap, paired bootstrap
  errors.py                  error analysis (OOV, frequency, segmentation, confusions, sampling)
  lm.py                      word trigram LM written directly as ARPA (no lmplz needed)
  inference.py               loads any saved run for the app
notebooks/                   01 audit -> 02 EDA/splits -> 03 zero-shot -> 04 BiLSTM -> 05 fine-tuning -> 06 evaluation
app/app.py                   Gradio demo serving the selected model
tests/                       unit tests, synthetic smoke-test data, training sanity check
```

## 3. Running on Kaggle (recommended)

1. **Code:** push this repo to GitHub and set `REPO_URL` in the first cell of each notebook (Internet ON),
   *or* upload the repo as a private Kaggle dataset and attach it.
2. **Data:** upload the Mozilla Data Collective download as a **private** Kaggle dataset (never public).
3. Run the notebooks in order. Each notebook writes to `/kaggle/working`; attach a notebook's output as an
   input of the next one (**Add Input → Notebook Output**).

| notebook | accelerator | produces | needs |
|---|---|---|---|
| `01_data_audit` | CPU | `manifest.csv`, `audit_report.md`, `audit_summary.json`, `codeswitch_candidates.csv` | dataset |
| `02_eda_and_splits` | CPU | `splits.csv`, `split_report.json`, figures | 01 |
| `03_baseline_zero_shot` | GPU | `results/mms_zeroshot/` | 02 |
| `04_neural_baseline_bilstm` | GPU | `results/bilstm_ctc/` | 02 |
| `05_pretrained_finetuning` | GPU, one run per `EXPERIMENT` | `results/<experiment>/` | 02 |
| `06_evaluation_and_error_analysis` | CPU | `evaluation/` (tables, figures, LM ablation, summary) | 02 + all runs |

**Code-switching verification (you, as a Dholuo speaker):** fill the `verified` column of
`codeswitch_candidates.csv` (yes/no), save it as `codeswitch_candidates_verified.csv`, and attach it before
running notebook 02. Otherwise Experiment E is reported as *unverified*.

## 4. Local installation

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install --no-deps pyctcdecode pygtrie            # pyctcdecode pins an old numpy; skip its deps
pip install https://github.com/kpu/kenlm/archive/master.zip   # optional: LM ablation
pytest -q                                            # unit tests (~2 s)
python tests/overfit_check.py bilstm                 # training-loop sanity check (~3 min CPU)
```

Smoke-test every notebook on synthetic data (no real data needed):

```bash
python tests/make_fake_dataset.py /tmp/fake
export DHONAM_ROOT=/tmp/fake DHOLUO_OUT=/tmp/smoke
cd notebooks
EXPERIMENT=smoke_hf jupyter nbconvert --to notebook --execute 05_pretrained_finetuning.ipynb  # etc.
```

## 5. Experiments (all defined in `configs/experiments.yaml`)

| experiment | runs | single factor varied |
|---|---|---|
| A data size | `xlsr_full_frac10/25/50`, `xlsr_full` | number of training **speakers** (nested subsets, same update budget) |
| B normalisation | `xlsr_full_minimal_text` vs `xlsr_full` | training transcript view (scored with the same evaluation normaliser) |
| C model family | `mms_zeroshot`, `bilstm_ctc`, `xlsr_full`, `mms_adapter` | pre-training / architecture |
| D dialect | subgroup analysis of the selected model | Milambo vs Nyanduat test speakers |
| E code-switching | subgroup analysis | flagged vs unflagged utterances (verified flags only) |
| F leakage | `xlsr_full_random_split` | row-level random split vs speaker-independent split |
| G adaptation | `xlsr_frozen`, `xlsr_lora`, `xlsr_full`, `mms_adapter` | trainable parameters / compute |
| Prompt-disjoint | `xlsr_full_prompt_disjoint` | removes training sentences that occur in val/test |
| Ablation | LM decoding in notebook 06 | decoder only (greedy / beam / beam + trigram LM), acoustic model fixed |
| Variance | `xlsr_full_seed2/3` | random seed |

**Evaluation rules enforced in code:** model selection on validation only; no speaker in more than one split
(asserted); the random split is reported only as a leakage demonstration; models compared on identical test
utterances with a paired speaker bootstrap; LM trained on training transcripts only and also reported on
the unseen-prompt subset.

## 6. Inference and deployment

```bash
pip install gradio
MODEL_DIR=results/xlsr_full/model TEST_WER=0.XX python app/app.py      # http://localhost:7860
```

For Hugging Face Spaces: copy `app/`, `src/` and the selected `model/` folder, set `MODEL_DIR`. Before hosting
model weights publicly, confirm that NOODL-1.0 permits distributing a model trained on DhoNam; if unclear, keep
the Space private or ask the dataset maintainers. The app states its scope, shows a confidence value that is
explicitly *not* a guarantee, refuses audio over 30 s and does not store audio.

## 7. Reproducibility

* Every run saves its fully resolved `config.json`, `env.json` (package versions, GPU, git commit),
  `train_log.csv`, predictions, log-probabilities and metrics.
* Seeds: split seed 42 (notebook 02), training seeds in the config. The speaker split is deterministic and
  independent of row order (unit-tested).
* Tested with Python 3.13, torch 2.14, transformers 5.19, peft 0.21, numpy 2.5, pandas 3.0 (CPU smoke tests);
  Kaggle's preinstalled versions are recorded per run in `env.json`.

## 8. Limitations (known before any results)

* 59 speakers in total, so the test set holds only a handful per dialect: speaker-bootstrap intervals will be wide.
* Read speech of prompted sentences; prompts recur across speakers. Results on spontaneous Dholuo will be worse.
* Dholuo is tonal and tone is not written, so homographs cannot be disambiguated from text alone.
* Code-switching detection is a word-list heuristic until verified by a speaker.
* Kaggle's ~30 GPU-hours/week limits the number of seeds and the size of models that can be fully fine-tuned.

## 9. External resources vs original work

**External:** DhoNam dataset (MCAAI/CIPIT, Mozilla Data Collective); pretrained checkpoints
`facebook/wav2vec2-xls-r-300m` and `facebook/mms-1b-all`; PyTorch, Hugging Face Transformers, PEFT,
pyctcdecode, KenLM (loading only), soundfile/librosa, NumPy/SciPy/pandas, matplotlib, Gradio, jiwer (tests).
Cite the corresponding papers/licences in the report.

**Original to this project:** the audit, splitting, leakage analysis, transcript normalisation policy,
BiLSTM-CTC baseline, training/evaluation code, ARPA LM writer, error-analysis framework, experiment design,
app and all documentation in this repository.

## Licence

Code: MIT (see `LICENSE`). The DhoNam data is **not** covered by this licence and remains under NOODL-1.0.

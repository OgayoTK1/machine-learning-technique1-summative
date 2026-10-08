# Data access — DhoNam: Dholuo Speech dataset

This folder intentionally contains **no data**. The dataset is licensed under NOODL-1.0 and users agree not
to re-host or re-share it, so audio and transcripts must never be committed to this repository or published
on Kaggle/Hugging Face.

## How to obtain it
1. Create an account on Mozilla Data Collective and open the dataset page for
   "DhoNam: Dholuo Speech dataset".
2. Read and accept the NOODL-1.0 licence terms; download the archive.
3. **Kaggle:** *Datasets → New Dataset → upload the archive (or extracted folder)* with visibility **Private**.
   Attach it to `notebooks/01_data_audit.ipynb`; the notebook auto-detects the folder (or set `DATA_ROOT`).
4. **Local:** extract anywhere outside the repo and set `DHONAM_ROOT=/path/to/extracted/folder`.

## Synthetic stand-in for testing
`python tests/make_fake_dataset.py OUT_DIR` writes a tiny synthetic corpus (tones/noise and placeholder text,
with deliberate defects) that exercises every notebook. It is not Dholuo data and produces no results.

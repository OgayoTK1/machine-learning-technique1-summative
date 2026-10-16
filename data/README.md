# Data

No data is stored in this repository. The project uses the **Utavu Foundation Dholuo Speech Dataset** (`MCAA1-MSU/twb_luo_ke` on Hugging Face, revision `5cbcd61d844e29e80e5e574fa14a7c54603052c3`), created by the Utavu Foundation and CLEAR Global and stewarded by the Maseno Centre for Applied AI. It is gated and licensed under NOODL-1.0.

## Access

1. Sign in to Hugging Face, open the dataset page and accept the access conditions.
2. Create a read token (Settings, Access Tokens).
3. Add it as a secret named `HF_TOKEN` in Kaggle (Add-ons, Secrets) or Colab (the key icon), or set it as an environment variable.

The notebook downloads only the four small CSV tables (free-form and image-prompt transcripts and speaker metadata), not the 5.85 GB of audio. The files are located from the repository listing, because the real paths (`<subset>/Files/transcripts.csv`, `<subset>/Files/meta.csv`) differ from the dataset README.

## What the repository does contain

* `reports/results/splits.csv`: the speaker-disjoint train/validation/test assignment (file names and pseudonymous speaker IDs only), so the split can be reused.
* Result tables and figures in `reports/`, which contain metrics and identifiers, not transcript text.

Voice recordings are biometric data. Do not attempt to identify contributors, and follow the dataset's conditions before redistributing anything derived from it.

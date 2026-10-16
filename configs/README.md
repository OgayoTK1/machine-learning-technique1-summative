# Configuration

`config_resolved.json` is the exact configuration of the reported run, saved by the notebook after every validation-based choice was made: seeds, split settings, eligibility thresholds, TF-IDF grids, BiLSTM and transformer hyperparameters, and the values selected on validation.

The notebook defines the same settings in one dictionary, `CFG` (section 1). To reproduce the reported run, keep `RUN_PROFILE = "full"` and the values below unchanged. The paths (`/kaggle/working/...`) are the notebook's output folders and can be changed without affecting results.

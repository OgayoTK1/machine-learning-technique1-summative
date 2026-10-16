## Dataset & split
| split   |   rows |   speakers |   classes_present |   Nyanduat_spk |   Milambo_spk |   female_spk |   cs_rows |   mean_words |
|:--------|-------:|-----------:|------------------:|---------------:|--------------:|-------------:|----------:|-------------:|
| train   |    810 |         13 |                23 |             10 |             3 |            6 |       113 |         52   |
| val     |    188 |          5 |                23 |              2 |             3 |            2 |        38 |         79.6 |
| test    |    250 |          5 |                23 |              2 |             3 |            3 |        52 |         70.9 |

Leakage: speaker overlap 0; test rows whose prompt appears in train: 101/250; near-duplicate test rows (char cosine > 0.9): 0.

## Experiment C - model family (official test, mean over seeds; CI for best-val run)
| model                      |   runs |   val_macro_f1 |   val_sd |   test_macro_f1 |   test_sd |   test_acc |   test_weighted_f1 | row_boot_95ci   | speaker_boot_95ci   |
|:---------------------------|-------:|---------------:|---------:|----------------:|----------:|-----------:|-------------------:|:----------------|:--------------------|
| svm_wordchar               |      1 |         0.4324 |   0      |          0.4129 |    0      |     0.448  |             0.4144 | [0.345, 0.459]  | [0.367, 0.425]      |
| lora_afroxlmr_large_114L   |      3 |         0.4301 |   0.0177 |          0.4155 |    0.0051 |     0.436  |             0.4198 | [0.352, 0.470]  | [0.394, 0.454]      |
| lr_word                    |      1 |         0.4271 |   0      |          0.3957 |    0      |     0.424  |             0.4087 | [0.329, 0.442]  | [0.375, 0.425]      |
| lr_char                    |      1 |         0.4225 |   0      |          0.3705 |    0      |     0.408  |             0.373  | [0.306, 0.414]  | [0.329, 0.396]      |
| full_afroxlmr_large_114L   |      3 |         0.3604 |   0.014  |          0.4041 |    0.0199 |     0.4307 |             0.4172 | [0.362, 0.472]  | [0.376, 0.453]      |
| bilstm                     |      3 |         0.3374 |   0.0266 |          0.3066 |    0.0218 |     0.3387 |             0.3183 | [0.234, 0.331]  | [0.258, 0.294]      |
| frozen_afroxlmr_large_114L |      1 |         0.2904 |   0      |          0.2273 |    0      |     0.256  |             0.2392 | [0.177, 0.266]  | [0.178, 0.267]      |
| dummy_stratified           |      1 |         0.0386 |   0      |          0.0354 |    0      |     0.048  |             0.0432 | [0.017, 0.055]  | [0.023, 0.045]      |
| full_xlmr_large            |      3 |         0.031  |   0.0179 |          0.0461 |    0.0421 |     0.0893 |             0.0546 | [0.055, 0.119]  | [0.067, 0.103]      |
| dummy_most_frequent        |      1 |         0.0052 |   0      |          0.0046 |    0      |     0.056  |             0.0059 | [0.002, 0.007]  | [0.003, 0.006]      |

**Selected model:** svm_wordchar (run `main|svm_wordchar|normalized|0`)

## Experiment A - data size
|                                    |   n_train |   val_f1 |   val_sd |   test_f1 |   test_sd |
|:-----------------------------------|----------:|---------:|---------:|----------:|----------:|
| ('bilstm', 0.1)                    |        82 |   0.0749 |   0.0295 |    0.0702 |    0.0119 |
| ('bilstm', 0.25)                   |       203 |   0.1947 |   0.0283 |    0.178  |    0.0208 |
| ('bilstm', 0.5)                    |       408 |   0.2571 |   0.0247 |    0.2442 |    0.0234 |
| ('bilstm', 0.75)                   |       607 |   0.2822 |   0.0274 |    0.2454 |    0.0359 |
| ('bilstm', 1.0)                    |       810 |   0.3374 |   0.0266 |    0.3066 |    0.0218 |
| ('full_afroxlmr_large_114L', 0.1)  |        82 |   0.0388 | nan      |    0.0393 |  nan      |
| ('full_afroxlmr_large_114L', 0.25) |       203 |   0.1539 | nan      |    0.1551 |  nan      |
| ('full_afroxlmr_large_114L', 0.5)  |       408 |   0.3148 | nan      |    0.3312 |  nan      |
| ('full_afroxlmr_large_114L', 0.75) |       607 |   0.3628 | nan      |    0.3294 |  nan      |
| ('full_afroxlmr_large_114L', 1.0)  |       810 |   0.3447 | nan      |    0.3942 |  nan      |
| ('lr_word', 0.1)                   |        82 |   0.2295 |   0.0292 |    0.2173 |    0.0234 |
| ('lr_word', 0.25)                  |       203 |   0.2802 |   0.0332 |    0.2881 |    0.0214 |
| ('lr_word', 0.5)                   |       408 |   0.3615 |   0.0192 |    0.355  |    0.0382 |
| ('lr_word', 0.75)                  |       607 |   0.3824 |   0.0144 |    0.3689 |    0.0079 |
| ('lr_word', 1.0)                   |       810 |   0.4271 |   0      |    0.3957 |    0      |

## Experiment B - preprocessing
|                                  |   runs |   val_f1 |   val_sd |   test_f1 |   test_sd |
|:---------------------------------|-------:|---------:|---------:|----------:|----------:|
| ('bilstm', 'minimal')            |      3 |   0.3032 |   0.0324 |    0.3074 |    0.0154 |
| ('bilstm', 'normalized')         |      3 |   0.3374 |   0.0266 |    0.3066 |    0.0218 |
| ('bilstm', 'normalized_notags')  |      3 |   0.3284 |   0.0154 |    0.3199 |    0.0242 |
| ('lr_word', 'minimal')           |      1 |   0.4244 | nan      |    0.3918 |  nan      |
| ('lr_word', 'normalized')        |      1 |   0.4271 | nan      |    0.3957 |  nan      |
| ('lr_word', 'normalized_notags') |      1 |   0.423  | nan      |    0.3911 |  nan      |

## Ablation - class weighting
|                    |   val_f1 |   test_f1 |   test_sd |   test_macro_r |   test_acc |
|:-------------------|---------:|----------:|----------:|---------------:|-----------:|
| ('bilstm', False)  |   0.3277 |    0.2972 |    0.028  |         0.3087 |     0.332  |
| ('bilstm', True)   |   0.3374 |    0.3066 |    0.0218 |         0.3328 |     0.3387 |
| ('lr_word', False) |   0.3708 |    0.3809 |    0      |         0.4001 |     0.424  |
| ('lr_word', True)  |   0.4271 |    0.3957 |    0      |         0.424  |     0.424  |

## Experiment F - speaker-grouped vs random CV (dev speakers)
| model   |   random rows |   speaker-grouped |   inflation_random_minus_speaker |
|:--------|--------------:|------------------:|---------------------------------:|
| bilstm  |        0.2796 |            0.2593 |                           0.0203 |
| lr_word |        0.4041 |            0.3684 |                           0.0357 |

## Experiment D - dialect (test)
| model                      | dialect   |   n |   speakers |    acc |   macro_f1 | macro_f1_95ci   |
|:---------------------------|:----------|----:|-----------:|-------:|-----------:|:----------------|
| svm_wordchar               | Milambo   | 177 |          3 | 0.4576 |     0.412  | [0.334, 0.470]  |
| svm_wordchar               | Nyanduat  |  73 |          2 | 0.4247 |     0.3948 | [0.309, 0.521]  |
| lora_afroxlmr_large_114L   | Milambo   | 177 |          3 | 0.435  |     0.4064 | [0.325, 0.453]  |
| lora_afroxlmr_large_114L   | Nyanduat  |  73 |          2 | 0.4384 |     0.4448 | [0.327, 0.546]  |
| lr_word                    | Milambo   | 177 |          3 | 0.435  |     0.3947 | [0.313, 0.446]  |
| lr_word                    | Nyanduat  |  73 |          2 | 0.3973 |     0.3949 | [0.307, 0.526]  |
| lr_char                    | Milambo   | 177 |          3 | 0.4181 |     0.3698 | [0.299, 0.421]  |
| lr_char                    | Nyanduat  |  73 |          2 | 0.3836 |     0.3901 | [0.293, 0.497]  |
| full_afroxlmr_large_114L   | Milambo   | 177 |          3 | 0.4463 |     0.4006 | [0.332, 0.447]  |
| full_afroxlmr_large_114L   | Nyanduat  |  73 |          2 | 0.4932 |     0.4583 | [0.358, 0.591]  |
| bilstm                     | Milambo   | 177 |          3 | 0.3333 |     0.284  | [0.216, 0.330]  |
| bilstm                     | Nyanduat  |  73 |          2 | 0.3288 |     0.3035 | [0.205, 0.397]  |
| frozen_afroxlmr_large_114L | Milambo   | 177 |          3 | 0.2712 |     0.2245 | [0.167, 0.275]  |
| frozen_afroxlmr_large_114L | Nyanduat  |  73 |          2 | 0.2192 |     0.1973 | [0.117, 0.279]  |
| full_xlmr_large            | Milambo   | 177 |          3 | 0.113  |     0.0803 | [0.047, 0.111]  |
| full_xlmr_large            | Nyanduat  |  73 |          2 | 0.1781 |     0.0855 | [0.042, 0.142]  |

## Experiment E - code-switching (test)
| model                      | cs_group      |   n |   speakers |    acc |   macro_f1 | macro_f1_95ci   |
|:---------------------------|:--------------|----:|-----------:|-------:|-----------:|:----------------|
| svm_wordchar               | Dholuo only   | 198 |          5 | 0.4444 |     0.3939 | [0.315, 0.442]  |
| svm_wordchar               | code-switched |  52 |          5 | 0.4615 |     0.4496 | [0.318, 0.570]  |
| lora_afroxlmr_large_114L   | Dholuo only   | 198 |          5 | 0.4242 |     0.394  | [0.314, 0.449]  |
| lora_afroxlmr_large_114L   | code-switched |  52 |          5 | 0.4808 |     0.4569 | [0.338, 0.591]  |
| lr_word                    | Dholuo only   | 198 |          5 | 0.4394 |     0.3961 | [0.320, 0.444]  |
| lr_word                    | code-switched |  52 |          5 | 0.3654 |     0.3648 | [0.233, 0.479]  |
| lr_char                    | Dholuo only   | 198 |          5 | 0.4091 |     0.3597 | [0.285, 0.408]  |
| lr_char                    | code-switched |  52 |          5 | 0.4038 |     0.4027 | [0.263, 0.522]  |
| full_afroxlmr_large_114L   | Dholuo only   | 198 |          5 | 0.4646 |     0.421  | [0.348, 0.476]  |
| full_afroxlmr_large_114L   | code-switched |  52 |          5 | 0.4423 |     0.3874 | [0.281, 0.531]  |
| bilstm                     | Dholuo only   | 198 |          5 | 0.3283 |     0.2619 | [0.208, 0.302]  |
| bilstm                     | code-switched |  52 |          5 | 0.3462 |     0.3061 | [0.197, 0.431]  |
| frozen_afroxlmr_large_114L | Dholuo only   | 198 |          5 | 0.2626 |     0.2225 | [0.167, 0.265]  |
| frozen_afroxlmr_large_114L | code-switched |  52 |          5 | 0.2308 |     0.211  | [0.107, 0.310]  |
| full_xlmr_large            | Dholuo only   | 198 |          5 | 0.1364 |     0.0782 | [0.049, 0.104]  |
| full_xlmr_large            | code-switched |  52 |          5 | 0.1154 |     0.0904 | [0.031, 0.152]  |

## Experiment H - prompt overlap
| model                      | prompt_group         |   n |   speakers |    acc |   macro_f1 | macro_f1_95ci   |
|:---------------------------|:---------------------|----:|-----------:|-------:|-----------:|:----------------|
| svm_wordchar               | prompt seen in train | 101 |          5 | 0.5545 |     0.5014 | [0.399, 0.587]  |
| svm_wordchar               | prompt unseen        | 149 |          5 | 0.3758 |     0.3436 | [0.273, 0.402]  |
| lora_afroxlmr_large_114L   | prompt seen in train | 101 |          5 | 0.4554 |     0.36   | [0.278, 0.442]  |
| lora_afroxlmr_large_114L   | prompt unseen        | 149 |          5 | 0.4228 |     0.4159 | [0.329, 0.483]  |
| lr_word                    | prompt seen in train | 101 |          5 | 0.495  |     0.421  | [0.328, 0.509]  |
| lr_word                    | prompt unseen        | 149 |          5 | 0.3758 |     0.3606 | [0.275, 0.422]  |
| lr_char                    | prompt seen in train | 101 |          5 | 0.5446 |     0.4913 | [0.384, 0.563]  |
| lr_char                    | prompt unseen        | 149 |          5 | 0.3154 |     0.2922 | [0.222, 0.346]  |
| full_afroxlmr_large_114L   | prompt seen in train | 101 |          5 | 0.5644 |     0.5067 | [0.411, 0.614]  |
| full_afroxlmr_large_114L   | prompt unseen        | 149 |          5 | 0.3893 |     0.3725 | [0.292, 0.436]  |
| bilstm                     | prompt seen in train | 101 |          5 | 0.4356 |     0.3775 | [0.283, 0.457]  |
| bilstm                     | prompt unseen        | 149 |          5 | 0.2617 |     0.234  | [0.173, 0.285]  |
| frozen_afroxlmr_large_114L | prompt seen in train | 101 |          5 | 0.3069 |     0.2284 | [0.159, 0.299]  |
| frozen_afroxlmr_large_114L | prompt unseen        | 149 |          5 | 0.2215 |     0.2144 | [0.150, 0.273]  |
| full_xlmr_large            | prompt seen in train | 101 |          5 | 0.1683 |     0.101  | [0.064, 0.143]  |
| full_xlmr_large            | prompt unseen        | 149 |          5 | 0.1074 |     0.0708 | [0.036, 0.105]  |

|                                        |   n_train |   test_f1 |   test_sd |   test_acc |
|:---------------------------------------|----------:|----------:|----------:|-----------:|
| ('bilstm', 'speaker+prompt-disjoint')  |       709 |    0.2738 |    0.005  |     0.3013 |
| ('bilstm', 'speaker-disjoint')         |       810 |    0.3066 |    0.0218 |     0.3387 |
| ('lr_word', 'speaker+prompt-disjoint') |       709 |    0.3111 |    0      |     0.352  |
| ('lr_word', 'speaker-disjoint')        |       810 |    0.3957 |    0      |     0.424  |

Prompt echo (audit): {'median_share_of_prompt_words_in_transcript': 0.6, 'share_of_transcripts_covering_>=80%_of_prompt_words': 0.149, 'share_of_transcripts_opening_with_the_prompt': 0.335}

## Experiment G - fine-tuning strategy
| model                      |   val_f1 |   val_f1_sd |   test_f1 |   test_f1_sd |   trainable_params |   train_time_s |   peak_gpu_gb |   runs |
|:---------------------------|---------:|------------:|----------:|-------------:|-------------------:|---------------:|--------------:|-------:|
| frozen_afroxlmr_large_114L |   0.2904 |    nan      |    0.2273 |     nan      |    23575           |         58.9   |        nan    |      1 |
| full_afroxlmr_large_114L   |   0.3604 |      0.014  |    0.4041 |       0.0199 |        3.03912e+08 |        577.167 |          7.15 |      3 |
| lora_afroxlmr_large_114L   |   0.4301 |      0.0177 |    0.4155 |       0.0051 |        2.64604e+06 |        346.033 |          4.51 |      3 |

## Tokenizer fertility


## Error analysis
| flag                                               |   rows_flagged |   err_rate_flagged | ci_flagged   |   err_rate_rest | ci_rest      |   relative_risk |   share_of_all_errors |
|:---------------------------------------------------|---------------:|-------------------:|:-------------|----------------:|:-------------|----------------:|----------------------:|
| prompt unseen in training                          |            149 |              0.624 | [0.54, 0.70] |           0.446 | [0.35, 0.54] |           1.401 |                 0.674 |
| gold class has low training support                |             61 |              0.639 | [0.51, 0.75] |           0.524 | [0.45, 0.59] |           1.221 |                 0.283 |
| long (>= train Q3 words)                           |            142 |              0.556 | [0.47, 0.64] |           0.546 | [0.45, 0.64] |           1.018 |                 0.572 |
| code-switched                                      |             52 |              0.538 | [0.41, 0.67] |           0.556 | [0.49, 0.62] |           0.969 |                 0.203 |
| dialect = Milambo                                  |            177 |              0.542 | [0.47, 0.61] |           0.575 | [0.46, 0.68] |           0.943 |                 0.696 |
| gender = Female                                    |            116 |              0.534 | [0.44, 0.62] |           0.567 | [0.48, 0.65] |           0.942 |                 0.449 |
| transcript was editor-corrected                    |            115 |              0.522 | [0.43, 0.61] |           0.578 | [0.49, 0.66] |           0.903 |                 0.435 |
| high OOV (> train-median OOV of test)              |            123 |              0.52  | [0.43, 0.61] |           0.583 | [0.50, 0.66] |           0.893 |                 0.464 |
| short (<= train Q1 words)                          |             21 |              0.429 | [0.24, 0.63] |           0.563 | [0.50, 0.63] |           0.761 |                 0.065 |
| answer barely echoes prompt (<30% of prompt words) |              9 |              0.333 | [0.12, 0.65] |           0.56  | [0.50, 0.62] |           0.595 |                 0.022 |
| speech-rate outlier (possible transcript issue)    |              0 |            nan     | [nan, nan]   |           0.552 | [0.49, 0.61] |         nan     |                 0     |

Spearman(similarity, confusion) = 0.247 (p=7.15e-05)

## Key findings
- **C · Does model sophistication pay off?** Best test macro-F1: lora_afroxlmr_large_114L (0.416); selected on validation: svm_wordchar (0.413); majority baseline 0.005. svm_wordchar - lora_afroxlmr_large_114L: Δ = -0.007 (95% CI [-0.074, +0.056]; interval includes 0 → within noise) · svm_wordchar - lr_word: Δ = +0.017 (95% CI [-0.031, +0.061]; interval includes 0 → within noise) · svm_wordchar - lr_char: Δ = +0.042 (95% CI [+0.006, +0.076]; interval excludes 0) · svm_wordchar - full_afroxlmr_large_114L: Δ = -0.014 (95% CI [-0.073, +0.041]; interval includes 0 → within noise) · svm_wordchar - bilstm: Δ = +0.124 (95% CI [+0.055, +0.187]; interval excludes 0) · svm_wordchar - frozen_afroxlmr_large_114L: Δ = +0.186 (95% CI [+0.120, +0.243]; interval excludes 0) · svm_wordchar - full_xlmr_large: Δ = +0.323 (95% CI [+0.255, +0.379]; interval excludes 0)
- **A · How fast does performance grow with data?** bilstm: 0.070 at 82 rows → 0.307 at 810 rows; last step +0.061 (still rising - more Dholuo data would likely help); full_afroxlmr_large_114L: 0.039 at 82 rows → 0.394 at 810 rows; last step +0.065 (still rising - more Dholuo data would likely help); lr_word: 0.217 at 82 rows → 0.396 at 810 rows; last step +0.027 (still rising - more Dholuo data would likely help)
- **B · Does normalisation help?** bilstm: test macro-F1 0.307–0.320 across variants (spread 0.013; differences are within seed noise); lr_word: test macro-F1 0.391–0.396 across variants (spread 0.005; best variant on test: normalized). Variant used for all models (chosen on validation): normalized.
- **Ablation · Does class weighting help?** bilstm: 0.297 → 0.307 with weighting (+0.009); macro recall 0.309 → 0.333; lr_word: 0.381 → 0.396 with weighting (+0.015); macro recall 0.400 → 0.424
- **F · How much does speaker leakage inflate scores?** bilstm: random-row CV 0.280 vs speaker-grouped 0.259 (random minus speaker: +0.020); lr_word: random-row CV 0.404 vs speaker-grouped 0.368 (random minus speaker: +0.036) → a random split overstates performance, so speaker-independent evaluation is necessary.
- **D · Does dialect matter?** bilstm: Milambo 0.287 [0.232, 0.332] vs Nyanduat 0.285 [0.251, 0.311]; lr_word: Milambo 0.373 [0.310, 0.421] vs Nyanduat 0.372 [0.340, 0.403] (dev speaker-grouped OOF). Per-speaker accuracy spans 0.21; speaker-to-speaker spread is larger than the dialect gap, so individual speakers matter more than dialect at this scale.
- **E · Does code-switching cause failures?** bilstm: Dholuo-only 0.279 vs code-switched 0.302 (CIs overlap → no systematic failure detected); lr_word: Dholuo-only 0.367 vs code-switched 0.355 (CIs overlap → no systematic failure detected) (dev speaker-grouped OOF).
- **G · Best fine-tuning strategy?** frozen_afroxlmr_large_114L: test 0.227 ± 0.000, 23,575 trainable params, 59s/run; full_afroxlmr_large_114L: test 0.404 ± 0.020, 303,911,959 trainable params, 577s/run; lora_afroxlmr_large_114L: test 0.415 ± 0.005, 2,646,039 trainable params, 346s/run → best test mean: lora_afroxlmr_large_114L.
- **H · Is the model reading the answer or recognising the question?** svm_wordchar: macro-F1 0.501 on answers to prompts seen in training vs 0.344 on unseen prompts. Removing shared prompts from training: bilstm: 0.307 → 0.274 (-0.033); lr_word: 0.396 → 0.311 (-0.085). Median answer repeats 60% of its prompt's words → part of the headline score comes from recognising question wording; report the unseen-prompt figure as the realistic estimate for new questions.
- **I · Do existing tools already solve it?** nllb+zeroshot_bart: test 0.181 [0.133, 0.219]; nllb+tfidf_lr: test 0.318 [0.260, 0.363]; svm_wordchar: test 0.413 [0.345, 0.459]; lr_word: test 0.396 [0.329, 0.442] · svm_wordchar - nllb+zeroshot_bart: Δ = +0.232 (95% CI [+0.161, +0.297]; interval excludes 0) · svm_wordchar - nllb+tfidf_lr: Δ = +0.095 (95% CI [+0.035, +0.150]; interval excludes 0) → the Dholuo-trained model clearly beats the existing-tools routes, so the task is not already solved by MT + English NLP.
- **Errors · What drives mistakes?** Error rate 55.2%. Highest relative risks: prompt unseen in training (RR 1.40, n=149, CI [0.54, 0.70] vs [0.35, 0.54]); gold class has low training support (RR 1.22, n=61, CI [0.51, 0.75] vs [0.45, 0.59]); long (>= train Q3 words) (RR 1.02, n=142, CI [0.47, 0.64] vs [0.45, 0.64]). Lexically similar classes are confused more (Spearman ρ = 0.25, p = 7.1e-05). 74 of 250 test answers are wrong for every model - candidates for label/topic ambiguity (read these first when filling `why (human)`).

## Experiment I - existing tools
| system             | uses Dholuo training data   |   val_macro_f1 |   test_macro_f1 | test_95ci      |   test_acc |
|:-------------------|:----------------------------|---------------:|----------------:|:---------------|-----------:|
| nllb+zeroshot_bart | False                       |         0.2438 |          0.1814 | [0.133, 0.219] |      0.224 |
| nllb+tfidf_lr      | True                        |         0.4232 |          0.3183 | [0.260, 0.363] |      0.364 |
| svm_wordchar       | True                        |         0.4324 |          0.4129 | [0.345, 0.459] |      0.448 |
| lr_word            | True                        |         0.4271 |          0.3957 | [0.329, 0.442] |      0.424 |

## Hyperparameters
|                         | TF-IDF + LR (word)          | TF-IDF + SVM                | BiLSTM + attention   | Transformer full FT          | Transformer LoRA            |
|:------------------------|:----------------------------|:----------------------------|:---------------------|:-----------------------------|:----------------------------|
| ngram                   | [1, 1]                      |                             |                      |                              |                             |
| min_df                  | 1                           |                             |                      |                              |                             |
| C                       | 1                           | 0.1                         |                      |                              |                             |
| class_weight            | balanced                    | balanced                    | balanced             | balanced                     |                             |
| solver                  | lbfgs, max_iter 5000        |                             |                      |                              |                             |
| selection               | grid on validation macro-F1 | grid on validation macro-F1 |                      |                              |                             |
| emb_dim                 |                             |                             | 128.0                |                              |                             |
| hidden                  |                             |                             | 128.0                |                              |                             |
| dropout                 |                             |                             | 0.4                  |                              |                             |
| lr                      |                             |                             | 0.002                | 2e-05                        | 0.0002                      |
| weight_decay            |                             |                             | 0.0001               | 0.01                         |                             |
| batch_size              |                             |                             | 32.0                 | 8.0                          |                             |
| max_epochs              |                             |                             | 40.0                 | 20.0                         | 20.0                        |
| patience                |                             |                             | 6.0                  | 5.0                          | 5.0                         |
| optimizer               |                             |                             | AdamW                | AdamW                        |                             |
| scheduler               |                             |                             | none (constant LR)   | linear warmup + linear decay |                             |
| max_len                 |                             |                             | 200.0                | 256.0                        |                             |
| min_freq                |                             |                             | 2.0                  |                              |                             |
| grad_clip               |                             |                             | 1.0                  |                              |                             |
| seeds                   |                             |                             | [13, 42, 2026]       | [13, 42, 2026]               | [13, 42, 2026]              |
| checkpoint              |                             |                             |                      | Davlan/afro-xlmr-large-114L  | Davlan/afro-xlmr-large-114L |
| grad_accum              |                             |                             |                      | 2.0                          |                             |
| effective_batch         |                             |                             |                      | 16.0                         |                             |
| warmup_ratio            |                             |                             |                      | 0.1                          |                             |
| fp16                    |                             |                             |                      | True                         |                             |
| frozen_input_embeddings |                             |                             |                      | True                         |                             |
| gradient_checkpointing  |                             |                             |                      | True                         |                             |
| lora_r                  |                             |                             |                      |                              | 16.0                        |
| lora_alpha              |                             |                             |                      |                              | 32.0                        |
| lora_dropout            |                             |                             |                      |                              | 0.1                         |
| target_modules          |                             |                             |                      |                              | ['query', 'value']          |

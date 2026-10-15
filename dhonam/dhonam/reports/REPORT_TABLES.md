# Result tables exported by the notebook

Copied verbatim (apart from punctuation) from the output of section 18 of `notebooks/dholuo_topic_classification_final.ipynb`.

## Dataset & split
| split   |   rows |   speakers |   classes_present |   Nyanduat_spk |   Milambo_spk |   female_spk |   cs_rows |   mean_words |
|:--------|-------:|-----------:|------------------:|---------------:|--------------:|-------------:|----------:|-------------:|
| train   |    810 |         13 |                23 |             10 |             3 |            6 |       113 |         52   |
| val     |    188 |          5 |                23 |              2 |             3 |            2 |        38 |         79.6 |
| test    |    250 |          5 |                23 |              2 |             3 |            3 |        52 |         70.9 |

Leakage: speaker overlap 0; test rows whose prompt appears in train: 101/250; near-duplicate test rows (char cosine > 0.9): 0.

## Experiment C: model family (official test, mean over seeds; CI for best-val run)
| model                      |   runs |   val_macro_f1 |   val_sd |   test_macro_f1 |   test_sd |   test_acc |   test_weighted_f1 | row_boot_95ci   | speaker_boot_95ci   |
|:---------------------------|-------:|---------------:|---------:|----------------:|----------:|-----------:|-------------------:|:----------------|:--------------------|
| lora_afroxlmr_large_114L   |      3 |         0.4396 |   0.0194 |          0.4358 |    0.0599 |     0.452  |             0.4343 | [0.309, 0.425]  | [0.314, 0.420]      |
| svm_wordchar               |      1 |         0.4324 |   0      |          0.4129 |    0      |     0.448  |             0.4144 | [0.345, 0.459]  | [0.367, 0.425]      |
| lr_word                    |      1 |         0.4271 |   0      |          0.3957 |    0      |     0.424  |             0.4087 | [0.329, 0.442]  | [0.375, 0.425]      |
| lr_char                    |      1 |         0.4225 |   0      |          0.3705 |    0      |     0.408  |             0.373  | [0.306, 0.414]  | [0.329, 0.396]      |
| full_afroxlmr_large_114L   |      3 |         0.3859 |   0.0253 |          0.4162 |    0.0247 |     0.4347 |             0.4228 | [0.328, 0.441]  | [0.330, 0.457]      |
| bilstm                     |      3 |         0.3374 |   0.0266 |          0.3066 |    0.0218 |     0.3387 |             0.3183 | [0.234, 0.331]  | [0.258, 0.294]      |
| frozen_afroxlmr_large_114L |      1 |         0.2904 |   0      |          0.2273 |    0      |     0.256  |             0.2392 | [0.177, 0.266]  | [0.178, 0.267]      |
| full_xlmr_large            |      3 |         0.112  |   0.1421 |          0.1009 |    0.1354 |     0.1493 |             0.1069 | [0.202, 0.296]  | [0.222, 0.276]      |
| dummy_stratified           |      1 |         0.0386 |   0      |          0.0354 |    0      |     0.048  |             0.0432 | [0.017, 0.055]  | [0.023, 0.045]      |
| dummy_most_frequent        |      1 |         0.0052 |   0      |          0.0046 |    0      |     0.056  |             0.0059 | [0.002, 0.007]  | [0.003, 0.006]      |

**Selected model:** lr_word (run `main|lr_word|normalized|0`)

## Experiment A: data size
|                                    |   n_train |   val_f1 |   val_sd |   test_f1 |   test_sd |
|:-----------------------------------|----------:|---------:|---------:|----------:|----------:|
| ('bilstm', 0.1)                    |        82 |   0.0749 |   0.0295 |    0.0702 |    0.0119 |
| ('bilstm', 0.25)                   |       203 |   0.1947 |   0.0283 |    0.178  |    0.0208 |
| ('bilstm', 0.5)                    |       408 |   0.2571 |   0.0247 |    0.2442 |    0.0234 |
| ('bilstm', 0.75)                   |       607 |   0.2822 |   0.0274 |    0.2454 |    0.0359 |
| ('bilstm', 1.0)                    |       810 |   0.3374 |   0.0266 |    0.3066 |    0.0218 |
| ('full_afroxlmr_large_114L', 0.1)  |        82 |   0.0388 | nan      |    0.0393 |  nan      |
| ('full_afroxlmr_large_114L', 0.25) |       203 |   0.0957 | nan      |    0.0686 |  nan      |
| ('full_afroxlmr_large_114L', 0.5)  |       408 |   0.303  | nan      |    0.3009 |  nan      |
| ('full_afroxlmr_large_114L', 0.75) |       607 |   0.3403 | nan      |    0.3204 |  nan      |
| ('full_afroxlmr_large_114L', 1.0)  |       810 |   0.3602 | nan      |    0.4071 |  nan      |
| ('lr_word', 0.1)                   |        82 |   0.2295 |   0.0292 |    0.2173 |    0.0234 |
| ('lr_word', 0.25)                  |       203 |   0.2802 |   0.0332 |    0.2881 |    0.0214 |
| ('lr_word', 0.5)                   |       408 |   0.3615 |   0.0192 |    0.355  |    0.0382 |
| ('lr_word', 0.75)                  |       607 |   0.3824 |   0.0144 |    0.3689 |    0.0079 |
| ('lr_word', 1.0)                   |       810 |   0.4271 |   0      |    0.3957 |    0      |

## Experiment B: preprocessing
|                                  |   runs |   val_f1 |   val_sd |   test_f1 |   test_sd |
|:---------------------------------|-------:|---------:|---------:|----------:|----------:|
| ('bilstm', 'minimal')            |      3 |   0.3032 |   0.0324 |    0.3074 |    0.0154 |
| ('bilstm', 'normalized')         |      3 |   0.3374 |   0.0266 |    0.3066 |    0.0218 |
| ('bilstm', 'normalized_notags')  |      3 |   0.3284 |   0.0154 |    0.3199 |    0.0242 |
| ('lr_word', 'minimal')           |      1 |   0.4244 | nan      |    0.3918 |  nan      |
| ('lr_word', 'normalized')        |      1 |   0.4271 | nan      |    0.3957 |  nan      |
| ('lr_word', 'normalized_notags') |      1 |   0.423  | nan      |    0.3911 |  nan      |

## Ablation: class weighting
|                    |   val_f1 |   test_f1 |   test_sd |   test_macro_r |   test_acc |
|:-------------------|---------:|----------:|----------:|---------------:|-----------:|
| ('bilstm', False)  |   0.3277 |    0.2972 |    0.028  |         0.3087 |     0.332  |
| ('bilstm', True)   |   0.3374 |    0.3066 |    0.0218 |         0.3328 |     0.3387 |
| ('lr_word', False) |   0.3708 |    0.3809 |    0      |         0.4001 |     0.424  |
| ('lr_word', True)  |   0.4271 |    0.3957 |    0      |         0.424  |     0.424  |

## Experiment F: speaker-grouped vs random CV (dev speakers)
| model   |   random rows |   speaker-grouped |   inflation_random_minus_speaker |
|:--------|--------------:|------------------:|---------------------------------:|
| bilstm  |        0.2796 |            0.2593 |                           0.0203 |
| lr_word |        0.4041 |            0.3684 |                           0.0357 |

## Experiment D: dialect (test)
| model                      | dialect   |   n |   speakers |    acc |   macro_f1 | macro_f1_95ci   |
|:---------------------------|:----------|----:|-----------:|-------:|-----------:|:----------------|
| lora_afroxlmr_large_114L   | Milambo   | 177 |          3 | 0.3955 |     0.3618 | [0.286, 0.412]  |
| lora_afroxlmr_large_114L   | Nyanduat  |  73 |          2 | 0.411  |     0.3931 | [0.273, 0.508]  |
| svm_wordchar               | Milambo   | 177 |          3 | 0.4576 |     0.412  | [0.334, 0.470]  |
| svm_wordchar               | Nyanduat  |  73 |          2 | 0.4247 |     0.3948 | [0.309, 0.521]  |
| lr_word                    | Milambo   | 177 |          3 | 0.435  |     0.3947 | [0.313, 0.446]  |
| lr_word                    | Nyanduat  |  73 |          2 | 0.3973 |     0.3949 | [0.307, 0.526]  |
| lr_char                    | Milambo   | 177 |          3 | 0.4181 |     0.3698 | [0.299, 0.421]  |
| lr_char                    | Nyanduat  |  73 |          2 | 0.3836 |     0.3901 | [0.293, 0.497]  |
| full_afroxlmr_large_114L   | Milambo   | 177 |          3 | 0.3842 |     0.3518 | [0.280, 0.397]  |
| full_afroxlmr_large_114L   | Nyanduat  |  73 |          2 | 0.5068 |     0.4646 | [0.369, 0.589]  |
| bilstm                     | Milambo   | 177 |          3 | 0.3333 |     0.284  | [0.216, 0.330]  |
| bilstm                     | Nyanduat  |  73 |          2 | 0.3288 |     0.3035 | [0.205, 0.397]  |
| frozen_afroxlmr_large_114L | Milambo   | 177 |          3 | 0.2712 |     0.2245 | [0.167, 0.275]  |
| frozen_afroxlmr_large_114L | Nyanduat  |  73 |          2 | 0.2192 |     0.1973 | [0.117, 0.279]  |
| full_xlmr_large            | Milambo   | 177 |          3 | 0.2881 |     0.2454 | [0.183, 0.289]  |
| full_xlmr_large            | Nyanduat  |  73 |          2 | 0.3014 |     0.2515 | [0.182, 0.349]  |

## Experiment E: code-switching (test)
| model                      | cs_group      |   n |   speakers |    acc |   macro_f1 | macro_f1_95ci   |
|:---------------------------|:--------------|----:|-----------:|-------:|-----------:|:----------------|
| lora_afroxlmr_large_114L   | Dholuo only   | 198 |          5 | 0.404  |     0.3712 | [0.295, 0.429]  |
| lora_afroxlmr_large_114L   | code-switched |  52 |          5 | 0.3846 |     0.3655 | [0.232, 0.470]  |
| svm_wordchar               | Dholuo only   | 198 |          5 | 0.4444 |     0.3939 | [0.315, 0.442]  |
| svm_wordchar               | code-switched |  52 |          5 | 0.4615 |     0.4496 | [0.318, 0.570]  |
| lr_word                    | Dholuo only   | 198 |          5 | 0.4394 |     0.3961 | [0.320, 0.444]  |
| lr_word                    | code-switched |  52 |          5 | 0.3654 |     0.3648 | [0.233, 0.479]  |
| lr_char                    | Dholuo only   | 198 |          5 | 0.4091 |     0.3597 | [0.285, 0.408]  |
| lr_char                    | code-switched |  52 |          5 | 0.4038 |     0.4027 | [0.263, 0.522]  |
| full_afroxlmr_large_114L   | Dholuo only   | 198 |          5 | 0.4242 |     0.389  | [0.321, 0.439]  |
| full_afroxlmr_large_114L   | code-switched |  52 |          5 | 0.4038 |     0.3889 | [0.241, 0.513]  |
| bilstm                     | Dholuo only   | 198 |          5 | 0.3283 |     0.2619 | [0.208, 0.302]  |
| bilstm                     | code-switched |  52 |          5 | 0.3462 |     0.3061 | [0.197, 0.431]  |
| frozen_afroxlmr_large_114L | Dholuo only   | 198 |          5 | 0.2626 |     0.2225 | [0.167, 0.265]  |
| frozen_afroxlmr_large_114L | code-switched |  52 |          5 | 0.2308 |     0.211  | [0.107, 0.310]  |
| full_xlmr_large            | Dholuo only   | 198 |          5 | 0.3232 |     0.2764 | [0.211, 0.317]  |
| full_xlmr_large            | code-switched |  52 |          5 | 0.1731 |     0.1942 | [0.080, 0.272]  |

## Experiment H: prompt overlap
| model                      | prompt_group         |   n |   speakers |    acc |   macro_f1 | macro_f1_95ci   |
|:---------------------------|:---------------------|----:|-----------:|-------:|-----------:|:----------------|
| lora_afroxlmr_large_114L   | prompt seen in train | 101 |          5 | 0.495  |     0.4127 | [0.329, 0.497]  |
| lora_afroxlmr_large_114L   | prompt unseen        | 149 |          5 | 0.3356 |     0.3276 | [0.250, 0.391]  |
| svm_wordchar               | prompt seen in train | 101 |          5 | 0.5545 |     0.5014 | [0.399, 0.587]  |
| svm_wordchar               | prompt unseen        | 149 |          5 | 0.3758 |     0.3436 | [0.273, 0.402]  |
| lr_word                    | prompt seen in train | 101 |          5 | 0.495  |     0.421  | [0.328, 0.509]  |
| lr_word                    | prompt unseen        | 149 |          5 | 0.3758 |     0.3606 | [0.275, 0.422]  |
| lr_char                    | prompt seen in train | 101 |          5 | 0.5446 |     0.4913 | [0.384, 0.563]  |
| lr_char                    | prompt unseen        | 149 |          5 | 0.3154 |     0.2922 | [0.222, 0.346]  |
| full_afroxlmr_large_114L   | prompt seen in train | 101 |          5 | 0.5149 |     0.4428 | [0.356, 0.542]  |
| full_afroxlmr_large_114L   | prompt unseen        | 149 |          5 | 0.3557 |     0.3291 | [0.262, 0.387]  |
| bilstm                     | prompt seen in train | 101 |          5 | 0.4356 |     0.3775 | [0.283, 0.457]  |
| bilstm                     | prompt unseen        | 149 |          5 | 0.2617 |     0.234  | [0.173, 0.285]  |
| frozen_afroxlmr_large_114L | prompt seen in train | 101 |          5 | 0.3069 |     0.2284 | [0.159, 0.299]  |
| frozen_afroxlmr_large_114L | prompt unseen        | 149 |          5 | 0.2215 |     0.2144 | [0.150, 0.273]  |
| full_xlmr_large            | prompt seen in train | 101 |          5 | 0.3267 |     0.2675 | [0.187, 0.334]  |
| full_xlmr_large            | prompt unseen        | 149 |          5 | 0.2685 |     0.2358 | [0.173, 0.286]  |

|                                        |   n_train |   test_f1 |   test_sd |   test_acc |
|:---------------------------------------|----------:|----------:|----------:|-----------:|
| ('bilstm', 'speaker+prompt-disjoint')  |       709 |    0.2738 |    0.005  |     0.3013 |
| ('bilstm', 'speaker-disjoint')         |       810 |    0.3066 |    0.0218 |     0.3387 |
| ('lr_word', 'speaker+prompt-disjoint') |       709 |    0.3111 |    0      |     0.352  |
| ('lr_word', 'speaker-disjoint')        |       810 |    0.3957 |    0      |     0.424  |

Prompt echo (audit): {'median_share_of_prompt_words_in_transcript': 0.6, 'share_of_transcripts_covering_>=80%_of_prompt_words': 0.149, 'share_of_transcripts_opening_with_the_prompt': 0.335}

## Experiment G: fine-tuning strategy
| model                      |   val_f1 |   val_f1_sd |   test_f1 |   test_f1_sd |   trainable_params |   train_time_s |   peak_gpu_gb |   runs |
|:---------------------------|---------:|------------:|----------:|-------------:|-------------------:|---------------:|--------------:|-------:|
| frozen_afroxlmr_large_114L |   0.2904 |    nan      |    0.2273 |     nan      |    23575           |        103.6   |        nan    |      1 |
| full_afroxlmr_large_114L   |   0.3859 |      0.0253 |    0.4162 |       0.0247 |        3.03912e+08 |        609.567 |          7.15 |      3 |
| lora_afroxlmr_large_114L   |   0.4396 |      0.0194 |    0.4358 |       0.0599 |        2.64604e+06 |        330.6   |          4.51 |      3 |

## Tokenizer fertility
|                     |   subwords_per_word |   share_words_split_3plus |   share_words_single_piece |   vocab_size | example                               |
|:--------------------|--------------------:|--------------------------:|---------------------------:|-------------:|:--------------------------------------|
| afroxlmr_large_114L |                 1.7 |                     0.125 |                      0.448 |       250002 | ▁jo pur | ▁ny ithi ndo | ▁mang ' en y |
| xlmr_large          |                 1.7 |                     0.125 |                      0.448 |       250002 | ▁jo pur | ▁ny ithi ndo | ▁mang ' en y |

## Error analysis
| flag                                               |   rows_flagged |   err_rate_flagged | ci_flagged   |   err_rate_rest | ci_rest      |   relative_risk |   share_of_all_errors |
|:---------------------------------------------------|---------------:|-------------------:|:-------------|----------------:|:-------------|----------------:|----------------------:|
| prompt unseen in training                          |            149 |              0.624 | [0.54, 0.70] |           0.505 | [0.41, 0.60] |           1.236 |                 0.646 |
| long (>= train Q3 words)                           |            142 |              0.62  | [0.54, 0.70] |           0.519 | [0.43, 0.61] |           1.195 |                 0.611 |
| gold class has low training support                |             61 |              0.656 | [0.53, 0.76] |           0.55  | [0.48, 0.62] |           1.192 |                 0.278 |
| code-switched                                      |             52 |              0.635 | [0.50, 0.75] |           0.561 | [0.49, 0.63] |           1.132 |                 0.229 |
| high OOV (> train-median OOV of test)              |            123 |              0.585 | [0.50, 0.67] |           0.567 | [0.48, 0.65] |           1.033 |                 0.5   |
| dialect = Milambo                                  |            177 |              0.565 | [0.49, 0.64] |           0.603 | [0.49, 0.71] |           0.937 |                 0.694 |
| gender = Female                                    |            116 |              0.543 | [0.45, 0.63] |           0.604 | [0.52, 0.68] |           0.898 |                 0.438 |
| transcript was editor-corrected                    |            115 |              0.539 | [0.45, 0.63] |           0.607 | [0.52, 0.69] |           0.888 |                 0.431 |
| short (<= train Q1 words)                          |             21 |              0.429 | [0.24, 0.63] |           0.59  | [0.52, 0.65] |           0.727 |                 0.062 |
| answer barely echoes prompt (<30% of prompt words) |              9 |              0.333 | [0.12, 0.65] |           0.585 | [0.52, 0.65] |           0.57  |                 0.021 |
| speech-rate outlier (possible transcript issue)    |              0 |            nan     | [nan, nan]   |           0.576 | [0.51, 0.64] |         nan     |                 0     |

Spearman(similarity, confusion) = 0.300 (p=1.14e-06)

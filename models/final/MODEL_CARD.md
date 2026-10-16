# Dholuo agricultural topic classifier - svm_wordchar

**What it does:** assigns a Dholuo (Luo, Kenya) agricultural text - written or transcribed speech - to one of 23 agricultural sub-domains and returns class probabilities.

**What it does not do:** it does not transcribe audio, translate, give agricultural advice, or handle topics outside these 23 sub-domains. It always returns one of the 23 labels, even for off-topic input.

**Training data:** Utavu Foundation Dholuo Speech Dataset (MCAA1-MSU/twb_luo_ke, revision 5cbcd61d844e29e80e5e574fa14a7c54603052c3), freeform-prompts subset, 810 training transcripts from 13 speakers. Licence NOODL-1.0.

**Evaluation (speaker-disjoint, 250 test transcripts from 5 unseen speakers):**
macro-F1 0.413 (row bootstrap 95% CI [0.345, 0.459], speaker bootstrap [0.367, 0.425]), accuracy 0.448, weighted F1 0.414. Validation macro-F1 0.432.

**Prompt overlap:** 101 of 250 test answers respond to a prompt also answered in training. On the 149 test answers to unseen prompts, macro-F1 is 0.344 - the more realistic figure for new questions (Experiment H).

**Selection:** chosen by validation macro-F1 with a prefer-simpler rule (see notebook §12); run `main|svm_wordchar|normalized|0`.

**Labels:** Access to government programs and subsidies, Agricultural education, Animal feeding and nutrition, Automation and AI in farming, Climate resilience and adaptation strategies, Contemporary issues in Agriculture, Crop Yield and quality, Distribution channels (local, regional, international), Economic impact of agricultural practices, Effect of climate change on Agriculture, Farming Methods and Systems, Fish farming, Land ownership and tenure issues, Livestock production and productivity, Personal Agricultural day to day experience, Post harvesting technology, Poultry farming, Precision agriculture tools (drones, GPS, sensors), Storage and Transportation, Use of machinery and equipment, Value addition (processing, packaging), Water Usage and Irrigation Methods, Water and Land Conservation

**Known limitations:** 23 speakers in total, most aged 18-29 (dataset card); agriculture only; transcripts from one transcription team with inconsistent spelling and tags; dialect effects cannot be separated from individual speakers at this scale. Not for high-stakes use (e.g. deciding who receives support or subsidies).

**Preprocessing:** `preprocess.py`, variant `normalized` - identical to evaluation.

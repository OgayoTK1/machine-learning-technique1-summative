"""Sanity check that the CTC training loop learns: each character is rendered as its own
tone, so a correct pipeline must drive training CER towards zero within a few hundred
steps. Not a unit test (takes ~1-3 min on CPU). Run: python tests/overfit_check.py [bilstm|hf]
"""
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from dholuo_asr.data import ASRDataset, HFCTCCollator  # noqa: E402
from dholuo_asr.metrics import corpus_rates, score_frame  # noqa: E402
from dholuo_asr.utils import set_seed  # noqa: E402
from dholuo_asr.vocab import CharTokenizer, build_char_vocab, save_vocab  # noqa: E402

WORDS = "an ber dala ng'ato ok gi".split()


def synth(text, sr=16000):
    chars = sorted(set("".join(WORDS)) | {" "})
    freq = {c: 250 + 330 * i for i, c in enumerate(chars)}  # well-separated tones
    seg = []
    for c in text:
        t = np.arange(int(0.12 * sr)) / sr
        # the word boundary gets its own tone: this check tests the plumbing, not acoustics
        seg.append(0.3 * np.sin(2 * np.pi * freq[c] * t))
        seg.append(np.zeros(int(0.03 * sr)))
    return np.concatenate([np.zeros(1600)] + seg + [np.zeros(1600)]).astype(np.float32)


def main(kind="bilstm"):
    set_seed(0)
    rng = np.random.default_rng(0)
    tmp = Path(tempfile.mkdtemp())
    rows = []
    for i in range(48):
        text = " ".join(rng.choice(WORDS, size=rng.integers(1, 4)))
        p = tmp / f"{i}.wav"
        w = synth(text)
        sf.write(p, w, 16000)
        rows.append({"utt_id": str(i), "audio_path": str(p), "text_train": text, "text_raw": text,
                     "duration_s": len(w) / 16000, "speaker": f"s{i % 4}"})
    df = pd.DataFrame(rows)
    vocab = build_char_vocab(df["text_train"])
    ds = ASRDataset(df, "text_train")
    from dholuo_asr.training import BiLSTMAdapter, HFAdapter, build_hf_model, build_processor, train_ctc
    if kind == "bilstm":
        from dholuo_asr.bilstm import BiLSTMCollator, BiLSTMCTC
        tok = CharTokenizer(vocab)
        adapter = BiLSTMAdapter(BiLSTMCTC(len(tok), hidden=96, layers=2, dropout=0.0, specaug=False), tok)
        coll = BiLSTMCollator(tok)
        lr, steps = 3e-3, 1200
    else:
        save_vocab(vocab, tmp / "vocab.json")
        proc = build_processor(tmp / "vocab.json")
        m = build_hf_model({"checkpoint": "tiny-random", "strategy": "full"}, len(proc.tokenizer),
                           proc.tokenizer.pad_token_id)
        adapter, coll = HFAdapter(m, proc), HFCTCCollator(proc)
        lr, steps = 2e-3, 1000
    best = train_ctc(adapter, ds, ds, coll,
                     {"lr": lr, "seed": 0, "max_steps": steps, "epochs": 1000, "max_batch_seconds": 20,
                      "eval_batch_seconds": 60, "eval_steps": 200, "num_workers": 0, "warmup_ratio": 0.05,
                      "early_stopping_patience": 100},
                     tmp / "run", eval_ref_col="text_raw")
    print(kind, "best train-set (memorisation) CER:", round(best["val_cer"], 3), "WER:", round(best["val_wer"], 3))
    assert best["val_cer"] < 0.25, "training loop failed to learn a trivially learnable task"
    print("OK")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "bilstm")

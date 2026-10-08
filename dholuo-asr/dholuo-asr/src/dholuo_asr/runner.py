"""One function that turns a resolved experiment config into a complete, saved run.

results/<run_name>/
    config.json  env.json  vocab.json  data_report.json  train_log.csv
    predictions_val.csv  predictions_test.csv  logprobs_{val,test}.npz
    metrics.json  model/   (weights + processor/tokenizer, if output.save_model)
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .data import ASRDataset, HFCTCCollator
from .metrics import corpus_rates, group_rates, score_frame, speaker_bootstrap
from .splits import subsample_train_speakers
from .text import apply_view
from .utils import dump_json, env_info, set_seed
from .vocab import CharTokenizer, build_char_vocab, oov_chars, save_vocab


def prepare_data(cfg: dict, splits: pd.DataFrame) -> dict:
    d = cfg["data"]
    split_col = {"speaker": "split", "random": "split_random", "strict": "split_strict"}[d["split_scheme"]]
    df = splits.copy()
    df["text_train"] = df["text_raw"].map(lambda t: apply_view(t, d["text_view"]))
    df = df[df["text_train"].str.len() > 0]
    train = df[df[split_col] == "train"]
    if d["train_fraction"] < 1.0:
        if d["split_scheme"] == "random":
            raise ValueError("train_fraction < 1 is not defined for the random split")
        train = subsample_train_speakers(train, d["train_fraction"], seed=cfg["train"]["seed"])
    if len(train) == 0:
        raise ValueError(f"no training utterances for split_scheme={d['split_scheme']!r} "
                         f"(train_fraction={d['train_fraction']})")
    out = {"train": train, "val": df[df[split_col] == "val"], "test": df[df[split_col] == "test"],
           "split_col": split_col}
    out["report"] = {
        s: {"utts": int(len(out[s])), "speakers": int(out[s]["speaker"].nunique()),
            "hours": float(out[s]["duration_s"].sum() / 3600)}
        for s in ("train", "val", "test")}
    out["report"]["train_skipped_too_long"] = int(
        (train["duration_s"] > d["max_train_seconds"]).sum())
    return out


def _save_preds(df: pd.DataFrame, path: Path):
    keep = [c for c in ["utt_id", "speaker", "dialect", "gender", "age", "domain", "duration_s",
                        "prompt_in_train", "prompt_in_train_random", "cs_candidate", "ref",
                        "text_train", "hyp"] if c in df]
    df[keep].to_csv(path, index=False)


def run_experiment(cfg: dict, splits: pd.DataFrame, out_root: str | Path, log=print) -> dict:
    out = Path(out_root) / cfg["name"]
    out.mkdir(parents=True, exist_ok=True)
    set_seed(cfg["train"]["seed"])
    dump_json(cfg, out / "config.json")
    dump_json(env_info(), out / "env.json")

    data = prepare_data(cfg, splits)
    vocab = build_char_vocab(data["train"]["text_train"])
    save_vocab(vocab, out / "vocab.json")
    rep = data["report"]
    for s in ("val", "test"):
        rep[f"{s}_oov_chars"] = dict(oov_chars(data[s]["text_train"], vocab))
        tw = Counter(w for t in data["train"]["text_train"] for w in t.split())
        sw = [w for t in data[s]["text_train"] for w in t.split()]
        rep[f"{s}_word_oov_rate"] = float(np.mean([w not in tw for w in sw])) if sw else None
    dump_json(rep, out / "data_report.json")
    log(json.dumps({k: v for k, v in rep.items() if not k.endswith("oov_chars")}, indent=1))

    d, m, t = cfg["data"], cfg["model"], cfg["train"]
    train_ds = ASRDataset(data["train"], "text_train", d["cache_dir"], d["max_train_seconds"])
    val_ds = ASRDataset(data["val"], "text_train", d["cache_dir"])
    test_ds = ASRDataset(data["test"], "text_train", d["cache_dir"])

    from .training import (BiLSTMAdapter, HFAdapter, build_hf_model, build_processor,
                           param_counts, predict, train_ctc)
    if m["family"] == "hf":
        processor = build_processor(out / "vocab.json")
        model = build_hf_model(m, len(processor.tokenizer), processor.tokenizer.pad_token_id)
        adapter = HFAdapter(model, processor)
        collator = HFCTCCollator(processor)
    elif m["family"] == "bilstm":
        from .bilstm import BiLSTMCollator, BiLSTMCTC
        tok = CharTokenizer(vocab)
        model = BiLSTMCTC(len(tok), hidden=m["hidden"], layers=m["layers"], dropout=m["dropout"])
        adapter = BiLSTMAdapter(model, tok)
        collator = BiLSTMCollator(tok)
    else:
        raise ValueError(m["family"])
    pc = param_counts(model)
    log(f"parameters: {pc}")

    best = train_ctc(adapter, train_ds, val_ds, collator, t, out, eval_ref_col="text_raw", log=log)

    metrics = {"name": cfg["name"], **pc, "best": best, "data": rep}
    for split, ds in (("val", val_ds), ("test", test_ds)):
        preds, lps = predict(adapter, ds, collator, max_batch_seconds=t["eval_batch_seconds"],
                             num_workers=t.get("num_workers", 2),
                             keep_logprobs=cfg["output"]["keep_logprobs"], ref_col="text_raw")
        _save_preds(preds, out / f"predictions_{split}.csv")
        if lps:
            np.savez_compressed(out / f"logprobs_{split}.npz", **lps)
        scored = score_frame(preds)
        metrics[split] = {**corpus_rates(scored), **speaker_bootstrap(scored)}
        if "dialect" in scored:
            metrics[split]["by_dialect"] = group_rates(scored, "dialect")[["wer", "cer", "n_utts"]].to_dict("index")
        log(f"{split}: WER={metrics[split]['wer']:.4f} CER={metrics[split]['cer']:.4f} "
            f"WER 95% CI (speaker bootstrap)={tuple(round(x, 4) for x in metrics[split]['wer_ci'])}")
    dump_json(metrics, out / "metrics.json")

    if cfg["output"]["save_model"]:
        save_model(cfg, adapter, out / "model")
    return metrics


def save_model(cfg: dict, adapter, path: Path):
    path.mkdir(parents=True, exist_ok=True)
    m = cfg["model"]
    if adapter.kind == "bilstm":
        torch.save({"state_dict": adapter.model.state_dict(), "vocab": adapter.tok.vocab,
                    "hparams": {k: m[k] for k in ("hidden", "layers", "dropout")}},
                   path / "bilstm_ctc.pt")
        return
    model = adapter.model
    adapter.proc.save_pretrained(path)
    if m["strategy"] == "lora":
        model = model.merge_and_unload()
    if m["strategy"] == "adapter":
        from safetensors.torch import save_file
        save_file({k: v.contiguous() for k, v in model._get_adapters().items()},
                  path / "adapter.luo.safetensors")
        dump_json({"base_checkpoint": m["checkpoint"], "adapter_file": "adapter.luo.safetensors",
                   "vocab_size": model.config.vocab_size}, path / "adapter_info.json")
        return
    model.save_pretrained(path)

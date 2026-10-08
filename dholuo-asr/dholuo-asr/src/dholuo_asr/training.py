"""Model construction (fine-tuning strategies) and a single, explicit CTC training loop.

A hand-written loop is used instead of ``transformers.Trainer`` so that every
experiment uses identical logic across library versions, model families (HF wav2vec2
models and the from-scratch BiLSTM) and fine-tuning strategies, and so model selection
is unambiguously done on the VALIDATION set only.
"""
from __future__ import annotations

import copy
import csv
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .data import DurationBatchSampler
from .metrics import corpus_rates, score_frame

# ----------------------------------------------------------------------------- models

def build_processor(vocab_path: str):
    from transformers import Wav2Vec2CTCTokenizer, Wav2Vec2FeatureExtractor, Wav2Vec2Processor
    # bos/eos disabled: CTC needs no sentence markers, and leaving them on would add two
    # extra output units the model could emit but never sees as targets.
    tok = Wav2Vec2CTCTokenizer(str(vocab_path), unk_token="[UNK]", pad_token="[PAD]",
                               word_delimiter_token="|", bos_token=None, eos_token=None)
    fe = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0,
                                  do_normalize=True, return_attention_mask=True)
    return Wav2Vec2Processor(feature_extractor=fe, tokenizer=tok)


def _tiny_config(vocab_size: int, pad_id: int, adapter: bool):
    """Randomly initialised miniature wav2vec2 used ONLY for the offline smoke test."""
    from transformers import Wav2Vec2Config
    return Wav2Vec2Config(
        vocab_size=vocab_size, pad_token_id=pad_id, hidden_size=32, num_hidden_layers=2,
        num_attention_heads=2, intermediate_size=64, conv_dim=(32, 32, 32),
        conv_stride=(5, 4, 4), conv_kernel=(10, 8, 8), num_conv_pos_embeddings=16,
        num_conv_pos_embedding_groups=4, feat_extract_norm="layer", do_stable_layer_norm=True,
        ctc_loss_reduction="mean", ctc_zero_infinity=True, mask_time_prob=0.0,
        adapter_attn_dim=16 if adapter else None)


def build_hf_model(mcfg: dict, vocab_size: int, pad_id: int):
    """Create a wav2vec2-family CTC model and apply the fine-tuning ``strategy``.

    strategy:
      full    – all transformer layers trained, CNN feature encoder frozen (standard)
      frozen  – entire pretrained encoder frozen; only the new CTC head is trained
      lora    – LoRA on attention projections + trainable CTC head (peft)
      adapter – MMS-style per-layer language adapters + CTC head (MMS checkpoints only)
    """
    from transformers import Wav2Vec2ForCTC
    strategy = mcfg["strategy"]
    if mcfg["checkpoint"] == "tiny-random":
        model = Wav2Vec2ForCTC(_tiny_config(vocab_size, pad_id, strategy == "adapter"))
    else:
        model = Wav2Vec2ForCTC.from_pretrained(
            mcfg["checkpoint"], vocab_size=vocab_size, pad_token_id=pad_id,
            ignore_mismatched_sizes=True, ctc_loss_reduction="mean", ctc_zero_infinity=True,
            attention_dropout=mcfg.get("attention_dropout", 0.0),
            hidden_dropout=mcfg.get("hidden_dropout", 0.0),
            feat_proj_dropout=mcfg.get("feat_proj_dropout", 0.0),
            layerdrop=mcfg.get("layerdrop", 0.0),
            mask_time_prob=mcfg.get("mask_time_prob", 0.05),
            mask_feature_prob=mcfg.get("mask_feature_prob", 0.0),
        )
    if mcfg.get("gradient_checkpointing") and torch.cuda.is_available():
        model.gradient_checkpointing_enable()

    if strategy == "full":
        model.freeze_feature_encoder()
    elif strategy == "frozen":
        model.freeze_base_model()
        for p in model.lm_head.parameters():
            p.requires_grad = True
    elif strategy == "adapter":
        model.init_adapter_layers()
        model.freeze_base_model()
        for p in model._get_adapters().values():
            p.requires_grad = True
    elif strategy == "lora":
        from peft import LoraConfig, get_peft_model
        model.freeze_feature_encoder()
        lcfg = LoraConfig(r=mcfg.get("lora_r", 16), lora_alpha=mcfg.get("lora_alpha", 32),
                          lora_dropout=mcfg.get("lora_dropout", 0.05),
                          target_modules=mcfg.get("lora_targets", ["q_proj", "k_proj", "v_proj", "out_proj"]),
                          modules_to_save=["lm_head"], bias="none")
        model = get_peft_model(model, lcfg)
    else:
        raise ValueError(f"unknown strategy {strategy}")
    return model


def param_counts(model) -> dict:
    total = sum(p.numel() for p in model.parameters())
    train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"params_total": total, "params_trainable": train,
            "trainable_frac": train / max(total, 1)}


# ----------------------------------------------------------------------------- adapters
# Thin wrappers so the loop below treats HF and BiLSTM models identically.

class HFAdapter:
    kind = "hf"

    def __init__(self, model, processor):
        self.model, self.proc = model, processor

    def loss(self, batch, dev):
        out = self.model(input_values=batch["input_values"].to(dev),
                         attention_mask=batch["attention_mask"].to(dev),
                         labels=batch["labels"].to(dev))
        return out.loss

    @torch.no_grad()
    def logprobs(self, batch, dev):
        out = self.model(input_values=batch["input_values"].to(dev),
                         attention_mask=batch["attention_mask"].to(dev))
        lp = out.logits.float().log_softmax(-1)
        base = self.model.get_base_model() if hasattr(self.model, "get_base_model") else self.model
        lens = base._get_feat_extract_output_lengths(batch["attention_mask"].sum(-1)).cpu()
        return lp.cpu(), lens

    def greedy(self, lp, lens) -> list[str]:
        ids = lp.argmax(-1)
        texts = []
        for i in range(len(ids)):
            texts.append(self.proc.decode(ids[i, : int(lens[i])].tolist(), skip_special_tokens=True))
        return [" ".join(t.split()) for t in texts]


class BiLSTMAdapter:
    kind = "bilstm"

    def __init__(self, model, tokenizer):
        self.model, self.tok = model, tokenizer
        self.ctc = torch.nn.CTCLoss(blank=tokenizer.blank_id, zero_infinity=True)

    def loss(self, batch, dev):
        lp, out_lens = self.model(batch["wav"].to(dev), batch["lengths"].to(dev))
        return self.ctc(lp.float().transpose(0, 1), batch["targets"].to(dev),
                        out_lens, batch["target_lengths"].to(dev))

    @torch.no_grad()
    def logprobs(self, batch, dev):
        lp, lens = self.model(batch["wav"].to(dev), batch["lengths"].to(dev))
        return lp.float().cpu(), lens.cpu()

    def greedy(self, lp, lens) -> list[str]:
        ids = lp.argmax(-1)
        return [self.tok.decode_ids(ids[i, : int(lens[i])].tolist()) for i in range(len(ids))]


# ----------------------------------------------------------------------------- inference

def predict(adapter, ds, collator, *, max_batch_seconds: float, num_workers: int = 2,
            keep_logprobs: bool = False, ref_col: str | None = None):
    """Greedy-decode a dataset. Returns (predictions frame, {utt_id: log-probs})."""
    dev = next(adapter.model.parameters()).device
    adapter.model.eval()
    sampler = DurationBatchSampler(ds.durations, max_batch_seconds, shuffle=False)
    dl = DataLoader(ds, batch_sampler=sampler, collate_fn=collator, num_workers=num_workers)
    hyps, lps = {}, {}
    use_amp = dev.type == "cuda"
    for batch in dl:
        with torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=use_amp):
            lp, lens = adapter.logprobs(batch, dev)
        texts = adapter.greedy(lp, lens)
        for k, i in enumerate(batch["idx"].tolist()):
            hyps[i] = texts[k]
            if keep_logprobs:
                lps[ds.df.iloc[i]["utt_id"]] = lp[k, : int(lens[k])].numpy().astype(np.float16)
    df = ds.df.copy()
    df["hyp"] = [hyps[i] for i in range(len(df))]
    df["ref"] = df[ref_col or ds.text_col]
    return df, lps


# ----------------------------------------------------------------------------- training

def train_ctc(adapter, train_ds, val_ds, collator, tcfg: dict, out_dir: str | Path,
              eval_ref_col: str = "text_raw", log=print) -> dict:
    """Train with AdamW + linear warmup/decay, fp16 autocast on GPU, gradient
    accumulation, periodic validation WER, best-checkpoint selection on validation WER,
    early stopping and a wall-clock budget (Kaggle sessions are capped)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = adapter.model.to(dev)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=tcfg["lr"], weight_decay=tcfg.get("weight_decay", 0.0),
                            betas=(0.9, 0.98), eps=1e-8)
    sampler = DurationBatchSampler(train_ds.durations, tcfg["max_batch_seconds"], shuffle=True,
                                   seed=tcfg["seed"])
    dl = DataLoader(train_ds, batch_sampler=sampler, collate_fn=collator,
                    num_workers=tcfg.get("num_workers", 2), persistent_workers=False)
    accum = tcfg.get("grad_accum", 1)
    steps_per_epoch = math.ceil(len(sampler) / accum)
    max_steps = tcfg.get("max_steps") or steps_per_epoch * tcfg["epochs"]
    warmup = int(tcfg.get("warmup_ratio", 0.1) * max_steps)

    def lr_at(step):
        if step < warmup:
            return (step + 1) / max(1, warmup)
        return max(0.0, (max_steps - step) / max(1, max_steps - warmup))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)
    use_amp = dev.type == "cuda" and tcfg.get("fp16", True)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    eval_every = tcfg.get("eval_steps") or steps_per_epoch
    patience = tcfg.get("early_stopping_patience", 5)
    budget_s = tcfg.get("max_minutes", 1e9) * 60
    t0 = time.time()
    best = {"val_wer": float("inf"), "step": -1}
    best_state, bad_evals, step, epoch, done = None, 0, 0, 0, False
    log_rows, running = [], []
    log(f"steps/epoch={steps_per_epoch} max_steps={max_steps} warmup={warmup} "
        f"trainable={sum(p.numel() for p in params):,} device={dev}")

    while not done:
        sampler.set_epoch(epoch)
        model.train()
        opt.zero_grad(set_to_none=True)
        for b_i, batch in enumerate(dl):
            with torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=use_amp):
                loss = adapter.loss(batch, dev) / accum
            if not torch.isfinite(loss):
                log(f"non-finite loss at step {step}; batch skipped")
                opt.zero_grad(set_to_none=True)
                continue
            scaler.scale(loss).backward()
            running.append(loss.item() * accum)
            if (b_i + 1) % accum:
                continue
            scaler.unscale_(opt)
            gnorm = torch.nn.utils.clip_grad_norm_(params, tcfg.get("max_grad_norm", 1.0))
            scaler.step(opt)
            scaler.update()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            if step == 50:
                el = time.time() - t0
                log(f"throughput check: 50 steps in {el/60:.1f} min -> projected "
                    f"{el/50*max_steps/3600:.2f} h for {max_steps} steps (budget "
                    f"{budget_s/3600:.1f} h)")
            out_of_time = time.time() - t0 > budget_s
            if step % eval_every == 0 or step >= max_steps or out_of_time:
                preds, _ = predict(adapter, val_ds, collator,
                                   max_batch_seconds=tcfg["eval_batch_seconds"],
                                   num_workers=tcfg.get("num_workers", 2), ref_col=eval_ref_col)
                r = corpus_rates(score_frame(preds))
                row = {"step": step, "epoch": epoch, "lr": sched.get_last_lr()[0],
                       "train_loss": float(np.mean(running)) if running else float("nan"),
                       "grad_norm": float(gnorm), "val_wer": r["wer"], "val_cer": r["cer"],
                       "elapsed_min": (time.time() - t0) / 60}
                running = []
                log_rows.append(row)
                log(" ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items()))
                if r["wer"] < best["val_wer"] - 1e-6:
                    best = {"val_wer": r["wer"], "val_cer": r["cer"], "step": step}
                    best_state = {k: v.detach().to("cpu", copy=True)
                                  for k, v in model.state_dict().items()
                                  if k in _trainable_keys(model)}
                    bad_evals = 0
                else:
                    bad_evals += 1
                model.train()
                if bad_evals >= patience:
                    log(f"early stopping: no val-WER improvement in {patience} evaluations")
                    done = True
                if out_of_time:
                    log("time budget reached")
                    done = True
            if step >= max_steps:
                done = True
            if done:
                break
        epoch += 1

    with open(out_dir / "train_log.csv", "w", newline="") as f:
        if log_rows:
            w = csv.DictWriter(f, fieldnames=list(log_rows[0]))
            w.writeheader()
            w.writerows(log_rows)
    if best_state is not None:
        model.load_state_dict(best_state, strict=False)
    best.update({"steps_run": step, "epochs_run": epoch, "train_minutes": (time.time() - t0) / 60,
                 "max_steps": max_steps})
    return best


def _trainable_keys(model) -> set[str]:
    names = {n for n, p in model.named_parameters() if p.requires_grad}
    # include buffers of modules that have trainable params (e.g. batchnorm stats)
    return names | {n for n, _ in model.named_buffers() if n.rsplit(".", 1)[0] in
                    {k.rsplit(".", 1)[0] for k in names}}

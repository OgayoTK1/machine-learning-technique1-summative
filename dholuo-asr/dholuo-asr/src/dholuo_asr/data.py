"""PyTorch datasets, duration-bucketed batching and collators."""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from .audio import TARGET_SR, load_audio


class AudioCache:
    """Optional on-disk cache of decoded 16 kHz float16 audio (speeds up epochs >1)."""

    def __init__(self, cache_dir: str | None):
        self.dir = Path(cache_dir) if cache_dir else None
        if self.dir:
            self.dir.mkdir(parents=True, exist_ok=True)

    def load(self, path: str) -> np.ndarray:
        if not self.dir:
            return load_audio(path, TARGET_SR)
        key = hashlib.sha1(path.encode()).hexdigest()[:20]
        f = self.dir / f"{key}.npy"
        if f.exists():
            try:
                return np.load(f).astype(np.float32)
            except Exception:  # noqa: BLE001 - partial write from a killed worker
                pass
        wav = load_audio(path, TARGET_SR)
        tmp = f.with_suffix(f".{np.random.randint(1e9)}.tmp.npy")
        np.save(tmp, wav.astype(np.float16))
        tmp.replace(f)
        return wav


class ASRDataset(Dataset):
    """Rows of a manifest -> (waveform, target text, row index)."""

    def __init__(self, df, text_col: str, cache_dir: str | None = None,
                 max_seconds: float | None = None):
        df = df.reset_index(drop=True)
        if max_seconds is not None and "duration_s" in df:
            df = df[df["duration_s"] <= max_seconds].reset_index(drop=True)
        self.df = df
        self.text_col = text_col
        self.cache = AudioCache(cache_dir)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        return {"wav": self.cache.load(r["audio_path"]), "text": r[self.text_col], "idx": i}

    @property
    def durations(self) -> np.ndarray:
        return self.df["duration_s"].to_numpy(dtype=float)


class DurationBatchSampler(Sampler):
    """Pack utterances of similar length so that (longest * batch size) stays under
    ``max_batch_seconds``. Shuffled at chunk level each epoch (seeded) for training,
    deterministic sorted order for evaluation."""

    def __init__(self, durations, max_batch_seconds: float, shuffle: bool, seed: int = 0,
                 max_batch_size: int = 64):
        self.d = np.asarray(durations, dtype=float)
        self.max_s = max_batch_seconds
        self.shuffle = shuffle
        self.seed = seed
        self.max_bs = max_batch_size
        self.epoch = 0

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def _batches(self):
        n = len(self.d)
        if self.shuffle:
            rng = np.random.default_rng(self.seed + self.epoch)
            idx = rng.permutation(n)
            chunk = 2000
            idx = np.concatenate([c[np.argsort(self.d[c])] for c in np.array_split(idx, max(1, n // chunk))])
        else:
            idx = np.argsort(self.d, kind="stable")
        batches, cur, cur_max = [], [], 0.0
        for i in idx:
            m = max(cur_max, self.d[i])
            if cur and (m * (len(cur) + 1) > self.max_s or len(cur) >= self.max_bs):
                batches.append(cur)
                cur, m = [], self.d[i]
            cur.append(int(i))
            cur_max = m
        if cur:
            batches.append(cur)
        if self.shuffle:
            rng = np.random.default_rng(self.seed + 10_000 + self.epoch)
            batches = [batches[j] for j in rng.permutation(len(batches))]
        return batches

    def __iter__(self):
        yield from self._batches()

    def __len__(self):
        return len(self._batches())


class HFCTCCollator:
    """Collate for HF wav2vec2-style models: normalise + pad audio, pad labels with -100."""

    def __init__(self, processor, normalize: bool = True):
        self.proc = processor
        self.normalize = normalize

    def __call__(self, items):
        wavs = [it["wav"] for it in items]
        if self.normalize:
            wavs = [(w - w.mean()) / np.sqrt(w.var() + 1e-7) for w in wavs]
        lens = [len(w) for w in wavs]
        T = max(lens)
        x = np.zeros((len(wavs), T), dtype=np.float32)
        mask = np.zeros((len(wavs), T), dtype=np.int64)
        for i, w in enumerate(wavs):
            x[i, : len(w)] = w
            mask[i, : len(w)] = 1
        lab = [self.proc.tokenizer(it["text"]).input_ids for it in items]
        L = max(1, max(len(l) for l in lab))
        y = np.full((len(lab), L), -100, dtype=np.int64)
        for i, l in enumerate(lab):
            y[i, : len(l)] = l
        return {"input_values": torch.from_numpy(x), "attention_mask": torch.from_numpy(mask),
                "labels": torch.from_numpy(y), "idx": torch.tensor([it["idx"] for it in items])}

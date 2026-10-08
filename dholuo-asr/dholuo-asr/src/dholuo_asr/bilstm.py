"""From-scratch neural baseline: log-mel -> conv subsampling -> BiLSTM -> CTC.

No pre-training of any kind. This answers "what does a conventional neural ASR model
learn from DhoNam alone?", the reference point for the value of multilingual
self-supervised pre-training (Experiment C).
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn


def mel_filterbank(sr: int = 16000, n_fft: int = 400, n_mels: int = 80,
                   fmin: float = 20.0, fmax: float | None = None) -> np.ndarray:
    """Slaney-style triangular mel filters (HTK mel scale)."""
    fmax = fmax or sr / 2
    def hz2mel(f): return 2595.0 * np.log10(1.0 + f / 700.0)
    def mel2hz(m): return 700.0 * (10 ** (m / 2595.0) - 1.0)
    mels = np.linspace(hz2mel(fmin), hz2mel(fmax), n_mels + 2)
    hz = mel2hz(mels)
    bins = np.fft.rfftfreq(n_fft, 1.0 / sr)
    fb = np.zeros((n_mels, len(bins)), dtype=np.float32)
    for m in range(1, n_mels + 1):
        lo, c, hi = hz[m - 1], hz[m], hz[m + 1]
        up = (bins - lo) / max(c - lo, 1e-9)
        down = (hi - bins) / max(hi - c, 1e-9)
        fb[m - 1] = np.maximum(0, np.minimum(up, down))
    enorm = 2.0 / (hz[2:] - hz[:-2])
    return fb * enorm[:, None].astype(np.float32)


class LogMel(nn.Module):
    def __init__(self, sr=16000, n_fft=400, hop=160, n_mels=80):
        super().__init__()
        self.n_fft, self.hop = n_fft, hop
        self.register_buffer("window", torch.hann_window(n_fft), persistent=False)
        self.register_buffer("fb", torch.from_numpy(mel_filterbank(sr, n_fft, n_mels)), persistent=False)

    def forward(self, wav: torch.Tensor, lengths: torch.Tensor):
        spec = torch.stft(wav, self.n_fft, self.hop, window=self.window, return_complex=True)
        power = spec.abs() ** 2  # (B, F, T)
        mel = torch.matmul(self.fb, power).clamp_min(1e-10).log()
        frames = lengths // self.hop + 1
        # per-utterance mean/variance normalisation over valid frames
        T = mel.shape[-1]
        mask = (torch.arange(T, device=wav.device)[None, :] < frames[:, None]).unsqueeze(1)
        n = mask.sum(-1, keepdim=True).clamp_min(1)
        mu = (mel * mask).sum(-1, keepdim=True) / n
        var = (((mel - mu) * mask) ** 2).sum(-1, keepdim=True) / n
        mel = (mel - mu) / (var.sqrt() + 1e-5) * mask
        return mel.transpose(1, 2), frames  # (B, T, n_mels)


class SpecAugment(nn.Module):
    def __init__(self, freq_masks=2, freq_width=15, time_masks=2, time_width_frac=0.05):
        super().__init__()
        self.fm, self.fw, self.tm, self.tw = freq_masks, freq_width, time_masks, time_width_frac

    def forward(self, x):
        if not self.training:
            return x
        x = x.clone()
        B, T, F = x.shape
        for b in range(B):
            for _ in range(self.fm):
                w = np.random.randint(0, self.fw + 1)
                f0 = np.random.randint(0, max(1, F - w))
                x[b, :, f0:f0 + w] = 0
            for _ in range(self.tm):
                w = np.random.randint(0, max(1, int(T * self.tw)) + 1)
                t0 = np.random.randint(0, max(1, T - w))
                x[b, t0:t0 + w, :] = 0
        return x


class BiLSTMCTC(nn.Module):
    def __init__(self, vocab_size: int, n_mels: int = 80, conv_channels: int = 128,
                 hidden: int = 320, layers: int = 3, dropout: float = 0.2, specaug: bool = True):
        super().__init__()
        self.feats = LogMel(n_mels=n_mels)
        self.aug = SpecAugment() if specaug else nn.Identity()
        self.conv = nn.Sequential(  # 4x time subsampling (10 ms -> 40 ms frames)
            nn.Conv2d(1, conv_channels, 3, stride=2, padding=1), nn.GELU(),
            nn.Conv2d(conv_channels, conv_channels, 3, stride=2, padding=1), nn.GELU(),
        )
        conv_out = conv_channels * math.ceil(math.ceil(n_mels / 2) / 2)
        self.proj = nn.Linear(conv_out, hidden)
        self.lstm = nn.LSTM(hidden, hidden, num_layers=layers, bidirectional=True,
                            batch_first=True, dropout=dropout if layers > 1 else 0.0)
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(2 * hidden, vocab_size)

    @staticmethod
    def _sub_len(n):
        for _ in range(2):
            n = (n - 1) // 2 + 1
        return n

    def forward(self, wav, lengths):
        x, frames = self.feats(wav, lengths)
        x = self.aug(x)
        x = self.conv(x.unsqueeze(1))  # (B, C, T', F')
        B, C, T, F = x.shape
        x = self.proj(x.permute(0, 2, 1, 3).reshape(B, T, C * F))
        out_lens = self._sub_len(frames).clamp(max=T)
        x = nn.utils.rnn.pack_padded_sequence(x, out_lens.cpu(), batch_first=True, enforce_sorted=False)
        x, _ = self.lstm(x)
        x, _ = nn.utils.rnn.pad_packed_sequence(x, batch_first=True, total_length=T)
        return self.out(self.drop(x)).log_softmax(-1), out_lens


class BiLSTMCollator:
    def __init__(self, tokenizer):
        self.tok = tokenizer

    def __call__(self, items):
        lens = torch.tensor([len(it["wav"]) for it in items])
        x = torch.zeros(len(items), int(lens.max()))
        for i, it in enumerate(items):
            x[i, : len(it["wav"])] = torch.from_numpy(it["wav"])
        lab = [self.tok.encode(it["text"]) for it in items]
        y = torch.cat([torch.tensor(l, dtype=torch.long) for l in lab]) if lab else torch.zeros(0, dtype=torch.long)
        return {"wav": x, "lengths": lens, "targets": y,
                "target_lengths": torch.tensor([len(l) for l in lab]),
                "idx": torch.tensor([it["idx"] for it in items])}

"""Load a saved run's model (any family/strategy) and transcribe audio.

The deployed app uses exactly this code path, so the served model is the evaluated model.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from .audio import TARGET_SR, load_audio, resample


class Transcriber:
    def __init__(self, model_dir: str | Path, device: str | None = None):
        self.dir = Path(model_dir)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        if (self.dir / "bilstm_ctc.pt").exists():
            self._load_bilstm()
        else:
            self._load_hf()
        self.model.to(self.device).eval()

    # ------------------------------------------------------------------ loading
    def _load_bilstm(self):
        from .bilstm import BiLSTMCTC
        from .vocab import CharTokenizer
        ck = torch.load(self.dir / "bilstm_ctc.pt", map_location="cpu")
        self.tok = CharTokenizer(ck["vocab"])
        self.model = BiLSTMCTC(len(self.tok), specaug=False, **ck["hparams"])
        self.model.load_state_dict(ck["state_dict"])
        self.kind = "bilstm"

    def _load_hf(self):
        from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
        self.proc = Wav2Vec2Processor.from_pretrained(self.dir)
        info_f = self.dir / "adapter_info.json"
        if info_f.exists():                       # MMS base + our Dholuo adapter
            info = json.loads(info_f.read_text())
            if info["base_checkpoint"] == "tiny-random":   # offline smoke test only
                from .training import _tiny_config
                self.model = Wav2Vec2ForCTC(_tiny_config(info["vocab_size"], self.proc.tokenizer.pad_token_id, True))
            else:
                self.model = Wav2Vec2ForCTC.from_pretrained(
                    info["base_checkpoint"], vocab_size=info["vocab_size"],
                    pad_token_id=self.proc.tokenizer.pad_token_id, ignore_mismatched_sizes=True)
                self.model.init_adapter_layers()
            from safetensors.torch import load_file
            missing, unexpected = self.model.load_state_dict(load_file(self.dir / info["adapter_file"]), strict=False)
            if unexpected:
                raise RuntimeError(f"adapter keys not in model: {unexpected[:5]}")
        else:
            self.model = Wav2Vec2ForCTC.from_pretrained(self.dir)
        self.kind = "hf"

    # ------------------------------------------------------------------ inference
    @torch.no_grad()
    def transcribe(self, wav: np.ndarray, sr: int = TARGET_SR) -> dict:
        """Greedy CTC transcription.

        ``confidence`` is the mean posterior probability of the characters the model
        emitted (non-blank frames of the greedy path). It reflects the model's own
        certainty, NOT a calibrated probability that the transcript is correct.
        """
        wav = np.asarray(wav, dtype=np.float32)
        if wav.ndim == 2:
            wav = wav.mean(axis=1 if wav.shape[1] <= 2 else 0)
        if sr != TARGET_SR:
            wav = resample(wav, sr, TARGET_SR)
        if self.kind == "hf":
            x = (wav - wav.mean()) / np.sqrt(wav.var() + 1e-7)
            inp = torch.from_numpy(x)[None].to(self.device)
            lp = self.model(inp, attention_mask=torch.ones_like(inp, dtype=torch.long)).logits[0].float().log_softmax(-1).cpu()
            ids = lp.argmax(-1)
            text = " ".join(self.proc.decode(ids.tolist(), skip_special_tokens=True).split())
            blank = self.proc.tokenizer.pad_token_id
        else:
            t = torch.from_numpy(wav)[None].to(self.device)
            lp, lens = self.model(t, torch.tensor([len(wav)], device=self.device))
            lp = lp[0, : int(lens[0])].float().cpu()
            ids = lp.argmax(-1)
            text = self.tok.decode_ids(ids.tolist())
            blank = self.tok.blank_id
        probs = lp.exp().max(-1).values
        emitted = ids != blank
        conf = float(probs[emitted].mean()) if emitted.any() else 0.0
        return {"text": text, "confidence": conf, "duration_s": len(wav) / TARGET_SR}

    def transcribe_file(self, path: str) -> dict:
        return self.transcribe(load_audio(path, TARGET_SR), TARGET_SR)

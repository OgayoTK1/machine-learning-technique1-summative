"""Audio loading, resampling and per-file quality statistics."""
from __future__ import annotations

from math import gcd
from pathlib import Path

import numpy as np

TARGET_SR = 16_000


def _load_soundfile(path: str):
    import soundfile as sf
    wav, sr = sf.read(path, dtype="float32", always_2d=True)
    return wav, sr


def _load_librosa(path: str):
    import librosa
    wav, sr = librosa.load(path, sr=None, mono=False)
    wav = np.atleast_2d(wav).T.astype(np.float32)
    return wav, sr


def load_audio(path: str, target_sr: int = TARGET_SR) -> np.ndarray:
    """Load any supported file as mono float32 at ``target_sr``."""
    try:
        wav, sr = _load_soundfile(path)
    except Exception:  # noqa: BLE001 - e.g. m4a/webm or old libsndfile without mp3
        wav, sr = _load_librosa(path)
    wav = wav.mean(axis=1) if wav.ndim == 2 else wav
    if sr != target_sr:
        wav = resample(wav, sr, target_sr)
    return wav.astype(np.float32, copy=False)


def resample(wav: np.ndarray, sr: int, target_sr: int) -> np.ndarray:
    from scipy.signal import resample_poly
    g = gcd(int(sr), int(target_sr))
    return resample_poly(wav, target_sr // g, sr // g).astype(np.float32)


def _valid(path) -> bool:
    return isinstance(path, (str, Path)) and bool(str(path)) and Path(path).exists()


def file_info(path: str | None) -> dict:
    """Header-level info without decoding (fast). Falls back to decoding."""
    out = {"sr": np.nan, "channels": np.nan, "duration_s": np.nan, "format": None,
           "readable": False, "error": None}
    if not _valid(path):
        out["error"] = "missing"
        return out
    try:
        import soundfile as sf
        i = sf.info(path)
        out.update(sr=i.samplerate, channels=i.channels, duration_s=i.frames / i.samplerate,
                   format=f"{i.format}/{i.subtype}", readable=True)
        return out
    except Exception as e:  # noqa: BLE001
        out["error"] = f"sf: {e}"
    try:
        wav, sr = _load_librosa(path)
        out.update(sr=sr, channels=wav.shape[1], duration_s=len(wav) / sr,
                   format=Path(path).suffix.lstrip("."), readable=True, error=None)
    except Exception as e:  # noqa: BLE001
        out["error"] = f"{out['error']} | librosa: {e}"
    return out


def quality_stats(path: str | None, frame_ms: float = 25.0, silence_db: float = -40.0) -> dict:
    """Decode a file and compute simple, interpretable quality indicators.

    * rms_dbfs – overall loudness
    * peak – max |sample| (1.0 = full scale)
    * clip_frac – fraction of samples with |x| >= 0.999 (clipping)
    * lead_sil_s / trail_sil_s – leading / trailing silence (frames below ``silence_db``
      relative to the file's own max frame energy)
    * speech_frac – fraction of frames above that threshold
    * dc_offset – mean sample value
    """
    out = {"rms_dbfs": np.nan, "peak": np.nan, "clip_frac": np.nan, "lead_sil_s": np.nan,
           "trail_sil_s": np.nan, "speech_frac": np.nan, "dc_offset": np.nan,
           "decode_ok": False, "decode_error": None}
    if not _valid(path):
        out["decode_error"] = "missing"
        return out
    try:
        wav = load_audio(path, TARGET_SR)
    except Exception as e:  # noqa: BLE001
        out["decode_error"] = str(e)[:200]
        return out
    if wav.size == 0:
        out["decode_error"] = "empty"
        return out
    eps = 1e-10
    out["rms_dbfs"] = float(20 * np.log10(np.sqrt(np.mean(wav ** 2)) + eps))
    out["peak"] = float(np.max(np.abs(wav)))
    out["clip_frac"] = float(np.mean(np.abs(wav) >= 0.999))
    out["dc_offset"] = float(np.mean(wav))
    hop = int(TARGET_SR * frame_ms / 1000)
    n = len(wav) // hop
    if n >= 1:
        frames = wav[: n * hop].reshape(n, hop)
        e_db = 10 * np.log10(np.mean(frames ** 2, axis=1) + eps)
        active = e_db > (e_db.max() + silence_db)
        out["speech_frac"] = float(active.mean())
        if active.any():
            first, last = int(np.argmax(active)), int(n - 1 - np.argmax(active[::-1]))
            out["lead_sil_s"] = first * hop / TARGET_SR
            out["trail_sil_s"] = (n - 1 - last) * hop / TARGET_SR
    out["decode_ok"] = True
    return out

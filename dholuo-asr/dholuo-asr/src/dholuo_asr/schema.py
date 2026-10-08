"""Discover the DhoNam metadata layout and build a canonical manifest.

The exact file layout of the Mozilla Data Collective download is NOT hard-coded. The
audit notebook prints every metadata file and column it finds, then maps columns to the
canonical names below using candidate lists. Any mapping can be overridden explicitly
with ``column_map`` (canonical -> actual column name) if auto-detection is wrong.

Canonical manifest columns
--------------------------
utt_id, audio_path, text_raw, speaker, dialect, gender, age, domain, source_file
plus any unmapped original columns, prefixed ``orig_``.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path

import pandas as pd

AUDIO_EXTS = {".wav", ".mp3", ".flac", ".ogg", ".opus", ".webm", ".m4a"}
META_EXTS = {".csv", ".tsv", ".json", ".jsonl", ".parquet", ".xlsx"}

CANDIDATES: dict[str, list[str]] = {
    "audio_path": ["audio_path", "path", "audio", "file", "filename", "file_name",
                   "audio_file", "audio_filename", "recording", "clip", "wav", "file_path"],
    "text_raw": ["transcription", "transcript", "sentence", "text", "prompt", "utterance",
                 "normalized_text", "raw_transcription", "transcripts", "label"],
    "speaker": ["speaker_id", "speaker", "client_id", "participant_id", "contributor_id",
                "user_id", "speaker_name", "participant", "contributor", "spk"],
    "dialect": ["dialect", "variant", "accent", "accents", "sub_dialect", "dialects"],
    "gender": ["gender", "sex"],
    "age": ["age", "age_group", "age_range", "agegroup", "age_bracket"],
    "domain": ["domain", "category", "topic", "theme", "domains", "subject"],
    "utt_id": ["id", "utt_id", "utterance_id", "recording_id", "clip_id", "audio_id", "uid"],
}


def _norm_col(c: str) -> str:
    return str(c).strip().lower().replace(" ", "_").replace("-", "_")


def find_files(root: str | Path) -> tuple[list[Path], list[Path]]:
    """Return (metadata_files, audio_files) under ``root`` (recursive)."""
    meta, audio = [], []
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            p = Path(dirpath) / fn
            ext = p.suffix.lower()
            if ext in AUDIO_EXTS:
                audio.append(p)
            elif ext in META_EXTS:
                meta.append(p)
    return sorted(meta), sorted(audio)


def read_table(path: Path) -> pd.DataFrame:
    ext = path.suffix.lower()
    if ext == ".tsv":
        return pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, quoting=3)
    if ext == ".csv":
        return pd.read_csv(path, dtype=str, keep_default_na=False)
    if ext == ".jsonl":
        return pd.read_json(path, lines=True, dtype=False).astype(str)
    if ext == ".json":
        try:
            return pd.read_json(path, dtype=False).astype(str)
        except ValueError:
            return pd.read_json(path, lines=True, dtype=False).astype(str)
    if ext == ".parquet":
        return pd.read_parquet(path).astype(str)
    if ext == ".xlsx":
        return pd.read_excel(path, dtype=str).fillna("")
    raise ValueError(f"unsupported metadata file {path}")


def detect_columns(columns: list[str], column_map: dict | None = None) -> dict[str, str | None]:
    """Map canonical names to actual columns. Explicit ``column_map`` wins."""
    column_map = dict(column_map or {})
    normed = {_norm_col(c): c for c in columns}
    used: set[str] = set(v for v in column_map.values() if v)
    out: dict[str, str | None] = {}
    for canon, cands in CANDIDATES.items():
        if canon in column_map:
            out[canon] = column_map[canon]
            continue
        hit = None
        for cand in cands:
            if cand in normed and normed[cand] not in used:
                hit = normed[cand]
                break
        out[canon] = hit
        if hit:
            used.add(hit)
    return out


def looks_like_utterance_table(df: pd.DataFrame, column_map: dict | None = None) -> bool:
    m = detect_columns(list(df.columns), column_map)
    return m["audio_path"] is not None and m["text_raw"] is not None and len(df) > 0


def build_audio_index(audio_files: list[Path], root: str | Path) -> dict[str, list[Path]]:
    """Index audio files by relative path, basename and stem (all lower-cased)."""
    idx: dict[str, list[Path]] = {}
    root = Path(root)
    for p in audio_files:
        keys = {str(p.relative_to(root)).lower().replace("\\", "/"), p.name.lower(),
                p.stem.lower()}
        for k in keys:
            idx.setdefault(k, []).append(p)
    return idx


def resolve_audio(value: str, idx: dict[str, list[Path]]) -> tuple[str | None, str]:
    """Return (path, status) where status in {ok, missing, ambiguous}."""
    v = str(value).strip().lower().replace("\\", "/").lstrip("./")
    for key in (v, Path(v).name, Path(v).stem):
        hits = idx.get(key)
        if hits:
            if len(hits) > 1 and key != v:
                return str(hits[0]), "ambiguous"
            return str(hits[0]), "ok"
    return None, "missing"


def stable_id(*parts: str) -> str:
    return hashlib.sha1("||".join(map(str, parts)).encode()).hexdigest()[:16]


def build_manifest(root: str | Path, column_map: dict | None = None,
                   include_files: list[str] | None = None) -> tuple[pd.DataFrame, dict]:
    """Load every utterance-level metadata table under ``root`` into one manifest.

    Returns (manifest, info). ``info`` records which files were used/skipped and the
    detected column mapping for each, so the audit can document it.
    """
    root = Path(root)
    meta_files, audio_files = find_files(root)
    idx = build_audio_index(audio_files, root)
    info = {"root": str(root), "n_audio_files": len(audio_files), "files": []}
    frames = []
    for mf in meta_files:
        rel = str(mf.relative_to(root))
        if include_files and rel not in include_files and mf.name not in include_files:
            info["files"].append({"file": rel, "used": False, "reason": "not in include_files"})
            continue
        try:
            df = read_table(mf)
        except Exception as e:  # noqa: BLE001 - we want to log anything
            info["files"].append({"file": rel, "used": False, "reason": f"unreadable: {e}"})
            continue
        if not looks_like_utterance_table(df, column_map):
            info["files"].append({"file": rel, "used": False, "rows": len(df),
                                  "columns": list(df.columns),
                                  "reason": "no audio/text column detected"})
            continue
        m = detect_columns(list(df.columns), column_map)
        info["files"].append({"file": rel, "used": True, "rows": len(df),
                              "columns": list(df.columns), "mapping": m})
        out = pd.DataFrame(index=df.index)
        for canon, col in m.items():
            out[canon] = df[col].astype(str) if col else pd.NA
        for c in df.columns:
            if c not in m.values():
                out[f"orig_{_norm_col(c)}"] = df[c]
        out["source_file"] = rel
        out["audio_ref"] = out["audio_path"]
        frames.append(out)
    if not frames:
        raise RuntimeError(
            "No utterance-level metadata table found. Inspect info['files'] and pass "
            "column_map={'audio_path': ..., 'text_raw': ..., 'speaker': ...}.")
    man = pd.concat(frames, ignore_index=True)
    resolved = man["audio_ref"].map(lambda v: resolve_audio(v, idx))
    man["audio_path"] = resolved.map(lambda t: t[0])
    man["audio_status"] = resolved.map(lambda t: t[1])
    if man["utt_id"].isna().all():
        man["utt_id"] = [stable_id(a, t) for a, t in zip(man["audio_ref"], man["text_raw"])]
    man["utt_id"] = man["utt_id"].astype(str)
    for c in ["speaker", "dialect", "gender", "age", "domain"]:
        man[c] = man[c].astype("string").str.strip()
        man.loc[man[c].isin(["", "nan", "None", "NaN", "null"]), c] = pd.NA
    man["dialect"] = man["dialect"].str.title()
    man["gender"] = man["gender"].str.lower()
    # audio files present on disk but not referenced by any metadata row
    referenced = set(man["audio_path"].dropna())
    info["unreferenced_audio"] = [str(p) for p in audio_files if str(p) not in referenced]
    return man, info

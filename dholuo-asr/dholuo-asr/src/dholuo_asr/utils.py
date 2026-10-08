"""Seeding, environment capture, config handling and Kaggle artifact lookup."""
from __future__ import annotations

import copy
import json
import os
import platform
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np


def set_seed(seed: int, deterministic: bool = False):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def env_info() -> dict:
    info = {"python": sys.version.split()[0], "platform": platform.platform(),
            "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "kaggle": bool(os.environ.get("KAGGLE_KERNEL_RUN_TYPE"))}
    for mod in ["numpy", "pandas", "torch", "transformers", "peft", "soundfile", "librosa",
                "scipy", "sklearn", "pyctcdecode", "kenlm"]:
        try:
            info[mod] = __import__(mod).__version__
        except Exception:  # noqa: BLE001
            info[mod] = None
    try:
        import torch
        if torch.cuda.is_available():
            info["gpu"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
            info["cuda"] = torch.version.cuda
    except Exception:  # noqa: BLE001
        pass
    try:
        info["git_commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).parent, stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:  # noqa: BLE001
        info["git_commit"] = None
    return info


def deep_update(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_update(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_experiment(config_path: str | Path, name: str) -> dict:
    """Resolve ``base`` + named experiment overrides from configs/experiments.yaml."""
    import yaml
    cfg = yaml.safe_load(Path(config_path).read_text())
    if name not in cfg["experiments"]:
        raise KeyError(f"{name!r} not in {list(cfg['experiments'])}")
    exp = cfg["experiments"][name]
    parent = exp.get("extends")
    resolved = load_experiment(config_path, parent) if parent else copy.deepcopy(cfg["base"])
    resolved = deep_update(resolved, {k: v for k, v in exp.items() if k != "extends"})
    resolved["name"] = name
    return resolved


def find_artifact(name: str, search_roots=("/kaggle/working", "/kaggle/input", ".")) -> Path:
    """Find a file produced by an earlier notebook (current dir first, then inputs)."""
    for root in search_roots:
        root = Path(root)
        if not root.exists():
            continue
        direct = root / name
        if direct.exists():
            return direct
        hits = sorted(root.rglob(name))
        if hits:
            return hits[0]
    raise FileNotFoundError(
        f"{name} not found. On Kaggle: add the output of the notebook that creates it "
        f"via 'Add Input' -> 'Notebook output'.")


def dump_json(obj, path: str | Path):
    def default(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, (set, tuple)):
            return list(o)
        return str(o)
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=default))

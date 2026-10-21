"""Versioned output allocation with an injected atomic cross-container claim."""
import re
from pathlib import Path


def reserve_run(root, stem, claim):
    if not re.fullmatch(r"[a-zA-Z0-9_]+", stem):
        raise ValueError("Invalid run stem")
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    version = 1
    while True:
        name = f"{stem}_v{version}"
        path = root/name
        if not path.exists() and claim(name):
            path.mkdir(exist_ok=False)
            return name, path
        version += 1


def checked_run(root, name):
    if not re.fullmatch(r"rfdetr_dqk_(?:enhanced|smoke)_v[1-9][0-9]*", name):
        raise ValueError("Use an existing versioned DQK run name")
    path = Path(root)/name
    if not path.is_dir():
        raise FileNotFoundError(path)
    return path

"""Dataset name(s) -> merged {id, code, input, output} rows.

Add a converted dataset by running its folder's convert.py (saves ./data via
save_to_disk) and listing it in _LOCAL. cruxeval keeps its own Hub-fallback
loader and is held out entirely for eval (eval_cruxeval_*.py), never trained on.
"""

from __future__ import annotations

import os
from pathlib import Path

_LOCAL = {"mbpp": "MBPP", "humaneval": "HumanEval", "pyx": "PyX"}  # name -> folder, data in ./data


def load_cruxeval():
    """Full CRUXEval-O (the held-out test set). Prefer a local save_to_disk copy;
    the HF builder FileLock dies on NFS caches."""
    local_dir = os.environ.get("CRUXEVAL_DIR")
    if local_dir and os.path.isdir(local_dir):
        from datasets import load_from_disk

        return list(load_from_disk(local_dir))
    from datasets import load_dataset

    return list(load_dataset("cruxeval-org/cruxeval", split="test"))


def load_one(name: str) -> list[dict]:
    key = name.strip().lower()
    if key == "cruxeval":
        return load_cruxeval()
    if key in _LOCAL:
        from datasets import load_from_disk

        d = os.environ.get(key.upper() + "_DIR") or str(Path(__file__).parent / _LOCAL[key] / "data")
        return list(load_from_disk(d))
    raise ValueError(f"unknown data source {name!r}; pick from {['cruxeval', *_LOCAL]}")

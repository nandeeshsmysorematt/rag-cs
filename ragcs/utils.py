from __future__ import annotations

import json
import random
from typing import Iterable, Iterator, Optional

import numpy as np


def set_seed(seed: int = 42) -> None:
    """Fix seeds for Python, NumPy and (if installed) PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def resolve_device(device: Optional[str] = None) -> str:
    if device:
        return device
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def read_jsonl(path: str) -> Iterator[dict]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path: str, rows: Iterable[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

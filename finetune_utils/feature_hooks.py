"""Forward‑hook helpers to grab intermediate features."""
from __future__ import annotations
from typing import Dict, List
from contextlib import contextmanager
import torch
from torch import nn

_FEATURE_STORE: Dict[str, List[torch.Tensor]] = {}

def _save(key: str):
    def fn(_, __, out):
        _FEATURE_STORE.setdefault(key, []).append(out.detach())
    return fn

def register_hooks(model: nn.Module, names: List[str]):
    handles = []
    for name in names:
        m = model
        for part in name.split('.'):
            m = getattr(m, part)
        handles.append(m.register_forward_hook(_save(name)))
    return handles

def pop_features() -> Dict[str, torch.Tensor]:
    out = {k: torch.stack(v) for k, v in _FEATURE_STORE.items()}
    _FEATURE_STORE.clear()
    return out

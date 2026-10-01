"""Bounded cache for precomputed teacher features."""

import gc
from pathlib import Path

import numpy as np
import torch


def clear_gpu_cache():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()


class MemoryEfficientFeatureCache:
    def __init__(self, maxsize=64):
        self.cache = {}
        self.maxsize = maxsize
        self.access_order = []

    def get(self, path: Path):
        key = str(path)
        if key in self.cache:
            self.access_order.remove(key)
            self.access_order.append(key)
            return self.cache[key]
        arr = np.load(key)
        tensor = torch.from_numpy(arr).cuda(non_blocking=True)
        if len(self.cache) >= self.maxsize:
            oldest = self.access_order.pop(0)
            del self.cache[oldest]
        self.cache[key] = tensor
        self.access_order.append(key)
        return tensor

    def clear(self):
        self.cache.clear()
        self.access_order.clear()
        clear_gpu_cache()


feature_cache = MemoryEfficientFeatureCache()


def load_cached_npy_features(
    base: Path, teacher: str, split: str, stems: list[str], keys: list[str]
):
    stacked = {k: [] for k in keys}
    for stem in stems:
        for k in keys:
            fname = f"{stem}_{k.replace('.', '_').replace('[', '_').replace(']', '')}.npy"
            stacked[k].append(feature_cache.get(base / teacher / split / fname))
    return [torch.stack(stacked[k]) for k in keys]

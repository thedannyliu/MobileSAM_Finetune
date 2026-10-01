"""Learning-rate schedule used by the fine-tuning driver."""

import numpy as np
import torch


class WarmupCosineLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup, total, min_ratio=0.0, last_epoch=-1):
        self.warmup = warmup
        self.total = total
        self.min_ratio = min_ratio
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        cur = self.last_epoch + 1
        if cur < self.warmup:
            return [base_lr * cur / self.warmup for base_lr in self.base_lrs]
        prog = (cur - self.warmup) / max(1, (self.total - self.warmup))
        cos = 0.5 * (1 + np.cos(np.pi * prog))
        return [
            base_lr * (self.min_ratio + (1 - self.min_ratio) * cos) for base_lr in self.base_lrs
        ]

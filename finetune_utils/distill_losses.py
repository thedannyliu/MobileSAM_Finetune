# ───────────────────────── finetune_utils/distill_losses.py ─────────────────────────
"""Knowledge distillation losses (v2)

This revision implements the four new distillation objectives described in the
project documentation and removes the legacy attention / RKD losses.

  1. Encoder Patch Tokens      – L2  + 1-CosSim
  2. Prompt-conditioned Embeds – MSE + 1-CosSim
  3. Decoder Mask Tokens       – KLDiv on token logits
  4. Dense Mask Logits         –  KLDiv + Focal (soft targets)

Every function follows the signature
    loss = fn(student_feat, teacher_feat, **hyper_params)
returning a scalar tensor on the same device / dtype as the inputs.
"""

from __future__ import annotations
import torch
import torch.nn.functional as F
from torch import Tensor

from typing import List, Union


# ─── helper ───
def _permute_if_channel_last(t: Tensor) -> Tensor:
    if t.ndim == 4:
        # Heuristic: if the last dim is much larger than spatial dims, it's probably channels.
        if t.shape[3] > t.shape[1] and t.shape[3] > t.shape[2]:
            return t.permute(0, 3, 1, 2)
    return t


def _interpolate_if_needed(src: Tensor, ref: Tensor) -> Tensor:
    if src.ndim == 5:
        src = src.mean(1)
    if src.shape[2:] != ref.shape[2:]:
        src = F.interpolate(src, size=ref.shape[2:], mode="bilinear", align_corners=False)
    return src


def _stack(x: Union[Tensor, List[Tensor]]) -> Tensor:
    """Ensure input is a single Tensor (N,C,*) or (N,*) by cat / to(float)."""
    if isinstance(x, list):
        x = torch.cat([t if t.ndim == x[0].ndim else t.unsqueeze(0) for t in x], 0)
    return x.float()


def _gap(x: Union[Tensor, List[Tensor]]) -> Tensor:
    x = _stack(x)
    if x.ndim > 2:
        x = F.adaptive_avg_pool2d(x, 1).flatten(1)
    return x


def _match_channels(src: Tensor, tgt: Tensor) -> Tensor:
    """Align channel dims via average pooling or tiling so Cs == Ct."""
    Cs, Ct = src.size(1), tgt.size(1)
    if Cs == Ct:
        return src
    if Cs % Ct == 0:  # down-sample
        k = Cs // Ct
        return src.view(src.size(0), Ct, k, *src.shape[2:]).mean(2)
    if Ct % Cs == 0:  # up-sample (repeat)
        k = Ct // Cs
        return (
            src.unsqueeze(2)
            .repeat(1, 1, k, *([1] * (src.ndim - 2)))
            .view(src.size(0), Ct, *src.shape[2:])
        )
    return src.mean(1, keepdim=True).repeat(1, Ct, 1, 1)


def _flat(x: Tensor) -> Tensor:
    """Flatten all dims except batch for cosine / KL computations."""
    return x.flatten(1)


# ──────────────────── 1. Encoder patch tokens ────────────────────


def encoder_patch_loss(
    feat_s: Union[Tensor, List[Tensor]],
    feat_t: Union[Tensor, List[Tensor]],
    w_l2: float = 1.0,
    w_cos: float = 1.0,
) -> Tensor:
    """L = w_l2·MSE  +  w_cos·(1−cosine)"""
    fs, ft = _stack(feat_s), _stack(feat_t)
    ft = _interpolate_if_needed(ft, fs)
    fs = _match_channels(fs, ft)

    mse = F.mse_loss(fs, ft, reduction="mean")
    cos = 1.0 - F.cosine_similarity(_flat(fs), _flat(ft)).mean()
    return w_l2 * mse + w_cos * cos


# ──────────────────── 2. Prompt-conditioned embedding ────────────────────


def prompt_embed_loss(
    feat_s: Union[Tensor, List[Tensor]],
    feat_t: Union[Tensor, List[Tensor]],
    w_mse: float = 0.7,
    w_cos: float = 0.3,
) -> Tensor:
    fs, ft = _stack(feat_s), _stack(feat_t)
    ft = _interpolate_if_needed(ft, fs)
    fs = _match_channels(fs, ft)

    mse = F.mse_loss(F.normalize(fs, dim=1), F.normalize(ft, dim=1), reduction="mean")
    cos = 1.0 - F.cosine_similarity(_flat(fs), _flat(ft)).mean()
    return w_mse * mse + w_cos * cos


# ──────────────────── 3. Mask token logits ────────────────────


def mask_token_loss(
    token_s: Tensor,  # B × T × C, usually T=4
    token_t: Tensor,
    w_kl: float = 1.0,
    temperature: float = 0.5,
) -> Tensor:
    assert token_s.ndim == 3 and token_t.ndim == 3, "Expect (B,T,C) tensors"
    min_b = min(token_s.size(0), token_t.size(0))
    token_s, token_t = token_s[:min_b], token_t[:min_b]

    ps = F.log_softmax(token_s / temperature, dim=-1)
    pt = F.softmax(token_t / temperature, dim=-1)
    return w_kl * F.kl_div(ps, pt, reduction="batchmean") * (temperature**2)


# ──────────────────── 4. Dense mask logits ────────────────────


def dense_mask_logits_loss(
    logit_s: Tensor,  # B × 1 × H × W  (before sigmoid)
    logit_t: Tensor,
    w_kl: float = 0.6,
    w_focal: float = 0.4,
    gamma: float = 2.0,
) -> Tensor:
    logit_s, logit_t = logit_s.float(), logit_t.float()
    logit_t = _interpolate_if_needed(logit_t, logit_s)
    min_b = min(logit_s.size(0), logit_t.size(0))
    logit_s, logit_t = logit_s[:min_b], logit_t[:min_b]

    p_s = torch.sigmoid(logit_s)
    p_t = torch.sigmoid(logit_t).detach()

    # avoid numerical issues when taking log
    eps = 1e-6
    p_s = p_s.clamp(min=eps, max=1 - eps)
    p_t = p_t.clamp(min=eps, max=1 - eps)

    # KL divergence for Bernoulli distributions (pixel-wise)
    kl = F.kl_div(torch.log(p_s), p_t, reduction="mean")

    # Soft-label focal loss
    fl = ((1 - p_s) ** gamma * p_t * (-torch.log(p_s))).mean()

    return w_kl * kl + w_focal * fl

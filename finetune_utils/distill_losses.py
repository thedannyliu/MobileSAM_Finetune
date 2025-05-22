"""Knowledge–distillation loss utilities for MobileSAM fine-tune.

所有函式皆回傳 fp32 scalar Tensor。
"""
from __future__ import annotations
from typing import List, Union

import torch
import torch.nn.functional as F
from torch import Tensor

# --------------------------------------------------------------------------- #
# helpers                                                                     #
# --------------------------------------------------------------------------- #

def _interpolate_if_needed(src: Tensor, ref: Tensor) -> Tensor:
    """若空間尺寸不同，將 `src` resize 成 `ref` (bilinear)。"""
    if src.shape[2:] != ref.shape[2:]:
        src = F.interpolate(src, size=ref.shape[2:], mode="bilinear", align_corners=False)
    return src


def _stack(feat: Union[Tensor, List[Tensor]]) -> Tensor:
    """
    把 hook 取回的 list 堆疊成 (B, …)；若已是 Tensor 直接回傳。
    不做 flatten；保留空間/序列維度。
    """
    if isinstance(feat, list):
        # 各 element 可能 shape=(1,C,H,W) 或 (C,H,W)
        # 先確定有 batch 維 (1,…)；再 concat
        processed = []
        for f in feat:
            if f.dim() == len(feat[0].shape):      # (C,H,W)
                processed.append(f.unsqueeze(0))   # →(1,C,H,W)
            else:
                processed.append(f)                # 已是 (1,C,H,W)
        feat = torch.cat(processed, dim=0)         # (B,C,H,W)
    return feat.float()                            # 確保 fp32


def _collapse(feat: Union[Tensor, List[Tensor]]) -> Tensor:
    """
    專給 RKD：最後要比 (B,D)，故再 flatten(1)。
    """
    feat = _stack(feat)        # (B, …)
    if feat.ndim > 2:
        feat = feat.flatten(1) # (B,D)
    return feat

# --------------------------------------------------------------------------- #
# 1. Encoder feature matching                                                 #
# --------------------------------------------------------------------------- #

def encoder_matching_loss(
    feats_s: List[Union[Tensor, List[Tensor]]],
    feats_t: List[Tensor],
    lambda_mse: float = 1.0,
    lambda_kl:  float = 1.0,
    temperature: float = 1.0,
    **_ignored
) -> Tensor:
    """Encoder 特徵 distillation（MSE + KL）。"""
    mse_total, kl_total = 0.0, 0.0
    for fs_raw, ft in zip(feats_s, feats_t):
        fs = _stack(fs_raw)                # (B,C,H,W)
        ft = _interpolate_if_needed(ft, fs)
        mse_total += F.mse_loss(fs, ft, reduction="mean")

        ps = F.log_softmax(fs.flatten(1) / temperature, dim=-1)
        pt =  F.softmax(ft.flatten(1) / temperature, dim=-1)
        kl_total += F.kl_div(ps, pt, reduction="batchmean") * temperature**2

    return lambda_mse * mse_total + lambda_kl * kl_total

# --------------------------------------------------------------------------- #
# 2. Decoder pre-logits matching                                              #
# --------------------------------------------------------------------------- #

def decoder_matching_loss(
    feat_s_raw: Union[Tensor, List[Tensor]],
    feat_t: Tensor,
    lambda_mse: float = 1.0,
    lambda_cos: float = 1.0,
    lambda_kl:  float = 1.0,
    temperature: float = 1.0,
    **_ignored
) -> Tensor:
    """Mask-decoder pre-logits distillation（MSE + CosSim + KL）。"""
    feat_s = _stack(feat_s_raw)            # (B,C,H,W) 或 (B,D)
    feat_t = _interpolate_if_needed(feat_t, feat_s)

    mse = F.mse_loss(feat_s, feat_t, reduction="mean")
    cos = 1 - F.cosine_similarity(feat_s.flatten(1), feat_t.flatten(1)).mean()

    ps = F.log_softmax(feat_s.flatten(1) / temperature, dim=-1)
    pt =  F.softmax(feat_t.flatten(1) / temperature, dim=-1)
    kl = F.kl_div(ps, pt, reduction="batchmean") * temperature**2

    return lambda_mse * mse + lambda_cos * cos + lambda_kl * kl

# --------------------------------------------------------------------------- #
# 3. Attention map distillation                                               #
# --------------------------------------------------------------------------- #

def attention_matching_loss(
    attn_s: List[Union[Tensor, List[Tensor]]],
    attn_t: List[Tensor],
    lambda_attn: float = 1.0,
    temperature: float = 1.0,
    **_ignored
) -> Tensor:
    """ViT attention map distillation（KL）。"""
    loss = 0.0
    for as_raw, at in zip(attn_s, attn_t):
        as_ = _stack(as_raw)               # (B, h, N, N) 已經 batch 化
        ps = F.log_softmax(as_ / temperature, dim=-1)
        pt =  F.softmax(at / temperature,  dim=-1)
        loss += F.kl_div(ps, pt, reduction="batchmean") * temperature**2
    return lambda_attn * loss / max(1, len(attn_s))

# --------------------------------------------------------------------------- #
# 4. Relational knowledge distillation (RKD)                                  #
# --------------------------------------------------------------------------- #

def _pdist(e: Tensor) -> Tensor:
    return torch.cdist(e, e, p=2)

def rkd_loss(
    embed_s_raw: Union[Tensor, List[Tensor]],
    embed_t_raw: Union[Tensor, List[Tensor]],
    lambda_rkd: float = 1.0,
    dist_factor: float = 1.0,
    angle_factor: float = 1.0,
    **_ignored
) -> Tensor:
    """RKD：距離＋角度。"""
    embed_s = _collapse(embed_s_raw)       # (B,D)
    embed_t = _collapse(embed_t_raw)       # (B,D)

    # -- distance --
    ds, dt = _pdist(embed_s), _pdist(embed_t)
    mask = dt > 0
    ds = ds / (ds[mask].mean() + 1e-6)
    dt = dt / (dt[mask].mean() + 1e-6)
    loss_dist = F.smooth_l1_loss(ds, dt, reduction="mean")

    # -- angle --
    diff_s = embed_s.unsqueeze(2) - embed_s.unsqueeze(1)
    diff_t = embed_t.unsqueeze(2) - embed_t.unsqueeze(1)
    angle_s = F.normalize(diff_s, dim=-1)
    angle_t = F.normalize(diff_t, dim=-1)
    angle_s = torch.matmul(angle_s, angle_s.transpose(-1, -2))
    angle_t = torch.matmul(angle_t, angle_t.transpose(-1, -2))
    loss_ang = F.smooth_l1_loss(angle_s, angle_t, reduction="mean")

    return lambda_rkd * (dist_factor * loss_dist + angle_factor * loss_ang)

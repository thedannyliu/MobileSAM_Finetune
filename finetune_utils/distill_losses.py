"""Knowledge–distillation loss utilities for MobileSAM fine-tune."""
from __future__ import annotations
from typing import List, Union

import torch
import torch.nn.functional as F
from torch import Tensor

# --------------------------------------------------------------------------- #
# helpers                                                                     #
# --------------------------------------------------------------------------- #

def _interpolate_if_needed(src: Tensor, ref: Tensor) -> Tensor:
    """
    • 5D (B,K,C,H,W) → mean over K → (B,C,H,W)
    • If spatial dims differ, resize src to ref via bilinear
    """
    if src.ndim == 5:
        src = src.mean(dim=1)
    if src.shape[2:] != ref.shape[2:]:
        src = F.interpolate(src, size=ref.shape[2:], mode="bilinear", align_corners=False)
    return src


def _stack(feat: Union[Tensor, List[Tensor]]) -> Tensor:
    """
    Stack hook features:
    - If list of tensors, cat along dim 0.
    - Else assume Tensor of shape (B,...) or (C,H,W) and return.
    """
    if isinstance(feat, list):
        processed = []
        for f in feat:
            if f.dim() == feat[0].dim():  # same dims
                processed.append(f)
            else:
                processed.append(f.unsqueeze(0))
        feat = torch.cat(processed, dim=0)
    return feat.float()


def _gap(feat_raw: Union[Tensor, List[Tensor]]) -> Tensor:
    """
    Global avg pool + flatten for RKD to reduce dimension and avoid OOM.
    """
    feat = _stack(feat_raw)
    if feat.ndim > 2:
        feat = F.adaptive_avg_pool2d(feat, 1).flatten(1)
    return feat


def _match_channels(src: Tensor, tgt: Tensor) -> Tensor:
    """
    Align channel dimension of src to tgt:
    - If src channels divisible by tgt: group mean reduce.
    - If tgt divisible by src: repeat src.
    - Else average over channels then repeat.
    """
    Cs, Ct = src.size(1), tgt.size(1)
    if Cs == Ct:
        return src
    if Cs % Ct == 0:
        k = Cs // Ct
        return src.view(src.size(0), Ct, k, *src.shape[2:]).mean(dim=2)
    elif Ct % Cs == 0:
        k = Ct // Cs
        expanded = src.unsqueeze(2).repeat(1, 1, k, *([1] * (src.ndim - 2)))
        return expanded.view(src.size(0), Ct, *src.shape[2:])
    else:
        avg = src.mean(dim=1, keepdim=True)
        return avg.repeat(1, Ct, 1, 1)

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
    """
    Pixel/patch-level encoder feature distillation: MSE + KL.
    """
    mse_total, kl_total = 0.0, 0.0
    for fs_raw, ft in zip(feats_s, feats_t):
        fs = _stack(fs_raw)                  # (Bs,Cs,H,W)
        ft = _interpolate_if_needed(ft, fs)  # (Bt,Ct,H,W)
        b = min(fs.size(0), ft.size(0))      # batch align
        fs, ft = fs[:b], ft[:b]
        fs = _match_channels(fs, ft)

        mse_total += F.mse_loss(fs, ft, reduction="mean")
        ps = F.log_softmax(fs.flatten(1) / temperature, dim=-1)
        pt = F.softmax(ft.flatten(1) / temperature, dim=-1)
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
    """
    Mask-decoder pre-logits distillation: MSE + Cosine + KL.
    """
    fs = _stack(feat_s_raw)
    ft = _interpolate_if_needed(feat_t, fs)
    b = min(fs.size(0), ft.size(0))
    fs, ft = fs[:b], ft[:b]
    fs = _match_channels(fs, ft)

    mse = F.mse_loss(fs, ft, reduction="mean")
    cos = 1 - F.cosine_similarity(fs.flatten(1), ft.flatten(1)).mean()
    ps = F.log_softmax(fs.flatten(1) / temperature, dim=-1)
    pt = F.softmax(ft.flatten(1) / temperature, dim=-1)
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
    """
    ViT attention map distillation: KL divergence.
    """
    loss = 0.0
    for as_raw, at in zip(attn_s, attn_t):
        as_ = _stack(as_raw)               # (Bs,h,N,N)
        b = min(as_.size(0), at.size(0))
        as_, at = as_[:b], at[:b]
        ps = F.log_softmax(as_ / temperature, dim=-1)
        pt = F.softmax(at / temperature, dim=-1)
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
    """
    RKD: distance + angle matching.
    """
    es, et = _gap(embed_s_raw), _gap(embed_t_raw)  # (Bs,C) / (Bt,C)
    b = min(es.size(0), et.size(0))
    es, et = es[:b], et[:b]

    if es.size(0) < 2:
        return es.new_tensor(0.0)

    ds, dt = _pdist(es), _pdist(et)
    mask = dt > 0
    ds = ds / (ds[mask].mean() + 1e-6)
    dt = dt / (dt[mask].mean() + 1e-6)
    loss_dist = F.smooth_l1_loss(ds, dt, reduction="mean")

    diff_s = es.unsqueeze(2) - es.unsqueeze(1)
    diff_t = et.unsqueeze(2) - et.unsqueeze(1)
    norm_s = F.normalize(diff_s, dim=-1)
    norm_t = F.normalize(diff_t, dim=-1)
    angle_s = torch.matmul(norm_s, norm_s.transpose(-1, -2))
    angle_t = torch.matmul(norm_t, norm_t.transpose(-1, -2))
    loss_ang = F.smooth_l1_loss(angle_s, angle_t, reduction="mean")

    return lambda_rkd * (dist_factor * loss_dist + angle_factor * loss_ang)

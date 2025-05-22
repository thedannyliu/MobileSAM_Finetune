"""Knowledge–distillation loss utilities for MobileSAM fine‑tune.
Public API returns scalar loss tensors (fp32)."""
from __future__ import annotations
import torch
import torch.nn.functional as F
from torch import Tensor

# ----- helper -----

def _interpolate_if_needed(src: Tensor, ref: Tensor) -> Tensor:
    """Resize spatial dims of `src` to match `ref` via bilinear if needed."""
    if src.shape[2:] != ref.shape[2:]:
        src = F.interpolate(src, size=ref.shape[2:], mode="bilinear", align_corners=False)
    return src

# 1. Encoder feature matching -------------------------------------------------

def encoder_matching_loss(
    feats_s: list[Tensor],
    feats_t: list[Tensor],
    lambda_mse: float = 1.0,
    lambda_kl: float = 1.0,
    temperature: float = 1.0,
) -> Tensor:
    mse_total, kl_total = 0.0, 0.0
    for fs, ft in zip(feats_s, feats_t):
        ft = _interpolate_if_needed(ft, fs)
        mse_total += F.mse_loss(fs, ft, reduction="mean")
        ps = F.log_softmax(fs.flatten(1) / temperature, dim=-1)
        pt = F.softmax(ft.flatten(1) / temperature, dim=-1)
        kl_total += F.kl_div(ps, pt, reduction="batchmean") * temperature ** 2
    return lambda_mse * mse_total + lambda_kl * kl_total

# 2. Decoder pre‑logits matching ---------------------------------------------

def decoder_matching_loss(
    feat_s: Tensor,
    feat_t: Tensor,
    lambda_mse: float = 1.0,
    lambda_cos: float = 1.0,
    lambda_kl: float = 1.0,
    temperature: float = 1.0,
) -> Tensor:
    feat_t = _interpolate_if_needed(feat_t, feat_s)
    mse = F.mse_loss(feat_s, feat_t, reduction="mean")
    cos = 1 - F.cosine_similarity(feat_s.flatten(1), feat_t.flatten(1)).mean()
    ps = F.log_softmax(feat_s.flatten(1) / temperature, dim=-1)
    pt = F.softmax(feat_t.flatten(1) / temperature, dim=-1)
    kl = F.kl_div(ps, pt, reduction="batchmean") * temperature ** 2
    return lambda_mse * mse + lambda_cos * cos + lambda_kl * kl

# 3. Attention map distillation ----------------------------------------------

def attention_matching_loss(
    attn_s: list[Tensor],
    attn_t: list[Tensor],
    lambda_attn: float = 1.0,
    temperature: float = 1.0,
) -> Tensor:
    loss = 0.0
    for as_, at_ in zip(attn_s, attn_t):
        ps = F.log_softmax(as_ / temperature, dim=-1)
        pt = F.softmax(at_ / temperature, dim=-1)
        loss += F.kl_div(ps, pt, reduction="batchmean") * temperature ** 2
    return lambda_attn * loss / max(1, len(attn_s))

# 4. Relational knowledge distillation ---------------------------------------

def _pdist(e: Tensor) -> Tensor:
    return torch.cdist(e, e, p=2)

def rkd_loss(
    embed_s: Tensor,
    embed_t: Tensor,
    lambda_rkd: float = 1.0,
    dist_factor: float = 1.0,
    angle_factor: float = 1.0,
) -> Tensor:
    # ---- distance ----
    ds, dt = _pdist(embed_s), _pdist(embed_t)
    mask = dt > 0
    ds = ds / (ds[mask].mean() + 1e-6)
    dt = dt / (dt[mask].mean() + 1e-6)
    loss_dist = F.smooth_l1_loss(ds, dt, reduction="mean")
    # ---- angle ----
    diff_s = embed_s.unsqueeze(2) - embed_s.unsqueeze(1)
    diff_t = embed_t.unsqueeze(2) - embed_t.unsqueeze(1)
    norm_s = F.normalize(diff_s, dim=-1)
    norm_t = F.normalize(diff_t, dim=-1)
    angle_s = torch.matmul(norm_s, norm_s.transpose(-1, -2))
    angle_t = torch.matmul(norm_t, norm_t.transpose(-1, -2))
    loss_ang = F.smooth_l1_loss(angle_s, angle_t, reduction="mean")
    return lambda_rkd * (dist_factor * loss_dist + angle_factor * loss_ang)
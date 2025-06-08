# ───────────────────────── finetune_utils/distill_losses.py ─────────────────────────
"""Knowledge-distillation losses (revised)
   ✔ 支援 layer-count 歸一化
   ✔ KL 溫度公式修正
"""

from __future__ import annotations
from typing import List, Union
import torch
import torch.nn.functional as F
from torch import Tensor

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
    if isinstance(x, list):
        x = torch.cat([t.unsqueeze(0) if t.ndim == x[0].ndim else t for t in x], 0)
    return x.float()


def _gap(x: Union[Tensor, List[Tensor]]) -> Tensor:
    x = _stack(x)
    if x.ndim > 2:
        x = F.adaptive_avg_pool2d(x, 1).flatten(1)
    return x


def _match_channels(src: Tensor, tgt: Tensor):
    Cs, Ct = src.size(1), tgt.size(1)
    if Cs == Ct:
        return src
    if Cs % Ct == 0:
        k = Cs // Ct
        return src.view(src.size(0), Ct, k, *src.shape[2:]).mean(2)
    if Ct % Cs == 0:
        k = Ct // Cs
        return src.unsqueeze(2).repeat(1, 1, k, *([1] * (src.ndim - 2))).view(src.size(0), Ct, *src.shape[2:])
    avg = src.mean(1, keepdim=True)
    return avg.repeat(1, Ct, 1, 1)

# ─── 1. Encoder feature - MSE + KL ───
def encoder_matching_loss(
    feats_s: List[Union[Tensor, List[Tensor]]],
    feats_t: List[Tensor],
    lambda_mse=1.0,
    lambda_kl=1.0,
    temperature=1.0,
    n_layers: int = 1,
) -> Tensor:
    mse_tot, kl_tot = 0.0, 0.0
    for fs_raw, ft in zip(feats_s, feats_t):
        fs = _stack(fs_raw)
        
        # Permute if teacher feature seems to be channel-last
        if ft.ndim == 4 and fs.ndim == 4 and ft.shape[1:3] == fs.shape[2:4]:
            ft = ft.permute(0, 3, 1, 2)  # B, H, W, C -> B, C, H, W

        ft = _interpolate_if_needed(ft, fs)
        b = min(fs.size(0), ft.size(0))
        fs, ft = fs[:b], ft[:b]
        fs = _match_channels(fs, ft)

        mse_tot += F.mse_loss(fs, ft, reduction="mean")
        ps = F.log_softmax(fs.flatten(1) / temperature, dim=-1)
        pt = F.softmax(ft.flatten(1) / temperature, dim=-1)
        kl_tot += F.kl_div(ps, pt, reduction="batchmean") * temperature ** 2
    return (lambda_mse * mse_tot + lambda_kl * kl_tot) / max(1, n_layers)

# ─── 2. Decoder pre-logits ───
def decoder_matching_loss(
    feat_s_raw: Union[Tensor, List[Tensor]],
    feat_t: Tensor,
    lambda_mse=1.0,
    lambda_cos=1.0,
    lambda_kl=1.0,
    temperature=1.0,
) -> Tensor:
    fs = _stack(feat_s_raw)

    # Permute if teacher feature seems to be channel-last
    if feat_t.ndim == 4 and fs.ndim == 4 and feat_t.shape[1:3] == fs.shape[2:4]:
        feat_t = feat_t.permute(0, 3, 1, 2)  # B, H, W, C -> B, C, H, W

    ft = _interpolate_if_needed(feat_t, fs)
    b = min(fs.size(0), ft.size(0))
    fs, ft = fs[:b], ft[:b]
    fs = _match_channels(fs, ft)

    mse = F.mse_loss(fs, ft, reduction="mean")
    cos = 1 - F.cosine_similarity(fs.flatten(1), ft.flatten(1)).mean()
    ps = F.log_softmax(fs.flatten(1) / temperature, dim=-1)
    pt = F.softmax(ft.flatten(1) / temperature, dim=-1)
    kl = F.kl_div(ps, pt, reduction="batchmean") * temperature ** 2

    return lambda_mse * mse + lambda_cos * cos + lambda_kl * kl

# ─── 3. ViT attention map ───
def attention_matching_loss(
    attn_s: List[Union[Tensor, List[Tensor]]],
    attn_t: List[Tensor],
    lambda_attn=1.0,
    temperature=1.0,
    n_layers: int = 1,
) -> Tensor:
    loss = 0.0
    for as_raw, at_raw in zip(attn_s, attn_t):
        as_ = _stack(as_raw)
        at_ = _stack(at_raw)
        
        # Reshape to be 4D-compatible for interpolation and channel matching
        if as_.ndim == 3: as_ = as_.permute(0, 2, 1).unsqueeze(-1)
        if at_.ndim == 3: at_ = at_.permute(0, 2, 1).unsqueeze(-1)

        at_ = _interpolate_if_needed(at_, as_)
        
        b = min(as_.size(0), at_.size(0))
        as_, at_ = as_[:b], at_[:b]
        
        at_ = _match_channels(at_, as_)
        
        ps = F.log_softmax(as_.flatten(1) / temperature, dim=-1)
        pt = F.softmax(at_.flatten(1) / temperature, dim=-1)

        if ps.shape != pt.shape:
            continue

        loss += F.kl_div(ps, pt, reduction="batchmean") * (temperature ** 2)
    return lambda_attn * loss / max(1, n_layers)

# ─── 4. Relational KD ───
def _pdist(e):
    return torch.cdist(e, e, p=2)

def rkd_loss(
    embed_s_raw: Union[Tensor, List[Tensor]],
    embed_t_raw: Union[Tensor, List[Tensor]],
    lambda_rkd=1.0,
    dist_factor=1.0,
    angle_factor=2.0,
) -> Tensor:
    es_raw, et_raw = _stack(embed_s_raw), _stack(embed_t_raw)
    
    es_p = _permute_if_channel_last(es_raw)
    et_p = _permute_if_channel_last(et_raw)

    es, et = _gap(es_p), _gap(et_p)
    b = min(es.size(0), et.size(0))
    es, et = es[:b], et[:b]

    if es.shape[1] != et.shape[1]:
        et_4d = et.unsqueeze(-1).unsqueeze(-1)
        es_4d = es.unsqueeze(-1).unsqueeze(-1)
        et_matched_4d = _match_channels(et_4d, es_4d)
        et = et_matched_4d.squeeze(-1).squeeze(-1)

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

# train.py

import numpy as np
import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms as T
from torchvision.ops import sigmoid_focal_loss

from mobile_sam import sam_model_registry

import argparse
import gc
import json
import logging
import os
import traceback
from finetune_utils.datasets import ComponentDataset
from finetune_utils.distill_losses import (
    attention_matching_loss,
    decoder_matching_loss,
    encoder_matching_loss,
    rkd_loss,
)
from finetune_utils.feature_hooks import pop_features, register_hooks
from finetune_utils.visualization import overlay_mask_on_image
from pathlib import Path
from tqdm import tqdm

# ─────────────────── logging ───────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("train")


def log_gpu_memory(step_name=""):
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        cached = torch.cuda.memory_reserved() / 1024**3
        log.info(
            f"{step_name} GPU Memory - Allocated: {allocated:.2f}GB, Cached: {cached:.2f}GB"
        )


def clear_gpu_cache():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()


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
            base_lr * (self.min_ratio + (1 - self.min_ratio) * cos)
            for base_lr in self.base_lrs
        ]


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
    feats = []
    for stem in stems:
        this_img = []
        for k in keys:
            fname = (
                f"{stem}_{k.replace('.', '_').replace('[', '_').replace(']', '')}.npy"
            )
            this_img.append(feature_cache.get(base / teacher / split / fname))
        feats.append(torch.stack(this_img))
    return [
        torch.stack([feats[b][i] for b in range(len(stems))]) for i in range(len(keys))
    ]


def _parse_hw(x):
    return (int(x[0]), int(x[1])) if isinstance(x, torch.Tensor) else tuple(map(int, x))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = json.load(open(args.config))
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    if "lr" not in cfg["train"]:
        cfg["train"]["lr"] = 5e-5

    MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    tf_img = T.Compose([T.ToTensor(), T.Normalize(MEAN, STD)])
    tf_msk = T.Compose([T.ToTensor()])

    ds_cfg = cfg["dataset"]
    train_ds = ComponentDataset(
        ds_cfg["train_dataset"],
        (tf_img, tf_msk),
        max_bbox_shift=ds_cfg.get("max_bbox_shift", 20),
        prompt_mode=ds_cfg.get("prompt_mode", "point"),  # 或 "box"
        min_points=ds_cfg.get("min_points", 1),
        max_points=ds_cfg.get("max_points", 3),
        image_size=cfg["model"].get("image_size", 1024),
    )
    val_ds = ComponentDataset(
        ds_cfg["val_dataset"],
        (tf_img, tf_msk),
        max_bbox_shift=ds_cfg.get("max_bbox_shift", 20),
        prompt_mode=ds_cfg.get("prompt_mode", "point"),  # 或 "box"
        min_points=ds_cfg.get("min_points", 1),
        max_points=ds_cfg.get("max_points", 3),
        image_size=cfg["model"].get("image_size", 1024),
    )

    def sam_collate(batch):
        out = {}
        for k in batch[0].keys():
            vals = [d[k] for d in batch]
            if isinstance(vals[0], torch.Tensor) and all(v is not None for v in vals):
                out[k] = torch.stack(vals, 0)
            else:
                out[k] = vals
        return out

    tr_loader = DataLoader(
        train_ds,
        batch_size=cfg["train"]["batch_size"],
        shuffle=True,
        num_workers=ds_cfg.get("num_workers", 4),
        pin_memory=True,
        collate_fn=sam_collate,
    )
    va_loader = DataLoader(
        val_ds,
        batch_size=cfg["train"]["batch_size"],
        shuffle=False,
        num_workers=ds_cfg.get("num_workers", 4),
        pin_memory=True,
        collate_fn=sam_collate,
    )

    m_cfg = cfg["model"]
    student = sam_model_registry[m_cfg.get("type", "vit_t")](
        checkpoint=m_cfg.get("checkpoint_path")
    ).to(dev)

    if cfg["freeze"].get("freeze_image_encoder", True):
        student.image_encoder.requires_grad_(False)
    if cfg["freeze"].get("freeze_prompt_encoder", False):
        student.prompt_encoder.requires_grad_(False)
    if cfg["freeze"].get("freeze_mask_decoder", False):
        student.mask_decoder.requires_grad_(False)

    dist_cfg = cfg.get("distillation", {})
    use_distillation = dist_cfg.get("enable", False)
    hook_handles = []
    pot = {"enc": [], "dec": [], "attn": [], "rkd": []}
    if use_distillation:
        stype = m_cfg.get("type", "vit_t")
        if stype == "vit_t":
            pot["enc"] = ["image_encoder.neck"]
        else:
            pot["enc"] = [f"image_encoder.blocks.{i}" for i in (9, 10, 11, 12)]
            pot["dec"] = ["mask_decoder.pre_logits"]
            pot["attn"] = [f"image_encoder.blocks.{i}.attn" for i in range(12)]
        pot["rkd"] = ["image_encoder.patch_embed"]

        hook_layers = []
        for name, key in (
            ("encoder_matching", "enc"),
            ("decoder_matching", "dec"),
            ("attention_matching", "attn"),
            ("relational_KD", "rkd"),
        ):
            if dist_cfg.get(name, {}).get("enable"):
                hook_layers += pot[key]
        hook_layers = sorted(set(hook_layers))
        if hook_layers:
            hook_handles = register_hooks(student, hook_layers)
            log.info(f"Registered hooks for distillation: {hook_layers}")
    else:
        log.info("Distillation disabled - no hooks registered")

    log_gpu_memory("After model loading")

    tr_cfg = cfg["train"]
    enc_params, other_params = [], []
    for n, p in student.named_parameters():
        if not p.requires_grad:
            continue
        if n.startswith("image_encoder"):
            enc_params.append(p)
        else:
            other_params.append(p)

    opt = torch.optim.AdamW(
        [
            {"params": other_params, "lr": tr_cfg["lr"]},
            {"params": enc_params, "lr": tr_cfg["lr"] * 0.1},
        ],
        weight_decay=1e-4,
    )
    total_steps = tr_cfg["epochs"] * len(tr_loader)
    scheduler = WarmupCosineLR(
        opt,
        warmup=tr_cfg.get("warmup_step", 250),
        total=total_steps,
        min_ratio=tr_cfg.get("min_lr_ratio", 0.0),
    )

    scaler = GradScaler(enabled=not tr_cfg.get("bf16", False))
    writer = SummaryWriter(Path(m_cfg.get("save_path", "logs")) / "tb")

    lambda_coef = 1.0
    dyn_wait = 0
    best_score, stop_counter = -1, 0
    patience = tr_cfg.get("early_stop_patience", 20)
    unfreeze_epoch = cfg["freeze"].get("unfreeze_epoch", 10)

    teacher_cfgs = cfg.get("teachers", [])
    if teacher_cfgs:
        for t in teacher_cfgs:
            t.setdefault("weight", 1.0 / len(teacher_cfgs))

    try:
        global_step = 0
        for ep in range(tr_cfg["epochs"]):
            if ep == unfreeze_epoch:
                log.info(f"Unfreeze image_encoder at epoch {ep}")
                student.image_encoder.requires_grad_(True)
                opt.param_groups[1]["lr"] = tr_cfg["lr"] * 0.2

            if ep > 0:
                feature_cache.clear()
                clear_gpu_cache()
                log_gpu_memory(f"Epoch {ep} start (after cache clear)")

            student.train()
            tot_task, tot_dist, tot_iou = 0.0, 0.0, 0.0
            pbar = tqdm(tr_loader, desc=f"Train {ep}")
            opt.zero_grad()

            for step, batch in enumerate(pbar):
                # ─────────── 先初始化所有可能 later 會被用到的 loss 變數 ───────────
                bce = torch.tensor(0.0, device=dev)
                focal = torch.tensor(0.0, device=dev)
                dice_loss = torch.tensor(0.0, device=dev)
                dist_loss = torch.tensor(0.0, device=dev)
                task_loss = torch.tensor(0.0, device=dev)
                loss = torch.tensor(0.0, device=dev)
                #
                imgs = batch["image"].to(dev)  # [B,3,1024,1024]
                masks = batch["mask"].to(dev)  # [B,1,1024,1024]
                ids = batch["id"]
                osz = batch["original_size"]  # [B, 2] raw sizes

                batched_input = []
                for i in range(len(imgs)):
                    entry = {
                        "image": imgs[i],
                        "original_size": (int(osz[i][0]), int(osz[i][1])),
                    }
                    if batch["box_prompt"][i] is not None:
                        entry["boxes"] = batch["box_prompt"][i].to(dev).unsqueeze(0)
                    if batch["point_coords"][i] is not None:
                        entry["point_coords"] = (
                            batch["point_coords"][i].to(dev).unsqueeze(0)
                        )
                        entry["point_labels"] = (
                            batch["point_labels"][i].to(dev).unsqueeze(0)
                        )

                    if step % 200 == 0 and i == 0:
                        boxes = entry.get("boxes", None)
                        pts = entry.get("point_coords", None)
                        # 如果 point_coords 是 None，就不要做切片
                        pt0 = pts[:1] if pts is not None else None

                        log.info(
                            f"[PROMPT] ep{ep}_step{step} id={ids[i]}, "
                            f"orig={entry['original_size']}, "
                            f"box={boxes}, "
                            f"pt0={pt0}"
                        )

                    batched_input.append(entry)
                    
                with autocast(
                    dtype=torch.bfloat16 if tr_cfg.get("bf16", False) else torch.float16
                ):
                    out = student(batched_input=batched_input, multimask_output=True)

                    mask_list = []
                    iou_list = []
                    for o in out:
                        mask_up = F.interpolate(
                            o["masks"].to(torch.float32),
                            size=masks.shape[-2:],
                            mode="bilinear",
                            align_corners=False,
                        ).squeeze(0)
                        mask_list.append(mask_up)
                        iou_list.append(
                            o["iou_predictions"].squeeze(0).to(torch.float32)
                        )

                    pred_masks = torch.stack(mask_list, dim=0)
                    pred_ious = torch.stack(iou_list, dim=0)

                    best_indices = pred_ious.argmax(dim=1)
                    sel_masks = pred_masks[torch.arange(len(pred_masks)), best_indices]
                    sel_masks = sel_masks.unsqueeze(1)

                    bce = F.binary_cross_entropy_with_logits(sel_masks, masks)
                    focal = sigmoid_focal_loss(sel_masks, masks, reduction="mean")

                    prob = torch.sigmoid(sel_masks)
                    num = (prob * masks).sum((-2, -1)) * 2
                    den = prob.sum((-2, -1)) + masks.sum((-2, -1))
                    dice_loss = 1 - (num / (den + 1e-6)).mean()
                    task_loss = bce + 0.5 * focal + dice_loss

                    with torch.no_grad():
                        gt_bin = masks > 0.5
                        ious = []
                        for b in range(pred_masks.shape[0]):
                            preds_bin = (torch.sigmoid(pred_masks[b]) > 0.5).float()
                            inter = (preds_bin * gt_bin[b]).sum((-2, -1))
                            union = (
                                preds_bin.sum((-2, -1))
                                + gt_bin[b].sum((-2, -1))
                                - inter
                            )
                            ious.append(inter / (union + 1e-6))
                        gt_ious = torch.stack(ious, dim=0)

                    iou_loss = F.mse_loss(pred_ious, gt_ious)

                    dist_loss = torch.tensor(0.0, device=dev)
                    if use_distillation and hook_handles:
                        feat_student = pop_features() or {}
                        base_dir = Path(dist_cfg["precomputed_root"])

                        for t_cfg in teacher_cfgs:
                            weight = t_cfg["weight"]
                            tname = t_cfg["name"]

                            if dist_cfg.get("encoder_matching", {}).get("enable"):
                                enc_keys = pot["enc"]
                                if enc_keys:
                                    try:
                                        feat_teacher = load_cached_npy_features(
                                            base_dir, tname, "train", ids, enc_keys
                                        )
                                        enc_loss = encoder_matching_loss(
                                            [feat_student[k] for k in enc_keys],
                                            feat_teacher,
                                            **dist_cfg["encoder_matching"],
                                            n_layers=len(enc_keys),
                                        )
                                        dist_loss = dist_loss + weight * enc_loss
                                    except Exception as e:
                                        log.debug(f"Encoder matching error: {e}")

                            if (
                                dist_cfg.get("decoder_matching", {}).get("enable")
                                and pot["dec"]
                            ):
                                dk = pot["dec"][0]
                                if dk in feat_student:
                                    try:
                                        feat_teacher = load_cached_npy_features(
                                            base_dir, tname, "train", ids, [dk]
                                        )[0]
                                        dec_loss = decoder_matching_loss(
                                            feat_student[dk],
                                            feat_teacher,
                                            **dist_cfg["decoder_matching"],
                                        )
                                        dist_loss = dist_loss + weight * dec_loss
                                    except Exception as e:
                                        log.debug(f"Decoder matching error: {e}")

                            if (
                                dist_cfg.get("attention_matching", {}).get("enable")
                                and pot["attn"]
                            ):
                                try:
                                    attn_teacher = load_cached_npy_features(
                                        base_dir, tname, "train", ids, pot["attn"]
                                    )
                                    attn_loss = attention_matching_loss(
                                        [feat_student[k] for k in pot["attn"]],
                                        attn_teacher,
                                        **dist_cfg["attention_matching"],
                                        n_layers=len(pot["attn"]),
                                    )
                                    dist_loss = dist_loss + weight * attn_loss
                                except Exception as e:
                                    log.debug(f"Attention matching error: {e}")

                            if dist_cfg.get("relational_KD", {}).get("enable"):
                                rk = pot["rkd"][0]
                                if rk in feat_student:
                                    try:
                                        feat_teacher = load_cached_npy_features(
                                            base_dir, tname, "train", ids, [rk]
                                        )[0]
                                        rkd_loss_val = rkd_loss(
                                            feat_student[rk],
                                            feat_teacher,
                                            **dist_cfg["relational_KD"],
                                        )
                                        dist_loss = dist_loss + weight * rkd_loss_val
                                    except Exception as e:
                                        log.debug(f"RKD error: {e}")

                    loss = (
                        task_loss + iou_loss + lambda_coef * dist_loss
                    ) / tr_cfg.get("gradient_accumulation", 1)

                scaler.scale(loss).backward()
                # del low_res_logits, tmp, logit_up, prob, focal, dice_loss
                if use_distillation and "feat_student" in locals():
                    del feat_student

                if (step + 1) % tr_cfg.get("gradient_accumulation", 1) == 0 or (
                    step + 1
                ) == len(tr_loader):
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
                    scaler.step(opt)
                    scaler.update()
                    opt.zero_grad()
                    global_step += 1
                    scheduler.step()
                    torch.cuda.empty_cache()

                tot_task += task_loss.item()
                tot_dist += dist_loss.item()
                tot_iou += iou_loss.item()

                pbar.set_postfix(
                    bce=f"{bce.item():.3f}",
                    focal=f"{focal.item():.3f}",
                    dice=f"{dice_loss.item():.3f}",
                    iou=f"{iou_loss.item():.3f}",
                    dist=f"{dist_loss.item():.3f}",
                    total=f"{loss.item():.3f}",
                    lr=f"{scheduler.get_last_lr()[0]:.2e}",
                )

                if step % 100 == 0:
                    log_gpu_memory(f"Epoch {ep}, Step {step}")

            writer.add_scalar("train/task_loss", tot_task / len(tr_loader), ep)
            writer.add_scalar("train/dist_loss", tot_dist / len(tr_loader), ep)
            writer.add_scalar("train/iou_loss", tot_iou / len(tr_loader), ep)
            writer.add_scalar("train/lambda_coef", lambda_coef, ep)
            for i, g in enumerate(opt.param_groups):
                writer.add_scalar(f"lr/group{i}", g["lr"], ep)
            log_gpu_memory(f"Epoch {ep} training completed")

            if (ep + 1) % tr_cfg.get("val_freq", 1) != 0:
                continue

            student.eval()
            dices, ious = [], []
            with torch.no_grad():
                for bi, vb in enumerate(va_loader):
                    try:
                        imgs = vb["image"].to(dev)
                        masks = vb["mask"].to(dev)
                        original_sizes = vb["original_size"]

                        vinp = []
                        for i in range(len(imgs)):
                            entry = {
                                "image": imgs[i],
                                "original_size": (
                                    int(original_sizes[i][0]),
                                    int(original_sizes[i][1]),
                                ),
                            }
                            if vb["box_prompt"][i] is not None:
                                entry["boxes"] = (
                                    vb["box_prompt"][i].to(dev).unsqueeze(0)
                                )  # (1,4)
                            if vb["point_coords"][i] is not None:
                                entry["point_coords"] = (
                                    vb["point_coords"][i].to(dev).unsqueeze(0)
                                )  # (1,K,2)
                                entry["point_labels"] = (
                                    vb["point_labels"][i].to(dev).unsqueeze(0)
                                )  # (1,K)
                            vinp.append(entry)
                        vo = student(batched_input=vinp, multimask_output=True)

                        mask_list = []
                        iou_list = []
                        for o in vo:
                            mask_up = F.interpolate(
                                o["masks"].to(torch.float32),
                                size=masks.shape[-2:],
                                mode="bilinear",
                                align_corners=False,

                            ).squeeze(0)
                            mask_list.append(mask_up)
                            iou_list.append(
                                o["iou_predictions"].squeeze(0).to(torch.float32)
                            )


                        pred_masks = torch.stack(mask_list, dim=0)
                        pred_ious = torch.stack(iou_list, dim=0)

                        best_indices = pred_ious.argmax(dim=1)
                        probs = torch.sigmoid(

                            pred_masks[
                                torch.arange(len(pred_masks)), best_indices
                            ].unsqueeze(1)

                        )

                        for i in range(len(imgs)):
                            prob_i = probs[i]  # [1,1024,1024]
                            gt_i = masks[i]  # [1,1024,1024]

                            # Soft Dice for loss logging
                            num_soft = (prob_i * gt_i).sum((-2, -1)) * 2
                            den_soft = prob_i.sum((-2, -1)) + gt_i.sum((-2, -1))
                            soft_dice = (num_soft / (den_soft + 1e-6)).item()

                            # Binary metrics
                            pred_bin = (prob_i >= 0.5).float()
                            num_bin = (pred_bin * gt_i).sum((-2, -1)) * 2
                            den_bin = pred_bin.sum((-2, -1)) + gt_i.sum((-2, -1))
                            bin_dice = (num_bin / (den_bin + 1e-6)).item()

                            union = pred_bin + gt_i - pred_bin * gt_i
                            bin_iou = (
                                (pred_bin * gt_i).sum((-2, -1))
                                / (union.sum((-2, -1)) + 1e-6)
                            ).item()

                            dices.append(bin_dice)
                            ious.append(bin_iou)

                            # 印 debug: 第一張
                            if bi == 0 and i == 0:
                                log.info(
                                    f"[VAL] id={vb['id'][i]}, "
                                    f"soft_dice={soft_dice:.3f}, "
                                    f"bin_dice={bin_dice:.3f}, bin_iou={bin_iou:.3f}, "

                                    f"gt_sum={gt_i.sum().item():.0f}, "
                                    f"pred_sum={pred_bin.sum().item():.0f}"

                                )

                            # Visualization
                            if (
                                bi == 0
                                and cfg["visual"].get("status", False)
                                and (
                                    ep % cfg["visual"].get("save_every_n_epochs", 10)
                                    == 0
                                    or ((np.mean(dices) + np.mean(ious)) / 2)
                                    > best_score
                                )
                            ):
                                cur_path = (
                                    Path(cfg["visual"]["save_path"]) / f"epoch_{ep}"
                                )
                                cur_path.mkdir(parents=True, exist_ok=True)
                                img_denorm = (
                                    imgs[i]
                                    * torch.tensor(STD, device=dev)[:, None, None]
                                    + torch.tensor(MEAN, device=dev)[:, None, None]
                                )
                                img_denorm = img_denorm.clamp(0, 1).cpu()

                                pred_mask = probs[i].squeeze(0).cpu()

                                box = vb["box_prompt"][i]
                                pts = vb["point_coords"][i]
                                lbl = vb["point_labels"][i]
                                if box is not None:
                                    box = box.to(torch.float32)
                                if pts is not None:
                                    pts = pts.to(torch.float32)
                                if lbl is not None:
                                    lbl = lbl.to(torch.long)
                                # orig_h, orig_w = int(original_sizes[i][0]), int(original_sizes[i][1])

                                overlay_mask_on_image(
                                    image_tensor=img_denorm,
                                    mask_tensor=pred_mask,
                                    bbox_tensor=box.cpu() if box is not None else None,
                                    point_coords=pts.cpu() if pts is not None else None,
                                    point_labels=lbl.cpu() if lbl is not None else None,
                                    # original_size=(orig_h, orig_w),
                                    original_size=None,
                                    threshold=cfg["visual"].get("IOU_threshold", 0.5),
                                    save_dir=str(cur_path),

                                    filename_info=(
                                        f"ep{ep}_id{vb['id'][i]}_b{bi}_s{i}"
                                    ),

                                )
                    except Exception as e:
                        log.error(f"Error in validation step {bi}: {e}")
                        clear_gpu_cache()
                        continue

                v_dice, v_iou = np.mean(dices), np.mean(ious)
                v_score = 0.5 * (v_dice + v_iou)
                writer.add_scalar("val/dice", v_dice, ep)
                writer.add_scalar("val/iou", v_iou, ep)
                writer.add_scalar("val/score", v_score, ep)
                log.info(

                    (
                        f"Epoch {ep}  Bin-Dice={v_dice:.4f} "
                        f"Bin-IoU={v_iou:.4f} Score={v_score:.4f}"
                    )

                )
                log_gpu_memory(f"Epoch {ep} validation completed")

                if use_distillation and dist_cfg.get("dynamic_lambda", {}).get(
                    "enable_plateau_scheduler"
                ):
                    if v_score > best_score + 1e-4:
                        dyn_wait = 0
                        lambda_coef = min(

                            lambda_coef / dist_cfg["dynamic_lambda"]["factor"],
                            1.0,

                        )
                    else:
                        dyn_wait += 1
                        if dyn_wait >= dist_cfg["dynamic_lambda"]["patience"]:
                            lambda_coef = max(

                                lambda_coef * dist_cfg["dynamic_lambda"]["factor"],
                                1e-3,

                            )
                            dyn_wait = 0
                            log.info(f"λ ↓ {lambda_coef:.3f}")

                if v_score > best_score:
                    best_score = v_score
                    stop_counter = 0
                    out_dir = Path(m_cfg.get("save_path", "./"))
                    out_dir.mkdir(parents=True, exist_ok=True)
                    torch.save(student.state_dict(), out_dir / "best_student.pth")
                    log.info(f"✨  Saved best model (score={best_score:.4f})")
                else:
                    stop_counter += 1
                    if stop_counter >= patience:
                        log.info("Early stop triggered.")
                        break
    except Exception as e:
        log.error(f"Training error: {e}")
        log.error(traceback.format_exc())
        raise
    finally:
        for h in hook_handles:
            h.remove()
        feature_cache.clear()
        writer.close()
        clear_gpu_cache()
        log_gpu_memory("Final cleanup completed")


if __name__ == "__main__":
    main()

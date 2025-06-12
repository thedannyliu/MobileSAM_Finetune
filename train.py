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
from mobile_sam.utils.amg import batch_iterator

import argparse
import gc
import json
import logging
import os
import traceback
import yaml
from finetune_utils.datasets import ComponentDataset, SegmentEverythingDataset
from finetune_utils.distill_losses import (
    attention_matching_loss,
    decoder_matching_loss,
    encoder_matching_loss,
    rkd_loss,
)
from finetune_utils.feature_hooks import pop_features, register_hooks
from finetune_utils.visualization import overlay_mask_on_image, overlay_masks_on_image
from pathlib import Path
from scipy.optimize import linear_sum_assignment
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
        log.info(f"{step_name} GPU Memory - Allocated: {allocated:.2f}GB, Cached: {cached:.2f}GB")


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
            base_lr * (self.min_ratio + (1 - self.min_ratio) * cos) for base_lr in self.base_lrs
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
    stacked = {k: [] for k in keys}
    for stem in stems:
        for k in keys:
            fname = f"{stem}_{k.replace('.', '_').replace('[', '_').replace(']', '')}.npy"
            stacked[k].append(feature_cache.get(base / teacher / split / fname))
    return [torch.stack(stacked[k]) for k in keys]


def _parse_hw(x):
    return (int(x[0]), int(x[1])) if isinstance(x, torch.Tensor) else tuple(map(int, x))


def predict_from_grid(model, image, points, orig_size, batch_size=64):
    """Run the SAM model on a grid of points and return masks and IoU preds."""
    device = image.device
    inp = model.preprocess(image.unsqueeze(0))
    embedding = model.image_encoder(inp)
    dense_pe = model.prompt_encoder.get_dense_pe()

    all_masks = []
    all_ious = []
    for (pts,) in batch_iterator(batch_size, points):
        coords = torch.as_tensor(pts, dtype=torch.float, device=device)
        labels = torch.ones(coords.shape[0], dtype=torch.int, device=device)
        sparse, dense = model.prompt_encoder(
            points=(coords.unsqueeze(0), labels.unsqueeze(0)),
            boxes=None,
            masks=None,
        )
        low_res, iou_pred = model.mask_decoder(
            image_embeddings=embedding,
            image_pe=dense_pe,
            sparse_prompt_embeddings=sparse,
            dense_prompt_embeddings=dense,
            multimask_output=True,
        )
        masks = model.postprocess_masks(low_res, image.shape[-2:], orig_size).squeeze(0)
        all_masks.append(masks)
        all_ious.append(iou_pred.squeeze(0))
    return torch.cat(all_masks, dim=0), torch.cat(all_ious, dim=0)


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
    dataset_mode = ds_cfg.get("mode", "single")
    if dataset_mode == "everything":
        train_ds = SegmentEverythingDataset(
            ds_cfg["train_dataset"],
            (tf_img, tf_msk),
            grid_points=ds_cfg.get("grid_points", 32),
            image_size=cfg["model"].get("image_size", 1024),
        )
        val_ds = SegmentEverythingDataset(
            ds_cfg["val_dataset"],
            (tf_img, tf_msk),
            grid_points=ds_cfg.get("grid_points", 32),
            image_size=cfg["model"].get("image_size", 1024),
        )
    else:
        train_ds = ComponentDataset(
            ds_cfg["train_dataset"],
            (tf_img, tf_msk),
            max_bbox_shift=ds_cfg.get("max_bbox_shift", 20),
            prompt_mode=ds_cfg.get("prompt_mode", "point"),
            min_points=ds_cfg.get("min_points", 1),
            max_points=ds_cfg.get("max_points", 3),
            image_size=cfg["model"].get("image_size", 1024),
        )
        val_ds = ComponentDataset(
            ds_cfg["val_dataset"],
            (tf_img, tf_msk),
            max_bbox_shift=ds_cfg.get("max_bbox_shift", 20),
            prompt_mode=ds_cfg.get("prompt_mode", "point"),
            min_points=ds_cfg.get("min_points", 1),
            max_points=ds_cfg.get("max_points", 3),
            image_size=cfg["model"].get("image_size", 1024),
        )

    def sam_collate(batch):
        out = {}
        for k in batch[0].keys():
            vals = [d[k] for d in batch]
            if k in ("gt_masks", "point_coords", "point_labels"):
                out[k] = vals  # keep list due to variable lengths
                continue
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
    lambda_coef = dist_cfg.get("lambda_coef", 1.0)
    hook_handles = []

    def _build_pot(model_type: str):
        p = {"enc": [], "dec": [], "attn": [], "rkd": ["image_encoder.patch_embed"]}
        if model_type == "vit_t":
            p["enc"] = ["image_encoder.neck"]
            p["dec"] = ["mask_decoder.output_upscaling"]
            p["attn"] = [
                "image_encoder.layers.1.blocks.0.attn",
                "image_encoder.layers.1.blocks.1.attn",
                "image_encoder.layers.2.blocks.0.attn",
                "image_encoder.layers.2.blocks.1.attn",
                "image_encoder.layers.2.blocks.2.attn",
                "image_encoder.layers.2.blocks.3.attn",
                "image_encoder.layers.2.blocks.4.attn",
                "image_encoder.layers.2.blocks.5.attn",
                "image_encoder.layers.3.blocks.0.attn",
                "image_encoder.layers.3.blocks.1.attn",
            ]
        else:
            p["enc"] = [f"image_encoder.blocks.{i}" for i in (9, 10, 11, 12)]
            # Some official SAM models do not expose `pre_logits`. Fallback to
            # `output_upscaling` which exists in all variants.
            p["dec"] = ["mask_decoder.output_upscaling"]
            p["attn"] = [f"image_encoder.blocks.{i}.attn" for i in range(12)]
        return p

    pot = _build_pot(m_cfg.get("type", "vit_t"))

    teacher_models = []
    teacher_pots = {}
    if use_distillation:
        enabled_losses = [
            n
            for n in (
                "encoder_matching",
                "decoder_matching",
                "attention_matching",
                "relational_KD",
            )
            if dist_cfg.get(n, {}).get("enable")
        ]
        log.info(
            f"Distillation enabled. Methods: {', '.join(enabled_losses) if enabled_losses else 'none'}"
        )
        use_precomputed = dist_cfg.get("use_precomputed_features", False)
        if use_precomputed:
            log.info("Using precomputed teacher features")
        stype = m_cfg.get("type", "vit_t")

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
            log.info(f"Registered hooks for student: {hook_layers}")

        for t in cfg.get("teachers", []):
            try:
                with open(t["cfg"], "r") as f:
                    t_yaml = yaml.safe_load(f)
                mtype = t_yaml.get("model", {}).get("type", "vit_t")
            except Exception:
                mtype = "vit_t"
            teacher_pots[t["name"]] = _build_pot(mtype)

            if not use_precomputed:
                ckpt = t.get("checkpoint")
                if not ckpt or not os.path.exists(ckpt):
                    log.warning(f"Teacher checkpoint {ckpt} not found; skip {t['name']}")
                    continue
                builder = sam_model_registry.get(mtype)
                if builder is None:
                    log.warning(f"Unknown teacher model type {mtype}; skip {t['name']}")
                    continue
                model_t = builder(checkpoint=ckpt).to(dev).eval()
                hooks_t = []
                hook_layers_t = []
                for name, key in (
                    ("encoder_matching", "enc"),
                    ("decoder_matching", "dec"),
                    ("attention_matching", "attn"),
                    ("relational_KD", "rkd"),
                ):
                    if dist_cfg.get(name, {}).get("enable"):
                        hook_layers_t += teacher_pots[t["name"]][key]
                hook_layers_t = sorted(set(hook_layers_t))
                if hook_layers_t:
                    hooks_t = register_hooks(model_t, hook_layers_t)
                teacher_models.append(
                    {
                        "name": t["name"],
                        "model": model_t,
                        "hooks": hooks_t,
                        "weight": t["weight"],
                        "type": mtype,
                    }
                )
                log.info(f"Loaded teacher {t['name']} ({mtype}) with weight {t['weight']}")
            else:
                log.info(f"Using precomputed features for teacher {t['name']}")
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

    loss_weights = tr_cfg.get("loss_weights", {})
    w_bce = loss_weights.get("bce", 1.0)
    w_focal = loss_weights.get("focal", 0.5)
    w_dice = loss_weights.get("dice", 1.0)
    w_iou = loss_weights.get("iou", 1.0)
    w_cls = loss_weights.get("cls", 1.0)

    scaler = GradScaler(enabled=not tr_cfg.get("bf16", False))
    writer = SummaryWriter(Path(m_cfg.get("save_path", "logs")) / "tb")

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
                log.info(f"Unfreeze all modules at epoch {ep}")
                student.image_encoder.requires_grad_(True)
                student.prompt_encoder.requires_grad_(True)
                student.mask_decoder.requires_grad_(True)
                if len(opt.param_groups) > 1:
                    opt.param_groups[1]["lr"] = tr_cfg["lr"] * 0.2

            if ep > 0:
                feature_cache.clear()
                clear_gpu_cache()
                log_gpu_memory(f"Epoch {ep} start (after cache clear)")

            student.train()
            tot_task, tot_dist, tot_iou, tot_cls = 0.0, 0.0, 0.0, 0.0
            pbar = tqdm(tr_loader, desc=f"Train {ep}")
            opt.zero_grad()

            for step, batch in enumerate(pbar):
                # ─────────── 先初始化所有可能 later 會被用到的 loss 變數 ───────────
                bce = torch.tensor(0.0, device=dev)
                focal = torch.tensor(0.0, device=dev)
                dice_loss = torch.tensor(0.0, device=dev)
                dist_loss = torch.tensor(0.0, device=dev)
                enc_loss_val = torch.tensor(0.0, device=dev)
                dec_loss_val = torch.tensor(0.0, device=dev)
                attn_loss_val = torch.tensor(0.0, device=dev)
                rkd_loss_val = torch.tensor(0.0, device=dev)
                cls_loss_val = torch.tensor(0.0, device=dev)
                task_loss = torch.tensor(0.0, device=dev)
                loss = torch.tensor(0.0, device=dev)
                #
                imgs = batch["image"].to(dev)
                ids = batch["id"]
                osz = batch["original_size"]

                if dataset_mode == "single":
                    masks = batch["mask"].to(dev)

                    batched_input = []
                    for i in range(len(imgs)):
                        entry = {
                            "image": imgs[i],
                            "original_size": (int(osz[i][0]), int(osz[i][1])),
                        }
                        if batch["box_prompt"][i] is not None:
                            entry["boxes"] = batch["box_prompt"][i].to(dev).unsqueeze(0)
                        if batch["point_coords"][i] is not None:
                            entry["point_coords"] = batch["point_coords"][i].to(dev).unsqueeze(0)
                            entry["point_labels"] = batch["point_labels"][i].to(dev).unsqueeze(0)

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
                            iou_list.append(o["iou_predictions"].squeeze(0).to(torch.float32))

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
                        best_conf_logits = pred_ious.gather(1, best_indices.unsqueeze(1)).squeeze(1)
                        cls_loss_val += F.binary_cross_entropy_with_logits(
                            best_conf_logits, torch.ones_like(best_conf_logits), reduction="mean"
                        )
                        task_loss = (
                            w_bce * bce
                            + w_focal * focal
                            + w_dice * dice_loss
                            + w_cls * cls_loss_val
                        )

                        with torch.no_grad():
                            gt_bin = masks > 0.5
                            ious = []
                            for b in range(pred_masks.shape[0]):
                                preds_bin = (torch.sigmoid(pred_masks[b]) > 0.5).float()
                                inter = (preds_bin * gt_bin[b]).sum((-2, -1))
                                union = preds_bin.sum((-2, -1)) + gt_bin[b].sum((-2, -1)) - inter
                                ious.append(inter / (union + 1e-6))
                            gt_ious = torch.stack(ious, dim=0)

                        iou_loss = F.mse_loss(torch.sigmoid(pred_ious), gt_ious)
                else:
                    gt_masks = batch["gt_masks"]
                    point_coords = batch["point_coords"]
                    pred_masks_all = []
                    pred_ious_all = []
                    for bi in range(len(imgs)):
                        pm, pi = predict_from_grid(
                            student,
                            imgs[bi],
                            point_coords[bi].to(dev),
                            (int(imgs[bi].shape[-2]), int(imgs[bi].shape[-1])),
                        )
                        pred_masks_all.append(pm)
                        pred_ious_all.append(pi)

                    bce = torch.tensor(0.0, device=dev)
                    focal = torch.tensor(0.0, device=dev)
                    dice_loss = torch.tensor(0.0, device=dev)
                    iou_loss = torch.tensor(0.0, device=dev)
                    cls_total = 0
                    matched_cnt = 0

                    for bi in range(len(imgs)):
                        gt = gt_masks[bi].to(dev)  # [K,1,S,S]
                        preds = pred_masks_all[bi].reshape(-1, gt.shape[-2], gt.shape[-1])
                        ious_pred = pred_ious_all[bi].reshape(-1)
                        gt_flat = gt.squeeze(1)

                        # IoU matrix for Hungarian matching
                        pred_bin = (torch.sigmoid(preds) > 0.5).float()
                        inter = (pred_bin.unsqueeze(1) * gt_flat.unsqueeze(0)).sum((-2, -1))
                        union = (
                            pred_bin.unsqueeze(1).sum((-2, -1))
                            + gt_flat.unsqueeze(0).sum((-2, -1))
                            - inter
                        )
                        iou_mat = inter / (union + 1e-6)

                        cost = (-iou_mat).cpu().numpy()
                        row_ind, col_ind = linear_sum_assignment(cost)
                        assert len(set(col_ind)) == len(col_ind), "GT matched more than once"
                        assert len(set(row_ind)) == len(
                            row_ind
                        ), "Prediction matched more than once"

                        labels = torch.zeros(preds.shape[0], device=dev)
                        assigned_gt = torch.full(
                            (preds.shape[0],), -1, device=dev, dtype=torch.long
                        )
                        for r, c in zip(row_ind, col_ind):
                            labels[r] = 1.0 if iou_mat[r, c] >= 0.5 else 0.0
                            if iou_mat[r, c] >= 0.5:
                                assigned_gt[r] = c
                                matched_cnt += 1

                        cls_loss_val += F.binary_cross_entropy_with_logits(
                            ious_pred, labels, reduction="sum"
                        )
                        cls_total += labels.numel()

                        for r, c in zip(row_ind, col_ind):
                            if iou_mat[r, c] < 0.5:
                                continue
                            target = gt[c]
                            logit = preds[r].unsqueeze(0).unsqueeze(0)
                            target_exp = target.unsqueeze(0)
                            bce += F.binary_cross_entropy_with_logits(logit, target_exp)
                            focal += sigmoid_focal_loss(logit, target_exp, reduction="mean")
                            prob = torch.sigmoid(logit)
                            num = (prob * target_exp).sum((-2, -1)) * 2
                            den = prob.sum((-2, -1)) + target_exp.sum((-2, -1))
                            dice_loss += 1 - (num / (den + 1e-6)).mean()
                            iou_loss += F.mse_loss(torch.sigmoid(ious_pred[r]), iou_mat[r, c])

                    # Average using only the matched predictions to avoid loss dilution.
                    n_matched = matched_cnt
                    cls_loss_val = cls_loss_val / max(1, cls_total)
                    task_loss = (
                        w_bce * bce + w_focal * focal + w_dice * dice_loss + w_cls * cls_loss_val
                    )
                    task_loss = task_loss / max(1, n_matched)
                    iou_loss = iou_loss / max(1, n_matched)

                    dist_loss = torch.tensor(0.0, device=dev)
                    if use_distillation and hook_handles:
                        feat_student = pop_features() or {}
                        if step % 20 == 0:
                            log.info(
                                f"Student features: { {k: [tuple(t.shape) for t in v] for k, v in feat_student.items()} }"
                            )

                        use_precomputed = dist_cfg.get("use_precomputed_features", False)
                        base_dir = Path(dist_cfg.get("precomputed_root", "precomputed"))
                        teacher_feats = {}

                        if not use_precomputed:
                            for tinfo in teacher_models:
                                with torch.no_grad():
                                    _ = tinfo["model"](
                                        batched_input=batched_input,
                                        multimask_output=False,
                                    )
                                teacher_feats[tinfo["name"]] = pop_features() or {}
                            if step % 20 == 0:
                                for tn, feats in teacher_feats.items():
                                    log.info(
                                        f"Teacher {tn} features: { {k: [tuple(t.shape) for t in v] for k, v in feats.items()} }"
                                    )

                        for t_cfg in teacher_cfgs:
                            weight = t_cfg["weight"]
                            tname = t_cfg["name"]
                            tpot = teacher_pots.get(tname, pot)

                            def _get_loss_params(config_dict, rename_map=None):
                                params = {k: v for k, v in config_dict.items() if k != "enable"}
                                if rename_map:
                                    for old_key, new_key in rename_map.items():
                                        if old_key in params:
                                            params[new_key] = params.pop(old_key)
                                return params

                            if dist_cfg.get("encoder_matching", {}).get("enable"):
                                enc_keys_s = pot.get("enc", [])
                                enc_keys_t = tpot.get("enc", [])

                                if (
                                    enc_keys_s
                                    and enc_keys_t
                                    and all(k in feat_student for k in enc_keys_s)
                                ):
                                    try:
                                        num_layers_to_compare = len(enc_keys_s)
                                        selected_enc_keys_t = enc_keys_t[-num_layers_to_compare:]

                                        if use_precomputed:
                                            feat_teacher = load_cached_npy_features(
                                                base_dir,
                                                tname,
                                                "train",
                                                ids,
                                                selected_enc_keys_t,
                                            )
                                        else:
                                            feat_teacher = [
                                                teacher_feats[tname][k][0]
                                                for k in selected_enc_keys_t
                                            ]

                                        enc_loss = encoder_matching_loss(
                                            [feat_student[k][0] for k in enc_keys_s],
                                            feat_teacher,
                                            **_get_loss_params(dist_cfg["encoder_matching"]),
                                            n_layers=num_layers_to_compare,
                                        )
                                        w_enc = dist_cfg.get("encoder_matching", {}).get(
                                            "weight", 1.0
                                        )
                                        enc_loss_val += weight * w_enc * enc_loss
                                        if step % 20 == 0:
                                            log.info(f"enc_loss[{tname}]={enc_loss.item():.4f}")
                                    except Exception as e:
                                        log.warning(
                                            f"Encoder matching error for teacher {tname}: {e}"
                                        )

                            if dist_cfg.get("decoder_matching", {}).get("enable"):
                                dec_keys_s = pot.get("dec", [])
                                dec_keys_t = tpot.get("dec", [])
                                if (
                                    dec_keys_s
                                    and dec_keys_t
                                    and all(k in feat_student for k in dec_keys_s)
                                    and (
                                        use_precomputed
                                        or dec_keys_t[0] in teacher_feats.get(tname, {})
                                    )
                                ):
                                    try:
                                        if use_precomputed:
                                            feat_teacher = load_cached_npy_features(
                                                base_dir,
                                                tname,
                                                "train",
                                                ids,
                                                dec_keys_t,
                                            )[0]
                                        else:
                                            feat_teacher = teacher_feats[tname][dec_keys_t[0]][0]

                                        dec_loss = decoder_matching_loss(
                                            feat_student[dec_keys_s[0]][0],
                                            feat_teacher,
                                            **_get_loss_params(dist_cfg["decoder_matching"]),
                                        )
                                        w_dec = dist_cfg.get("decoder_matching", {}).get(
                                            "weight", 1.0
                                        )
                                        dec_loss_val += weight * w_dec * dec_loss
                                        if step % 20 == 0:
                                            log.info(f"dec_loss[{tname}]={dec_loss.item():.4f}")
                                    except Exception as e:
                                        log.warning(
                                            f"Decoder matching error for teacher {tname}: {e}"
                                        )

                            if dist_cfg.get("attention_matching", {}).get("enable"):
                                attn_keys_s = pot.get("attn", [])
                                attn_keys_t = tpot.get("attn", [])

                                if (
                                    attn_keys_s
                                    and attn_keys_t
                                    and all(k in feat_student for k in attn_keys_s)
                                ):
                                    try:
                                        num_layers_to_compare = len(attn_keys_s)
                                        selected_attn_keys_t = attn_keys_t[-num_layers_to_compare:]

                                        if use_precomputed:
                                            attn_teacher = load_cached_npy_features(
                                                base_dir,
                                                tname,
                                                "train",
                                                ids,
                                                selected_attn_keys_t,
                                            )
                                        else:
                                            attn_teacher = [
                                                teacher_feats[tname][k][0]
                                                for k in selected_attn_keys_t
                                            ]

                                        attn_loss = attention_matching_loss(
                                            [feat_student[k][0] for k in attn_keys_s],
                                            attn_teacher,
                                            **_get_loss_params(
                                                dist_cfg["attention_matching"],
                                                {"lambda": "lambda_attn"},
                                            ),
                                            n_layers=num_layers_to_compare,
                                        )
                                        w_attn = dist_cfg.get("attention_matching", {}).get(
                                            "weight", 1.0
                                        )
                                        attn_loss_val += weight * w_attn * attn_loss
                                        if step % 20 == 0:
                                            log.info(f"attn_loss[{tname}]={attn_loss.item():.4f}")
                                    except Exception as e:
                                        log.warning(
                                            f"Attention matching error for teacher {tname}: {e}"
                                        )

                            if dist_cfg.get("relational_KD", {}).get("enable"):
                                rk_keys_s = pot.get("rkd", [])
                                rk_keys_t = tpot.get("rkd", [])
                                if (
                                    rk_keys_s
                                    and rk_keys_t
                                    and all(k in feat_student for k in rk_keys_s)
                                    and (
                                        use_precomputed
                                        or rk_keys_t[0] in teacher_feats.get(tname, {})
                                    )
                                ):
                                    try:
                                        if use_precomputed:
                                            feat_teacher = load_cached_npy_features(
                                                base_dir, tname, "train", ids, rk_keys_t
                                            )[0]
                                        else:
                                            feat_teacher = teacher_feats[tname][rk_keys_t[0]][0]

                                        rk_loss = rkd_loss(
                                            feat_student[rk_keys_s[0]][0],
                                            feat_teacher,
                                            **_get_loss_params(
                                                dist_cfg["relational_KD"],
                                                {"lambda": "lambda_rkd"},
                                            ),
                                        )
                                        w_rkd = dist_cfg.get("relational_KD", {}).get("weight", 1.0)
                                        rkd_loss_val += weight * w_rkd * rk_loss
                                        if step % 20 == 0:
                                            log.info(f"rkd_loss[{tname}]={rk_loss.item():.4f}")
                                    except Exception as e:
                                        log.warning(f"RKD error for teacher {tname}: {e}")

                    dist_loss = enc_loss_val + dec_loss_val + attn_loss_val + rkd_loss_val
                    loss = (task_loss + w_iou * iou_loss + lambda_coef * dist_loss) / tr_cfg.get(
                        "gradient_accumulation", 1
                    )

                scaler.scale(loss).backward()
                # del low_res_logits, tmp, logit_up, prob, focal, dice_loss
                if use_distillation and "feat_student" in locals():
                    del feat_student

                if (step + 1) % tr_cfg.get("gradient_accumulation", 1) == 0 or (step + 1) == len(
                    tr_loader
                ):
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
                tot_cls += cls_loss_val.item()

                ga = tr_cfg.get("gradient_accumulation", 1)
                norm = n_matched if dataset_mode == "everything" else 1
                bce_c = w_bce * bce.item() / max(1, norm) / ga
                focal_c = w_focal * focal.item() / max(1, norm) / ga
                dice_c = w_dice * dice_loss.item() / max(1, norm) / ga
                iou_c = w_iou * iou_loss.item() / ga
                cls_c = w_cls * cls_loss_val.item() / max(1, norm) / ga
                enc_c = lambda_coef * enc_loss_val.item() / ga
                dec_c = lambda_coef * dec_loss_val.item() / ga
                attn_c = lambda_coef * attn_loss_val.item() / ga
                rkd_c = lambda_coef * rkd_loss_val.item() / ga
                dist_c = lambda_coef * dist_loss.item() / ga

                pbar.set_postfix(
                    bce=f"{bce_c:.3f}",
                    focal=f"{focal_c:.3f}",
                    dice=f"{dice_c:.3f}",
                    iou=f"{iou_c:.3f}",
                    cls=f"{cls_c:.3f}",
                    enc=f"{enc_c:.3f}",
                    dec=f"{dec_c:.3f}",
                    attn=f"{attn_c:.3f}",
                    rkd=f"{rkd_c:.3f}",
                    dist=f"{dist_c:.3f}",
                    total=f"{loss.item():.3f}",
                    lr=f"{scheduler.get_last_lr()[0]:.2e}",
                )

                if step % 100 == 0:
                    log_gpu_memory(f"Epoch {ep}, Step {step}")

            writer.add_scalar("train/task_loss", tot_task / len(tr_loader), ep)
            writer.add_scalar("train/dist_loss", tot_dist / len(tr_loader), ep)
            writer.add_scalar("train/iou_loss", tot_iou / len(tr_loader), ep)
            writer.add_scalar("train/cls_loss", tot_cls / len(tr_loader), ep)
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
                        original_sizes = vb["original_size"]

                        if dataset_mode == "single":
                            masks = vb["mask"].to(dev)

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
                                iou_list.append(o["iou_predictions"].squeeze(0).to(torch.float32))

                            pred_masks = torch.stack(mask_list, dim=0)
                            pred_ious = torch.stack(iou_list, dim=0)

                            best_indices = pred_ious.argmax(dim=1)
                            probs = torch.sigmoid(
                                pred_masks[torch.arange(len(pred_masks)), best_indices].unsqueeze(1)
                            )

                            for i in range(len(imgs)):
                                prob_i = probs[i]
                                gt_i = masks[i]

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
                                    (pred_bin * gt_i).sum((-2, -1)) / (union.sum((-2, -1)) + 1e-6)
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
                                        ep % cfg["visual"].get("save_every_n_epochs", 10) == 0
                                        or ((np.mean(dices) + np.mean(ious)) / 2) > best_score
                                    )
                                ):
                                    cur_path = Path(cfg["visual"]["save_path"]) / f"epoch_{ep}"
                                    cur_path.mkdir(parents=True, exist_ok=True)
                                    img_denorm = (
                                        imgs[i] * torch.tensor(STD, device=dev)[:, None, None]
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

                                    overlay_mask_on_image(
                                        image_tensor=img_denorm,
                                        mask_tensor=pred_mask,
                                        bbox_tensor=box.cpu() if box is not None else None,
                                        point_coords=pts.cpu() if pts is not None else None,
                                        point_labels=lbl.cpu() if lbl is not None else None,
                                        original_size=None,
                                        threshold=cfg["visual"].get("IOU_threshold", 0.5),
                                        save_dir=str(cur_path),
                                        filename_info=(f"ep{ep}_id{vb['id'][i]}_b{bi}_s{i}"),
                                    )
                        else:
                            # Segment-everything validation
                            gt_masks = vb["gt_masks"]
                            point_coords = vb["point_coords"]

                            for i in range(len(imgs)):
                                pm, _ = predict_from_grid(
                                    student,
                                    imgs[i],
                                    point_coords[i].to(dev),
                                    (int(imgs[i].shape[-2]), int(imgs[i].shape[-1])),
                                )

                                preds = pm.reshape(-1, gt_masks[i].shape[-2], gt_masks[i].shape[-1])
                                probs = torch.sigmoid(preds)
                                gt = gt_masks[i].to(dev)
                                gt_bin = gt.squeeze(1)
                                pred_bin = (probs >= 0.5).float()

                                inter = (pred_bin.unsqueeze(1) * gt_bin.unsqueeze(0)).sum((-2, -1))
                                union = (
                                    pred_bin.unsqueeze(1).sum((-2, -1))
                                    + gt_bin.unsqueeze(0).sum((-2, -1))
                                    - inter
                                )
                                iou_mat = inter / (union + 1e-6)

                                cost = (-iou_mat).cpu().numpy()
                                row_ind, col_ind = linear_sum_assignment(cost)
                                assert len(set(col_ind)) == len(
                                    col_ind
                                ), "GT matched more than once"
                                assert len(set(row_ind)) == len(
                                    row_ind
                                ), "Prediction matched more than once"
                                for r, c in zip(row_ind, col_ind):
                                    prob_i = probs[r]
                                    gt_i = gt[c]
                                    num_soft = (prob_i * gt_i).sum() * 2
                                    den_soft = prob_i.sum() + gt_i.sum()
                                    dice_val = (num_soft / (den_soft + 1e-6)).item()
                                    iou_val = iou_mat[r, c].item()
                                    dices.append(dice_val)
                                    ious.append(iou_val)

                                if (
                                    bi == 0
                                    and cfg["visual"].get("status", False)
                                    and (
                                        ep % cfg["visual"].get("save_every_n_epochs", 10) == 0
                                        or ((np.mean(dices) + np.mean(ious)) / 2) > best_score
                                    )
                                ):
                                    cur_path = Path(cfg["visual"]["save_path"]) / f"epoch_{ep}"
                                    cur_path.mkdir(parents=True, exist_ok=True)
                                    img_denorm = (
                                        (
                                            imgs[i] * torch.tensor(STD, device=dev)[:, None, None]
                                            + torch.tensor(MEAN, device=dev)[:, None, None]
                                        )
                                        .clamp(0, 1)
                                        .cpu()
                                    )
                                    best_masks = probs[row_ind].cpu()
                                    overlay_masks_on_image(
                                        image_tensor=img_denorm,
                                        masks=best_masks,
                                        grid_points=point_coords[i].cpu(),
                                        threshold=cfg["visual"].get("IOU_threshold", 0.5),
                                        save_dir=str(cur_path),
                                        filename_info=f"ep{ep}_id{vb['id'][i]}_b{bi}",
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
        for t in teacher_models:
            for h in t.get("hooks", []):
                h.remove()
        feature_cache.clear()
        writer.close()
        clear_gpu_cache()
        log_gpu_memory("Final cleanup completed")


if __name__ == "__main__":
    main()

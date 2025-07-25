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
    encoder_patch_loss,
    prompt_embed_loss,
    mask_token_loss,
    dense_mask_logits_loss,
)
from finetune_utils.feature_hooks import pop_features, register_hooks
from finetune_utils.visualization import overlay_mask_on_image, overlay_masks_on_image
from pathlib import Path
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm
from contextlib import nullcontext

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


def predict_from_grid(model, image, points, orig_size, input_size, batch_size=64, multimask_output=True):
    """Run the SAM model on a grid of points and return masks and IoU preds.
    input_size: (H_resized, W_resized) before padding, used to correctly crop padding.
    """
    device = image.device
    inp = model.preprocess(image.unsqueeze(0))
    embedding = model.image_encoder(inp)
    dense_pe = model.prompt_encoder.get_dense_pe()

    all_masks = []
    all_ious = []
    all_lowres = []
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
            multimask_output=multimask_output,
        )
        masks = model.postprocess_masks(low_res, input_size, orig_size).squeeze(0)
        all_masks.append(masks)
        all_lowres.append(low_res.squeeze(0))
        all_ious.append(iou_pred.squeeze(0))
    return torch.cat(all_masks, dim=0), torch.cat(all_ious, dim=0), torch.cat(all_lowres, dim=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = json.load(open(args.config))
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    # ─────────────────── Multimask setting ───────────────────
    # Allow users to control whether the mask decoder returns multiple masks
    # via `model.multimask_output` in the JSON config. Defaults to True for
    # backward-compatibility.
    multimask_output_cfg = cfg.get("model", {}).get("multimask_output", True)

    if "lr" not in cfg["train"]:
        cfg["train"]["lr"] = 5e-5

    # ─────────────────── Image Preprocessing ───────────────────
    # SAM's `preprocess` expects input pixels in the [0, 255] range and handles
    # normalization (subtracting ImageNet mean, dividing by std) internally.
    # Therefore, the DataLoader should **not** normalize the images beforehand.
    # Instead, the pipeline should be:
    #   1. Convert PIL Image to a tensor (0-1 range).
    #   2. Scale it directly to the 0-255 range.
    # This ensures alignment with the `preprocess` method.
    tf_img = T.Compose([T.ToTensor(), T.Lambda(lambda x: x * 255.0)])
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
            if k in (
                "gt_masks",
                "gt_masks_original",
                "mask_original",
                "point_coords",
                "point_labels",
                "box_prompt_raw",
                "point_coords_raw",
                "point_labels_raw",
            ):
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
    out_dir = Path(m_cfg.get("save_path", "./"))
    out_dir.mkdir(parents=True, exist_ok=True)
    log_file = out_dir / "training_log.txt"
    file_handler = logging.FileHandler(log_file, mode="a")
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%H:%M:%S")
    )
    log.addHandler(file_handler)

    # ─────────────────── Build student and ensure parameters are trainable ───────────────────
    student = sam_model_registry[m_cfg.get("type", "vit_t")](
        checkpoint=m_cfg.get("checkpoint_path")
    ).to(dev)
    # The builder internally calls .eval(), which doesn't change requires_grad,
    # but we enable gradients for all parameters just in case.
    student.requires_grad_(True)

    if cfg["freeze"].get("freeze_image_encoder", True):
        student.image_encoder.requires_grad_(False)
    if cfg["freeze"].get("freeze_prompt_encoder", False):
        student.prompt_encoder.requires_grad_(False)
    if cfg["freeze"].get("freeze_mask_decoder", False):
        student.mask_decoder.requires_grad_(False)

    dist_cfg = cfg.get("distillation", {})
    # The base flag from config decides default behaviour. However, if any
    # stage in `stage_schedule` explicitly enables distillation, we must still
    # prepare teacher models & hooks even when the base flag is False.
    any_stage_distill = any(
        st.get("distillation", False)
        for st in cfg.get("stage_schedule", [])
    )
    use_distillation = dist_cfg.get("enable", False) or any_stage_distill
    lambda_coef = dist_cfg.get("lambda_coef", 1.0)
    hook_handles = []

    def _build_pot(model_type: str):
        """Return layers to hook for each distillation component."""
        pot = {
            "enc_patch": [],
            "prompt_embed": [],
            "mask_token": [],
        }

        if model_type == "vit_t":
            pot["enc_patch"] = ["image_encoder.neck"]
        else:  # vit_h or other ViT variants
            pot["enc_patch"] = ["image_encoder.blocks.30"]

        # Both teacher & student share the same hook for prompt embed & mask token
        pot["prompt_embed"] = ["mask_decoder.transformer"]
        pot["mask_token"] = ["mask_decoder.transformer"]
        return pot

    pot = _build_pot(m_cfg.get("type", "vit_t"))

    teacher_models = []
    teacher_pots = {}
    if use_distillation:
        enabled_losses = [
            n
            for n in (
                "encoder_patch",
                "prompt_embed",
                "decoder_mask_token",
                "dense_mask_logits",
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
            ("encoder_patch", "enc_patch"),
            ("prompt_embed", "prompt_embed"),
            ("decoder_mask_token", "mask_token"),
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
                    ("encoder_patch", "enc_patch"),
                    ("prompt_embed", "prompt_embed"),
                    ("decoder_mask_token", "mask_token"),
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

    # ─────────────────── Mask-teacher supervision (segment-everything) ───────────────────
    mask_teacher_cfg = cfg.get("mask_teacher", {})
    mask_teacher_enable = mask_teacher_cfg.get("enable", False)
    mask_teacher_method = mask_teacher_cfg.get("method", "none").lower()
    mask_teacher_model = None
    soft_loss_type = (mask_teacher_cfg.get("soft_loss", {}) or {}).get("type", "l2").lower()
    soft_loss_weight = (mask_teacher_cfg.get("soft_loss", {}) or {}).get("weight", 1.0)
    mask_teacher_iou_thr = mask_teacher_cfg.get("iou_threshold", 0.3)
    gt_override_enable = mask_teacher_cfg.get("gt_override", False)
    if dataset_mode == "everything" and mask_teacher_enable and mask_teacher_method in ("replace_gt", "dual_supervision"):
        t_type = mask_teacher_cfg.get("teacher_type", "vit_h").lower()
        t_ckpt = mask_teacher_cfg.get("checkpoint_path")
        if t_type not in sam_model_registry:
            log.warning(f"Unknown mask teacher type {t_type}; disable mask_teacher supervision.")
            mask_teacher_enable = False
        elif not t_ckpt or not os.path.exists(t_ckpt):
            log.warning(f"Mask teacher checkpoint not found: {t_ckpt}; disable mask_teacher supervision.")
            mask_teacher_enable = False
        else:
            mask_teacher_model = sam_model_registry[t_type](checkpoint=t_ckpt).to(dev).eval()
            mask_teacher_model.requires_grad_(False)
            log.info(f"Loaded mask teacher ({t_type}) from {t_ckpt} | method={mask_teacher_method}")
    else:
        mask_teacher_enable = False

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

    # ─────────────────── Loss weights & stage scheduling ───────────────────
    # Users can optionally define a `stage_schedule` in the root of the JSON
    # config. Each stage is a dict containing:
    #   {
    #       "name": "distill_only",        # optional, just for logging
    #       "start_epoch": 0,               # inclusive
    #       "end_epoch": 100,               # exclusive
    #       "loss_weights": {               # same schema as train.loss_weights
    #           "bce": 0, "focal": 0, "dice": 0, "iou": 0, "cls": 0
    #       },
    #       "lambda_coef": 1.0,             # overrides distillation.lambda_coef
    #       "distillation": true            # toggles distillation in this stage
    #   }
    #
    # If no schedule is provided, the original single-stage behaviour is kept.
    stage_schedule = cfg.get("stage_schedule", [])

    base_loss_weights = tr_cfg.get("loss_weights", {})
    base_w_bce = base_loss_weights.get("bce", 1.0)
    base_w_focal = base_loss_weights.get("focal", 0.5)
    base_w_dice = base_loss_weights.get("dice", 1.0)
    base_w_iou = base_loss_weights.get("iou", 1.0)
    base_w_cls = base_loss_weights.get("cls", 1.0)

    # Initial (may be overridden each epoch)
    w_bce, w_focal, w_dice, w_iou, w_cls = (
        base_w_bce,
        base_w_focal,
        base_w_dice,
        base_w_iou,
        base_w_cls,
    )

    # Helper to fetch stage-specific settings at runtime
    def _apply_stage_overrides(epoch: int):
        nonlocal w_bce, w_focal, w_dice, w_iou, w_cls, lambda_coef, use_distillation

        # Reset to base values before applying overrides
        w_bce, w_focal, w_dice, w_iou, w_cls = (
            base_w_bce,
            base_w_focal,
            base_w_dice,
            base_w_iou,
            base_w_cls,
        )
        lambda_coef_stage = lambda_coef  # default keep previous value
        distill_stage_enable = use_distillation

        for st in stage_schedule:
            if st.get("start_epoch", 0) <= epoch < st.get("end_epoch", tr_cfg["epochs"]):
                lw = st.get("loss_weights", {})
                w_bce = lw.get("bce", w_bce)
                w_focal = lw.get("focal", w_focal)
                w_dice = lw.get("dice", w_dice)
                w_iou = lw.get("iou", w_iou)
                w_cls = lw.get("cls", w_cls)
                lambda_coef_stage = st.get("lambda_coef", lambda_coef_stage)
                distill_stage_enable = st.get("distillation", distill_stage_enable)
                break  # Assume at most one active stage per epoch

        lambda_coef = lambda_coef_stage
        use_distillation = distill_stage_enable

    # Apply once for epoch 0 before entering the loop
    _apply_stage_overrides(0)

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
            # Apply stage-specific overrides (loss weights, distillation toggle, lambda)
            _apply_stage_overrides(ep)

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
                # ─────────── Initialize all potential loss variables that might be used later ───────────
                bce = torch.tensor(0.0, device=dev, requires_grad=True)
                focal = torch.tensor(0.0, device=dev, requires_grad=True)
                dice_loss = torch.tensor(0.0, device=dev, requires_grad=True)
                dist_loss = torch.tensor(0.0, device=dev, requires_grad=True)
                enc_loss_val = torch.tensor(0.0, device=dev, requires_grad=True)
                prompt_loss_val = torch.tensor(0.0, device=dev, requires_grad=True)
                tok_loss_val = torch.tensor(0.0, device=dev, requires_grad=True)
                dense_loss_val = torch.tensor(0.0, device=dev, requires_grad=True)
                cls_loss_val = torch.tensor(0.0, device=dev, requires_grad=True)
                task_loss = torch.tensor(0.0, device=dev, requires_grad=True)
                loss = torch.tensor(0.0, device=dev, requires_grad=True)
                #
                imgs = batch["image"].to(dev)
                ids = batch["id"]
                osz = batch["original_size"]

                if dataset_mode == "single":
                    masks = batch["mask"].to(dev)  # low-res GT (1/4)
                    masks_orig = [m.to(dev) for m in batch["mask_original"]]

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
                        if "input_size" in batch:
                            entry["input_size"] = (
                                int(batch["input_size"][i][0]),
                                int(batch["input_size"][i][1]),
                            )

                        batched_input.append(entry)

                    with autocast(
                        dtype=torch.bfloat16 if tr_cfg.get("bf16", False) else torch.float16
                    ):
                        out = student(batched_input=batched_input, multimask_output=multimask_output_cfg)

                        # ---------------- Distillation Forward ----------------
                        if use_distillation:
                            # Pop intermediate features from the student's forward pass
                            feat_student = pop_features()
                            teacher_outs = []  # Store forward results for each teacher (list of list[dict])
                            teacher_feats = []  # Store features captured by hooks (list[dict])
                            for t in teacher_models:
                                with torch.no_grad():
                                    _out_t = t["model"](batched_input=batched_input, multimask_output=multimask_output_cfg)
                                teacher_outs.append(_out_t)
                                teacher_feats.append(pop_features())

                            # Initialize distillation loss accumulators
                            enc_loss_val = torch.tensor(0.0, device=dev, requires_grad=True)
                            prompt_loss_val = torch.tensor(0.0, device=dev, requires_grad=True)
                            tok_loss_val = torch.tensor(0.0, device=dev, requires_grad=True)
                            dense_loss_val = torch.tensor(0.0, device=dev, requires_grad=True)

                            for t_idx, t in enumerate(teacher_models):
                                w_t = t["weight"]
                                feat_t = teacher_feats[t_idx]

                                # 1. Encoder patch tokens
                                if dist_cfg.get("encoder_patch", {}).get("enable"):
                                    l_s = pot["enc_patch"][0]
                                    l_t = teacher_pots[t["name"]]["enc_patch"][0]
                                    enc_cfg = dist_cfg["encoder_patch"]
                                    enc_loss_val = enc_loss_val + w_t * encoder_patch_loss(
                                        feat_student[l_s][0],
                                        feat_t[l_t][0],
                                        w_l2=enc_cfg.get("w_l2", 1.0),
                                        w_cos=enc_cfg.get("w_cos", 1.0),
                                    )

                                # 2. Prompt-conditioned embeddings
                                if dist_cfg.get("prompt_embed", {}).get("enable"):
                                    l_s = pot["prompt_embed"][0]
                                    l_t = teacher_pots[t["name"]]["prompt_embed"][0]
                                    pe_cfg = dist_cfg["prompt_embed"]
                                    prompt_loss_val = prompt_loss_val + w_t * prompt_embed_loss(
                                        feat_student[l_s][0],
                                        feat_t[l_t][0],
                                        w_mse=pe_cfg.get("w_mse", 0.7),
                                        w_cos=pe_cfg.get("w_cos", 0.3),
                                    )

                                # 3. Mask token logits
                                if dist_cfg.get("decoder_mask_token", {}).get("enable"):
                                    l_s = pot["mask_token"][0]
                                    l_t = teacher_pots[t["name"]]["mask_token"][0]
                                    tok_cfg = dist_cfg["decoder_mask_token"]
                                    tok_s = feat_student[l_s][0][:, 1 : 1 + student.mask_decoder.num_mask_tokens, :]
                                    tok_t = feat_t[l_t][0][:, 1 : 1 + student.mask_decoder.num_mask_tokens, :]
                                    tok_loss_val = tok_loss_val + w_t * mask_token_loss(
                                        tok_s,
                                        tok_t,
                                        w_kl=tok_cfg.get("w_kl", 1.0),
                                        temperature=tok_cfg.get("temperature", 0.5),
                                    )

                        bce = torch.tensor(0.0, device=dev, requires_grad=True)
                        focal = torch.tensor(0.0, device=dev, requires_grad=True)
                        dice_loss = torch.tensor(0.0, device=dev, requires_grad=True)
                        cls_loss_val = torch.tensor(0.0, device=dev, requires_grad=True)
                        iou_loss = torch.tensor(0.0, device=dev, requires_grad=True)

                        for i, o in enumerate(out):
                            low_res = o["low_res_logits"].to(torch.float32).squeeze(0)
                            iou_pred = o["iou_predictions"].squeeze(0).to(torch.float32)
                            mask_orig_pred = o["masks"].to(torch.float32).squeeze(0)

                            best_idx = iou_pred.argmax()
                            sel_logit = low_res[best_idx].unsqueeze(0)

                            bce = bce + F.binary_cross_entropy_with_logits(sel_logit, masks[i])
                            focal = focal + sigmoid_focal_loss(sel_logit, masks[i], reduction="mean")
                            prob = torch.sigmoid(sel_logit)
                            num = (prob * masks[i]).sum((-2, -1)) * 2
                            den = prob.sum((-2, -1)) + masks[i].sum((-2, -1))
                            dice_loss = dice_loss + (1 - (num / (den + 1e-6)).mean())

                            best_conf = iou_pred[best_idx]
                            cls_loss_val = cls_loss_val + F.binary_cross_entropy_with_logits(
                                best_conf, torch.ones_like(best_conf)
                            )

                            with torch.no_grad():
                                gt_bin = masks_orig[i] > 0.5
                                ious = []
                                for c in range(mask_orig_pred.shape[0]):
                                    pred_bin = (mask_orig_pred[c] > 0.5).float()
                                    inter = (pred_bin * gt_bin).sum()
                                    union = pred_bin.sum() + gt_bin.sum() - inter
                                    ious.append(inter / (union + 1e-6))
                                gt_ious = torch.stack(ious)
                            iou_loss = iou_loss + F.mse_loss(torch.sigmoid(iou_pred), gt_ious)

                            # -------- distillation: dense mask logits (per-sample) --------
                            if use_distillation and dist_cfg.get("dense_mask_logits", {}).get("enable"):
                                dl_cfg = dist_cfg["dense_mask_logits"]
                                for t_idx, t in enumerate(teacher_models):
                                    t_out_i = teacher_outs[t_idx][i]
                                    t_low_res = t_out_i["low_res_logits"].to(torch.float32).squeeze(0)
                                    t_iou_pred = t_out_i["iou_predictions"].squeeze(0).to(torch.float32)
                                    t_best_idx = t_iou_pred.argmax()
                                    t_sel_logit = t_low_res[t_best_idx].unsqueeze(0)
                                    dense_loss_val = dense_loss_val + teacher_models[t_idx]["weight"] * dense_mask_logits_loss(
                                        sel_logit,
                                        t_sel_logit,
                                        w_kl=dl_cfg.get("w_kl", 0.6),
                                        w_focal=dl_cfg.get("w_focal", 0.4),
                                        gamma=dl_cfg.get("gamma", 2.0),
                                    )

                        bce = bce / len(out)
                        focal = focal / len(out)
                        dice_loss = dice_loss / len(out)
                        cls_loss_val = cls_loss_val / len(out)
                        iou_loss = iou_loss / len(out)

                        # ---------------- distillation: dense mask logits ----------------
                        if use_distillation and dist_cfg.get("dense_mask_logits", {}).get("enable"):
                            dl_cfg = dist_cfg["dense_mask_logits"]
                            for t_idx, t in enumerate(teacher_models):
                                t_out_i = teacher_outs[t_idx][i]
                                t_low_res = t_out_i["low_res_logits"].to(torch.float32).squeeze(0)
                                t_iou_pred = t_out_i["iou_predictions"].squeeze(0).to(torch.float32)
                                t_best_idx = t_iou_pred.argmax()
                                t_sel_logit = t_low_res[t_best_idx].unsqueeze(0)
                                dense_loss_val = dense_loss_val + teacher_models[t_idx]["weight"] * dense_mask_logits_loss(
                                    sel_logit,
                                    t_sel_logit,
                                    w_kl=dl_cfg.get("w_kl", 0.6),
                                    w_focal=dl_cfg.get("w_focal", 0.4),
                                    gamma=dl_cfg.get("gamma", 2.0),
                                )

                        task_loss = (
                            w_bce * bce
                            + w_focal * focal
                            + w_dice * dice_loss
                            + w_cls * cls_loss_val
                        )

                        # Calculate total loss (for single-prompt mode)
                        dist_loss = enc_loss_val + prompt_loss_val + tok_loss_val + dense_loss_val

                        # The total loss must be calculated here for single-prompt mode.
                        # Otherwise, if the loss remains 0, gradients will be zero, and
                        # it won't display correctly in the tqdm progress bar.
                        loss = (
                            task_loss.float()
                            + w_iou * iou_loss.float()
                            + lambda_coef * dist_loss.float()
                        ) / tr_cfg.get("gradient_accumulation", 1)
                else:
                    gt_masks = batch["gt_masks"]
                    point_coords = batch["point_coords"]
                    pred_lowres_all = []
                    pred_ious_all = []
                    # ─────────────────── Teacher predictions (optional) ───────────────────
                    teacher_lowres_all = []
                    for bi in range(len(imgs)):
                        raw_sz_tuple = (
                            int(osz[bi][0]),
                            int(osz[bi][1]),
                        )
                        in_sz_tuple = (
                            int(batch["input_size"][bi][0])
                            if "input_size" in batch else raw_sz_tuple[0],
                            int(batch["input_size"][bi][1])
                            if "input_size" in batch else raw_sz_tuple[1],
                        )
                        pm, pi, lo = predict_from_grid(
                            student,
                            imgs[bi],
                            point_coords[bi].to(dev),
                            raw_sz_tuple,
                            in_sz_tuple,
                            multimask_output=multimask_output_cfg,
                        )
                        # ---------------- Distillation (segment-everything) ----------------
                        if use_distillation:
                            feat_student = pop_features()

                            # Lazily initialise per-step distillation loss if first time here
                            # (they were set to 0 earlier already)

                            teacher_lowres_per_teacher = []  # store low-res logits
                            teacher_iou_per_teacher = []

                            for t_idx, t in enumerate(teacher_models):
                                # Teacher forward with hooks
                                with torch.no_grad():
                                    _pm_t, _pi_t, _lo_t = predict_from_grid(
                                        t["model"],
                                        imgs[bi],
                                        point_coords[bi].to(dev),
                                        raw_sz_tuple,
                                        in_sz_tuple,
                                        multimask_output=multimask_output_cfg,
                                    )
                                feat_t = pop_features()
                                teacher_lowres_per_teacher.append(_lo_t.detach())
                                teacher_iou_per_teacher.append(_pi_t.detach())

                                w_t = t["weight"]

                                # 1. Encoder patch tokens
                                if dist_cfg.get("encoder_patch", {}).get("enable"):
                                    l_s = pot["enc_patch"][0]
                                    l_t = teacher_pots[t["name"]]["enc_patch"][0]
                                    enc_cfg = dist_cfg["encoder_patch"]
                                    enc_loss_val = enc_loss_val + w_t * encoder_patch_loss(
                                        feat_student[l_s][0],
                                        feat_t[l_t][0],
                                        w_l2=enc_cfg.get("w_l2", 1.0),
                                        w_cos=enc_cfg.get("w_cos", 1.0),
                                    )

                                # 2. Prompt-conditioned embeddings
                                if dist_cfg.get("prompt_embed", {}).get("enable"):
                                    l_s = pot["prompt_embed"][0]
                                    l_t = teacher_pots[t["name"]]["prompt_embed"][0]
                                    pe_cfg = dist_cfg["prompt_embed"]
                                    prompt_loss_val = prompt_loss_val + w_t * prompt_embed_loss(
                                        feat_student[l_s][0],
                                        feat_t[l_t][0],
                                        w_mse=pe_cfg.get("w_mse", 0.7),
                                        w_cos=pe_cfg.get("w_cos", 0.3),
                                    )

                                # 3. Mask token logits
                                if dist_cfg.get("decoder_mask_token", {}).get("enable"):
                                    l_s = pot["mask_token"][0]
                                    l_t = teacher_pots[t["name"]]["mask_token"][0]
                                    tok_cfg = dist_cfg["decoder_mask_token"]
                                    tok_s = feat_student[l_s][0][:, 1 : 1 + student.mask_decoder.num_mask_tokens, :]
                                    tok_t = feat_t[l_t][0][:, 1 : 1 + student.mask_decoder.num_mask_tokens, :]
                                    tok_loss_val = tok_loss_val + w_t * mask_token_loss(
                                        tok_s,
                                        tok_t,
                                        w_kl=tok_cfg.get("w_kl", 1.0),
                                        temperature=tok_cfg.get("temperature", 0.5),
                                    )
                        pred_lowres_all.append(lo)
                        pred_ious_all.append(pi)

                        # teacher inference (no grad)
                        if mask_teacher_enable:
                            with torch.no_grad():
                                _pm_t, _pi_t, lo_t = predict_from_grid(
                                    mask_teacher_model,
                                    imgs[bi],
                                    point_coords[bi].to(dev),
                                    raw_sz_tuple,
                                    in_sz_tuple,
                                    multimask_output=multimask_output_cfg,
                                )
                            teacher_lowres_all.append(lo_t)
                        else:
                            teacher_lowres_all.append(None)

                    bce = torch.tensor(0.0, device=dev, requires_grad=True)
                    focal = torch.tensor(0.0, device=dev, requires_grad=True)
                    dice_loss = torch.tensor(0.0, device=dev, requires_grad=True)
                    iou_loss = torch.tensor(0.0, device=dev, requires_grad=True)
                    cls_total = 0
                    matched_cnt = 0
                    soft_loss_val = torch.tensor(0.0, device=dev, requires_grad=True)
                    matched_soft = 0

                    for bi in range(len(imgs)):
                        # ------------------- choose reference (GT or teacher) -------------------
                        original_gt = gt_masks[bi].to(dev)  # [K,1,S,S]
                        preds = pred_lowres_all[bi].reshape(-1, original_gt.shape[-2], original_gt.shape[-1])
                        ious_pred = pred_ious_all[bi].reshape(-1)

                        if (
                            mask_teacher_enable
                            and mask_teacher_method == "replace_gt"
                            and teacher_lowres_all[bi] is not None
                            and gt_override_enable
                        ):
                            # ─────────────────── Hybrid teacher / GT supervision ───────────────────
                            # 1) Use the teacher's soft mask as the default target.
                            # 2) If a teacher mask's IoU with any GT mask exceeds a threshold, replace it with that GT mask.
                            #    This ensures the corresponding point is on a GT object and maintains consistency.
                            teacher_pred_flat = teacher_lowres_all[bi].reshape(-1, original_gt.shape[-2], original_gt.shape[-1]).detach()

                            teacher_soft = torch.sigmoid(teacher_pred_flat)  # [K,S,S] in [0,1]

                            # Convert ground-truth to binary for IoU calculation
                            gt_bin_all = original_gt.squeeze(1)  # [G,S,S]

                            # Binarize teacher_soft in advance for IoU calculation
                            teacher_bin = (teacher_soft > 0.5).float()

                            # Create final reference masks (copy teacher, then conditionally overwrite)
                            ref_masks = teacher_soft.clone()  # float in [0,1], [K,S,S]

                            # Calculate IoU matrix (K x G)
                            inter_tg = (teacher_bin.unsqueeze(1) * gt_bin_all.unsqueeze(0)).sum((-2, -1))
                            union_tg = (
                                teacher_bin.unsqueeze(1).sum((-2, -1))
                                + gt_bin_all.unsqueeze(0).sum((-2, -1))
                                - inter_tg
                            )
                            iou_mat_tg = inter_tg / (union_tg + 1e-6)  # [K,G]

                            # For each teacher mask, check the best IoU; if it exceeds the threshold, replace with the corresponding GT.
                            for t_idx in range(iou_mat_tg.shape[0]):
                                best_iou_val, best_gt_idx = torch.max(iou_mat_tg[t_idx], dim=0)
                                if best_iou_val.item() >= mask_teacher_iou_thr:
                                    ref_masks[t_idx] = gt_bin_all[best_gt_idx]

                            # Use ref_masks as the supervision target
                            gt = ref_masks.unsqueeze(1)  # [K,1,S,S]
                            gt_flat = ref_masks  # [K,S,S]
                        elif (
                            mask_teacher_enable
                            and mask_teacher_method == "replace_gt"
                            and teacher_lowres_all[bi] is not None
                            and not gt_override_enable
                        ):
                            # Original replace_gt: use the teacher's mask completely (soft label)
                            teacher_pred_flat = teacher_lowres_all[bi].reshape(-1, original_gt.shape[-2], original_gt.shape[-1]).detach()
                            teacher_soft = torch.sigmoid(teacher_pred_flat)
                            gt = teacher_soft.unsqueeze(1)
                            gt_flat = teacher_soft
                        else:
                            gt = original_gt
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

                        THRESH_SE = 0.3  # segment-everything positive threshold
                        labels = torch.zeros(preds.shape[0], device=dev)
                        assigned_gt = torch.full(
                            (preds.shape[0],), -1, device=dev, dtype=torch.long
                        )
                        for r, c in zip(row_ind, col_ind):
                            labels[r] = 1.0 if iou_mat[r, c] >= THRESH_SE else 0.0
                            if iou_mat[r, c] >= THRESH_SE:
                                assigned_gt[r] = c
                                matched_cnt += 1

                        cls_loss_val = cls_loss_val + F.binary_cross_entropy_with_logits(
                            ious_pred, labels, reduction="sum"
                        )
                        cls_total += labels.numel()

                        for r, c in zip(row_ind, col_ind):
                            if iou_mat[r, c] < THRESH_SE:
                                continue
                            target = gt[c]
                            logit = preds[r].unsqueeze(0).unsqueeze(0)
                            target_exp = target.unsqueeze(0)
                            bce = bce + F.binary_cross_entropy_with_logits(logit, target_exp)
                            focal = focal + sigmoid_focal_loss(logit, target_exp, reduction="mean")
                            prob = torch.sigmoid(logit)
                            num = (prob * target_exp).sum((-2, -1)) * 2
                            den = prob.sum((-2, -1)) + target_exp.sum((-2, -1))
                            dice_loss = dice_loss + (1 - (num / (den + 1e-6)).mean())
                            iou_loss = iou_loss + F.mse_loss(torch.sigmoid(ious_pred[r]), iou_mat[r, c])

                            # -------- dense mask logits distillation (segment-everything) --------
                            if dist_cfg.get("dense_mask_logits", {}).get("enable"):
                                dl_cfg = dist_cfg["dense_mask_logits"]
                                stu_logit = preds[r].unsqueeze(0)  # 1xHlowxWlow

                                for t_idx_iter, t_iter in enumerate(teacher_models):
                                    t_lo_tensor = teacher_lowres_per_teacher[t_idx_iter][r].unsqueeze(0)
                                    w_tt = t_iter["weight"]
                                    _d_loss = dense_mask_logits_loss(
                                        stu_logit,
                                        t_lo_tensor,
                                        w_kl=dl_cfg.get("w_kl", 0.6),
                                        w_focal=dl_cfg.get("w_focal", 0.4),
                                        gamma=dl_cfg.get("gamma", 2.0),
                                    )
                                    spatial_norm = stu_logit.shape[-2] * stu_logit.shape[-1]
                                    dense_loss_val = dense_loss_val + w_tt * (_d_loss / spatial_norm)

                        # ---------- soft label supervision (teacher vs student logits) ----------
                        if mask_teacher_enable and mask_teacher_method == "dual_supervision" and teacher_lowres_all[bi] is not None:
                            teacher_pred_flat = teacher_lowres_all[bi].reshape(-1, preds.shape[-2], preds.shape[-1]).detach()
                            teacher_bin = (torch.sigmoid(teacher_pred_flat) > 0.5).float()
                            # Hungarian between student and teacher
                            inter_st = (pred_bin.unsqueeze(1) * teacher_bin.unsqueeze(0)).sum((-2, -1))
                            union_st = pred_bin.unsqueeze(1).sum((-2, -1)) + teacher_bin.unsqueeze(0).sum((-2, -1)) - inter_st
                            iou_mat_st = inter_st / (union_st + 1e-6)
                            cost_st = (-iou_mat_st).cpu().numpy()
                            row_t, col_t = linear_sum_assignment(cost_st)
                            for r_t, c_t in zip(row_t, col_t):
                                if iou_mat_st[r_t, c_t] < THRESH_SE:
                                    continue
                                s_logit = preds[r_t].unsqueeze(0)
                                t_logit = teacher_pred_flat[c_t].unsqueeze(0)
                                if soft_loss_type == "kl":
                                    s_prob = torch.sigmoid(s_logit)
                                    t_prob = torch.sigmoid(t_logit)
                                    soft_loss_val = soft_loss_val + F.kl_div(torch.log(s_prob + 1e-6), t_prob, reduction="batchmean")
                                else:
                                    soft_loss_val = soft_loss_val + F.mse_loss(s_logit, t_logit)
                                matched_soft += 1

                    # Average using only the matched predictions to avoid loss dilution.
                    n_matched = matched_cnt
                    cls_loss_val = cls_loss_val / max(1, cls_total)
                    if mask_teacher_enable and mask_teacher_method == "dual_supervision" and matched_soft > 0:
                        soft_loss_val = soft_loss_val / matched_soft
                    else:
                        soft_loss_val = torch.tensor(0.0, device=dev, requires_grad=True)
                    task_loss = (
                        w_bce * bce + w_focal * focal + w_dice * dice_loss + w_cls * cls_loss_val
                    )

                    if mask_teacher_enable and mask_teacher_method == "dual_supervision":
                        task_loss = task_loss + soft_loss_weight * soft_loss_val

                    dist_loss = enc_loss_val + prompt_loss_val + tok_loss_val + dense_loss_val

                    # Normalize dense logits loss by matched mask count to avoid explosion
                    if dist_cfg.get("dense_mask_logits", {}).get("enable") and n_matched > 0:
                        dense_loss_val = dense_loss_val / n_matched
                    dist_loss = enc_loss_val + prompt_loss_val + tok_loss_val + dense_loss_val

                    # The total loss must be calculated here for single-prompt mode.
                    # Otherwise, if the loss remains 0, gradients will be zero, and
                    # it won't display correctly in the tqdm progress bar.
                    loss = (
                        task_loss.float()
                        + w_iou * iou_loss.float()
                        + lambda_coef * dist_loss.float()
                    ) / tr_cfg.get("gradient_accumulation", 1)

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

                ga = tr_cfg.get("gradient_accumulation", 1)
                norm = 1 if dataset_mode == "single" else max(1, n_matched)

                tot_task += task_loss.item() / norm / ga
                tot_dist += dist_loss.item()
                tot_iou += iou_loss.item()
                tot_cls += cls_loss_val.item()

                bce_c = w_bce * bce.item() / norm / ga
                focal_c = w_focal * focal.item() / norm / ga
                dice_c = w_dice * dice_loss.item() / norm / ga
                iou_c = w_iou * iou_loss.item() / ga
                cls_c = w_cls * cls_loss_val.item() / norm / ga
                enc_c = lambda_coef * enc_loss_val.item() / ga
                pe_c  = lambda_coef * prompt_loss_val.item() / ga
                tok_c = lambda_coef * tok_loss_val.item() / ga
                dense_c = lambda_coef * dense_loss_val.item() / ga

                soft_c = (
                    soft_loss_weight * soft_loss_val.item() / ga
                    if mask_teacher_enable and mask_teacher_method == "dual_supervision"
                    else 0.0
                )

                # dist_c directly uses lambda_coef * dist_loss for checking; sub-components are not added again.
                dist_c = lambda_coef * dist_loss.item() / ga

                # Directly display the actual loss that was backpropagated to avoid inconsistency with gradients.
                total_c = loss.item()

                pbar.set_postfix(
                    bce=f"{bce_c:.3f}", focal=f"{focal_c:.3f}", dice=f"{dice_c:.3f}",
                    iou=f"{iou_c:.3f}", cls=f"{cls_c:.3f}",
                    enc=f"{enc_c:.3f}", pe=f"{pe_c:.3f}", tok=f"{tok_c:.3f}", dense=f"{dense_c:.3f}",
                    soft=f"{soft_c:.3f}", dist=f"{dist_c:.3f}", total=f"{total_c:.3f}",
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
            log.info(
                (
                    f"Epoch {ep} train: task={tot_task/len(tr_loader):.4f} "
                    f"dist={tot_dist/len(tr_loader):.4f} "
                    f"iou={tot_iou/len(tr_loader):.4f} "
                    f"cls={tot_cls/len(tr_loader):.4f}"
                )
            )
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
                            masks_orig = [m.to(dev) for m in vb["mask_original"]]

                            vinp = []
                            for i in range(len(imgs)):
                                entry = {
                                    "image": imgs[i],
                                    "original_size": (
                                        int(original_sizes[i][0]),
                                        int(original_sizes[i][1]),
                                    ),
                                }
                                if vb.get("box_prompt", [None]*len(imgs))[i] is not None:
                                    entry["boxes"] = vb["box_prompt"][i].to(dev).unsqueeze(0)
                                if vb.get("point_coords", [None]*len(imgs))[i] is not None:
                                    entry["point_coords"] = vb["point_coords"][i].to(dev).unsqueeze(0)
                                    entry["point_labels"] = vb["point_labels"][i].to(dev).unsqueeze(0)
                                if "input_size" in vb:
                                    entry["input_size"] = (
                                        int(vb["input_size"][i][0]),
                                        int(vb["input_size"][i][1]),
                                    )
                                vinp.append(entry)
                            vo = student(batched_input=vinp, multimask_output=multimask_output_cfg)

                            for i in range(len(imgs)):
                                out_i = vo[i]
                                preds_orig = out_i["masks"].to(torch.float32).squeeze(0)
                                iou_pred = out_i["iou_predictions"].squeeze(0).to(torch.float32)
                                best_idx = iou_pred.argmax()
                                prob_i = torch.sigmoid(preds_orig[best_idx]).unsqueeze(0)
                                gt_i = masks_orig[i]

                                # Soft Dice for loss logging
                                num_soft = (prob_i * gt_i).sum() * 2
                                den_soft = prob_i.sum() + gt_i.sum()
                                soft_dice = (num_soft / (den_soft + 1e-6)).item()

                                # Binary metrics
                                pred_bin = (prob_i >= 0.5).float()
                                num_bin = (pred_bin * gt_i).sum() * 2
                                den_bin = pred_bin.sum() + gt_i.sum()
                                bin_dice = (num_bin / (den_bin + 1e-6)).item()

                                union = pred_bin + gt_i - pred_bin * gt_i
                                bin_iou = ((pred_bin * gt_i).sum() / (union.sum() + 1e-6)).item()

                                dices.append(bin_dice)
                                ious.append(bin_iou)

                                # Print debug info for the first image
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
                                    # Denormalize to 0-1 range for visualization
                                    img_denorm = (imgs[i] / 255.0).clamp(0, 1).cpu()

                                    box = vb.get("box_prompt_raw", [None]*len(imgs))[i]
                                    pts = vb.get("point_coords_raw", [None]*len(imgs))[i]
                                    lbl = vb.get("point_labels_raw", [None]*len(imgs))[i]
                                    if box is not None:
                                        box = box.to(torch.float32)
                                    if pts is not None:
                                        pts = pts.to(torch.float32)
                                    if lbl is not None:
                                        lbl = lbl.to(torch.long)

                                    overlay_mask_on_image(
                                        image_tensor=img_denorm,
                                        mask_tensor=prob_i.squeeze(0).cpu(),
                                        bbox_tensor=box.cpu() if box is not None else None,
                                        point_coords=pts.cpu() if pts is not None else None,
                                        point_labels=lbl.cpu() if lbl is not None else None,
                                        original_size=(
                                            int(original_sizes[i][0]),
                                            int(original_sizes[i][1]),
                                        ),
                                        threshold=cfg["visual"].get("IOU_threshold", 0.5),
                                        save_dir=str(cur_path),
                                        filename_info=(f"ep{ep}_id{vb['id'][i]}_b{bi}_s{i}"),
                                    )
                        else:
                            # Segment-everything validation
                            gt_masks = vb["gt_masks_original"]
                            point_coords = vb["point_coords"]

                            for i in range(len(imgs)):
                                raw_sz_tuple = (
                                    int(original_sizes[i][0]),
                                    int(original_sizes[i][1]),
                                )
                                in_sz_tuple = (
                                    int(vb["input_size"][i][0])
                                    if "input_size" in vb else raw_sz_tuple[0],
                                    int(vb["input_size"][i][1])
                                    if "input_size" in vb else raw_sz_tuple[1],
                                )
                                pm, _, _ = predict_from_grid(
                                    student,
                                    imgs[i],
                                    point_coords[i].to(dev),
                                    raw_sz_tuple,
                                    in_sz_tuple,
                                    multimask_output=multimask_output_cfg,
                                )

                                preds = pm  # already BxHxW, where B = num prompts * 3
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
                                    # Denormalize to 0-1 range for visualization
                                    img_denorm = (imgs[i] / 255.0).clamp(0, 1).cpu()
                                    best_masks = probs[row_ind].cpu()
                                    overlay_masks_on_image(
                                        image_tensor=img_denorm,
                                        masks=best_masks,
                                        original_size=(
                                            int(original_sizes[i][0]),
                                            int(original_sizes[i][1]),
                                        ),
                                        grid_points=vb.get("point_coords_raw", point_coords)[i].cpu(),
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

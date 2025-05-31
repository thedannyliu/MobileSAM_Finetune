import argparse, json, pathlib, logging, traceback, os, functools, gc
import numpy as np
import torch, torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.ops import sigmoid_focal_loss
from tqdm import tqdm
from torchvision import transforms as T

from mobile_sam import sam_model_registry
from finetune_utils.datasets import ComponentDataset
from finetune_utils.distill_losses import (
    encoder_matching_loss, decoder_matching_loss,
    attention_matching_loss, rkd_loss
)
from finetune_utils.feature_hooks import register_hooks, pop_features
from finetune_utils.visualization import overlay_mask_on_image

# ───────────────────────────── logging ──────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("train")

# ───────────────────────────── helpers ──────────────────────────────
def _parse_hw(x):  # (H,W)
    return (int(x[0]), int(x[1])) if isinstance(x, torch.Tensor) else tuple(map(int, x))

def log_gpu_memory(step_name=""):
    """記錄GPU記憶體使用情況"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        cached = torch.cuda.memory_reserved() / 1024**3
        log.info(f"{step_name} GPU Memory - Allocated: {allocated:.2f}GB, Cached: {cached:.2f}GB")

def clear_gpu_cache():
    """清理GPU快取"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()

class WarmupCosineLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, opt, warmup, total, min_ratio=0.0, last_epoch=-1):
        self.warmup, self.total, self.min_ratio = warmup, total, min_ratio
        super().__init__(opt, last_epoch)

    def get_lr(self):
        cur = self.last_epoch + 1
        if cur < self.warmup:
            return [b * cur / self.warmup for b in self.base_lrs]
        prog = (cur - self.warmup) / max(1, (self.total - self.warmup))
        cos = 0.5 * (1 + np.cos(np.pi * prog))
        return [b * (self.min_ratio + (1 - self.min_ratio) * cos) for b in self.base_lrs]

# ─── feature cache (修正版LRU，定期清理) ───
class MemoryEfficientFeatureCache:
    """記憶體友善的特徵快取系統"""
    def __init__(self, maxsize=64):  # 降低快取大小
        self.cache = {}
        self.maxsize = maxsize
        self.access_order = []
    
    def get(self, path: pathlib.Path):
        key = str(path)
        if key in self.cache:
            # 更新訪問順序
            self.access_order.remove(key)
            self.access_order.append(key)
            return self.cache[key]
        
        # 載入新特徵
        arr = np.load(key)
        tensor = torch.from_numpy(arr).cuda(non_blocking=True)
        
        # 檢查快取大小
        if len(self.cache) >= self.maxsize:
            # 移除最舊的項目
            oldest = self.access_order.pop(0)
            del self.cache[oldest]
        
        self.cache[key] = tensor
        self.access_order.append(key)
        return tensor
    
    def clear(self):
        """清空快取"""
        self.cache.clear()
        self.access_order.clear()
        clear_gpu_cache()

# 全域快取實例
feature_cache = MemoryEfficientFeatureCache()

def load_cached_npy_features(base: pathlib.Path, teacher: str, split: str,
                             stems: list[str], keys: list[str]):
    feats = []
    for stem in stems:
        this_img = []
        for k in keys:
            fname = f"{stem}_{k.replace('.','_').replace('[','_').replace(']','')}.npy"
            this_img.append(feature_cache.get(base/teacher/split/fname))
        feats.append(torch.stack(this_img))
    # (B,K,C,H,W) or (B,K,D)
    return [torch.stack([feats[b, i] for b in range(len(stems))]) for i in range(len(keys))]

# ─── custom collate_fn ───
def sam_collate(batch):
    """
    • Tensor 型別 → 直接 stack
    • 允許 None / list 保留原樣
    """
    elem = batch[0]
    out = {}
    for k in elem.keys():
        vals = [d[k] for d in batch]
        if isinstance(vals[0], torch.Tensor) and all(v is not None for v in vals):
            out[k] = torch.stack(vals, 0)
        else:
            out[k] = vals  # list / None
    return out

# ───────────────────────────── main ──────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    cfg = json.load(open(parser.parse_args().config))
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    # 初始記憶體狀態
    log_gpu_memory("Initial")

    # ─── transforms (單一 Resize 已在 dataset 完成；此處只 Normalize) ───
    MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    tf_img = T.Compose([T.ToTensor(), T.Normalize(MEAN, STD)])
    tf_msk = T.Compose([T.ToTensor()])

    # ─── dataset / dataloader ───
    ds_cfg = cfg["dataset"]
    train_ds = ComponentDataset(ds_cfg["train_dataset"], (tf_img, tf_msk),
                                max_bbox_shift=ds_cfg.get("max_bbox_shift", 20),
                                prompt_mode=ds_cfg.get("prompt_mode", "mixed"),
                                min_points=ds_cfg.get("min_points", 1),
                                max_points=ds_cfg.get("max_points", 3),
                                image_size=cfg["model"].get("image_size", 1024))
    val_ds = ComponentDataset(ds_cfg["val_dataset"], (tf_img, tf_msk),
                              max_bbox_shift=ds_cfg.get("max_bbox_shift", 20),
                              prompt_mode=ds_cfg.get("prompt_mode", "mixed"),
                              min_points=ds_cfg.get("min_points", 1),
                              max_points=ds_cfg.get("max_points", 3),
                              image_size=cfg["model"].get("image_size", 1024))

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

    # ─── model ───
    m_cfg = cfg["model"]
    student = sam_model_registry[m_cfg.get("type", "vit_t")](
        checkpoint=m_cfg.get("checkpoint_path")
    ).to(dev)

    # 初始凍結：image_encoder
    if cfg["freeze"].get("freeze_image_encoder", True):
        student.image_encoder.requires_grad_(False)

    if cfg["freeze"].get("freeze_prompt_encoder", False):
        student.prompt_encoder.requires_grad_(False)
    if cfg["freeze"].get("freeze_mask_decoder", False):
        student.mask_decoder.requires_grad_(False)

    # hooks for distillation - 修正版
    dist_cfg = cfg.get("distillation", {})
    use_distillation = dist_cfg.get("enable", False)
    hook_handles = []
    
    if use_distillation:
        stype = m_cfg.get("type", "vit_t")
        pot = {"enc": [], "dec": [], "attn": [], "rkd": ["image_encoder.patch_embed"]}
        if stype == "vit_t":
            pot["enc"] = ["image_encoder.neck"]
        else:
            pot["enc"] = [f"image_encoder.blocks.{i}" for i in (9, 10, 11, 12)]
            pot["dec"] = ["mask_decoder.pre_logits"]
            pot["attn"] = [f"image_encoder.blocks.{i}.attn" for i in range(12)]

        hook_layers = []
        for n, k in (
            ("encoder_matching", "enc"),
            ("decoder_matching", "dec"),
            ("attention_matching", "attn"),
            ("relational_KD", "rkd"),
        ):
            if dist_cfg.get(n, {}).get("enable"):
                hook_layers += pot[k]
        hook_layers = sorted(set(hook_layers))
        
        if hook_layers:
            hook_handles = register_hooks(student, hook_layers)
            log.info(f"Registered hooks for distillation: {hook_layers}")
    else:
        log.info("Distillation disabled - no hooks registered")

    log_gpu_memory("After model loading")

    # ─── optimizer (param groups) ───
    tr_cfg = cfg["train"]
    enc_params, others = [], []
    for n, p in student.named_parameters():
        if not p.requires_grad:
            continue
        if n.startswith("image_encoder"):
            enc_params.append(p)
        else:
            others.append(p)

    opt = torch.optim.AdamW(
        [
            {"params": others, "lr": tr_cfg["lr"]},
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
    writer = SummaryWriter(pathlib.Path(m_cfg.get("save_path", "logs")) / "tb")

    # ─── dynamic λ & early-stop setting ───
    lambda_coef = 1.0
    dyn_wait = 0
    best_score, stop_counter = -1, 0
    patience = tr_cfg.get("early_stop_patience", 20)

    # 逐 epoch 解凍 image_encoder
    unfreeze_epoch = cfg["freeze"].get("unfreeze_epoch", 10)

    # teacher weighting
    teacher_cfgs = cfg.get("teachers", [])
    if teacher_cfgs:
        for t in teacher_cfgs:
            t.setdefault("weight", 1.0 / len(teacher_cfgs))

    # ─── training loop ───
    grad_acc = tr_cfg.get("gradient_accumulation", 1)
    val_freq = tr_cfg.get("val_freq", 1)
    save_vis_cfg = cfg.get("visual", {})
    vis_base = pathlib.Path(save_vis_cfg.get("save_path", "images"))
    save_vis_base_n = save_vis_cfg.get("save_every_n_epochs", 10)

    # 記憶體監控頻率
    memory_log_freq = 100  # 每100步記錄一次

    try:
        global_step = 0
        for ep in range(tr_cfg["epochs"]):
            # --- 解凍策略 ---
            if ep == unfreeze_epoch:
                log.info(f"Unfreeze image_encoder at epoch {ep}")
                student.image_encoder.requires_grad_(True)
                # 將 lr group 提高
                opt.param_groups[1]["lr"] = tr_cfg["lr"] * 0.2

            # 每個epoch開始時清理快取
            if ep > 0:
                feature_cache.clear()
                clear_gpu_cache()
                log_gpu_memory(f"Epoch {ep} start (after cache clear)")

            student.train()
            tot_task, tot_dist = 0.0, 0.0
            pbar = tqdm(tr_loader, desc=f"Train {ep}")
            opt.zero_grad()

            for step, batch in enumerate(pbar):
                try:
                    imgs = batch["image"].to(dev, non_blocking=True)
                    masks = batch["mask"].to(dev, non_blocking=True)
                    ids = batch["id"]
                    osz = batch["original_size"]

                    # 構建 batched_input
                    batched_input = []
                    for i in range(len(imgs)):
                        entry = {
                            "image": imgs[i],
                            "original_size": _parse_hw(osz[i]),
                        }
                        if batch["box_prompt"][i] is not None:
                            entry["boxes"] = batch["box_prompt"][i].to(dev, non_blocking=True)
                        if batch["point_coords"][i] is not None:
                            entry["point_coords"] = batch["point_coords"][i].to(dev, non_blocking=True)
                            entry["point_labels"] = batch["point_labels"][i].to(dev, non_blocking=True)
                        batched_input.append(entry)

                    with autocast(dtype=torch.bfloat16 if tr_cfg.get("bf16", False) else torch.float16):
                        out = student(batched_input=batched_input, multimask_output=False)
                        logit = torch.stack([o["low_res_logits"] for o in out]).squeeze(1)
                        logit_up = F.interpolate(logit, size=masks.shape[-2:], mode="bilinear", align_corners=False)

                        focal = sigmoid_focal_loss(logit_up, masks, reduction="mean")
                        # safe dice:
                        prob = torch.sigmoid(logit_up)
                        num  = (prob * masks).sum((-2, -1)) * 2
                        den  = prob.sum((-2, -1)) + masks.sum((-2, -1))
                        dice = 1 - (num / (den + 1e-6)).mean()
                        task_loss = 2.0 * focal + dice

                        # ─ distillation (只在啟用時執行) ─
                        dist_loss = torch.tensor(0., device=dev, requires_grad=False)
                        if use_distillation and hook_handles:
                            feat_student = pop_features() or {}
                            base_dir = pathlib.Path(dist_cfg["precomputed_root"])

                            for t_cfg in teacher_cfgs:
                                weight = t_cfg["weight"]
                                tname = t_cfg["name"]
                                
                                # encoder
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

                                # decoder
                                if dist_cfg.get("decoder_matching", {}).get("enable") and pot["dec"]:
                                    dk = pot["dec"][0]
                                    if dk in feat_student:
                                        try:
                                            feat_teacher = load_cached_npy_features(base_dir, tname, "train", ids, [dk])[0]
                                            dec_loss = decoder_matching_loss(
                                                feat_student[dk],
                                                feat_teacher,
                                                **dist_cfg["decoder_matching"],
                                            )
                                            dist_loss = dist_loss + weight * dec_loss
                                        except Exception as e:
                                            log.debug(f"Decoder matching error: {e}")

                                # attention
                                if dist_cfg.get("attention_matching", {}).get("enable") and pot["attn"]:
                                    try:
                                        attn_teacher = load_cached_npy_features(base_dir, tname, "train", ids, pot["attn"])
                                        attn_loss = attention_matching_loss(
                                            [feat_student[k] for k in pot["attn"]],
                                            attn_teacher,
                                            **dist_cfg["attention_matching"],
                                            n_layers=len(pot["attn"]),
                                        )
                                        dist_loss = dist_loss + weight * attn_loss
                                    except Exception as e:
                                        log.debug(f"Attention matching error: {e}")

                                # RKD
                                if dist_cfg.get("relational_KD", {}).get("enable"):
                                    rk = pot["rkd"][0]
                                    if rk in feat_student:
                                        try:
                                            feat_teacher = load_cached_npy_features(base_dir, tname, "train", ids, [rk])[0]
                                            rkd_loss_val = rkd_loss(
                                                feat_student[rk],
                                                feat_teacher,
                                                **dist_cfg["relational_KD"],
                                            )
                                            dist_loss = dist_loss + weight * rkd_loss_val
                                        except Exception as e:
                                            log.debug(f"RKD error: {e}")

                        loss = task_loss + lambda_coef * dist_loss
                        loss = loss / grad_acc  # 梯度累積平均

                    # 反向傳播
                    scaler.scale(loss).backward()
                    
                    # 明確釋放中間變量
                    del logit, logit_up, prob, focal, dice
                    if use_distillation and 'feat_student' in locals():
                        del feat_student
                    
                    # 梯度累積步
                    if (step + 1) % grad_acc == 0 or (step + 1) == len(tr_loader):
                        # gradient clipping
                        scaler.unscale_(opt)
                        torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)

                        scaler.step(opt)
                        scaler.update()
                        opt.zero_grad()
                        global_step += 1
                        scheduler.step()
                        
                        # 強制清理梯度累積
                        torch.cuda.empty_cache()

                    tot_task += task_loss.item()
                    tot_dist += dist_loss.item()
                    total_loss += loss.item()
                    
                    pbar.set_postfix(
                        task_loss=f"{task_loss.item():.3f}",
                        dist_loss=f"{dist_loss.item():.3f}",
                        total_loss=f"{loss.item():.3f}",
                        λ=f"{lambda_coef:.2f}",
                        lr=f"{scheduler.get_last_lr()[0]:.2e}",
                    )

                    # 定期記憶體監控
                    if step % memory_log_freq == 0:
                        log_gpu_memory(f"Epoch {ep}, Step {step}")

                except Exception as e:
                    log.error(f"Error in training step {step}: {e}")
                    # 緊急記憶體清理
                    clear_gpu_cache()
                    raise

            # ─── log train epoch ───
            writer.add_scalar("train/task_loss", tot_task / len(tr_loader), ep)
            writer.add_scalar("train/dist_loss", tot_dist / len(tr_loader), ep)
            writer.add_scalar("train/total_loss", total_loss / len(tr_loader), ep)
            writer.add_scalar("train/lambda", lambda_coef, ep)
            for i, g in enumerate(opt.param_groups):
                writer.add_scalar(f"lr/group{i}", g["lr"], ep)

            log_gpu_memory(f"Epoch {ep} training completed")

            # ─── validation ───
            if (ep + 1) % val_freq != 0:
                continue

            student.eval()
            dices, ious = [], []
            with torch.no_grad():
                for bi, vb in enumerate(va_loader):
                    try:
                        imgs = vb["image"].to(dev, non_blocking=True)
                        masks = vb["mask"].to(dev, non_blocking=True)
                        vinp = [
                            {"image": imgs[i], "original_size": _parse_hw(vb["original_size"][i])}
                            for i in range(len(imgs))
                        ]
                        vo = student(batched_input=vinp, multimask_output=False)
                        vl = torch.stack([o["low_res_logits"] for o in vo]).squeeze(1)
                        vl_up = F.interpolate(vl, size=masks.shape[-2:], mode="bilinear", align_corners=False)
                        probs = torch.sigmoid(vl_up)

                        num  = (probs * masks).sum((-2, -1)) * 2
                        den  = probs.sum((-2, -1)) + masks.sum((-2, -1))
                        dice = (num / (den + 1e-6)).mean().item()

                        pred = (probs >= 0.5).float()
                        union = pred + masks - pred * masks
                        iou   = ((pred * masks).sum((-2, -1)) /
                                (union.sum((-2, -1)) + 1e-6)).mean().item()
                        
                        dices.append(dice)
                        ious.append(iou)

                        # 清理validation變量
                        del vl, vl_up, probs, pred, union
                        
                        # save visualization 只有最佳時 & 每 N epoch
                        if (
                            bi == 0
                            and save_vis_cfg.get("status", False)
                            and (ep % save_vis_base_n == 0 or (dice + iou) / 2 > best_score)
                        ):
                            cur_path = vis_base / f"epoch_{ep}"
                            cur_path.mkdir(parents=True, exist_ok=True)
                            for i in range(min(3, len(imgs))):
                                img_denorm = imgs[i] * torch.tensor(STD, device=dev)[:, None, None] + torch.tensor(
                                    MEAN, device=dev
                                )[:, None, None]
                                overlay_mask_on_image(
                                    img_denorm.cpu(),
                                    torch.sigmoid(torch.stack([o["low_res_logits"] for o in vo]).squeeze(1)[i]).cpu(),
                                    None,
                                    threshold=save_vis_cfg.get("IOU_threshold", 0.5),
                                    save_dir=str(cur_path),
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
            log.info(f"Epoch {ep}  Dice={v_dice:.4f} IoU={v_iou:.4f} Score={v_score:.4f}")

            log_gpu_memory(f"Epoch {ep} validation completed")

            # ─── dynamic λ (plateau + 回升) ───
            if use_distillation and dist_cfg.get("dynamic_lambda", {}).get("enable_plateau_scheduler"):
                if v_score > best_score + 1e-4:
                    dyn_wait = 0
                    # 若 λ 曾下降且現在表現回升，緩慢提高 λ
                    lambda_coef = min(lambda_coef / dist_cfg["dynamic_lambda"]["factor"], 1.0)
                else:
                    dyn_wait += 1
                    if dyn_wait >= dist_cfg["dynamic_lambda"]["patience"]:
                        lambda_coef = max(lambda_coef * dist_cfg["dynamic_lambda"]["factor"], 1e-3)
                        dyn_wait = 0
                        log.info(f"λ ↓ {lambda_coef:.3f}")

            # ─── early-stop / checkpoint ───
            if v_score > best_score:
                best_score = v_score
                stop_counter = 0
                out_dir = pathlib.Path(m_cfg.get("save_path", "./"))
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
        # 清理資源
        for h in hook_handles:
            h.remove()
        feature_cache.clear()
        writer.close()
        clear_gpu_cache()
        log_gpu_memory("Final cleanup completed")


if __name__ == "__main__":
    main()
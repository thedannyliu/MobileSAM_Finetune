from finetune_utils.datasets import ComponentDataset
from torchvision import transforms as T
import argparse, json, pathlib, os, numpy as np, torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from mobile_sam import sam_model_registry
from finetune_utils.distill_losses import (
    encoder_matching_loss, decoder_matching_loss,
    attention_matching_loss, rkd_loss
)
from finetune_utils.feature_hooks import register_hooks, pop_features


def load_cached_npz(root: pathlib.Path, teacher: str, ids: list[str], keys: list[str]):
    res = [[] for _ in keys]
    for img_id in ids:
        arr = np.load(root / teacher / f"{img_id}.npz")
        for i, k in enumerate(keys):
            res[i].append(torch.from_numpy(arr[k]).cuda(non_blocking=True))
    return [torch.stack(v) for v in res]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    cfg = json.load(open(args.config))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---------- Dataset ----------

    ds_cfg = cfg["dataset"]
    train_root = ds_cfg["train_dataset"]   # "./datasets/train"
    val_root   = ds_cfg["val_dataset"]     # "./datasets/val"

    # 基本影像 / mask 轉換
    img_tfms  = T.ToTensor()
    mask_tfms = T.ToTensor()

    train_set = ComponentDataset(
        root_dir=train_root,
        transform=(img_tfms, mask_tfms),
        max_bbox_shift = ds_cfg.get("max_bbox_shift", 20),
        prompt_mode    = ds_cfg.get("prompt_mode", "box"),
        min_points     = ds_cfg.get("min_points", 1),
        max_points     = ds_cfg.get("max_points", 3),
        image_size     = cfg["model"].get("image_size", 1024)
    )

    val_set   = ComponentDataset(
        root_dir=val_root,
        transform=(img_tfms, mask_tfms),
        max_bbox_shift = ds_cfg.get("max_bbox_shift", 20),
        prompt_mode    = ds_cfg.get("prompt_mode", "box"),
        min_points     = ds_cfg.get("min_points", 1),
        max_points     = ds_cfg.get("max_points", 3),
        image_size     = cfg["model"].get("image_size", 1024)
    )

    train_loader = DataLoader(train_set, batch_size=cfg["train"]["batch_size"],
                            shuffle=True, num_workers=ds_cfg.get("num_workers",4),
                            pin_memory=True)
    val_loader   = DataLoader(val_set,   batch_size=cfg["train"]["batch_size"],
                            shuffle=False, num_workers=ds_cfg.get("num_workers",4),
                            pin_memory=True)


    # ---------- Student ----------
    student_config_data = cfg.get("model") # 從載入的 JSON 設定檔中獲取 "model" 部分
    if student_config_data is None:
        raise ValueError("Model configuration ('model') not found in the config file.")

    # 從學生模型的設定中獲取類型和初始權重路徑
    # 預設學生模型為 'vit_t' (MobileSAM)，如果設定檔中沒有明確指定
    student_model_type = student_config_data.get("type", "vit_t")
    # 學生模型的初始權重，可以為 None (從頭訓練或由建構函數處理預設權重)
    student_initial_checkpoint = student_config_data.get("checkpoint", None)

    # 從 sam_model_registry 中獲取對應的模型建構函數
    if student_model_type not in sam_model_registry:
        raise ValueError(f"Student model type '{student_model_type}' not found in sam_model_registry. "
                         f"Available types: {list(sam_model_registry.keys())}")
    
    model_builder_func = sam_model_registry[student_model_type]

    print(f"Building student model of type '{student_model_type}' with initial checkpoint: {student_initial_checkpoint}")
    # 使用獲取到的建構函數和權重路徑來建立學生模型
    student = model_builder_func(checkpoint=student_initial_checkpoint).to(device)
    
    optimizer = torch.optim.AdamW(student.parameters(), lr=cfg["train"]["lr"], weight_decay=1e-4)

    # ---------- Hooks ----------
    enc_layers = [f"image_encoder.blocks.{i}" for i in (9,10,11,12)]
    dec_layer  = ["mask_decoder.pre_logits"]
    attn_layers= [f"image_encoder.blocks.{i}.attn" for i in range(12)]
    rkd_layer  = ["image_encoder.patch_embed"]
    hook_names = enc_layers + dec_layer + attn_layers + rkd_layer
    hook_handles = register_hooks(student, hook_names)

    # ---------- Early Stopping ----------
    best_metric = -1
    patience = cfg["train"].get("early_stop_patience", 0)
    stop_counter = 0

    for epoch in range(cfg["train"]["epochs"]):
        student.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        for batch in pbar:
            imgs, masks, ids = batch["image"].to(device), batch["mask"].to(device), batch["id"]
            optimizer.zero_grad()
            pred = student(imgs)
            # --- task loss (Dice + BCE) ---
            bce = F.binary_cross_entropy_with_logits(pred, masks)
            inter = (torch.sigmoid(pred) * masks).sum([2,3]) * 2
            union = torch.sigmoid(pred).sum([2,3]) + masks.sum([2,3]) + 1e-6
            dice = 1 - (inter / union).mean()
            task_loss = 20 * bce + dice

            # --- distillation ---
            dist_loss = 0.0
            if cfg["distillation"]["enable"]:
                feats_s = pop_features()
                root = pathlib.Path(cfg["distillation"]["precomputed_root"])
                # 以第一位教師 (SAM_vitH) 為示範
                te_name = cfg["teachers"][0]["name"]
                # enc
                if cfg["distillation"]["encoder_matching"]["enable"]:
                    te_enc = load_cached_npz(root, te_name, ids, enc_layers)
                    dist_loss += encoder_matching_loss(
                        [feats_s[l] for l in enc_layers], te_enc,
                        **cfg["distillation"]["encoder_matching"]
                    )
                # dec
                if cfg["distillation"]["decoder_matching"]["enable"]:
                    te_dec = load_cached_npz(root, te_name, ids, dec_layer)[0]
                    dist_loss += decoder_matching_loss(
                        feats_s[dec_layer[0]], te_dec,
                        **cfg["distillation"]["decoder_matching"]
                    )
                # attn
                if cfg["distillation"]["attention_matching"]["enable"]:
                    te_attn = load_cached_npz(root, te_name, ids, attn_layers)
                    dist_loss += attention_matching_loss(
                        [feats_s[l] for l in attn_layers], te_attn,
                        **cfg["distillation"]["attention_matching"]
                    )
                # rkd
                if cfg["distillation"]["relational_KD"]["enable"]:
                    te_rkd = load_cached_npz(root, te_name, ids, rkd_layer)[0]
                    dist_loss += rkd_loss(
                        feats_s[rkd_layer[0]], te_rkd,
                        **cfg["distillation"]["relational_KD"]
                    )
            total_loss = task_loss + dist_loss
            total_loss.backward()
            optimizer.step()
            pbar.set_postfix({"task": task_loss.item(), "dist": dist_loss.item(), "tot": total_loss.item()})

        # ---------- validation ----------
        student.eval(); dices = []
        with torch.no_grad():
            for batch in val_loader:
                img, msk = batch["image"].to(device), batch["mask"].to(device)
                pr = student(img)
                inter = (torch.sigmoid(pr) * msk).sum([2,3]) * 2
                union = torch.sigmoid(pr).sum([2,3]) + msk.sum([2,3]) + 1e-6
                dices.append((inter/union).mean().item())
        val_dice = float(np.mean(dices))
        print(f"Epoch {epoch}  ▸  val Dice={val_dice:.4f}")

        # ---------- early‑stop monitor ----------
        if patience > 0:
            if val_dice > best_metric:
                best_metric = val_dice; stop_counter = 0
                torch.save(student.state_dict(), "best_student.pth")
            else:
                stop_counter += 1
                if stop_counter >= patience:
                    print(f"Early stopping triggered ▸ best Dice={best_metric:.4f}")
                    break

    for h in hook_handles:
        h.remove()

if __name__ == "__main__":
    main()
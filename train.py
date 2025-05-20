# train.py
import os
import sys
import argparse
import datetime
import time
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import GradScaler, autocast # 用於混合精度訓練

# 載入設定、日誌、儲存點等輔助工具
from finetune_utils.load_config import load_config, CfgNode as CN # 引入 CfgNode
from finetune_utils.load_logger import get_logger
from finetune_utils.save_checkpoint import save_checkpoint
from finetune_utils.schedular import build_scheduler # 學習率排程器

# 載入模型、資料集、損失函數
from mobile_sam import sam_model_registry
from finetune_utils.datasets import DatasetFinetune # 確保這是更新後的版本
from finetune_utils.loss import FocalLoss, DiceLoss, IoULoss, FeatureDistillationLoss # 引入 FeatureDistillationLoss

# Collate function - 使用 PyTorch 預設的即可，除非有特殊需求
# from torch.utils.data.dataloader import default_collate

def parse_option():
    parser = argparse.ArgumentParser(description="MobileSAM 微調腳本")
    parser.add_argument('--config', type=str, required=True, help="設定檔的路徑 (例如 configs/mobileSAM.json)")
    # 可以保留其他的 parser arguments，如果您的原始 train.py 有的話
    args = parser.parse_args()
    config = load_config(args.config) # load_config 應該返回 CfgNode 物件
    return config

def get_prompt_points_from_bboxes(bboxes_batch, device):
    """
    從邊界框批次中獲取中心點作為提示。
    bboxes_batch: (B, N_boxes, 4)，格式為 [x1, y1, x2, y2]
    返回:
        point_coords_batch: (B, N_boxes, 1, 2) - 中心點座標
        point_labels_batch: (B, N_boxes, 1) - 標籤 (前景點為1)
    """
    if bboxes_batch is None or bboxes_batch.numel() == 0:
        return None, None

    # 計算中心點: cx = (x1+x2)/2, cy = (y1+y2)/2
    # bboxes_batch shape: (B, N_boxes_per_image, 4)
    center_x = (bboxes_batch[..., 0] + bboxes_batch[..., 2]) / 2
    center_y = (bboxes_batch[..., 1] + bboxes_batch[..., 3]) / 2
    
    # 堆疊中心點座標
    # (B, N_boxes_per_image, 2)
    point_coords = torch.stack([center_x, center_y], dim=-1)
    
    # SAM 期望的點座標格式是 (B, N_prompts, N_points_per_prompt, 2)
    # 這裡我們每個 bbox 生成一個點提示，所以 N_points_per_prompt = 1
    point_coords_batch = point_coords.unsqueeze(2) # (B, N_boxes_per_image, 1, 2)
    
    # 產生對應的標籤 (前景點為 1)
    # (B, N_boxes_per_image, 1)
    point_labels_batch = torch.ones_like(point_coords_batch[..., 0], dtype=torch.int, device=device)
    
    return point_coords_batch, point_labels_batch

def main(config):
    # --- 設定 GPU 裝置 ---
    if torch.cuda.is_available():
        device = torch.device(config.TRAIN.DEVICE if hasattr(config.TRAIN, 'DEVICE') else 'cuda')
    elif torch.backends.mps.is_available(): # 針對 Apple Silicon
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f"使用裝置: {device}")

    # --- 建立輸出目錄和日誌記錄器 ---
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    logger = get_logger(config.OUTPUT_DIR, config.MODEL.NAME)
    writer = SummaryWriter(log_dir=os.path.join(config.OUTPUT_DIR, 'tensorboard_logs'))
    logger.info(f"設定檔內容:\n{config}")

    # --- 載入模型 ---
    logger.info(f"載入模型: {config.MODEL.TYPE}")
    # 確保模型權重路徑正確
    if not os.path.exists(config.MODEL.CHECKPOINT):
        logger.error(f"找不到模型權重檔案: {config.MODEL.CHECKPOINT}")
        # 嘗試從相對路徑 (相對於設定檔的目錄) 尋找
        config_dir = os.path.dirname(config.CONFIG_PATH) # 假設 config 物件有 CONFIG_PATH 屬性
        alt_checkpoint_path = os.path.join(config_dir, config.MODEL.CHECKPOINT)
        if os.path.exists(alt_checkpoint_path):
            logger.info(f"嘗試使用相對路徑的權重檔案: {alt_checkpoint_path}")
            config.defrost() # 允許修改設定
            config.MODEL.CHECKPOINT = alt_checkpoint_path
            config.freeze()
        else:
            logger.error(f"也找不到相對路徑的權重檔案: {alt_checkpoint_path}。請檢查 MODEL.CHECKPOINT 設定。")
            sys.exit(1)

    model_finetune = sam_model_registry[config.MODEL.TYPE](checkpoint=config.MODEL.CHECKPOINT)
    model_finetune.to(device)
    logger.info("模型載入完成")

    # --- 設定優化器 ---
    logger.info(f"設定優化器: {config.TRAIN.OPTIMIZER.NAME}")
    if config.TRAIN.OPTIMIZER.NAME.lower() == 'adamw':
        optimizer = torch.optim.AdamW(
            model_finetune.parameters(),
            lr=config.TRAIN.OPTIMIZER.LR,
            weight_decay=config.TRAIN.OPTIMIZER.WEIGHT_DECAY
        )
    elif config.TRAIN.OPTIMIZER.NAME.lower() == 'adam':
        optimizer = torch.optim.Adam(
            model_finetune.parameters(),
            lr=config.TRAIN.OPTIMIZER.LR,
            weight_decay=config.TRAIN.OPTIMIZER.WEIGHT_DECAY
        )
    else:
        logger.error(f"不支援的優化器: {config.TRAIN.OPTIMIZER.NAME}")
        sys.exit(1)
    logger.info(f"優化器 LR: {config.TRAIN.OPTIMIZER.LR}, Weight Decay: {config.TRAIN.OPTIMIZER.WEIGHT_DECAY}")

    # --- 梯度縮放器 (用於混合精度訓練) ---
    use_amp = getattr(config.TRAIN, 'AMP', False) # 預設為 False 如果沒設定
    scaler = GradScaler() if use_amp else None
    logger.info(f"使用混合精度訓練 (AMP): {use_amp}")

    # --- 設定損失函數 ---
    criterion = {} # 用字典儲存多個損失函數
    # 預設分割損失
    criterion['focal'] = FocalLoss()
    criterion['iou'] = IoULoss()
    # criterion['dice'] = DiceLoss() # 如果您也想用 Dice Loss

    # 新增：蒸餾損失函數
    # 從設定檔讀取蒸餾相關參數 (如果不存在，則設定預設值來自 CfgNode 的預設)
    # 確保 DISTILLATION 區塊存在於 config 中 (已在 load_config.py 中定義預設)
    distill_loss_type = config.DISTILLATION.LOSS_TYPE
    distill_kl_temp = config.DISTILLATION.KL_TEMP
    criterion['distillation'] = FeatureDistillationLoss(loss_type=distill_loss_type, kl_temp=distill_kl_temp)
    logger.info(f"蒸餾損失類型: {distill_loss_type}, KL 溫度 (如果適用): {distill_kl_temp}")

    distill_weight_mobilesam = config.DISTILLATION.WEIGHT_MOBILESAM
    distill_weight_sam_h = config.DISTILLATION.WEIGHT_SAM_H
    logger.info(f"原始 MobileSAM 蒸餾權重: {distill_weight_mobilesam}")
    logger.info(f"SAM-H 蒸餾權重: {distill_weight_sam_h}")

    # --- 載入資料集 ---
    logger.info("載入訓練資料集...")
    # 確保資料集路徑正確
    if not os.path.exists(config.DATA.DATASET_JSON):
        logger.error(f"找不到資料集 JSON 檔案: {config.DATA.DATASET_JSON}")
        sys.exit(1)
    if not os.path.exists(config.DATA.DATA_ROOT):
        logger.error(f"找不到資料根目錄: {config.DATA.DATA_ROOT}")
        sys.exit(1)

    train_dataset = DatasetFinetune(
        config,
        json_key=config.DATA.TRAIN_JSON_KEY,
        precomputed_feat_dir_mobilesam=config.DISTILLATION.PRECOMPUTED_FEAT_DIR_MOBILESAM if distill_weight_mobilesam > 0 else None,
        precomputed_feat_dir_sam_h=config.DISTILLATION.PRECOMPUTED_FEAT_DIR_SAM_H if distill_weight_sam_h > 0 else None
    )
    if len(train_dataset) == 0:
        logger.error("訓練資料集為空，請檢查資料集路徑和 JSON 檔案。")
        sys.exit(1)
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.DATA.BATCH_SIZE,
        shuffle=True,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=True, # 如果 GPU 訓練，建議開啟
        # collate_fn=collate_fn_distillation # 如果需要自訂 collate_fn
    )
    logger.info(f"訓練資料集大小: {len(train_dataset)}, DataLoader 批次大小: {config.DATA.BATCH_SIZE}")
    # ... (驗證資料集的載入邏輯，如果需要) ...

    # --- 學習率排程器 ---
    logger.info(f"設定學習率排程器: {config.TRAIN.SCHEDULER.NAME}")
    scheduler = build_scheduler(config, optimizer, len(train_dataloader))

    # --- 訓練迴圈 ---
    logger.info("開始訓練...")
    max_epochs = config.TRAIN.EPOCHS
    global_step = 0

    for epoch_num in range(max_epochs):
        model_finetune.train() # 設定為訓練模式
        
        epoch_total_loss_sum = 0.0
        epoch_original_loss_sum = 0.0
        epoch_distill_mobilesam_loss_sum = 0.0
        epoch_distill_sam_h_loss_sum = 0.0
        epoch_focal_loss_sum = 0.0
        epoch_iou_loss_sum = 0.0
        
        # 進度條
        pbar = tqdm(train_dataloader, desc=f"Epoch {epoch_num + 1}/{max_epochs}", file=sys.stdout)

        for batch_idx, batch_data in enumerate(pbar):
            try:
                # 從 DataLoader 獲取資料並移至裝置
                # DatasetFinetune 返回: image_padded, transformed_bbox, gt_mask_padded, features_mobilesam, features_sam_h
                image_batch = batch_data[0].to(device)
                bboxes_batch = batch_data[1].to(device)       # (B, 1, 4) - [x1,y1,x2,y2]
                gt_masks_batch = batch_data[2].to(device)     # (B, 1, H, W)
                
                teacher_feats_mobilesam_batch = batch_data[3] # 可能在 CPU 上，或為空 Tensor
                teacher_feats_sam_h_batch = batch_data[4]     # 可能在 CPU 上，或為空 Tensor

                # 將教師特徵移至裝置 (如果存在且非空)
                if isinstance(teacher_feats_mobilesam_batch, torch.Tensor) and teacher_feats_mobilesam_batch.numel() > 0:
                    teacher_feats_mobilesam_batch = teacher_feats_mobilesam_batch.to(device)
                else:
                    teacher_feats_mobilesam_batch = None # 標記為 None 以便後續處理

                if isinstance(teacher_feats_sam_h_batch, torch.Tensor) and teacher_feats_sam_h_batch.numel() > 0:
                    teacher_feats_sam_h_batch = teacher_feats_sam_h_batch.to(device)
                else:
                    teacher_feats_sam_h_batch = None # 標記為 None

            except Exception as e:
                logger.error(f"在 Epoch {epoch_num+1}, Batch {batch_idx} 解包 batch_data 或移至裝置時出錯: {e}")
                logger.error(f"Batch data structure hint: len={len(batch_data) if hasattr(batch_data, '__len__') else 'N/A'}")
                if hasattr(batch_data, '__len__'):
                    for i, item in enumerate(batch_data):
                         logger.error(f"Item {i} type: {type(item)}, shape: {item.shape if hasattr(item, 'shape') else 'N/A'}, device: {item.device if hasattr(item, 'device') else 'N/A'}")
                continue # 跳過這個批次

            # --- 前向傳播 ---
            # autocast 用於混合精度
            with autocast(enabled=use_amp):
                # 1. 獲取學生模型的圖像編碼器特徵 (用於蒸餾)
                student_image_features = model_finetune.image_encoder(image_batch)
                
                # 2. 獲取分割預測 (使用提示)
                # 從邊界框生成點提示 (SAM 的標準輸入)
                point_coords_batch, point_labels_batch = get_prompt_points_from_bboxes(bboxes_batch, device)

                if point_coords_batch is None: # 如果沒有有效的邊界框/提示
                    logger.warning(f"Epoch {epoch_num+1}, Batch {batch_idx}: 無有效提示，跳過此批次。")
                    continue

                # SAM.forward(image, point_coords, point_labels)
                # masks_pred: (B, num_masks_per_image (e.g.1), H, W)
                # iou_pred: (B, num_masks_per_image)
                masks_pred, iou_pred, _ = model_finetune(image_batch, point_coords_batch, point_labels_batch)

                # 計算原始分割損失
                loss_focal = criterion['focal'](masks_pred, gt_masks_batch)
                loss_iou = criterion['iou'](masks_pred, gt_masks_batch)
                original_loss = loss_focal + loss_iou
                # if 'dice' in criterion:
                #     loss_dice = criterion['dice'](masks_pred, gt_masks_batch)
                #     original_loss += loss_dice
                
                # 計算蒸餾損失
                current_distill_mobilesam_loss = torch.tensor(0.0, device=device)
                if distill_weight_mobilesam > 0 and teacher_feats_mobilesam_batch is not None:
                    current_distill_mobilesam_loss = criterion['distillation'](student_image_features, teacher_feats_mobilesam_batch)
                
                current_distill_sam_h_loss = torch.tensor(0.0, device=device)
                if distill_weight_sam_h > 0 and teacher_feats_sam_h_batch is not None:
                    current_distill_sam_h_loss = criterion['distillation'](student_image_features, teacher_feats_sam_h_batch)
                    
                # 計算總損失
                total_loss = original_loss + \
                             distill_weight_mobilesam * current_distill_mobilesam_loss + \
                             distill_weight_sam_h * current_distill_sam_h_loss

            # --- 反向傳播與優化 ---
            optimizer.zero_grad()
            if scaler is not None: # 混合精度
                scaler.scale(total_loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else: # 全精度
                total_loss.backward()
                optimizer.step()
            
            global_step += 1
            
            # --- 累加損失用於 epoch 平均 ---
            epoch_total_loss_sum += total_loss.item()
            epoch_original_loss_sum += original_loss.item()
            epoch_focal_loss_sum += loss_focal.item()
            epoch_iou_loss_sum += loss_iou.item()
            epoch_distill_mobilesam_loss_sum += current_distill_mobilesam_loss.item()
            epoch_distill_sam_h_loss_sum += current_distill_sam_h_loss.item()
            
            # 更新進度條顯示
            pbar.set_postfix({
                'Total': f"{total_loss.item():.4f}",
                'Orig': f"{original_loss.item():.4f}",
                'D_MSAM': f"{current_distill_mobilesam_loss.item():.4f}",
                'D_SAMH': f"{current_distill_sam_h_loss.item():.4f}",
                'LR': f"{optimizer.param_groups[0]['lr']:.2e}"
            })

            # TensorBoard 記錄 (每個 step) - 可以選擇性地減少記錄頻率以加速訓練
            if global_step % config.TRAIN.LOG_FREQ == 0: # LOG_FREQ 需在設定檔中定義，例如 10 或 50
                writer.add_scalar('Loss_step/Total', total_loss.item(), global_step)
                writer.add_scalar('Loss_step/Original_Segmentation', original_loss.item(), global_step)
                writer.add_scalar('Loss_step/Focal', loss_focal.item(), global_step)
                writer.add_scalar('Loss_step/IoU', loss_iou.item(), global_step)
                if distill_weight_mobilesam > 0:
                    writer.add_scalar('Loss_step/Distill_MobileSAM', current_distill_mobilesam_loss.item(), global_step)
                if distill_weight_sam_h > 0:
                    writer.add_scalar('Loss_step/Distill_SAM_H', current_distill_sam_h_loss.item(), global_step)
                writer.add_scalar('Learning_Rate', optimizer.param_groups[0]['lr'], global_step)
        
        # --- Epoch 結束 ---
        # 更新學習率 (通常在每個 step 或每個 epoch 後更新，取決於排程器類型)
        if scheduler is not None :
             # 大部分排程器在 optimizer.step() 之後、每個 step 呼叫 scheduler.step()
             # 有些如 ReduceLROnPlateau 在每個 epoch 結束時，基於驗證指標呼叫 scheduler.step(val_metric)
             # 這裡假設 build_scheduler 返回的排程器適合在 epoch 結束時更新，或者已在 step 內部處理
             # 如果是 step-wise scheduler，這一行可能不需要或需要調整
             if not isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau): # PolyLR 等在 step 更新
                 pass # Step-wise schedulers already stepped.
             # else: scheduler.step(avg_epoch_total_loss) # Example for ReduceLROnPlateau

        num_batches_in_epoch = len(train_dataloader)
        avg_epoch_total_loss = epoch_total_loss_sum / num_batches_in_epoch
        avg_epoch_original_loss = epoch_original_loss_sum / num_batches_in_epoch
        avg_epoch_focal_loss = epoch_focal_loss_sum / num_batches_in_epoch
        avg_epoch_iou_loss = epoch_iou_loss_sum / num_batches_in_epoch
        avg_epoch_distill_mobilesam_loss = epoch_distill_mobilesam_loss_sum / num_batches_in_epoch
        avg_epoch_distill_sam_h_loss = epoch_distill_sam_h_loss_sum / num_batches_in_epoch

        logger.info(f"Epoch [{epoch_num + 1}/{max_epochs}] 完成. "
                    f"Avg Total Loss: {avg_epoch_total_loss:.4f}, "
                    f"Avg Original Loss: {avg_epoch_original_loss:.4f} (Focal: {avg_epoch_focal_loss:.4f}, IoU: {avg_epoch_iou_loss:.4f}), "
                    f"Avg Distill_MobileSAM: {avg_epoch_distill_mobilesam_loss:.4f}, "
                    f"Avg Distill_SAM_H: {avg_epoch_distill_sam_h_loss:.4f}, "
                    f"LR: {optimizer.param_groups[0]['lr']:.2e}")
        
        # TensorBoard 記錄 (每個 epoch)
        writer.add_scalar('Loss_epoch/Total', avg_epoch_total_loss, epoch_num + 1)
        writer.add_scalar('Loss_epoch/Original_Segmentation', avg_epoch_original_loss, epoch_num + 1)
        writer.add_scalar('Loss_epoch/Focal', avg_epoch_focal_loss, epoch_num + 1)
        writer.add_scalar('Loss_epoch/IoU', avg_epoch_iou_loss, epoch_num + 1)
        writer.add_scalar('Loss_epoch/Distill_MobileSAM', avg_epoch_distill_mobilesam_loss, epoch_num + 1)
        writer.add_scalar('Loss_epoch/Distill_SAM_H', avg_epoch_distill_sam_h_loss, epoch_num + 1)

        # --- 儲存模型 ---
        if (epoch_num + 1) % config.TRAIN.SAVE_FREQ == 0 or (epoch_num + 1) == max_epochs:
            checkpoint_data = {
                'epoch': epoch_num + 1,
                'model_state_dict': model_finetune.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'config': config.to_dict() # 儲存設定以供後續參考
            }
            if scheduler is not None:
                checkpoint_data['scheduler_state_dict'] = scheduler.state_dict()
            if scaler is not None:
                checkpoint_data['scaler_state_dict'] = scaler.state_dict()

            save_path = os.path.join(config.OUTPUT_DIR, f"epoch_{epoch_num + 1}.pth")
            save_checkpoint(checkpoint_data, save_path) # 使用 save_checkpoint 函式
            logger.info(f"模型檢查點已儲存至 {save_path}")

        # --- 執行驗證 (如果需要，此處未實現) ---
        # if config.TRAIN.EVAL_FREQ > 0 and (epoch_num + 1) % config.TRAIN.EVAL_FREQ == 0:
        #     logger.info(f"Epoch [{epoch_num+1}/{max_epochs}]: 開始驗證...")
        #     # evaluate_model(...) # 您需要實現或引入驗證邏輯

    writer.close()
    logger.info("訓練完成！")

if __name__ == '__main__':
    config = parse_option() # 載入 yaml/json 設定檔，返回 CfgNode

    # 確保 DISTILLATION 和其他必要的設定在 CfgNode 中有預設值
    # (這應該在 load_config.py 中處理)
    if not hasattr(config, 'DISTILLATION'):
        config.defrost()
        config.DISTILLATION = CN()
        config.DISTILLATION.PRECOMPUTED_FEAT_DIR_MOBILESAM = ""
        config.DISTILLATION.PRECOMPUTED_FEAT_DIR_SAM_H = ""
        config.DISTILLATION.WEIGHT_MOBILESAM = 0.0
        config.DISTILLATION.WEIGHT_SAM_H = 0.0
        config.DISTILLATION.LOSS_TYPE = "mse"
        config.DISTILLATION.KL_TEMP = 1.0
        config.freeze()
    if not hasattr(config.TRAIN, 'LOG_FREQ'): # 設定日誌記錄頻率的預設值
        config.defrost()
        config.TRAIN.LOG_FREQ = 50 # 每 50 個 global step 記錄一次
        config.freeze()
    if not hasattr(config.TRAIN, 'AMP'): # 設定混合精度訓練的預設值
        config.defrost()
        config.TRAIN.AMP = False
        config.freeze()
    if not hasattr(config, 'CONFIG_PATH'): # 儲存設定檔路徑，方便後續使用
        # parse_option 中沒有直接傳遞 args.config 給 config 物件
        # 這裡假設 load_config 會處理或我們可以手動添加
        # args = argparse.ArgumentParser().parse_args() # 重新解析以獲取 config path (不是好方法)
        # 最好是在 load_config 中將 config 檔案的路徑存入 CfgNode
        # config.defrost()
        # config.CONFIG_PATH = args.config # 這裡的 args 是 main 外面的，作用域問題
        # config.freeze()
        # 為了簡單，假設 load_config.py 中的 load_config(filepath) 會將 filepath 存到 CfgNode.CONFIG_PATH
        pass


    main(config)
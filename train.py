import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast, GradScaler

from torchvision import transforms
from torchvision.ops import sigmoid_focal_loss

from pathlib import Path
from tqdm import tqdm

from finetune_utils.load_config import get_config
from finetune_utils.load_logger import Logger
from finetune_utils.load_checkpoint import get_sam_vit_t
from finetune_utils.datasets import ComponentDataset
from finetune_utils.loss import DiceLoss, batch_iou
from finetune_utils.visualization import overlay_mask_on_image
from finetune_utils.save_checkpoint import save_checkpoint
from finetune_utils.schedular import LinearWarmup

torch.backends.cudnn.benchmark = True

args = None
MEAN = None
STD = None
IMAGE_SIZE = None

def main(config_args):
    global args, MEAN, STD, IMAGE_SIZE
    args = config_args

    assert torch.cuda.is_available(), "CUDA is not available."

    IMAGE_SIZE = (args.model.image_size, args.model.image_size)
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]

    transform_img = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD)
    ])
    transform_mask = transforms.Compose([
        transforms.Resize(IMAGE_SIZE, interpolation=transforms.InterpolationMode.NEAREST),
        transforms.ToTensor(),
    ])

    train_dataset = ComponentDataset(
        root_dir=args.dataset.train_dataset,
        transform=[transform_img, transform_mask],
        max_bbox_shift=args.dataset.max_bbox_shift,
        prompt_mode=args.dataset.prompt_mode,
        min_points=args.dataset.min_points,
        max_points=args.dataset.max_points,
        image_size=args.model.image_size
    )
    val_dataset = ComponentDataset(
        root_dir=args.dataset.val_dataset,
        transform=[transform_img, transform_mask],
        max_bbox_shift=0, # 通常驗證集不使用 bbox shift
        prompt_mode=args.dataset.prompt_mode, # 或者固定為 'box' or 'point'
        min_points=args.dataset.min_points,
        max_points=args.dataset.max_points,
        image_size=args.model.image_size
    )

    train_loader = DataLoader(
        train_dataset, batch_size=args.train.batch_size,
        num_workers=args.dataset.num_workers, shuffle=True,
        pin_memory=True, persistent_workers=True if args.dataset.num_workers > 0 else False
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.train.batch_size,
        num_workers=args.dataset.num_workers, shuffle=False,
        pin_memory=True, persistent_workers=True if args.dataset.num_workers > 0 else False
    )

    checkpoint_path = Path(args.model.checkpoint_path)
    save_path = Path(args.model.save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    logger = Logger(save_path / 'training.log').get_logger()
    logger.info(f"Training with config: {args}")

    scaler = GradScaler(enabled=args.train.bf16)
    model = get_sam_vit_t(checkpoint=checkpoint_path, resume=args.train.resume).cuda()

    for name, param in model.named_parameters():
        if args.freeze.freeze_image_encoder and 'image_encoder' in name:
            param.requires_grad = False
        if args.freeze.freeze_prompt_encoder and 'prompt_encoder' in name:
            param.requires_grad = False
        if args.freeze.freeze_mask_decoder and 'mask_decoder' in name:
            param.requires_grad = False
    
    num_unfrozen_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Number of unfrozen parameters: {num_unfrozen_params}")

    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.train.learning_rate)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.train.epochs * len(train_loader))
    warmup_scheduler = LinearWarmup(optimizer, warmup_period=args.train.warmup_step)

    criterion_MSE = nn.MSELoss()
    criterion_Dice = DiceLoss(sigmoid=True, squared_pred=True, reduction='mean')
    writer = SummaryWriter(log_dir=str(save_path / 'tensorboard_logs'))

    best_val_loss = float('inf')
    
    # --- Early Stopping 初始化 ---
    early_stopping_enabled = args.early_stopping.enabled
    early_stopping_patience = args.early_stopping.patience
    early_stopping_min_delta = args.early_stopping.min_delta
    epochs_no_improve = 0 # 計數器：記錄驗證損失沒有改善的驗證週期數
    # --- Early Stopping 初始化結束 ---

    logger.info("Starting training...")
    for epoch in range(args.train.epochs):
        train_loss = train_epoch(train_loader, model, optimizer, criterion_MSE, criterion_Dice, epoch, writer, scaler, lr_scheduler, warmup_scheduler, logger)
        logger.info(f"Epoch {epoch+1}/{args.train.epochs}, Train Loss: {train_loss:.4f}")

        if (epoch + 1) % args.train.val_freq == 0:
            val_loss = val_epoch(val_loader, model, criterion_MSE, criterion_Dice, epoch, writer, scaler, logger)
            logger.info(f"Epoch {epoch+1}/{args.train.epochs}, Val Loss: {val_loss:.4f}")

            # --- Early Stopping 邏輯 ---
            if early_stopping_enabled:
                # 檢查是否有改善 (新的 val_loss 是否比 best_val_loss 小了至少 min_delta)
                if val_loss < best_val_loss - early_stopping_min_delta:
                    logger.info(f"Validation loss improved from {best_val_loss:.4f} to {val_loss:.4f}.")
                    best_val_loss = val_loss
                    epochs_no_improve = 0 # 重置計數器
                    # 保存最佳模型 (原本的邏輯)
                    save_checkpoint({
                        'epoch': epoch + 1,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'best_val_loss': best_val_loss,
                        'args': args
                    }, True, save_path) # is_best = True
                else:
                    epochs_no_improve += 1
                    logger.info(f"Validation loss did not improve significantly for {epochs_no_improve} validation period(s). Current best: {best_val_loss:.4f}.")
                    # 即使沒有改善，如果設定檔中 `save_checkpoint` 的 is_best 邏輯是基於嚴格小於，
                    # 這裡還是可以呼叫 save_checkpoint，但 is_best 應為 False。
                    # 原本的 save_checkpoint 邏輯已經在 is_best 參數中處理這個。
                    # 我們只需要在 val_loss < best_val_loss 時才更新 best_val_loss 和 is_best=True 的儲存。
                    # 如果只是沒有顯著改善，但仍想儲存最新模型 (非最佳)，則需調整 save_checkpoint。
                    # 目前邏輯是：只有嚴格變好才更新 best_val_loss 並標記為 is_best=True 儲存。
                    # 如果只是沒有 "顯著" 改善 (即改善幅度 < min_delta 但仍比 best_val_loss 小)，
                    # 也應該更新 best_val_loss 並儲存。
                    # 調整如下：
                    if val_loss < best_val_loss : # 只要有任何一點點改善 (即使小於 min_delta)，也更新 best_val_loss
                        best_val_loss = val_loss # 並且儲存 (但 epochs_no_improve 仍然增加，因為改善不夠 "顯著")
                        save_checkpoint({
                            'epoch': epoch + 1,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'best_val_loss': best_val_loss, # 用新的 best_val_loss
                            'args': args
                        }, True, save_path) # 標記為 best，因為它是目前為止最好的
                    else: # 如果完全沒有改善 (val_loss >= best_val_loss)
                         save_checkpoint({ # 仍儲存當前 epoch 的模型，但不是 best
                            'epoch': epoch + 1,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'best_val_loss': best_val_loss, # best_val_loss 維持舊的
                            'args': args
                        }, False, save_path) # is_best = False


                if epochs_no_improve >= early_stopping_patience:
                    logger.info(f"Early stopping triggered after {epochs_no_improve} validation periods without significant improvement.")
                    break # 跳出訓練迴圈
            else: # Early stopping 未啟用，維持原本的儲存邏輯
                is_best = val_loss < best_val_loss
                if is_best:
                    best_val_loss = val_loss
                save_checkpoint({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_val_loss': best_val_loss,
                    'args': args
                }, is_best, save_path)
            # --- Early Stopping 邏輯結束 ---
        
        # 檢查是否因為 Early Stopping 而跳出迴圈
        if early_stopping_enabled and epochs_no_improve >= early_stopping_patience:
            break


    writer.close()
    logger.info("Training finished.")

# train_epoch 和 val_epoch 函數保持不變，這裡省略以節省篇幅
# ... (train_epoch 和 val_epoch 函數定義) ...
# (請確保 train_epoch 和 val_epoch 的定義與您先前版本一致)

def train_epoch(dataloader, model, optimizer, criterion_MSE, criterion_Dice, epoch, writer, scaler, lr_scheduler, warmup_scheduler, logger):
    """Main training function for one epoch."""
    model.train()
    total_loss = 0.0
    num_batches = len(dataloader)
    progress_bar = tqdm(dataloader, desc=f"Training Epoch {epoch+1}", total=num_batches)

    for batch_idx, (image_batch, mask_batch, box_prompt_batch, point_coords_batch, point_labels_batch, is_point_prompt_flags) in enumerate(progress_bar):
        image_batch = image_batch.cuda(non_blocking=True)
        mask_batch = mask_batch.cuda(non_blocking=True)
        
        boxes_for_model = box_prompt_batch.cuda(non_blocking=True).unsqueeze(1) 
        points_coords_for_model = point_coords_batch.cuda(non_blocking=True)
        points_labels_for_model = point_labels_batch.cuda(non_blocking=True)
        points_for_model = (points_coords_for_model, points_labels_for_model)

        with autocast(enabled=args.train.bf16, dtype=torch.bfloat16 if args.train.bf16 else torch.float16):
            pred_mask_logits, pred_iou_values = model(image_batch, boxes=boxes_for_model, points=points_for_model)
            actual_iou = batch_iou(torch.sigmoid(pred_mask_logits), mask_batch)
            loss_focal = sigmoid_focal_loss(pred_mask_logits, mask_batch, reduction='mean', alpha=0.25, gamma=2.0)
            loss_dice = criterion_Dice(pred_mask_logits, mask_batch)
            loss_mse = criterion_MSE(pred_iou_values, actual_iou)
            loss = (args.loss.focal_weight * loss_focal) + \
                   (args.loss.dice_weight * loss_dice) + \
                   (args.loss.iou_weight * loss_mse)

        scaler.scale(loss).backward()
        
        if (batch_idx + 1) % args.train.gradient_accumulation == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        
        with warmup_scheduler.dampening():
            lr_scheduler.step()
        
        total_loss += loss.item()
        progress_bar.set_postfix(loss=loss.item(), lr=optimizer.param_groups[0]['lr'])

    average_loss = total_loss / num_batches
    writer.add_scalar('Loss/Train', average_loss, epoch)
    writer.add_scalar('Learning_rate', optimizer.param_groups[0]['lr'], epoch)
    return average_loss

def val_epoch(dataloader, model, criterion_MSE, criterion_Dice, epoch, writer, scaler, logger):
    model.eval()
    total_loss = 0.0
    num_batches = len(dataloader)
    progress_bar = tqdm(dataloader, desc=f"Validating Epoch {epoch+1}", total=num_batches)
    first_batch_visualized = False

    with torch.no_grad():
        for batch_idx, (image_batch, mask_batch, box_prompt_batch, point_coords_batch, point_labels_batch, is_point_prompt_flags) in enumerate(progress_bar):
            image_batch = image_batch.cuda(non_blocking=True)
            mask_batch = mask_batch.cuda(non_blocking=True)
            
            boxes_for_model = box_prompt_batch.cuda(non_blocking=True).unsqueeze(1)
            points_coords_for_model = point_coords_batch.cuda(non_blocking=True)
            points_labels_for_model = point_labels_batch.cuda(non_blocking=True)
            points_for_model = (points_coords_for_model, points_labels_for_model)

            with autocast(enabled=args.train.bf16, dtype=torch.bfloat16 if args.train.bf16 else torch.float16):
                pred_mask_logits, pred_iou_values = model(image_batch, boxes=boxes_for_model, points=points_for_model)
                actual_iou = batch_iou(torch.sigmoid(pred_mask_logits), mask_batch)
                loss_focal = sigmoid_focal_loss(pred_mask_logits, mask_batch, reduction='mean', alpha=0.25, gamma=2.0)
                loss_dice = criterion_Dice(pred_mask_logits, mask_batch)
                loss_mse = criterion_MSE(pred_iou_values, actual_iou)
                loss = (args.loss.focal_weight * loss_focal) + \
                       (args.loss.dice_weight * loss_dice) + \
                       (args.loss.iou_weight * loss_mse)

            total_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())

            if args.visual.status and not first_batch_visualized and batch_idx == 0:
                try:
                    vis_image = image_batch[0].cpu()
                    vis_pred_mask_sigmoid = torch.sigmoid(pred_mask_logits[0]).cpu()
                    item_is_point_prompt = is_point_prompt_flags[0].item()
                    prompt_info_for_vis = None
                    if item_is_point_prompt:
                        num_pts = int(torch.sum(point_labels_batch[0] > 0).item())
                        actual_points = point_coords_batch[0][:num_pts].cpu()
                        prompt_info_for_vis = actual_points
                    else:
                        prompt_info_for_vis = box_prompt_batch[0].cpu()

                    img_mean = torch.tensor(MEAN, device=vis_image.device).view(3, 1, 1)
                    img_std = torch.tensor(STD, device=vis_image.device).view(3, 1, 1)
                    vis_image_unnorm = vis_image * img_std + img_mean
                    
                    vis_save_dir = Path(args.visual.save_path) / f"epoch_{epoch+1}"
                    vis_save_dir.mkdir(parents=True, exist_ok=True)

                    overlay_mask_on_image(
                        image_tensor=vis_image_unnorm,
                        mask_tensor=vis_pred_mask_sigmoid,
                        bbox_tensor=prompt_info_for_vis if not item_is_point_prompt else None,
                        points_tensor=prompt_info_for_vis if item_is_point_prompt else None,
                        gt_mask_tensor=mask_batch[0].cpu(),
                        threshold=args.visual.IOU_threshold,
                        save_dir=vis_save_dir,
                        filename_info=f"val_batch_{batch_idx}_item_0_prompt_{'point' if item_is_point_prompt else 'box'}"
                    )
                    first_batch_visualized = True
                    logger.info(f"Saved visualization for epoch {epoch+1} to {vis_save_dir}")
                except Exception as e:
                    logger.error(f"Error during visualization: {e}")
                    first_batch_visualized = True
    average_loss = total_loss / num_batches
    writer.add_scalar('Loss/Validation', average_loss, epoch)
    return average_loss


if __name__ == '__main__':
    config = get_config()
    if not hasattr(config, 'loss'): # 設定預設損失權重 (如果設定檔中沒有)
        config.loss = type('LossArgs', (), {})() # 建立一個簡單的命名空間物件
        config.loss.focal_weight = 20.0
        config.loss.dice_weight = 1.0
        config.loss.iou_weight = 1.0
    
    if not hasattr(config, 'early_stopping'): # 設定預設 Early Stopping (如果設定檔中沒有)
        config.early_stopping = type('EarlyStoppingArgs', (), {})()
        config.early_stopping.enabled = False # 預設關閉
        config.early_stopping.patience = 10
        config.early_stopping.min_delta = 0.0
        config.early_stopping.monitor_metric = "val_loss"

    main(config)
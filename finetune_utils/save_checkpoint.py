# finetune_utils/save_checkpoint.py
import torch
import logging
from pathlib import Path

def save_checkpoint(state: dict, is_best: bool, checkpoint_dir: Path):
    """
    儲存目前的訓練檢查點。

    參數:
        state (dict): 一個包含模型狀態、優化器狀態等的字典。
                      預期包含 'model_state_dict' (模型權重), 'optimizer_state_dict', 'epoch', 'best_val_loss', 'args'。
        is_best (bool): 一個布林標誌，指示目前的檢查點是否基於驗證損失為最佳。
        checkpoint_dir (Path): 儲存檢查點的目錄路徑。
    """
    checkpoint_dir = Path(checkpoint_dir) # 確保是 Path 物件
    checkpoint_dir.mkdir(parents=True, exist_ok=True) # 確保目錄存在

    # --- 修改副檔名 ---
    last_path = checkpoint_dir / 'last.pth' # 從 'last.pth.tar' 改為 'last.pth'
    best_path = checkpoint_dir / 'best.pth' # 從 'best.pth.tar' 改為 'best.pth'
    # --- 副檔名修改結束 ---

    try:
        # 永遠儲存包含所有資訊的 "last" checkpoint
        torch.save(state, last_path)
        logging.info(f"Checkpoint saved successfully at {last_path}")

        if is_best:
            # 如果是最佳模型，儲存完整的 state 字典到 best_path
            # 這樣 best.pth 和 last.pth 具有相同的結構，都包含完整的訓練狀態信息。
            torch.save(state, best_path)
            logging.info(f"New best checkpoint (full state) saved successfully at {best_path}")

    except OSError as e:
        logging.error(f"Saving checkpoint failed: {e}", exc_info=True)
        # raise # 如果希望儲存失敗時終止程式，則取消註解此行
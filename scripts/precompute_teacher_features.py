"""
# 預計算原始 MobileSAM 的特徵
python precompute_teacher_features.py \
    --config ./configs/mobileSAM.json \
    --dataset_json_key train \
    --teacher_model_type mobile_sam \
    --teacher_checkpoint path/to/original/mobile_sam_weights.pt \
    --output_dir ./precomputed_features/original_mobilesam_train

# 預計算 SAM-H 的特徵 (假設您有 SAM-H ViT-H 的權重)
python precompute_teacher_features.py \
    --config ./configs/mobileSAM.json \
    --dataset_json_key train \
    --teacher_model_type sam_h \
    --teacher_checkpoint path/to/sam_vit_h_weights.pth \
    --output_dir ./precomputed_features/sam_h_train
"""

import torch
import os
import argparse
from tqdm import tqdm
from torchvision import transforms
import cv2

# 假設您的 MobileSAM 和 SAM-H 模型相關函式庫路徑正確
# MobileSAM 的模型載入
from mobile_sam import sam_model_registry as mobile_sam_model_registry
from mobile_sam.utils.transforms import ResizeLongestSide

# SAM-H 的模型載入 (這裡假設您有 Meta AI 的 segment-anything 庫或者類似的載入方式)
# 您可能需要安裝 'segment-anything' 套件: pip install segment-anything
try:
    from segment_anything import sam_model_registry as hq_sam_model_registry
    from segment_anything.utils.transforms import ResizeLongestSide as ResizeLongestSideHQ
    SAM_H_AVAILABLE = True
except ImportError:
    print("警告：segment-anything 套件未找到，SAM-H 特徵提取將不可用。")
    print("請執行 'pip install segment-anything' 並準備 SAM-H (ViT-H) 的權重檔。")
    SAM_H_AVAILABLE = False


# 從您的專案中引入 DatasetFinetune，我們只用它來遍歷圖片
# 為了簡化，這裡我們直接寫一個輕量級的圖片讀取邏輯
# 如果您的 DatasetFinetune 有複雜的預處理，您可能需要調整
from finetune_utils.datasets import DatasetFinetune # 用於獲取圖片列表和路徑結構
from finetune_utils.load_config import load_config


def get_image_paths_from_dataset_json(dataset_json_path, data_root_path, json_key='train'):
    import json
    with open(dataset_json_path, 'r') as f:
        data_info = json.load()
    
    image_paths = []
    base_names = []
    for item in data_info[json_key]:
        image_name = item['image_name']
        image_path = os.path.join(data_root_path, image_name)
        if os.path.exists(image_path):
            image_paths.append(image_path)
            base_names.append(os.path.splitext(image_name.split('/')[-1])[0]) # 取不含副檔名的檔案名
        else:
            print(f"警告：圖片 {image_path} 不存在，將跳過。")
    return image_paths, base_names

def load_teacher_encoder(model_type, checkpoint_path, device):
    """
    載入教師模型的圖像編碼器。
    model_type: 'mobile_sam' 或 'sam_h'
    checkpoint_path: 模型權重檔案路徑
    """
    if model_type == "mobile_sam":
        model = mobile_sam_model_registry["vit_t"](checkpoint=checkpoint_path)
        encoder = model.image_encoder.to(device).eval()
        image_size = model.image_encoder.img_size
        transform = ResizeLongestSide(image_size)
        return encoder, transform, image_size
    elif model_type == "sam_h":
        if not SAM_H_AVAILABLE:
            raise RuntimeError("SAM-H 模型無法載入，請檢查 segment-anything 套件和相關設定。")
        # 假設 sam_model_registry["vit_h"] 是 SAM-H (ViT-H) 的模型類型
        # 您需要提供正確的 SAM-H 權重路徑
        model = hq_sam_model_registry["vit_h"](checkpoint=checkpoint_path)
        encoder = model.image_encoder.to(device).eval()
        image_size = model.image_encoder.img_size # 通常是 1024
        transform = ResizeLongestSideHQ(image_size) # SAM-H 可能使用不同的 ResizeLongestSide
        return encoder, transform, image_size
    else:
        raise ValueError(f"不支援的教師模型類型: {model_type}")

def preprocess_image(image_path, transform_fn, target_size, device):
    """
    預處理單張圖片。
    """
    image = cv2.imread(image_path)
    if image is None:
        print(f"警告：無法讀取圖片 {image_path}，將跳過。")
        return None
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    original_size = image.shape[:2]
    image_transformed = transform_fn.apply_image(image)
    image_tensor = torch.as_tensor(image_transformed, device=device)
    # PyTorch 的 Conv2d 期望 (N, C, H, W)
    image_tensor = image_tensor.permute(2, 0, 1).contiguous()[None, :, :, :]

    # Pad to target size (e.g., 1024x1024 for SAM)
    h, w = image_tensor.shape[-2:]
    pad_h = target_size - h
    pad_w = target_size - w
    image_padded = torch.nn.functional.pad(image_tensor, (0, pad_w, 0, pad_h))
    return image_padded, original_size


def main():
    parser = argparse.ArgumentParser(description="預先計算教師模型的特徵圖")
    parser.add_argument('--config', type=str, required=True, help='設定檔路徑 (例如 configs/mobileSAM.json)')
    parser.add_argument('--dataset_json_key', type=str, default='train', help='設定檔中指定資料集資訊的鍵名 (例如 "train" 或 "val")')
    
    parser.add_argument('--teacher_model_type', type=str, required=True, choices=['mobile_sam', 'sam_h'], help='教師模型類型')
    parser.add_argument('--teacher_checkpoint', type=str, required=True, help='教師模型權重路徑')
    parser.add_argument('--output_dir', type=str, required=True, help='儲存特徵圖的目錄')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='計算設備 (cuda/cpu)')
    args = parser.parse_args()

    # 載入設定檔以獲取資料集路徑
    # 注意：這裡的 config 是用來獲取 dataset_json 和 data_root，而不是 train.py 裡面的 CfgNode 物件
    # 你也可以直接傳入 dataset_json_path 和 data_root_path
    import json
    with open(args.config, 'r') as f:
        config_json = json.load(f)

    dataset_json_path = config_json['DATA']['DATASET_JSON']
    data_root_path = config_json['DATA']['DATA_ROOT']
    json_key = args.dataset_json_key


    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)

    print(f"正在載入教師模型: {args.teacher_model_type} 從 {args.teacher_checkpoint}")
    teacher_encoder, transform_fn, target_image_size = load_teacher_encoder(args.teacher_model_type, args.teacher_checkpoint, device)
    print(f"教師模型 {args.teacher_model_type} 的目標圖像大小: {target_image_size}")

    print(f"從 {dataset_json_path} (鍵: {json_key}) 讀取圖片列表...")
    image_paths, image_basenames = get_image_paths_from_dataset_json(dataset_json_path, data_root_path, json_key)
    
    print(f"找到 {len(image_paths)} 張圖片。開始提取特徵...")
    for img_path, basename in tqdm(zip(image_paths, image_basenames), total=len(image_paths)):
        try:
            input_image, _ = preprocess_image(img_path, transform_fn, target_image_size, device)
            if input_image is None:
                continue

            with torch.no_grad():
                features = teacher_encoder(input_image) # 應為 [1, 256, H_feat, W_feat] e.g. [1, 256, 64, 64]
            
            # 儲存特徵 (移除批次維度)
            output_filename = os.path.join(args.output_dir, f"{basename}_feat.pt")
            torch.save(features.squeeze(0).cpu(), output_filename)
        except Exception as e:
            print(f"處理圖片 {img_path} 時發生錯誤: {e}")
            continue
            
    print(f"特徵提取完成，已儲存至 {args.output_dir}")

if __name__ == '__main__':
    main()

# finetune_utils/datasets.py
import os
import json
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
from pycocotools import mask as mask_util # 從 COCO API 引入 mask 工具

from mobile_sam.utils.transforms import ResizeLongestSide # 從 MobileSAM 引入轉換函式

class DatasetFinetune(Dataset):
    def __init__(self, config, json_key='train',
                 precomputed_feat_dir_mobilesam=None,
                 precomputed_feat_dir_sam_h=None):
        """
        用於 MobileSAM 微調的資料集類別。
        Args:
            config: 設定物件，包含資料路徑和模型設定。
            json_key: 從 JSON 檔案中讀取哪個部分 (例如 'train' 或 'val')。
            precomputed_feat_dir_mobilesam (str, optional): 預計算的原始 MobileSAM 特徵圖目錄路徑。
            precomputed_feat_dir_sam_h (str, optional): 預計算的 SAM-H 特徵圖目錄路徑。
        """
        self.config = config
        self.dataset_json_path = config.DATA.DATASET_JSON
        self.data_root_path = config.DATA.DATA_ROOT
        self.json_key = json_key

        if not os.path.exists(self.dataset_json_path):
            raise FileNotFoundError(f"找不到資料集 JSON 檔案: {self.dataset_json_path}")
        if not os.path.exists(self.data_root_path):
            raise FileNotFoundError(f"找不到資料根目錄: {self.data_root_path}")

        with open(self.dataset_json_path, 'r') as f:
            data_info = json.load()

        if self.json_key not in data_info:
            raise KeyError(f"'{self.json_key}' 不在 JSON 檔案的鍵中。可用的鍵: {list(data_info.keys())}")

        self.image_data = data_info[self.json_key]
        self.image_paths = []
        self.annotations_list = [] # 儲存每個圖片的標註列表

        for item in self.image_data:
            image_name = item.get('image_name')
            annotations = item.get('annotations')

            if image_name is None or annotations is None:
                print(f"警告：項目 {item} 缺少 'image_name' 或 'annotations'，將跳過。")
                continue

            image_path = os.path.join(self.data_root_path, image_name)
            if not os.path.exists(image_path):
                print(f"警告：圖片 {image_path} 不存在，將跳過包含此圖片的項目。")
                continue

            self.image_paths.append(image_path)
            self.annotations_list.append(annotations)


        if not self.image_paths:
            raise ValueError(f"在指定的 JSON 鍵 '{self.json_key}' 下沒有找到有效的圖片和標註資料。")

        # 新增：預計算特徵相關
        self.precomputed_feat_dir_mobilesam = precomputed_feat_dir_mobilesam
        self.precomputed_feat_dir_sam_h = precomputed_feat_dir_sam_h

        self.image_size = config.MODEL.IMAGE_SIZE
        self.transform = ResizeLongestSide(self.image_size)

        # 檢查預計算特徵目錄是否存在 (如果提供的話)
        if self.precomputed_feat_dir_mobilesam and not os.path.isdir(self.precomputed_feat_dir_mobilesam):
            print(f"警告：預計算 MobileSAM 特徵目錄不存在: {self.precomputed_feat_dir_mobilesam}")
            # self.precomputed_feat_dir_mobilesam = None # 或者拋出錯誤，取決於是否強制要求
        if self.precomputed_feat_dir_sam_h and not os.path.isdir(self.precomputed_feat_dir_sam_h):
            print(f"警告：預計算 SAM-H 特徵目錄不存在: {self.precomputed_feat_dir_sam_h}")
            # self.precomputed_feat_dir_sam_h = None

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        annotations = self.annotations_list[index] # 該圖片的所有標註

        image = cv2.imread(image_path)
        if image is None:
            raise IOError(f"無法讀取圖片: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        original_image_size = image.shape[:2] # (H, W)

        # 預處理圖像 (與 MobileSAM 相同的方式)
        image_transformed = self.transform.apply_image(image) # 調整大小
        image_torch = torch.as_tensor(image_transformed, dtype=torch.float)
        image_torch = image_torch.permute(2, 0, 1).contiguous() # HWC to CHW

        # Pad to target size (e.g., 1024x1024 for SAM)
        h, w = image_torch.shape[1:]
        pad_h = self.image_size - h
        pad_w = self.image_size - w
        image_padded = F.pad(image_torch, (0, pad_w, 0, pad_h), value=0.0) # 用 0 填充
        # image_padded 的 shape 應為 (3, image_size, image_size)

        # 隨機選擇一個標註進行訓練
        if not annotations:
            # 如果圖片沒有任何標註，這是一個問題。
            # 處理方式可以是在 __init__ 中過濾掉這些圖片，或者在這裡返回一個錯誤/空樣本。
            # 為了與原始 MobileSAM_Finetune 的行為一致，這裡可能期望總是有標註。
            # 這裡我們產生一個假的 "無物件" 標註，讓訓練流程可以繼續，但這可能不是最佳選擇。
            # 理想情況下，您的資料集 JSON 應確保每個用於訓練的圖像至少有一個標註。
            print(f"警告: 圖片 {image_path} 沒有標註。將產生一個空的遮罩和邊界框。")
            # 產生一個全為零的遮罩和一個無效的邊界框
            gt_mask_padded = torch.zeros((1, self.image_size, self.image_size), dtype=torch.float)
            # 邊界框格式 [x1, y1, x2, y2]
            transformed_bbox = torch.tensor([[0, 0, 0, 0]], dtype=torch.float)
        else:
            selected_ann = annotations[np.random.randint(len(annotations))]

            # 處理邊界框 (BBox)
            # 假設 selected_ann['bbox'] 是 [x,y,w,h] 格式
            bbox_xywh = np.array(selected_ann['bbox'])
            # 轉換 bbox 為 [x_min, y_min, x_max, y_max] for SAM
            # ResizeLongestSide.apply_boxes 需要 (N, 4) 的輸入
            transformed_bbox_np = self.transform.apply_boxes(bbox_xywh.reshape(-1, 4), original_image_size)
            transformed_bbox = torch.as_tensor(transformed_bbox_np, dtype=torch.float).squeeze() # 確保是 (4,) 或 (1,4)

            # 處理遮罩 (Mask)
            # 假設 selected_ann['segmentation'] 是 COCO RLE 或多邊形格式
            segmentation = selected_ann['segmentation']
            # pycocotools.mask.decode 需要 RLE 格式
            # 如果是多邊形，需要先轉換成 RLE: rle = mask_util.frPyObjects(segmentation, original_image_size[0], original_image_size[1])
            # 這裡假設 segmentation 已經是 RLE 格式或者 frPyObjects 可以處理它
            if isinstance(segmentation, list) and all(isinstance(s, list) for s in segmentation): # 多邊形格式
                rle = mask_util.frPyObjects(segmentation, original_image_size[0], original_image_size[1])
            elif isinstance(segmentation, dict) and 'counts' in segmentation and 'size' in segmentation: # RLE 格式
                rle = [segmentation] # frPyObjects 期望一個列表
            else:
                raise ValueError(f"未知的標註格式 for segmentation: {type(segmentation)}")

            gt_mask_original = mask_util.decode(rle) # (H_orig, W_orig, num_masks) or (H_orig, W_orig)
            if gt_mask_original.ndim == 3: # 如果有多個遮罩 (例如 RLE 列表)，取第一個
                gt_mask_original = gt_mask_original[..., 0]
            
            gt_mask_transformed = self.transform.apply_image(gt_mask_original) # (image_size_transformed_h, image_size_transformed_w)
            gt_mask_torch = torch.as_tensor(gt_mask_transformed, dtype=torch.float)
            
            # Pad mask to target size
            h_mask, w_mask = gt_mask_torch.shape
            pad_h_mask = self.image_size - h_mask
            pad_w_mask = self.image_size - w_mask
            gt_mask_padded = F.pad(gt_mask_torch, (0, pad_w_mask, 0, pad_h_mask), value=0.0) # (image_size, image_size)
            gt_mask_padded = gt_mask_padded.unsqueeze(0) # (1, image_size, image_size)

        # 確保 transformed_bbox 是 (1,4) 的形狀，以方便批次化
        if transformed_bbox.ndim == 1:
            transformed_bbox = transformed_bbox.unsqueeze(0)


        # 載入預計算的特徵
        features_mobilesam = torch.empty(0) # 預設為空 tensor
        features_sam_h = torch.empty(0)     # 預設為空 tensor

        # 從 image_path 獲取 basename (不含副檔名和路徑)
        # 例如: /path/to/image.jpg -> image
        #       /path/to/image.seg.jpg -> image.seg (如果檔名本身包含點)
        # 這裡我們假設檔名中最後一個點之後的是副檔名
        basename = os.path.splitext(os.path.basename(image_path))[0]

        if self.precomputed_feat_dir_mobilesam:
            feat_path_mobilesam = os.path.join(self.precomputed_feat_dir_mobilesam, f"{basename}_feat.pt")
            if os.path.exists(feat_path_mobilesam):
                try:
                    features_mobilesam = torch.load(feat_path_mobilesam, map_location='cpu') # 載入到 CPU 以節省 GPU 記憶體
                except Exception as e:
                    print(f"警告：無法載入 MobileSAM 特徵檔案 {feat_path_mobilesam}: {e}")
            else:
                print(f"警告：找不到 MobileSAM 特徵檔案 {feat_path_mobilesam} (基於圖片 {image_path})")

        if self.precomputed_feat_dir_sam_h:
            feat_path_sam_h = os.path.join(self.precomputed_feat_dir_sam_h, f"{basename}_feat.pt")
            if os.path.exists(feat_path_sam_h):
                try:
                    features_sam_h = torch.load(feat_path_sam_h, map_location='cpu') # 載入到 CPU
                except Exception as e:
                    print(f"警告：無法載入 SAM-H 特徵檔案 {feat_path_sam_h}: {e}")
            else:
                print(f"警告：找不到 SAM-H 特徵檔案 {feat_path_sam_h} (基於圖片 {image_path})")

        # 返回:
        # 1. image_padded: (3, image_size, image_size) - 學生模型的輸入圖像
        # 2. transformed_bbox: (1, 4) - 用於生成提示的邊界框 [x1,y1,x2,y2]
        # 3. gt_mask_padded: (1, image_size, image_size) - 真值遮罩
        # 4. features_mobilesam: (C_ms, H_feat, W_feat) 或 空 tensor - 教師1的特徵
        # 5. features_sam_h: (C_sh, H_feat, W_feat) 或 空 tensor - 教師2的特徵
        return image_padded, transformed_bbox, gt_mask_padded, features_mobilesam, features_sam_h


# 針對 DataLoader 的 Collate 函式 (如果需要自訂的話)
# 預設的 torch.utils.data.dataloader.default_collate 通常可以處理這種情況，
# 只要 features_mobilesam 和 features_sam_h 在批次中都是 Tensor (即使是空 Tensor)。
# 如果某些樣本的特徵是 None 或其他非 Tensor 類型，則需要自訂 collate_fn。
# 由於我們將缺失的特徵初始化為 torch.empty(0)，default_collate 應該沒問題，
# 但在 train.py 中使用這些特徵前需要檢查它們是否為空 (numel() > 0)。

# 例如，一個簡單的 collate_fn，如果預期會有 None (這裡我們已經避免了 None)：
# def collate_fn_distillation(batch):
#     images, bboxes, masks, feats_ms, feats_sh = zip(*batch)
#     images = torch.stack(images, 0)
#     bboxes = torch.stack(bboxes, 0)
#     masks = torch.stack(masks, 0)
    
#     # 處理可能為空的特徵列表
#     # 如果確定都是 Tensor (即使是 empty tensor)，default_collate 更好
#     # 這裡假設如果特徵存在，它們的形狀是一致的
#     if all(f.numel() > 0 for f in feats_ms if isinstance(f, torch.Tensor)):
#         feats_ms = torch.stack([f for f in feats_ms if isinstance(f, torch.Tensor) and f.numel() > 0], 0) if any(isinstance(f, torch.Tensor) and f.numel() > 0 for f in feats_ms) else torch.empty(0)
#     else: # 如果混雜了空tensor和非空tensor，或者全為空，處理起來較複雜，最好確保一致性
#         feats_ms = torch.empty(0) # 或者返回一個包含空tensor的列表，讓train.py處理

#     if all(f.numel() > 0 for f in feats_sh if isinstance(f, torch.Tensor)):
#         feats_sh = torch.stack([f for f in feats_sh if isinstance(f, torch.Tensor) and f.numel() > 0], 0) if any(isinstance(f, torch.Tensor) and f.numel() > 0 for f in feats_sh) else torch.empty(0)
#     else:
#         feats_sh = torch.empty(0)
        
#     return images, bboxes, masks, feats_ms, feats_sh
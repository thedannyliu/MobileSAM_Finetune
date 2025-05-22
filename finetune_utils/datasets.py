import torch
from torch.utils.data import Dataset
import numpy as np
import random
from pathlib import Path
from PIL import Image
# torchvision.transforms 需要單獨導入，如果要在 __getitem__ 中動態使用
from torchvision import transforms # <--- 新增導入

class ComponentDataset(Dataset):
    """
    ComponentDataset for loading images and multiple object masks for finetuning SAM.
    Each image can have multiple associated object masks, and each image-mask pair
    will be treated as a distinct sample.
    Prompts (box or points) are generated based on the selected mask and mode.
    """
    def __init__(self, root_dir, transform=None, max_bbox_shift=10,
                 prompt_mode='mixed', min_points=1, max_points=3, image_size=1024):
        """
        Args:
            root_dir (string): Directory containing 'image' and 'mask' subdirectories.
            transform (tuple, optional): A tuple of two optional transforms to be applied
                on an image and its mask respectively. transform[0] for image, transform[1] for mask.
                These are typically T.ToTensor(). Resize is handled internally.
            max_bbox_shift (int, optional): Max random perturbation for bounding box coordinates.
            prompt_mode (str, optional): 'box', 'point', or 'mixed'.
            min_points (int, optional): Minimum number of points for a point prompt.
            max_points (int, optional): Maximum number of points for a point prompt.
            image_size (int, optional): The target size to which images and masks are resized.
        """
        self.root_dir = Path(root_dir)
        self.image_dir = self.root_dir / 'image'
        self.mask_dir = self.root_dir / 'mask'

        if not self.image_dir.is_dir() or not self.mask_dir.is_dir():
            raise FileNotFoundError(f"Image or mask directory not found in {root_dir}. "
                                    f"Expected subdirectories 'image' and 'mask'.")

        self.transform_image = transform[0] if transform and len(transform) > 0 else transforms.ToTensor()
        self.transform_mask = transform[1] if transform and len(transform) > 1 else transforms.ToTensor()
        
        self.max_bbox_shift = max_bbox_shift
        self.prompt_mode = prompt_mode
        self.min_points = min_points
        self.max_points = max_points
        self.image_size = image_size # Target size for resizing

        self.samples = []
        image_files = list(self.image_dir.glob('*.jpg')) + \
                      list(self.image_dir.glob('*.jpeg')) + \
                      list(self.image_dir.glob('*.png'))

        for img_path in image_files:
            img_stem = img_path.stem
            corresponding_mask_subdir = self.mask_dir / img_stem
            if corresponding_mask_subdir.is_dir():
                mask_files_in_subdir = sorted(list(corresponding_mask_subdir.glob('*.png')))
                if mask_files_in_subdir:
                    for mask_path in mask_files_in_subdir:
                        # 將圖像 ID (檔名，不含擴展名) 也存儲起來，以便 train.py 使用
                        self.samples.append({'image': img_path, 'mask': mask_path, 'id': img_stem})
                else:
                    print(f"Warning: No mask PNGs found in {corresponding_mask_subdir} for image {img_path.name}")
        
        if not self.samples:
            raise ValueError(f"No image-mask pairs found. Check dataset structure in {root_dir}.")
        
        print(f"Initialized ComponentDataset with {len(self.samples)} samples. Prompt mode: {self.prompt_mode}. Target image size: {self.image_size}")

    def __len__(self):
        return len(self.samples)

    def compute_bbox(self, mask_tensor):
        if mask_tensor.ndim == 3:
            mask_tensor = mask_tensor.squeeze(0)
        thresholded_mask = mask_tensor > 0.5 
        rows_any_white = torch.any(thresholded_mask, dim=1)
        cols_any_white = torch.any(thresholded_mask, dim=0)
        rows_white_indices = torch.where(rows_any_white)[0]
        cols_white_indices = torch.where(cols_any_white)[0]
        if rows_white_indices.nelement() == 0 or cols_white_indices.nelement() == 0:
            return torch.tensor([0, 0, 0, 0], dtype=torch.float)
        y_min, y_max = rows_white_indices[0].item(), rows_white_indices[-1].item()
        x_min, x_max = cols_white_indices[0].item(), cols_white_indices[-1].item()
        return torch.tensor([x_min, y_min, x_max, y_max], dtype=torch.float)

    def compute_point_prompts(self, mask_tensor):
        if mask_tensor.ndim == 3:
            mask_tensor = mask_tensor.squeeze(0)
        foreground_pixels_yx = torch.argwhere(mask_tensor > 0.5)
        point_coords_padded = torch.zeros((self.max_points, 2), dtype=torch.float)
        point_labels_padded = torch.zeros(self.max_points, dtype=torch.long)
        if foreground_pixels_yx.shape[0] == 0:
            return point_coords_padded, point_labels_padded
        num_points_to_sample = random.randint(self.min_points, self.max_points)
        num_actual_points = min(num_points_to_sample, foreground_pixels_yx.shape[0])
        if num_actual_points > 0:
            indices = torch.randperm(foreground_pixels_yx.shape[0])[:num_actual_points]
            sampled_points_yx = foreground_pixels_yx[indices]
            sampled_points_xy = torch.flip(sampled_points_yx, dims=[1]).float()
            point_coords_padded[:num_actual_points] = sampled_points_xy
            point_labels_padded[:num_actual_points] = 1
        return point_coords_padded, point_labels_padded

    def __getitem__(self, idx):
        sample_info = self.samples[idx]
        img_path = sample_info['image']
        mask_path = sample_info['mask']
        img_id = sample_info['id'] # 獲取圖像 ID

        image_pil = Image.open(img_path).convert("RGB")
        mask_pil = Image.open(mask_path).convert("L") # 確保以灰階模式載入遮罩

        # --- MODIFICATION START: Get original image size before resizing ---
        original_w, original_h = image_pil.size 
        original_size_tuple = torch.tensor([original_h, original_w], dtype=torch.int) # SAM expects (H, W) format
        # --- MODIFICATION END ---

        # --- 在轉換為 Tensor 之前，統一調整大小 ---
        target_size = (self.image_size, self.image_size) # 例如 (1024, 1024)
        
        image_resizer = transforms.Resize(target_size, interpolation=transforms.InterpolationMode.BILINEAR)
        mask_resizer = transforms.Resize(target_size, interpolation=transforms.InterpolationMode.NEAREST) # 遮罩用最近鄰

        image_pil_resized = image_resizer(image_pil)
        mask_pil_resized = mask_resizer(mask_pil)
        # --- 修改結束於此 ---

        if self.transform_image:
            transformed_image = self.transform_image(image_pil_resized) # 對調整大小後的圖像進行轉換
        else: # Fallback (如果 train.py 中 transform=None，雖然不太可能)
            transformed_image = transforms.ToTensor()(image_pil_resized)

        if self.transform_mask:
            transformed_mask = self.transform_mask(mask_pil_resized) # 對調整大小後的遮罩進行轉換
        else: # Fallback
            transformed_mask = transforms.ToTensor()(mask_pil_resized)
        
        # 確保遮罩是二值的 [0,1]
        transformed_mask = (transformed_mask > 0.5).float()

        box_prompt_coords = torch.zeros(4, dtype=torch.float)
        point_coords_padded = torch.zeros((self.max_points, 2), dtype=torch.float)
        point_labels_padded = torch.zeros(self.max_points, dtype=torch.long)
        is_point_prompt_item = False

        current_item_prompt_type = self.prompt_mode
        if self.prompt_mode == 'mixed':
            current_item_prompt_type = random.choice(['box', 'point'])

        if current_item_prompt_type == 'box':
            raw_bbox = self.compute_bbox(transformed_mask)
            if not torch.all(raw_bbox == 0):
                h, w = transformed_mask.shape[-2], transformed_mask.shape[-1]
                bbox_width = raw_bbox[2] - raw_bbox[0]
                bbox_height = raw_bbox[3] - raw_bbox[1]
                noise_w_val = 0
                noise_h_val = 0
                if bbox_width.item() > 0 and self.max_bbox_shift > 0:
                    noise_w_val = torch.clamp(torch.randn(1) * bbox_width * 0.1, min=-self.max_bbox_shift, max=self.max_bbox_shift).round().int().item()
                if bbox_height.item() > 0 and self.max_bbox_shift > 0:
                    noise_h_val = torch.clamp(torch.randn(1) * bbox_height * 0.1, min=-self.max_bbox_shift, max=self.max_bbox_shift).round().int().item()
                x_min_shifted = max(0, raw_bbox[0].item() + noise_w_val)
                y_min_shifted = max(0, raw_bbox[1].item() + noise_h_val)
                x_max_shifted = min(w, raw_bbox[2].item() + noise_w_val)
                y_max_shifted = min(h, raw_bbox[3].item() + noise_h_val)
                if x_max_shifted <= x_min_shifted: x_max_shifted = x_min_shifted + 1
                if y_max_shifted <= y_min_shifted: y_max_shifted = y_min_shifted + 1
                x_max_shifted = min(w, x_max_shifted)
                y_max_shifted = min(h, y_max_shifted)
                box_prompt_coords = torch.tensor([x_min_shifted, y_min_shifted, x_max_shifted, y_max_shifted], dtype=torch.float)
            else:
                box_prompt_coords = raw_bbox
            is_point_prompt_item = False
        elif current_item_prompt_type == 'point':
            point_coords_padded, point_labels_padded = self.compute_point_prompts(transformed_mask)
            is_point_prompt_item = True
            
        # 返回一個包含圖像 ID 的字典，train.py 中的 collate_fn 可以處理它
        return {
            "image": transformed_image,
            "mask": transformed_mask,
            "box_prompt": box_prompt_coords, # SAM 期望的 box 格式
            "point_coords": point_coords_padded, # SAM 期望的 point 格式
            "point_labels": point_labels_padded, # SAM 期望的 label 格式
            "is_point_prompt": is_point_prompt_item, # 布林值，指示是否為點提示
            "id": img_id, # 圖像 ID，用於載入教師特徵
            "original_size": original_size_tuple # --- MODIFICATION: Added original_size ---
        }
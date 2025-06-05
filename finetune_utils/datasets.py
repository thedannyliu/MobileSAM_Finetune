# finetune_utils/datasets.py

import torch
from torch.utils.data import Dataset
from torchvision import transforms

import random
from pathlib import Path
from PIL import Image


class ComponentDataset(Dataset):
    """
    SAM Fine-tune 多物件資料集 (Option B)
    • 保留原始尺寸，先在原圖上計算 prompt (box / point)，
      再縮放到 1024×1024，並回傳原始尺寸給 SAM。
    """

    def __init__(
        self,
        root_dir,
        transform=None,
        max_bbox_shift=10,
        prompt_mode="mixed",
        min_points=1,
        max_points=3,
        image_size=1024,
    ):
        self.root_dir = Path(root_dir)
        self.image_dir = self.root_dir / "image"
        self.mask_dir = self.root_dir / "mask"

        if not self.image_dir.is_dir() or not self.mask_dir.is_dir():
            raise FileNotFoundError(f"{root_dir} 缺少 image/ 或 mask/")

        # transform: (transform_image, transform_mask) for AFTER resize
        self.transform_image = transform[0] if transform else transforms.ToTensor()
        self.transform_mask = transform[1] if transform else transforms.ToTensor()

        self.max_bbox_shift = max_bbox_shift
        self.prompt_mode = prompt_mode
        self.min_points = min_points
        self.max_points = max_points
        self.image_size = image_size

        self.samples = []
        image_files = (
            list(self.image_dir.glob("*.jpg"))
            + list(self.image_dir.glob("*.jpeg"))
            + list(self.image_dir.glob("*.png"))
        )

        for img_path in image_files:
            img_stem = img_path.stem
            mask_sub = self.mask_dir / img_stem
            if mask_sub.is_dir():
                for mask_path in sorted(mask_sub.glob("*.png")):
                    self.samples.append({"image": img_path, "mask": mask_path, "id": img_stem})

        if not self.samples:
            raise ValueError(f"{root_dir} 找不到任何樣本")

        print(f"ComponentDataset: {len(self.samples)} samples, prompt={self.prompt_mode}")

        # 用於後續 Resize
        self.img_resize = transforms.Resize(
            (image_size, image_size), transforms.InterpolationMode.BILINEAR
        )
        self.msk_resize = transforms.Resize(
            (image_size, image_size), transforms.InterpolationMode.NEAREST
        )

    @staticmethod
    def _morph_close(mask_tensor: torch.Tensor, k: int = 3):
        if mask_tensor.ndim == 3:
            mask_tensor = mask_tensor.squeeze(0)
        kernel = torch.ones((1, 1, k, k), dtype=torch.float32, device=mask_tensor.device)
        m = mask_tensor.unsqueeze(0).unsqueeze(0)
        dil = torch.nn.functional.conv2d(m, kernel, padding=k // 2)
        dil = (dil > 0).float()
        ero = torch.nn.functional.conv2d(dil, kernel, padding=k // 2)
        ero = (ero >= k * k).float()
        return ero.squeeze()

    def compute_bbox_raw(self, mask_tensor: torch.Tensor):
        """
        在原始尺寸 mask_tensor ([1, H_raw, W_raw]) 上計算 tight bbox。
        回傳 [xmin, ymin, xmax, ymax] (raw scale)。
        """
        m = mask_tensor.squeeze(0)
        m_close = self._morph_close(m)
        rows = torch.any(m_close, dim=1)
        cols = torch.any(m_close, dim=0)
        if not rows.any() or not cols.any():
            return torch.zeros(4, dtype=torch.float)
        y_idx = torch.where(rows)[0]
        x_idx = torch.where(cols)[0]
        y_min, y_max = y_idx[0].item(), y_idx[-1].item()
        x_min, x_max = x_idx[0].item(), x_idx[-1].item()
        if (x_max - x_min) * (y_max - y_min) < 16:
            return torch.zeros(4, dtype=torch.float)
        return torch.tensor([x_min, y_min, x_max, y_max], dtype=torch.float)

    def _jitter_box(self, box: torch.Tensor, h: int, w: int) -> torch.Tensor:
        """Apply random jitter so the box still covers the object."""
        if box.sum() == 0:
            return box

        x_min, y_min, x_max, y_max = box.tolist()

        dx1 = random.randint(-self.max_bbox_shift, self.max_bbox_shift)
        dy1 = random.randint(-self.max_bbox_shift, self.max_bbox_shift)
        dx2 = random.randint(-self.max_bbox_shift, self.max_bbox_shift)
        dy2 = random.randint(-self.max_bbox_shift, self.max_bbox_shift)

        j_xmin = x_min + dx1
        j_ymin = y_min + dy1
        j_xmax = x_max + dx2
        j_ymax = y_max + dy2

        x_min_new = max(0, min(j_xmin, x_min))
        y_min_new = max(0, min(j_ymin, y_min))
        x_max_new = min(w - 1, max(j_xmax, x_max))
        y_max_new = min(h - 1, max(j_ymax, y_max))

        return torch.tensor([x_min_new, y_min_new, x_max_new, y_max_new], dtype=torch.float)

    def compute_point_prompts_raw(self, mask_tensor: torch.Tensor):
        """
        在原始尺寸 mask_tensor ([1, H_raw, W_raw]) 隨機挑 k 個正點 (mask>0.5)，
        其餘填 (-1,-1)，label 填 -1(忽略)。
        """
        m = mask_tensor.squeeze(0)
        fg = torch.argwhere(m > 0.5)  # [[y,x], ...]
        point_coords = torch.full((self.max_points, 2), -1.0, dtype=torch.float)
        point_labels = torch.full((self.max_points,), -1, dtype=torch.long)
        if fg.shape[0] == 0:
            return point_coords, point_labels
        k = random.randint(self.min_points, self.max_points)
        idx = torch.randperm(fg.shape[0])[:k]
        samp = fg[idx].float()  # yx
        samp = torch.flip(samp, dims=[1])  # 轉成 (x,y)
        point_coords[:k] = samp
        point_labels[:k] = 1
        return point_coords, point_labels

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        meta = self.samples[idx]
        img_pil = Image.open(meta["image"]).convert("RGB")
        msk_pil = Image.open(meta["mask"]).convert("L")

        orig_w, orig_h = img_pil.size
        raw_size = torch.tensor([orig_h, orig_w], dtype=torch.int)

        # 1. 先把 raw mask 轉成 tensor 做 prompt
        msk_raw = transforms.ToTensor()(msk_pil)  # [1, H_raw, W_raw]
        msk_raw = (msk_raw > 0.5).float()

        # 計算 raw prompt
        box_prompt_raw = torch.zeros(4, dtype=torch.float)
        point_coords_raw = torch.full((self.max_points, 2), -1.0, dtype=torch.float)
        point_labels_raw = torch.full((self.max_points,), -1, dtype=torch.long)

        cur_type = self.prompt_mode
        if self.prompt_mode == "mixed":
            cur_type = random.choice(["box", "point"])

        if cur_type == "box":
            box_prompt_raw = self.compute_bbox_raw(msk_raw)
            box_prompt_raw = self._jitter_box(box_prompt_raw, orig_h, orig_w)
        else:
            point_coords_raw, point_labels_raw = self.compute_point_prompts_raw(msk_raw)

        # 2. Resize image & mask to image_size × image_size
        img_resized = self.img_resize(img_pil)
        msk_resized = self.msk_resize(msk_pil)

        img_tensor = self.transform_image(img_resized)  # [3, 1024,1024]
        msk_tensor = self.transform_mask(msk_resized)  # [1,1024,1024]
        msk_tensor = (msk_tensor > 0.5).float()

        # 3. 把 raw prompt 縮放到 1024×1024
        scale_x = self.image_size / orig_w
        scale_y = self.image_size / orig_h

        if cur_type == "box":
            if box_prompt_raw.sum() != 0:
                box_prompt = box_prompt_raw.clone()
                box_prompt[0] = box_prompt_raw[0] * scale_x
                box_prompt[1] = box_prompt_raw[1] * scale_y
                box_prompt[2] = box_prompt_raw[2] * scale_x
                box_prompt[3] = box_prompt_raw[3] * scale_y
            else:
                box_prompt = torch.zeros(4, dtype=torch.float)
            point_coords, point_labels = None, None
        else:
            if (point_coords_raw >= 0).any():
                pc = point_coords_raw.clone()
                pc[:, 0] = point_coords_raw[:, 0] * scale_x
                pc[:, 1] = point_coords_raw[:, 1] * scale_y
                point_coords = pc
                point_labels = point_labels_raw.clone()
            else:
                point_coords = torch.full((self.max_points, 2), -1.0, dtype=torch.float)
                point_labels = torch.full((self.max_points,), -1, dtype=torch.long)
            box_prompt = None

        # 4. DEBUG: 隨機小機率印一次 prompt 原始 & 縮放值
        if random.random() < 0.002:
            print(
                f"[DBG] idx={idx}, id={meta['id']}, raw_size={(orig_h,orig_w)}, "
                f"prompt_type={cur_type}, "
                f"box_raw={box_prompt_raw.tolist() if cur_type=='box' else None}, "
                f"box_scaled={box_prompt.tolist() if cur_type=='box' else None}, "
                f"pt_raw={point_coords_raw[:1].tolist() if cur_type=='point' else None}, "
                f"pt_scaled={(point_coords[:1].tolist() if cur_type=='point' else None)}"
            )

        return {
            "image": img_tensor,  # [3,1024,1024]
            "mask": msk_tensor,  # [1,1024,1024]
            "box_prompt": box_prompt if cur_type == "box" else None,
            "point_coords": point_coords if cur_type == "point" else None,
            "point_labels": point_labels if cur_type == "point" else None,
            "id": meta["id"],
            "original_size": raw_size,  # (H_raw, W_raw)
        }

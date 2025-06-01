# ───────────────────────── finetune_utils/datasets.py ─────────────────────────
import random
from pathlib import Path
from typing import Tuple, List

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode


class ComponentDataset(Dataset):
    """
    多物件 (mask) 樣本 -> MobileSAM Fine-tune
    • 影像統一 resize 到 (image_size, image_size) 以減少 GPU 消耗
    • prompt 皆在 **原圖座標** 計算，SAM 會自行做座標縮放
    """

    def __init__(
        self,
        root_dir: str | Path,
        transform: Tuple[transforms.Compose, transforms.Compose] | None = None,
        max_bbox_shift: int = 10,
        prompt_mode: str = "mixed",          # "box" / "point" / "mixed"
        min_points: int = 1,
        max_points: int = 3,
        image_size: int = 1024,
    ):
        self.root_dir = Path(root_dir)
        self.image_dir = self.root_dir / "image"
        self.mask_dir = self.root_dir / "mask"

        if not self.image_dir.is_dir() or not self.mask_dir.is_dir():
            raise FileNotFoundError(f"{root_dir} 缺少 image/ 或 mask/")

        self.transform_img = transform[0] if transform else transforms.ToTensor()
        self.transform_msk = transform[1] if transform else transforms.ToTensor()

        self.max_bbox_shift = max_bbox_shift
        self.prompt_mode = prompt_mode
        self.min_points = min_points
        self.max_points = max_points
        self.image_size = image_size

        # 收集 <img, mask> 配對
        self.samples: List[dict] = []
        img_files = list(self.image_dir.glob("*.jpg")) + \
                    list(self.image_dir.glob("*.jpeg")) + \
                    list(self.image_dir.glob("*.png"))
        for img_path in img_files:
            stem = img_path.stem
            sub_mask_dir = self.mask_dir / stem
            if sub_mask_dir.is_dir():
                for m in sorted(sub_mask_dir.glob("*.png")):
                    self.samples.append({"image": img_path, "mask": m, "id": stem})

        if not self.samples:
            raise ValueError(f"{root_dir} 找不到任何 mask 樣本")

        print(f"ComponentDataset: {len(self.samples)} samples, prompt={self.prompt_mode}")

        # 影像 / mask -> 1024×1024（SAM 預設 1024）
        self.resize_img = transforms.Resize(
            (image_size, image_size), InterpolationMode.BILINEAR
        )
        self.resize_msk = transforms.Resize(
            (image_size, image_size), InterpolationMode.NEAREST
        )

    # ------- 形態學 closing，避免長條/單點 bbox 失效 -------
    @staticmethod
    def _morph_close(mask: torch.Tensor, k: int = 3) -> torch.Tensor:
        if mask.ndim == 3:
            mask = mask.squeeze(0)
        kernel = torch.ones((1, 1, k, k), dtype=torch.float32, device=mask.device)
        m = mask.unsqueeze(0).unsqueeze(0)
        dil = torch.nn.functional.conv2d(m, kernel, padding=k // 2)
        dil = (dil > 0).float()
        ero = torch.nn.functional.conv2d(dil, kernel, padding=k // 2)
        ero = (ero >= k * k).float()
        return ero.squeeze()

    # ------- bbox / point in ORIGINAL coordinate -------
    def _compute_bbox(self, mask_tensor: torch.Tensor) -> torch.Tensor:
        mask = self._morph_close(mask_tensor)
        rows, cols = torch.any(mask, 1), torch.any(mask, 0)
        if not rows.any() or not cols.any():
            return torch.zeros(4)
        y_idx, x_idx = torch.where(rows)[0], torch.where(cols)[0]
        y_min, y_max = y_idx[0].item(), y_idx[-1].item()
        x_min, x_max = x_idx[0].item(), x_idx[-1].item()
        if (x_max - x_min) * (y_max - y_min) < 16:        # tiny object
            return torch.zeros(4)
        return torch.tensor([x_min, y_min, x_max, y_max], dtype=torch.float)

    def _compute_points(self, mask_tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mask = mask_tensor.squeeze(0)
        fg = torch.argwhere(mask > 0.5)           # 前景
        max_pt = self.max_points

        # 預設用 -1 代表「此點無效」
        point_coords = torch.full((max_pt, 2), -1.0, dtype=torch.float)
        point_labels = torch.full((max_pt,), -1, dtype=torch.long)

        if fg.numel() == 0:
            return point_coords, point_labels

        # 正點 (label = 1)
        k_pos = random.randint(self.min_points, max_pt)
        idx = torch.randperm(fg.size(0))[:k_pos]
        samp = torch.flip(fg[idx].float(), dims=[1])   # yx → xy
        point_coords[:k_pos] = samp
        point_labels[:k_pos] = 1

        # 隨機加 0–2 個負點 (label = 0)
        bg = torch.argwhere(mask == 0)
        if bg.size(0) > 0:
            k_neg = random.randint(0, max(0, max_pt - k_pos))
            if k_neg > 0:
                neg_idx = torch.randperm(bg.size(0))[:k_neg]
                neg_samp = torch.flip(bg[neg_idx].float(), dims=[1])
                point_coords[k_pos:k_pos + k_neg] = neg_samp
                point_labels[k_pos:k_pos + k_neg] = 0
        return point_coords, point_labels

    # ------------------------------------------------------------------
    def __getitem__(self, idx: int):
        meta = self.samples[idx]

        # -------- 原圖 / 原尺寸 mask --------
        img_pil = Image.open(meta["image"]).convert("RGB")
        msk_pil = Image.open(meta["mask"]).convert("L")

        orig_w, orig_h = img_pil.size
        original_size = torch.tensor([orig_h, orig_w], dtype=torch.int)

        # 把 mask 轉 tensor (0/1) 以便計算 prompt
        msk_tensor_orig = torch.from_numpy(np.array(msk_pil)).float() / 255.0
        msk_tensor_orig = msk_tensor_orig.unsqueeze(0)   # 1×H×W

        # -------- prompt 決定 --------
        box_prompt = torch.zeros(4)
        point_coords = torch.full((self.max_points, 2), -1.0)
        point_labels = torch.full((self.max_points,), -1, dtype=torch.long)

        mode_cur = self.prompt_mode
        if self.prompt_mode == "mixed":
            mode_cur = random.choice(["box", "point"])

        if mode_cur == "box":
            box_prompt = self._compute_bbox(msk_tensor_orig)
        else:
            point_coords, point_labels = self._compute_points(msk_tensor_orig)

        # -------- 影像 / mask resize → 1024 --------
        img_rs = self.resize_img(img_pil)
        msk_rs = self.resize_msk(msk_pil)

        img_tensor = self.transform_img(img_rs)
        msk_tensor = self.transform_msk(msk_rs)
        msk_tensor = (msk_tensor > 0.5).float()    # binary

        return {
            "image": img_tensor,
            "mask": msk_tensor,
            "box_prompt": box_prompt if mode_cur == "box" else None,
            "point_coords": point_coords if mode_cur == "point" else None,
            "point_labels": point_labels if mode_cur == "point" else None,
            "id": meta["id"],
            "original_size": original_size,        # ← 原圖尺寸 (H, W)
        }

    def __len__(self):
        return len(self.samples)

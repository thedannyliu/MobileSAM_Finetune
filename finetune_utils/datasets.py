# ───────────────────────── finetune_utils/datasets.py ─────────────────────────
import torch
from torch.utils.data import Dataset
import random
from pathlib import Path
from PIL import Image
from torchvision import transforms
class ComponentDataset(Dataset):
    """
    針對 SAM fine-tune 的多物件資料集
    ▶ 單一 Resize → (image_size, image_size)
    ▶ 支援 box / point / mixed prompt
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

        self.transform_image = transform[0] if transform else transforms.ToTensor()
        self.transform_mask = transform[1] if transform else transforms.ToTensor()

        self.max_bbox_shift = max_bbox_shift
        self.prompt_mode = prompt_mode
        self.min_points = min_points
        self.max_points = max_points
        self.image_size = image_size

        self.samples = []
        image_files = list(self.image_dir.glob("*.jpg")) + list(self.image_dir.glob("*.jpeg")) + list(
            self.image_dir.glob("*.png")
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

        self.img_resize = transforms.Resize((image_size, image_size), transforms.InterpolationMode.BILINEAR)
        self.msk_resize = transforms.Resize((image_size, image_size), transforms.InterpolationMode.NEAREST)

    def __len__(self):
        return len(self.samples)

    # ─── util ───
    @staticmethod
    def _morph_close(mask_tensor: torch.Tensor, k: int = 3):
        """
        簡易形態學 closing，避免 tiny bbox
        """
        if mask_tensor.ndim == 3:
            mask_tensor = mask_tensor.squeeze(0)
        kernel = torch.ones((1, 1, k, k), dtype=torch.float32, device=mask_tensor.device)
        m = mask_tensor.unsqueeze(0).unsqueeze(0)
        dil = torch.nn.functional.conv2d(m, kernel, padding=k // 2)
        dil = (dil > 0).float()
        ero = torch.nn.functional.conv2d(dil, kernel, padding=k // 2)
        ero = (ero >= k * k).float()
        return ero.squeeze()

    def compute_bbox(self, mask_tensor: torch.Tensor):
        mask = self._morph_close(mask_tensor)
        rows, cols = torch.any(mask, 1), torch.any(mask, 0)
        if not rows.any() or not cols.any():
            return torch.zeros(4, dtype=torch.float)
        y_idx, x_idx = torch.where(rows)[0], torch.where(cols)[0]
        y_min, y_max = y_idx[0].item(), y_idx[-1].item()
        x_min, x_max = x_idx[0].item(), x_idx[-1].item()
        # 面積過小則視為無效 bbox
        if (x_max - x_min) * (y_max - y_min) < 16:
            return torch.zeros(4, dtype=torch.float)
        return torch.tensor([x_min, y_min, x_max, y_max], dtype=torch.float)

    def compute_point_prompts(self, mask_tensor: torch.Tensor):
        mask = mask_tensor.squeeze(0)
        fg = torch.argwhere(mask > 0.5)
        point_coords = torch.zeros((self.max_points, 2), dtype=torch.float)
        point_labels = torch.zeros(self.max_points, dtype=torch.long)
        if fg.shape[0] == 0:
            return point_coords, point_labels
        k = random.randint(self.min_points, self.max_points)
        idx = torch.randperm(fg.shape[0])[:k]
        samp = fg[idx]
        samp = torch.flip(samp.float(), dims=[1])  # yx→xy
        point_coords[: k] = samp
        point_labels[: k] = 1
        return point_coords, point_labels

    def __getitem__(self, idx):
        meta = self.samples[idx]
        img_pil = Image.open(meta["image"]).convert("RGB")
        msk_pil = Image.open(meta["mask"]).convert("L")

        orig_w, orig_h = img_pil.size
        original_size = torch.tensor([orig_h, orig_w], dtype=torch.int)

        img_resized = self.img_resize(img_pil)
        msk_resized = self.msk_resize(msk_pil)

        img_tensor = self.transform_image(img_resized)
        msk_tensor = self.transform_mask(msk_resized)
        msk_tensor = (msk_tensor > 0.5).float()

        # prompts
        box_prompt = torch.zeros(4, dtype=torch.float)
        point_coords = torch.zeros((self.max_points, 2), dtype=torch.float)
        point_labels = torch.zeros(self.max_points, dtype=torch.long)
        cur_type = self.prompt_mode
        if self.prompt_mode == "mixed":
            cur_type = random.choice(["box", "point"])

        if cur_type == "box":
            box_prompt = self.compute_bbox(msk_tensor)
        else:
            point_coords, point_labels = self.compute_point_prompts(msk_tensor)

        return {
            "image": img_tensor,
            "mask": msk_tensor,
            "box_prompt": box_prompt if cur_type == "box" else None,
            "point_coords": point_coords if cur_type == "point" else None,
            "point_labels": point_labels if cur_type == "point" else None,
            "id": meta["id"],
            "original_size": original_size,
        }

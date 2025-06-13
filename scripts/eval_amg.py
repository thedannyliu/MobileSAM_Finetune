"""Evaluate a finetuned MobileSAM model using SamAutomaticMaskGenerator.

The dataset root should contain `image/` and `mask/` folders with matching
subfolders for each dataset.  For example:

```
root/
  image/dataset1/xxx.jpg
  mask/dataset1/xxx/object0.png
  mask/dataset1/xxx/object1.png
```

Every image has a directory of ground truth masks under `mask/<dataset>/<id>/`.
The script runs SamAutomaticMaskGenerator on each image, matches the predicted
masks to the ground truths one-to-one using a Hungarian assignment, and reports
mean IoU per dataset and overall.
"""

from __future__ import annotations
import numpy as np
import cv2  # type: ignore
import torch

from mobile_sam import SamAutomaticMaskGenerator, sam_model_registry

import argparse
from pathlib import Path
from scipy.optimize import linear_sum_assignment
from typing import List


def load_gt_masks(mask_dir: Path) -> torch.Tensor:
    """Load all PNG masks under ``mask_dir`` as a tensor ``[K, H, W]``."""
    masks = []
    for mp in sorted(mask_dir.glob("*.png")):
        arr = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
        if arr is None:
            continue
        masks.append(torch.from_numpy(arr > 127).float())
    if not masks:
        raise FileNotFoundError(f"No masks found in {mask_dir}")
    return torch.stack(masks)


def evaluate_image(amg, img_path: Path, mask_dir: Path) -> List[float]:
    """Return IoU scores for all matched objects in ``img_path``."""
    bgr = cv2.imread(str(img_path))
    if bgr is None:
        raise FileNotFoundError(img_path)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    preds = amg.generate(rgb)
    if not preds:
        return []
    pred_masks = torch.stack(
        [torch.from_numpy(p["segmentation"]).float() for p in preds]
    )  # [N, H, W]

    gt_masks = load_gt_masks(mask_dir)  # [K, H, W]

    pred_bin = pred_masks.unsqueeze(1)
    gt_bin = gt_masks.unsqueeze(1)

    inter = (pred_bin.unsqueeze(1) * gt_bin.unsqueeze(0)).sum((-2, -1))
    union = pred_bin.unsqueeze(1).sum((-2, -1)) + gt_bin.unsqueeze(0).sum((-2, -1)) - inter
    iou_mat = inter / (union + 1e-6)

    cost = (-iou_mat).cpu().numpy()
    row_ind, col_ind = linear_sum_assignment(cost)
    assert len(set(row_ind)) == len(row_ind)
    assert len(set(col_ind)) == len(col_ind)

    return [iou_mat[r, c].item() for r, c in zip(row_ind, col_ind)]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True, help="Path to finetuned weights")
    ap.add_argument("--data", required=True, help="Dataset root containing image/ and mask/")
    ap.add_argument("--model-type", default="vit_t", help="Model type used during training")
    ap.add_argument("--device", default="cuda", help="Device for inference")
    args = ap.parse_args()

    sam = sam_model_registry[args.model_type](checkpoint=args.checkpoint)
    sam.to(device=args.device)
    sam.eval()
    amg = SamAutomaticMaskGenerator(sam)

    root = Path(args.data)
    image_root = root / "image"
    mask_root = root / "mask"

    datasets = [d.name for d in image_root.iterdir() if d.is_dir()]
    if not datasets:
        raise FileNotFoundError(f"No datasets found under {image_root}")

    overall_ious: List[float] = []

    for ds in sorted(datasets):
        ds_images = image_root / ds
        ds_masks = mask_root / ds
        if not ds_masks.is_dir():
            print(f"Mask folder missing for {ds}, skipping")
            continue

        ds_ious: List[float] = []
        for img_file in sorted(ds_images.iterdir()):
            if img_file.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            gt_dir = ds_masks / img_file.stem
            if not gt_dir.is_dir():
                continue
            ious = evaluate_image(amg, img_file, gt_dir)
            ds_ious.extend(ious)
            overall_ious.extend(ious)

        if ds_ious:
            print(f"{ds}: mIoU={np.mean(ds_ious):.4f} over {len(ds_ious)} objects")
        else:
            print(f"{ds}: no valid samples")

    if overall_ious:
        print(f"Overall mIoU: {np.mean(overall_ious):.4f} over {len(overall_ious)} objects")
    else:
        print("No results to report")


if __name__ == "__main__":
    main()

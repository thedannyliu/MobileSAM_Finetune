"""Evaluate MobileSAM checkpoints on one or more datasets using ``SamAutomaticMaskGenerator``.

The script reads a YAML config specifying checkpoints and datasets::

    weights:
      - name: run1
        path: weights/mobilesam_run1.pth
    datasets:
      - name: val
        image_dir: datasets/val/images
        mask_dir: datasets/val/masks
    output_csv: results/eval.csv

Each mask directory contains subfolders named after the corresponding image
file (without extension) and stores binary PNG masks.  For every checkpoint the
script reports the mean IoU and average inference time for each dataset in the
CSV file.
"""

from __future__ import annotations
import numpy as np
import cv2  # type: ignore
import torch

from mobile_sam import SamAutomaticMaskGenerator, sam_model_registry

import argparse
import csv
import time
from pathlib import Path
from typing import List

import yaml


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


def evaluate_image(amg, img_path: Path, mask_dir: Path) -> tuple[List[float], float]:
    """Return IoU scores for ``img_path`` and the inference time in seconds."""
    bgr = cv2.imread(str(img_path))
    if bgr is None:
        raise FileNotFoundError(img_path)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    start = time.perf_counter()
    preds = amg.generate(rgb)
    elapsed = time.perf_counter() - start
    if not preds:
        return [], elapsed
    pred_masks = torch.stack(
        [torch.from_numpy(p["segmentation"]).float() for p in preds]
    )  # [N, H, W]

    gt_masks = load_gt_masks(mask_dir)  # [K, H, W]

    gt_bin = gt_masks.unsqueeze(1)
    pred_bin = pred_masks.unsqueeze(1)

    inter = (gt_bin.unsqueeze(1) * pred_bin.unsqueeze(0)).sum((-2, -1))
    union = gt_bin.unsqueeze(1).sum((-2, -1)) + pred_bin.unsqueeze(0).sum((-2, -1)) - inter
    iou_mat = inter / (union + 1e-6)

    max_iou, _ = iou_mat.max(dim=1)
    return max_iou.cpu().tolist(), elapsed


def evaluate_dataset(amg, image_dir: Path, mask_dir: Path) -> tuple[float, float]:
    """Return mean IoU and average inference time over all images in a dataset."""
    all_ious: List[float] = []
    times: List[float] = []
    for img_file in sorted(image_dir.iterdir()):
        if img_file.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        gt_dir = mask_dir / img_file.stem
        if not gt_dir.is_dir():
            continue
        ious, t = evaluate_image(amg, img_file, gt_dir)
        all_ious.extend(ious)
        times.append(t)
    if not all_ious:
        return 0.0, float("nan")
    return float(np.mean(all_ious)), float(np.mean(times))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True, help="Path to YAML config file")
    args = ap.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    model_type = cfg.get("model_type", "vit_t")
    device = cfg.get("device", "cuda")
    weights = cfg.get("weights", [])
    datasets = cfg.get("datasets", [])
    csv_path = Path(cfg.get("output_csv", "eval_results.csv"))

    rows = []
    for w in weights:
        name = w.get("name") or Path(w["path"]).stem
        sam = sam_model_registry[model_type](checkpoint=w["path"])
        sam.to(device=device)
        sam.eval()
        amg = SamAutomaticMaskGenerator(sam)

        result = {"name": name}
        for ds in datasets:
            miou, avg_t = evaluate_dataset(
                amg, Path(ds["image_dir"]), Path(ds["mask_dir"])
            )
            result[f"{ds['name']}_mIoU"] = f"{miou:.4f}"
            result[f"{ds['name']}_time"] = f"{avg_t:.4f}"
        rows.append(result)

    fieldnames = ["name"]
    for ds in datasets:
        fieldnames.append(f"{ds['name']}_mIoU")
        fieldnames.append(f"{ds['name']}_time")

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


if __name__ == "__main__":
    main()

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
    viz_dir: results/vis

Each mask directory contains subfolders named after the corresponding image
file (without extension) and stores binary PNG masks.  For every checkpoint the
script reports the mean IoU and average inference time for each dataset in the
CSV file.  If ``viz_dir`` is provided, the script also saves a side-by-side
visualisation of ``SamAutomaticMaskGenerator`` results for every image.
"""

from __future__ import annotations
import numpy as np
import cv2  # type: ignore
import torch

from mobile_sam import SamAutomaticMaskGenerator, sam_model_registry

import argparse
import csv
import time
import resource
from pathlib import Path
from typing import Any, Dict, List

from PIL import Image, ImageDraw, ImageFont
from finetune_utils.visualization import generate_distinct_colors
import yaml  # type: ignore


def mem_usage_mb() -> float:
    """Return the current process RSS in megabytes."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def build_model(model_type: str, checkpoint: str, device: str):
    """Load a SAM checkpoint with debug output and move to ``device``."""
    print(f"\n[build_model] loading {checkpoint} as {model_type}")
    start = time.perf_counter()
    sam = sam_model_registry[model_type](checkpoint=checkpoint)
    print(
        f"[build_model] checkpoint loaded in {time.perf_counter() - start:.2f}s; RSS {mem_usage_mb():.1f} MB"
    )
    sam.to(device=device)
    print(f"[build_model] moved model to {device}; RSS {mem_usage_mb():.1f} MB")
    sam.eval()
    return sam


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


def overlay_masks(rgb: np.ndarray, masks: torch.Tensor) -> Image.Image:
    """Overlay ``masks`` on ``rgb`` and return a PIL image."""
    base = Image.fromarray(rgb).convert("RGBA")
    if masks.ndim == 4:
        masks = masks.squeeze(1)
    colors = generate_distinct_colors(masks.shape[0])
    for idx, m in enumerate(masks):
        m_bin = (m > 0.5).cpu().numpy().astype(np.uint8) * 255
        if m_bin.max() == 0:
            continue
        mask_pil = Image.fromarray(m_bin, mode="L")
        layer = Image.new("RGBA", base.size, colors[idx])
        base.paste(layer, mask=mask_pil)
    return base.convert("RGB")


def save_collage(
    orig_rgb: np.ndarray, overlays: List[tuple[str, Image.Image]], out_path: Path
) -> None:
    """Save ``orig_rgb`` and ``overlays`` side-by-side with titles."""
    orig_img = Image.fromarray(orig_rgb)
    images = [orig_img] + [img for _, img in overlays]
    titles = ["original"] + [name for name, _ in overlays]
    w, h = orig_img.size
    canvas = Image.new("RGB", (w * len(images), h + 20), "white")
    font = ImageFont.load_default()
    for idx, (title, im) in enumerate(zip(titles, images)):
        canvas.paste(im, (idx * w, 20))
        draw = ImageDraw.Draw(canvas)
        tw, th = draw.textsize(title, font=font)
        draw.text((idx * w + (w - tw) / 2, 0), title, fill="black", font=font)
    canvas.save(out_path)


def evaluate_image(
    amg, img_path: Path, mask_dir: Path
) -> tuple[List[float], float, Image.Image]:
    """Return IoU scores, inference time, and overlay for ``img_path``."""
    bgr = cv2.imread(str(img_path))
    if bgr is None:
        raise FileNotFoundError(img_path)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    start = time.perf_counter()
    preds = amg.generate(rgb)
    elapsed = time.perf_counter() - start
    if not preds:
        return [], elapsed, Image.fromarray(rgb)
    pred_masks = torch.stack(
        [torch.from_numpy(p["segmentation"]).float() for p in preds]
    )  # [N, H, W]

    gt_masks = load_gt_masks(mask_dir)  # [K, H, W]

    gt_bin = gt_masks.unsqueeze(1)
    pred_bin = pred_masks.unsqueeze(1)

    inter = (gt_bin.unsqueeze(1) * pred_bin.unsqueeze(0)).sum((-2, -1))
    union = gt_bin.unsqueeze(1).sum((-2, -1)) + pred_bin.unsqueeze(0).sum((-2, -1)) - inter
    iou_mat = inter / (union + 1e-6)

    overlay = overlay_masks(rgb, pred_masks)

    max_iou, _ = iou_mat.max(dim=1)
    return max_iou.cpu().tolist(), elapsed, overlay


def evaluate_dataset(
    amg,
    image_dir: Path,
    mask_dir: Path,
    weight_name: str,
    overlay_store: Dict[str, Dict[str, Image.Image]],
) -> tuple[float, float]:
    """Return mean IoU and average inference time over all images in a dataset."""
    all_ious: List[float] = []
    times: List[float] = []
    for img_file in sorted(image_dir.iterdir()):
        if img_file.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        gt_dir = mask_dir / img_file.stem
        if not gt_dir.is_dir():
            continue
        ious, t, overlay = evaluate_image(amg, img_file, gt_dir)
        overlay_store.setdefault(img_file.stem, {})[weight_name] = overlay
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
    viz_dir = cfg.get("viz_dir")

    ds_info: Dict[str, Dict[str, Any]] = {}
    overlay_cache: Dict[str, Dict[str, Dict[str, Image.Image]]] = {}
    for ds in datasets:
        name = ds["name"]
        img_dir = Path(ds["image_dir"])
        mask_dir = Path(ds["mask_dir"])
        images = {
            p.stem: p
            for p in sorted(img_dir.iterdir())
            if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
        }
        ds_info[name] = {"image_dir": img_dir, "mask_dir": mask_dir, "images": images}
        overlay_cache[name] = {k: {} for k in images.keys()}

    rows = []
    for w in weights:
        name = w.get("name") or Path(w["path"]).stem
        sam = build_model(model_type, w["path"], device)
        amg = SamAutomaticMaskGenerator(sam)

        result = {"name": name}
        for ds in datasets:
            ds_name = ds["name"]
            info = ds_info[ds_name]
            miou, avg_t = evaluate_dataset(
                amg,
                info["image_dir"],
                info["mask_dir"],
                name,
                overlay_cache[ds_name],
            )
            result[f"{ds_name}_mIoU"] = f"{miou:.4f}"
            result[f"{ds_name}_time"] = f"{avg_t:.4f}"
        rows.append(result)

    fieldnames = ["name"]
    for ds in datasets:
        ds_name = ds["name"]
        fieldnames.append(f"{ds_name}_mIoU")
        fieldnames.append(f"{ds_name}_time")

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    if viz_dir:
        viz_root = Path(viz_dir)
        for ds in datasets:
            ds_name = ds["name"]
            ds_viz = viz_root / ds_name
            ds_viz.mkdir(parents=True, exist_ok=True)
            info = ds_info[ds_name]
            for stem, img_path in info["images"].items():
                overlays = []
                for w in weights:
                    w_name = w.get("name") or Path(w["path"]).stem
                    ov = overlay_cache[ds_name].get(stem, {}).get(w_name)
                    if ov is not None:
                        overlays.append((w_name, ov))
                if not overlays:
                    continue
                bgr = cv2.imread(str(img_path))
                if bgr is None:
                    continue
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                out_path = ds_viz / f"{stem}_seg_every.jpg"
                save_collage(rgb, overlays, out_path)


if __name__ == "__main__":
    main()

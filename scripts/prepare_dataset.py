import os
import shutil
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

"""prepare_dataset.py

This script organizes COCO 2017 (YOLO segmentation label version) images and annotations
into the following hierarchy:

<root>/dataset/
    ├── train/
    │   ├── image/  # Original images (jpg)
    │   └── mask/   # A directory for each image, containing binary masks for each object (png)
    └── val/
        ├── image/
        └── mask/

Mask directory structure:
    dataset/<split>/mask/<image_id>/<object_idx>.png
    - Background is black (0)
    - Object pixels are white (255)

Assuming the original file structure is:
    <root>/train2017/*.jpg
    <root>/val2017/*.jpg
    <root>/coco/labels/train2017/*.txt  (YOLO seg annotations)
    <root>/coco/labels/val2017/*.txt

Usage:
    python scripts/prepare_dataset.py  # Default root is the script's parent directory
    python scripts/prepare_dataset.py --root /path/to/MobileSAM-fast-finetuning
"""

import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Re-arrange COCO segmentation dataset into image/mask folders.")
    parser.add_argument(
        "--root",
        type=str,
        default=str(Path(__file__).resolve().parent.parent),
        help="Project root directory (containing train2017, val2017, coco/labels, etc.)"
    )
    parser.add_argument("--verbose", action="store_true", help="Display detailed skip messages")
    parser.add_argument("--train-count", type=int, default=100, help="Number of train image-mask pairs (default: 100)")
    parser.add_argument("--val-count", type=int, default=50, help="Number of val image-mask pairs (default: 50)")
    parser.add_argument("--seed", type=int, default=42, help="Random sampling seed")
    return parser.parse_args()


def yolo_to_binary_masks(img_path: Path, label_path: Path, mask_out_dir: Path):
    """Generate a binary mask for each object from a YOLO segmentation label and save as PNG."""
    if not label_path.exists():
        return False  # No annotation

    # Read image dimensions
    img = cv2.imread(str(img_path))
    if img is None:
        raise RuntimeError(f"Could not read image: {img_path}")

    h, w = img.shape[:2]

    with open(label_path, "r") as f:
        lines = [ln.strip() for ln in f.readlines() if ln.strip()]

    if not lines:
        return False  # Empty annotation

    # Create output directory
    mask_out_dir.mkdir(parents=True, exist_ok=True)

    for idx, line in enumerate(lines):
        parts = line.split()
        if len(parts) < 7:
            # A polygon requires at least 3 points (6 coordinates) + class_id
            continue
        # parts[0] is the class_id
        coords = list(map(float, parts[1:]))
        pts = np.array([
            [coords[i * 2] * w, coords[i * 2 + 1] * h] for i in range(len(coords) // 2)
        ], dtype=np.int32)

        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, [pts], 255)

        out_path = mask_out_dir / f"{idx:03d}.png"
        cv2.imwrite(str(out_path), mask)

    return True


def copy_and_generate(img_paths, images_dir, labels_dir, out_image_dir, out_mask_root, verbose=False):
    """Copy specified images and generate their corresponding masks."""
    out_image_dir.mkdir(parents=True, exist_ok=True)
    out_mask_root.mkdir(parents=True, exist_ok=True)

    for img_path in tqdm(img_paths, desc=f"Writing to {out_image_dir.parent.name}"):
        stem = img_path.stem
        label_path = labels_dir / f"{stem}.txt"

        shutil.copy2(img_path, out_image_dir / img_path.name)

        mask_sub_dir = out_mask_root / stem
        success = yolo_to_binary_masks(img_path, label_path, mask_sub_dir)

        if not success:
            if verbose:
                print(f"[No annotation] for {label_path}, removing corresponding image.")
            (out_image_dir / img_path.name).unlink(missing_ok=True)
            shutil.rmtree(mask_sub_dir, ignore_errors=True)


def split_from_val(root: Path, train_count: int, val_count: int, seed: int = 42, verbose: bool = False):
    """Sample train/val images from val2017."""
    images_dir = root / "val2017"
    labels_dir = root / "coco" / "labels" / "val2017"

    if not images_dir.exists():
        raise FileNotFoundError(f"Directory not found: {images_dir}")

    # Collect all images that have valid annotations
    valid_images = []
    for img_path in tqdm(sorted(images_dir.glob("*.jpg")), desc="Scanning for valid annotations"):
        stem = img_path.stem
        label_path = labels_dir / f"{stem}.txt"
        if not label_path.exists():
            continue
        with open(label_path, "r") as f:
            if not any(line.strip() for line in f):
                continue
        valid_images.append(img_path)

    total_needed = train_count + val_count
    if len(valid_images) < total_needed:
        raise RuntimeError(f"Not enough images with valid annotations: needed {total_needed}, found {len(valid_images)}")

    import random
    rng = random.Random(seed)
    rng.shuffle(valid_images)

    train_imgs = valid_images[:train_count]
    val_imgs = valid_images[train_count:train_count + val_count]

    # Copy images and generate masks
    copy_and_generate(
        train_imgs,
        images_dir,
        labels_dir,
        root / "dataset-1" / "train" / "image",
        root / "dataset-1" / "train" / "mask",
        verbose=verbose,
    )

    copy_and_generate(
        val_imgs,
        images_dir,
        labels_dir,
        root / "dataset-1" / "val" / "image",
        root / "dataset-1" / "val" / "mask",
        verbose=verbose,
    )


def main():
    args = parse_args()
    root = Path(args.root).expanduser().resolve()

    split_from_val(
        root,
        train_count=args.train_count,
        val_count=args.val_count,
        seed=args.seed,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main() 
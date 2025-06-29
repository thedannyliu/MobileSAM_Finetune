import os
import shutil
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

"""prepare_dataset.py

這支腳本會將 COCO 2017 (YOLO segmentation label 版本) 的影像與標註整理成以下階層：

<root>/dataset/
    ├── train/
    │   ├── image/  # 原始影像 (jpg)
    │   └── mask/   # 每張影像對應一個資料夾，內含各物件的二值遮罩 (png)
    └── val/
        ├── image/
        └── mask/

mask 目錄說明：
    dataset/<split>/mask/<image_id>/<object_idx>.png
    - 背景為黑色 (0)
    - 物件像素為白色 (255)

假設原始檔案結構為：
    <root>/train2017/*.jpg
    <root>/val2017/*.jpg
    <root>/coco/labels/train2017/*.txt  (YOLO seg 標註)
    <root>/coco/labels/val2017/*.txt

用法：
    python prepare_dataset.py  # 預設 root 為腳本所在目錄
    python prepare_dataset.py --root /path/to/MobileSAM-fast-finetuning
"""

import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Re-arrange COCO segmentation dataset into image/mask folders.")
    parser.add_argument(
        "--root",
        type=str,
        default=str(Path(__file__).resolve().parent.parent),
        help="專案根目錄 (包含 train2017, val2017, coco/labels 等資料夾)"
    )
    parser.add_argument("--verbose", action="store_true", help="顯示詳細跳過訊息")
    parser.add_argument("--train-count", type=int, default=100, help="train 影像-遮罩 pair 數量 (預設100)")
    parser.add_argument("--val-count", type=int, default=50, help="val 影像-遮罩 pair 數量 (預設50)")
    parser.add_argument("--seed", type=int, default=42, help="隨機抽樣 seed")
    return parser.parse_args()


def yolo_to_binary_masks(img_path: Path, label_path: Path, mask_out_dir: Path):
    """根據 YOLO segmentation label 產生每個物件的二值遮罩，並存成 png。"""
    if not label_path.exists():
        return False  # 無標註

    # 讀取影像尺寸
    img = cv2.imread(str(img_path))
    if img is None:
        raise RuntimeError(f"無法讀取影像: {img_path}")

    h, w = img.shape[:2]

    with open(label_path, "r") as f:
        lines = [ln.strip() for ln in f.readlines() if ln.strip()]

    if not lines:
        return False  # 空標註

    # 建立輸出目錄
    mask_out_dir.mkdir(parents=True, exist_ok=True)

    for idx, line in enumerate(lines):
        parts = line.split()
        if len(parts) < 7:
            # 至少需要 class_id + 3 個點的 xy (6) 才能形成多邊形
            continue
        # parts[0] 是 class_id
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
    """將指定 img_paths 影像複製並產生遮罩。"""
    out_image_dir.mkdir(parents=True, exist_ok=True)
    out_mask_root.mkdir(parents=True, exist_ok=True)

    for img_path in tqdm(img_paths, desc=f"寫入 {out_image_dir.parent.name}"):
        stem = img_path.stem
        label_path = labels_dir / f"{stem}.txt"

        shutil.copy2(img_path, out_image_dir / img_path.name)

        mask_sub_dir = out_mask_root / stem
        success = yolo_to_binary_masks(img_path, label_path, mask_sub_dir)

        if not success:
            if verbose:
                print(f"[無標註] {label_path}，將移除對應影像。")
            (out_image_dir / img_path.name).unlink(missing_ok=True)
            shutil.rmtree(mask_sub_dir, ignore_errors=True)


def split_from_val(root: Path, train_count: int, val_count: int, seed: int = 42, verbose: bool = False):
    """從 val2017 中抽樣 train/val 影像。"""
    images_dir = root / "val2017"
    labels_dir = root / "coco" / "labels" / "val2017"

    if not images_dir.exists():
        raise FileNotFoundError(f"找不到 {images_dir}")

    # 收集所有具有有效遮罩的影像
    valid_images = []
    for img_path in tqdm(sorted(images_dir.glob("*.jpg")), desc="掃描有效標註"):
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
        raise RuntimeError(f"有效標註影像不足: 需要 {total_needed}，僅有 {len(valid_images)}")

    import random
    rng = random.Random(seed)
    rng.shuffle(valid_images)

    train_imgs = valid_images[:train_count]
    val_imgs = valid_images[train_count:train_count + val_count]

    # 複製與產生遮罩
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
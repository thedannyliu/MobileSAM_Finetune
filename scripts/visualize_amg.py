"""Visualize SamAutomaticMaskGenerator results with finetuned MobileSAM weights.

Example:
    python scripts/visualize_amg.py \
        --checkpoint weights/best_student.pth \
        --image path/to/image.jpg \
        --model-type vit_t \
        --output output.jpg
"""

import cv2  # type: ignore
import torch

from mobile_sam import SamAutomaticMaskGenerator, sam_model_registry

import argparse
from finetune_utils.visualization import overlay_masks_on_image
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Run SamAutomaticMaskGenerator using a finetuned MobileSAM model and "
        "save an overlay visualization of the masks."
    )
    ap.add_argument("--checkpoint", type=str, required=True, help="Path to finetuned weights")
    ap.add_argument(
        "--model-type",
        type=str,
        default="vit_t",
        help="Model type used during training (e.g. vit_t)",
    )
    ap.add_argument("--image", type=str, required=True, help="Input image path")
    ap.add_argument(
        "--output",
        type=str,
        default="amg_vis.jpg",
        help="Where to save the overlay visualization",
    )
    ap.add_argument("--device", type=str, default="cuda", help="Device for inference")
    args = ap.parse_args()

    print("Loading model...")
    sam = sam_model_registry[args.model_type](checkpoint=args.checkpoint)
    sam.to(device=args.device)
    sam.eval()

    mask_generator = SamAutomaticMaskGenerator(sam)

    bgr = cv2.imread(args.image)
    if bgr is None:
        raise FileNotFoundError(args.image)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    print("Generating masks...")
    masks = mask_generator.generate(rgb)
    if not masks:
        print("No masks generated")
        return

    mask_tensors = torch.stack(
        [torch.as_tensor(m["segmentation"], dtype=torch.float32) for m in masks]
    )
    img_tensor = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0

    out_path = Path(args.output)
    overlay_masks_on_image(
        image_tensor=img_tensor,
        masks=mask_tensors,
        save_dir=out_path.parent,
        filename_info=out_path.stem,
    )
    print(f"Visualization saved to {out_path}")


if __name__ == "__main__":
    main()

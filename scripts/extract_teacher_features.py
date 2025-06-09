"""
This script precomputes and saves features from a teacher model for a given dataset.
These precomputed features can then be used to accelerate the distillation process during student model training.

The script is designed to be run from the root directory of the MobileSAM-fast-finetuning project.

Example usage:

1. To precompute features for the SAM ViT-H teacher (using FP16 for less memory):
python scripts/extract_teacher_features.py \
    --teacher_name SAM_vitH \
    --teacher_cfg ./configs/sam_vith.yaml \
    --teacher_ckpt ./weights/sam_vit_h_4b8939.pth \
    --dataset_base_dir ./datasets \
    --output_base_dir ./precomputed/SAM_vitH \
    --splits train val \
    --fp16

2. To precompute features for the original MobileSAM teacher:
python scripts/extract_teacher_features.py \
    --teacher_name MobileSAM_orig \
    --teacher_cfg ./configs/mobile_sam_orig.yaml \
    --teacher_ckpt ./weights/mobile_sam.pt \
    --dataset_base_dir ./datasets \
    --output_base_dir ./precomputed/MobileSAM_orig \
    --splits train val
"""

import numpy as np
import torch
import torchvision.transforms as T

from mobile_sam import sam_model_registry

import argparse
import pathlib
import yaml
from finetune_utils.feature_hooks import pop_features, register_hooks
from PIL import Image
from tqdm import tqdm


# This function is a direct copy from `train.py` to ensure consistency.
def _build_pot(model_type: str):
    p = {"enc": [], "dec": [], "attn": [], "rkd": ["image_encoder.patch_embed"]}
    if model_type == "vit_t":
        p["enc"] = ["image_encoder.neck"]
        p["dec"] = ["mask_decoder.output_upscaling"]
        p["attn"] = [
            "image_encoder.layers.1.blocks.0.attn",
            "image_encoder.layers.1.blocks.1.attn",
            "image_encoder.layers.2.blocks.0.attn",
            "image_encoder.layers.2.blocks.1.attn",
            "image_encoder.layers.2.blocks.2.attn",
            "image_encoder.layers.2.blocks.3.attn",
            "image_encoder.layers.2.blocks.4.attn",
            "image_encoder.layers.2.blocks.5.attn",
            "image_encoder.layers.3.blocks.0.attn",
            "image_encoder.layers.3.blocks.1.attn",
        ]
    else:  # vit_b, vit_l, vit_h
        p["enc"] = [f"image_encoder.blocks.{i}" for i in (9, 10, 11, 12)]
        # Official SAM models do not expose `pre_logits`; reuse output_upscaling
        # for decoder matching to avoid missing features.
        p["dec"] = ["mask_decoder.output_upscaling"]
        p["attn"] = [f"image_encoder.blocks.{i}.attn" for i in range(12)]
    return p


def process_images_in_dir(
    model,
    image_dir_path: pathlib.Path,
    output_dir: pathlib.Path,
    capture_targets: list,
    transform: T.Compose,
    device: str,
):
    """
    Processes all images in a directory, extracts features, and saves them.
    """
    if not image_dir_path.is_dir():
        print(f"Info: Image directory {image_dir_path} does not exist. Skipping.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    img_extensions = ["*.[jJ][pP][gG]", "*.[jJ][pP][eE][gG]", "*.[pP][nN][gG]"]
    imgs = []
    for ext in img_extensions:
        imgs.extend(list(image_dir_path.glob(ext)))
    imgs = sorted(list(set(imgs)))

    if not imgs:
        print(f"Warning: No images (jpg, jpeg, png) found in {image_dir_path}.")
        return

    print(f"Processing {len(imgs)} images from {image_dir_path}...")
    handles = register_hooks(model, capture_targets)

    for p in tqdm(imgs, desc=f"Extracting features in {image_dir_path.name}"):
        try:
            pil_img = Image.open(p).convert("RGB")
            original_h, original_w = pil_img.height, pil_img.width

            # Use the same preprocessing as in training
            img_tensor = transform(pil_img).to(device)

            # If model is in half-precision, input tensor must also be converted.
            if next(model.parameters()).dtype == torch.float16:
                img_tensor = img_tensor.half()

            batched_item = {
                "image": img_tensor,  # Do NOT add batch dimension, model.forward will do it.
                "original_size": (original_h, original_w),
            }

            with torch.no_grad():
                # The model's internal preprocessor will handle resizing.
                # We only need to provide a normalized tensor.
                # For feature extraction, we don't need prompts or multimask output.
                _ = model(batched_input=[batched_item], multimask_output=False)

            feats = pop_features()
            if not feats:
                print(
                    f"Warning: No features were captured for image {p.name}. Check if `capture_targets` are correct for the model."
                )
                continue

            for key, captured_list_of_features in feats.items():
                if captured_list_of_features:
                    feature_tensor = captured_list_of_features[0]
                    if isinstance(feature_tensor, torch.Tensor):
                        numpy_feature = feature_tensor.cpu().squeeze(0).numpy()
                        sanitized_key = key.replace(".", "_").replace("[", "_").replace("]", "")
                        feature_filename = f"{p.stem}_{sanitized_key}.npy"
                        feature_save_path = output_dir / feature_filename
                        np.save(feature_save_path, numpy_feature)
                    else:
                        print(
                            f"Warning: Captured item for key '{key}' is not a tensor for image {p.name}."
                        )
                else:
                    print(
                        f"Warning: No features captured in the list for key '{key}' for image {p.name}."
                    )

        except Exception as e:
            print(f"Critical error processing file {p.name}: {e}")
            import traceback

            traceback.print_exc()

    for h in handles:
        h.remove()
    print(f"Finished processing images in {image_dir_path}. Features saved to {output_dir}")


def main():
    ap = argparse.ArgumentParser(
        description="Extract features from a teacher model for specified dataset splits."
    )
    ap.add_argument(
        "--teacher_name",
        required=True,
        help="Name of the teacher model (e.g., 'SAM_vitH', 'MobileSAM_orig').",
    )
    ap.add_argument(
        "--teacher_cfg", required=True, help="Path to the teacher model's YAML configuration file."
    )
    ap.add_argument(
        "--teacher_ckpt", required=True, help="Path to the teacher model's checkpoint file."
    )
    ap.add_argument(
        "--dataset_base_dir",
        required=True,
        type=pathlib.Path,
        help="Base directory for the datasets (e.g., './datasets').",
    )
    ap.add_argument(
        "--output_base_dir",
        required=True,
        type=pathlib.Path,
        help="Base directory for saving precomputed features. A subdirectory with `teacher_name` will be created here.",
    )
    ap.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val"],
        help="List of dataset splits to process (e.g., 'train' 'val'). Default: ['train', 'val']",
    )
    ap.add_argument(
        "--image_subdir_name",
        default="image",
        help="Subdirectory name under each split containing images. Default: 'image'",
    )
    ap.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use for computation. Default: 'cuda' if available, else 'cpu'",
    )
    ap.add_argument(
        "--fp16",
        action="store_true",
        help="Use half-precision (FP16) for inference to reduce memory usage.",
    )
    args = ap.parse_args()

    with open(args.teacher_cfg, "r") as f:
        teacher_config = yaml.safe_load(f)

    model_config_dict = teacher_config.get("model")
    if model_config_dict is None:
        raise ValueError(f"Error: 'model' key not found in the teacher config: {args.teacher_cfg}")

    model_type = model_config_dict.get("type")
    if model_type is None:
        raise ValueError(
            f"Error: 'type' not found under 'model' key in the teacher config: {args.teacher_cfg}"
        )

    # Use the same logic as train.py to determine which features to capture.
    pot = _build_pot(model_type)
    capture_targets = sorted(list(set(layer for layers in pot.values() for layer in layers)))
    print(f"Model type '{model_type}' requires capturing {len(capture_targets)} feature layers.")
    print("Capture targets:", capture_targets)

    try:
        model_builder = sam_model_registry[model_type]
    except KeyError:
        raise ValueError(
            f"Error: Unsupported 'model_type' ('{model_type}') in {args.teacher_cfg}. Available: {list(sam_model_registry.keys())}"
        )

    print(
        f"Building model '{model_type}' with checkpoint '{args.teacher_ckpt}' for teacher '{args.teacher_name}'"
    )
    model = model_builder(checkpoint=args.teacher_ckpt).to(args.device).eval()

    if args.fp16:
        if args.device == "cuda":
            print("-> Using half-precision (FP16) for inference.")
            model.half()
        else:
            print(
                "Warning: --fp16 was specified, but device is not 'cuda'. FP16 is only supported on CUDA devices. Ignoring."
            )

    # Resize images to the same size used during training to ensure that
    # precomputed features match the online pipeline.
    img_size = model_config_dict.get("image_size", 1024)
    transform = T.Compose([T.Resize((img_size, img_size)), T.ToTensor()])

    # The output directory will be named after the teacher
    output_root_path = args.output_base_dir

    for split_name in args.splits:
        print(f"\nProcessing split: {split_name}")
        image_dir_path = args.dataset_base_dir / split_name / args.image_subdir_name
        split_output_dir = output_root_path / split_name

        process_images_in_dir(
            model=model,
            image_dir_path=image_dir_path,
            output_dir=split_output_dir,
            capture_targets=capture_targets,
            transform=transform,
            device=args.device,
        )

    print(f"\n✅ Feature extraction process completed for teacher '{args.teacher_name}'.")
    print(f"Precomputed features saved in: {output_root_path}")


if __name__ == "__main__":
    main()

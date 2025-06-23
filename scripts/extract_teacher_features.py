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

For segment-everything training, add `--mode everything` and optionally
`--grid_points <step>` to match the training configuration.
"""

import numpy as np
import torch
import torchvision.transforms as T

from mobile_sam import sam_model_registry
from mobile_sam.utils.transforms import ResizeLongestSide

import argparse
import pathlib
import yaml
from finetune_utils.feature_hooks import pop_features, register_hooks
from PIL import Image
from tqdm import tqdm


def _build_pot(model_type: str):
    """Return list of layer names required for the new four-way distillation."""
    pot = {
        "enc_patch": [],
        "prompt_embed": ["mask_decoder.transformer"],
        "mask_token": ["mask_decoder.transformer"],
    }

    if model_type == "vit_t":
        pot["enc_patch"] = ["image_encoder.neck"]
    else:
        pot["enc_patch"] = ["image_encoder.blocks.30"]

    return pot


def process_images_in_dir(
    model,
    image_dir_path: pathlib.Path,
    output_dir: pathlib.Path,
    capture_targets: list,
    preprocess,
    resizer,
    device: str,
    mode: str = "single",
    grid_points: int = 32,
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
            img_tensor = preprocess(pil_img).to(device)

            # If model is in half-precision, input tensor must also be converted.
            if next(model.parameters()).dtype == torch.float16:
                img_tensor = img_tensor.half()

            if mode == "single":
                batched_item = {
                    "image": img_tensor,
                    "original_size": (original_h, original_w),
                }

                with torch.no_grad():
                    _ = model(batched_input=[batched_item], multimask_output=False)
            else:
                from mobile_sam.utils.amg import build_point_grid

                grid = torch.from_numpy(build_point_grid(grid_points)).float()
                grid[:, 0] *= original_w
                grid[:, 1] *= original_h
                grid = resizer.apply_coords_torch(grid, (original_h, original_w))

                from mobile_sam.utils.amg import batch_iterator

                inp = model.preprocess(img_tensor.unsqueeze(0))
                embedding = model.image_encoder(inp)
                dense_pe = model.prompt_encoder.get_dense_pe()

                with torch.no_grad():
                    for (pts,) in batch_iterator(64, grid):
                        coords = torch.as_tensor(pts, dtype=torch.float, device=device)
                        labels = torch.ones(coords.shape[0], dtype=torch.int, device=device)
                        sparse, dense = model.prompt_encoder(
                            points=(coords.unsqueeze(0), labels.unsqueeze(0)),
                            boxes=None,
                            masks=None,
                        )
                        _ = model.mask_decoder(
                            image_embeddings=embedding,
                            image_pe=dense_pe,
                            sparse_prompt_embeddings=sparse,
                            dense_prompt_embeddings=dense,
                            multimask_output=True,
                        )

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
    ap.add_argument(
        "--mode",
        choices=["single", "everything"],
        default="single",
        help="Dataset mode used during training. If 'everything', grid prompts are generated.",
    )
    ap.add_argument(
        "--grid_points",
        type=int,
        default=32,
        help="Grid size (in pixels) used when mode is 'everything'.",
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
    resizer = ResizeLongestSide(img_size)

    def preprocess_pil(img: Image.Image) -> torch.Tensor:
        h, w = img.height, img.width
        new_h, new_w = resizer.get_preprocess_shape(h, w, img_size)
        img_resized = img.resize((new_w, new_h), Image.BILINEAR)
        t = T.ToTensor()(img_resized)
        if new_h != img_size or new_w != img_size:
            pad_r = img_size - new_w
            pad_b = img_size - new_h
            t = torch.nn.functional.pad(t, (0, pad_r, 0, pad_b))
        return t

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
            preprocess=preprocess_pil,
            resizer=resizer,
            device=args.device,
            mode=args.mode,
            grid_points=args.grid_points,
        )

    print(f"\n✅ Feature extraction process completed for teacher '{args.teacher_name}'.")
    print(f"Precomputed features saved in: {output_root_path}")


if __name__ == "__main__":
    main()

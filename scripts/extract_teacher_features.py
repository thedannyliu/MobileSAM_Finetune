"""Extract features for *any* teacher (SAM 或 Original‑MobileSAM).
Usage:
    python scripts/extract_teacher_features.py \
        --teacher_name SAM_vitH \
        --teacher_cfg configs/sam_vith.yaml \
        --teacher_ckpt weights/sam_vith.pth \
        --dataset_dir datasets/train \
        --output_dir precomputed/SAM_vitH/train
可重複執行多教師。
"""
import argparse, pathlib, numpy as np, torch
from PIL import Image
import torchvision.transforms as T
from finetune_utils.feature_hooks import register_hooks, pop_features
from mobilesam import build_model  # adapt to real API

_CAPTURE = [
    "image_encoder.blocks.9", "image_encoder.blocks.10", "image_encoder.blocks.11", "image_encoder.blocks.12",
    "mask_decoder.pre_logits", "image_encoder.patch_embed"
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher_name")
    ap.add_argument("--teacher_cfg")
    ap.add_argument("--teacher_ckpt")
    ap.add_argument("--dataset_dir")
    ap.add_argument("--output_dir")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    model = build_model(args.teacher_cfg, args.teacher_ckpt).to(args.device).eval()
    handles = register_hooks(model, _CAPTURE)
    tfms = T.Compose([T.Resize(1024), T.ToTensor()])

    out_dir = pathlib.Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    imgs = sorted(list(pathlib.Path(args.dataset_dir).glob("*.*")))
    for p in imgs:
        img = tfms(Image.open(p).convert("RGB")).unsqueeze(0).to(args.device)
        with torch.no_grad():
            _ = model(img)
        feats = pop_features()
        np.savez_compressed(out_dir / f"{p.stem}.npz", **{k: v.cpu().squeeze(0).numpy() for k, v in feats.items()})
        print("saved", p.name)
    for h in handles:
        h.remove()

if __name__ == "__main__":
    main()

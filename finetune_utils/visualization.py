"""Visualization helpers for MobileSAM fine‑tuning
-------------------------------------------------
• Overlay predicted mask (red, 40 % opacity)
• Draw prompt box (lime) if provided
• Draw positive / negative point prompts
  ▸ positive (label==1)  → green solid circle
  ▸ negative (label==0) → red ‘×’ marker
• All prompts are accepted in *original‑image* coordinate system.
  If you pass the pair `original_size=(H_raw, W_raw)` we will
  automatically scale them to the (H_resize, W_resize) space of the
  visualised image (1024 × 1024 in our pipeline).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
from PIL import Image, ImageDraw
from torchvision.transforms.functional import to_pil_image

# ------------------------------------------------------------
# Main public API
# ------------------------------------------------------------

def overlay_mask_on_image(
    image_tensor: torch.Tensor,                 # C × H × W, 0‑1 or uint8
    mask_tensor: torch.Tensor,                  # H × W or 1×H×W (sigmoid)
    bbox_tensor: Optional[torch.Tensor] = None, # [xmin,ymin,xmax,ymax] in *orig* coords
    point_coords: Optional[torch.Tensor] = None,# N × 2  (x,y) in *orig* coords
    point_labels: Optional[torch.Tensor] = None,# N      1 = pos, 0 = neg, ‑1 = ignore
    *,
    original_size: Optional[Tuple[int, int]] = None,  # (H_raw, W_raw)
    threshold: float = 0.5,
    save_dir: str | Path = "./images",
    filename_info: str = "vis"
):
    """Render overlay and save to *save_dir* / <filename_info>.jpg.

    All tensors should reside on **CPU**; call `.cpu()` first if needed.
    Return value is the PIL.Image object for quick preview in notebook.
    """

    # --------------------------------------------------------
    # Resolve paths
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    out_path = save_dir / f"{filename_info}.jpg"

    # --------------------------------------------------------
    # Prepare base image (RGB)
    if image_tensor.dtype == torch.uint8:
        pil_img = to_pil_image(image_tensor, mode="RGB")
    else:  # float 0‑1
        pil_img = to_pil_image((image_tensor.clamp(0, 1) * 255).byte(), mode="RGB")

    w_rs, h_rs = pil_img.size        # resized spatial size (1024×1024)

    # --------------------------------------------------------
    # Prepare predicted mask → binary 0/1   (same spatial size as resized img)
    if mask_tensor.ndim == 3 and mask_tensor.shape[0] == 1:
        mask_2d = mask_tensor.squeeze(0)
    elif mask_tensor.ndim == 2:
        mask_2d = mask_tensor
    else:
        raise ValueError(f"mask_tensor shape must be H×W or 1×H×W, got {mask_tensor.shape}")

    if mask_2d.shape != (h_rs, w_rs):
        # interpolate to resized resolution (nearest keeps edge)
        mask_2d = torch.nn.functional.interpolate(
            mask_2d.unsqueeze(0).unsqueeze(0), size=(h_rs, w_rs), mode="nearest"
        ).squeeze()

    mask_bin = (mask_2d > threshold).cpu().numpy().astype(np.uint8) * 255
    mask_pil = Image.fromarray(mask_bin, mode="L")

    # --------------------------------------------------------
    # RGBA canvas → put mask with alpha
    canvas = pil_img.convert("RGBA")
    # red overlay, alpha = 100 (~40 % opaque)
    red_layer = Image.new("RGBA", canvas.size, (255, 0, 0, 100))
    canvas.paste(red_layer, mask=mask_pil)

    draw = ImageDraw.Draw(canvas)

    # --------------------------------------------------------
    # Coordinate scaling helper
    def _scale(x: torch.Tensor) -> torch.Tensor:
        """Scale coordinates from original to resized space."""
        if original_size is None:
            return x.clone()
        h_raw, w_raw = original_size if isinstance(original_size, (list, tuple)) else (
            int(original_size[0]), int(original_size[1])
        )
        sx, sy = w_rs / w_raw, h_rs / h_raw
        x = x.clone().float()
        if x.ndim == 1 and x.numel() == 4:            # bbox
            x[0::2] *= sx  # x coords
            x[1::2] *= sy  # y coords
        elif x.ndim == 2:                              # points N×2
            x[:, 0] *= sx
            x[:, 1] *= sy
        return x

    # --------------------------------------------------------
    # Draw bbox if provided
    if bbox_tensor is not None and bbox_tensor.numel() == 4:
        box = _scale(bbox_tensor).tolist()
        if not all(v == 0 for v in box):               # ignore 0‑box
            draw.rectangle(box, outline="lime", width=3)

    # --------------------------------------------------------
    # Draw points (pos/neg)
    if point_coords is not None and point_coords.numel() > 0:
        pts = _scale(point_coords)
        if point_labels is None:
            point_labels = torch.ones(pts.size(0), dtype=torch.long)
        for (x, y), lbl in zip(pts.tolist(), point_labels.tolist()):
            if lbl == -1:
                continue  # ignored
            r = 5
            if lbl == 1:             # positive
                draw.ellipse([(x - r, y - r), (x + r, y + r)], fill="green", outline="black")
            elif lbl == 0:           # negative
                # draw red cross
                draw.line([(x - r, y - r), (x + r, y + r)], fill="red", width=2)
                draw.line([(x - r, y + r), (x + r, y - r)], fill="red", width=2)

    # --------------------------------------------------------
    # Save & return
    final_rgb = canvas.convert("RGB")
    final_rgb.save(out_path)
    return final_rgb

# finetune_utils/visualization.py

import numpy as np
import torch
from torchvision.transforms.functional import to_pil_image

from pathlib import Path
from PIL import Image, ImageDraw
from typing import Optional, Tuple


def overlay_mask_on_image(
    image_tensor: torch.Tensor,  # C×H×W, float [0,1] or uint8
    mask_tensor: torch.Tensor,  # H×W or 1×H×W (sigmoid output)
    bbox_tensor: Optional[
        torch.Tensor
    ] = None,  # [xmin, ymin, xmax, ymax] in orig coords
    point_coords: Optional[torch.Tensor] = None,  # N×2 (x,y) in orig coords
    point_labels: Optional[torch.Tensor] = None,  # N, 1=positive, 0=negative, -1=ignore
    *,
    original_size: Optional[Tuple[int, int]] = None,  # (H_raw, W_raw)
    threshold: float = 0.5,
    save_dir: str | Path = "./images",
    filename_info: str = "vis",
):
    """Render overlay and save to {save_dir}/{filename_info}.jpg.

    All tensors must be on CPU. Returns the PIL.Image for preview.
    """

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    out_path = save_dir / f"{filename_info}.jpg"

    # Prepare base image (RGB)
    if image_tensor.dtype == torch.uint8:
        pil_img = to_pil_image(image_tensor, mode="RGB")
    else:
        pil_img = to_pil_image((image_tensor.clamp(0, 1) * 255).byte(), mode="RGB")

    w_rs, h_rs = pil_img.size  # resized size (e.g. 1024×1024)

    # Prepare predicted mask → binary
    if mask_tensor.ndim == 3 and mask_tensor.shape[0] == 1:
        mask_2d = mask_tensor.squeeze(0)
    elif mask_tensor.ndim == 2:
        mask_2d = mask_tensor
    else:
        raise ValueError(f"mask_tensor must be H×W or 1×H×W, got {mask_tensor.shape}")

    if mask_2d.shape != (h_rs, w_rs):
        mask_2d = torch.nn.functional.interpolate(
            mask_2d.unsqueeze(0).unsqueeze(0),
            size=(h_rs, w_rs),
            mode="nearest",  # 1×1×h'×w'
        ).squeeze()

    mask_bin = (mask_2d > threshold).cpu().numpy().astype(np.uint8) * 255
    mask_pil = Image.fromarray(mask_bin, mode="L")

    # RGBA canvas for overlay
    canvas = pil_img.convert("RGBA")
    red_layer = Image.new("RGBA", canvas.size, (255, 0, 0, 100))  # 40% opaque red
    canvas.paste(red_layer, mask=mask_pil)

    draw = ImageDraw.Draw(canvas)

    # Coordinate scaling helper
    def _scale(x: torch.Tensor) -> torch.Tensor:
        if original_size is None:
            return x.clone()
        h_raw, w_raw = original_size
        sx, sy = w_rs / w_raw, h_rs / h_raw
        x = x.clone().float()
        if x.ndim == 1 and x.numel() == 4:  # bbox
            x[0::2] *= sx
            x[1::2] *= sy
        elif x.ndim == 2:  # points N×2
            x[:, 0] *= sx
            x[:, 1] *= sy
        return x

    # Draw bbox if provided
    if bbox_tensor is not None and bbox_tensor.numel() == 4:
        box = _scale(bbox_tensor).tolist()
        if not all(v == 0 for v in box):
            draw.rectangle(box, outline="lime", width=3)

    # Draw point prompts
    if point_coords is not None and point_coords.numel() > 0:
        pts = _scale(point_coords)
        if point_labels is None:
            point_labels = torch.ones(pts.size(0), dtype=torch.long)
        for (x, y), lbl in zip(pts.tolist(), point_labels.tolist()):
            if lbl == -1:
                continue
            r = 5
            if lbl == 1:  # positive
                draw.ellipse(
                    [(x - r, y - r), (x + r, y + r)], fill="green", outline="black"
                )
            elif lbl == 0:  # negative
                draw.line([(x - r, y - r), (x + r, y + r)], fill="red", width=2)
                draw.line([(x - r, y + r), (x + r, y - r)], fill="red", width=2)

    final_rgb = canvas.convert("RGB")
    final_rgb.save(out_path)
    return final_rgb


def overlay_masks_on_image(
    image_tensor: torch.Tensor,
    masks: torch.Tensor,
    *,
    grid_points: Optional[torch.Tensor] = None,
    original_size: Optional[Tuple[int, int]] = None,
    threshold: float = 0.5,
    save_dir: str | Path = "./images",
    filename_info: str = "se_vis",
) -> Image.Image:
    """Overlay multiple masks with distinct colors.

    Parameters
    ----------
    image_tensor: torch.Tensor
        Base image in C×H×W format.
    masks: torch.Tensor
        Predicted masks at either the original size or the same size as the
        image.  If different they will be resized automatically.
    grid_points: Optional[torch.Tensor]
        Grid prompts in original coordinates.  They will be scaled if
        ``original_size`` is provided.
    original_size: Optional[Tuple[int, int]]
        Height and width of the original image used for prompt generation.
    """

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    out_path = save_dir / f"{filename_info}.jpg"

    if image_tensor.dtype == torch.uint8:
        base = to_pil_image(image_tensor, mode="RGB")
    else:
        base = to_pil_image((image_tensor.clamp(0, 1) * 255).byte(), mode="RGB")

    base = base.convert("RGBA")
    draw = ImageDraw.Draw(base)

    b_w, b_h = base.size

    def _resize_mask(t: torch.Tensor) -> torch.Tensor:
        if t.ndim == 3 and t.shape[0] == 1:
            t = t.squeeze(0)
        if t.shape != (b_h, b_w):
            t = torch.nn.functional.interpolate(
                t.unsqueeze(0).unsqueeze(0),
                size=(b_h, b_w),
                mode="nearest",
            ).squeeze()
        return t

    def _scale_points(pts: torch.Tensor) -> torch.Tensor:
        if original_size is None:
            return pts.clone()
        h_raw, w_raw = original_size
        sx, sy = b_w / w_raw, b_h / h_raw
        out = pts.clone().float()
        out[:, 0] *= sx
        out[:, 1] *= sy
        return out

    if masks.ndim == 4:
        masks = masks.squeeze(1)
    colors = [
        (255, 0, 0, 80),
        (0, 255, 0, 80),
        (0, 0, 255, 80),
        (255, 255, 0, 80),
        (255, 0, 255, 80),
        (0, 255, 255, 80),
    ]

    for idx, m in enumerate(masks):
        m_resized = _resize_mask(m)
        m_bin = (m_resized > threshold).cpu().numpy().astype(np.uint8) * 255
        if m_bin.max() == 0:
            continue
        color = colors[idx % len(colors)]
        mask_pil = Image.fromarray(m_bin, mode="L")
        layer = Image.new("RGBA", base.size, color)
        base.paste(layer, mask=mask_pil)

    if grid_points is not None and grid_points.numel() > 0:
        pts = _scale_points(grid_points)
        for x, y in pts.tolist():
            if x < 0 or y < 0:
                continue
            draw.ellipse(
                [(x - 2, y - 2), (x + 2, y + 2)],
                fill="blue",
                outline="black",
            )

    final_img = base.convert("RGB")
    final_img.save(out_path)
    return final_img

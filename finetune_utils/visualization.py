# finetune_utils/visualization.py

import numpy as np
import torch
from torchvision.transforms.functional import to_pil_image
import colorsys

from mobile_sam.utils.transforms import ResizeLongestSide

from pathlib import Path
from PIL import Image, ImageDraw
from typing import Optional, Tuple, List


def generate_distinct_colors(n: int, alpha: int = 80) -> List[tuple[int, int, int, int]]:
    """Generate ``n`` visually distinct RGBA colors."""
    colors: List[tuple[int, int, int, int]] = []
    golden_ratio = 0.618033988749895
    h = 0.0
    for _ in range(n):
        h = (h + golden_ratio) % 1.0
        r, g, b = colorsys.hsv_to_rgb(h, 1.0, 1.0)
        colors.append((int(r * 255), int(g * 255), int(b * 255), alpha))
    return colors


def overlay_mask_on_image(
    image_tensor: torch.Tensor,  # C×H×W, float [0,1] or uint8
    mask_tensor: torch.Tensor,  # H×W or 1×H×W (sigmoid output)
    bbox_tensor: Optional[torch.Tensor] = None,  # [xmin, ymin, xmax, ymax] in orig coords
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

    if original_size is not None:
        h_raw, w_raw = int(original_size[0]), int(original_size[1])
        resizer = ResizeLongestSide(max(image_tensor.shape[-2:]))
        new_h, new_w = resizer.get_preprocess_shape(h_raw, w_raw, resizer.target_length)
        image_tensor = image_tensor[:, :new_h, :new_w]
        image_tensor = torch.nn.functional.interpolate(
            image_tensor.unsqueeze(0),
            size=(h_raw, w_raw),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
        if mask_tensor.shape[-2:] != (h_raw, w_raw):
            mask_tensor = torch.nn.functional.interpolate(
                mask_tensor.unsqueeze(0),
                size=(h_raw, w_raw),
                mode="nearest",
            ).squeeze(0)

    if image_tensor.dtype == torch.uint8:
        pil_img = to_pil_image(image_tensor, mode="RGB")
    else:
        pil_img = to_pil_image((image_tensor.clamp(0, 1) * 255).byte(), mode="RGB")
    w_rs, h_rs = pil_img.size

    if mask_tensor.shape[-2:] != (h_rs, w_rs):
        mask_tensor = torch.nn.functional.interpolate(
            mask_tensor.unsqueeze(0),
            size=(h_rs, w_rs),
            mode="nearest",
        ).squeeze(0)

    # Prepare predicted mask → binary
    if mask_tensor.ndim == 3 and mask_tensor.shape[0] == 1:
        mask_2d = mask_tensor.squeeze(0)
    elif mask_tensor.ndim == 2:
        mask_2d = mask_tensor
    else:
        raise ValueError(f"mask_tensor must be H×W or 1×H×W, got {mask_tensor.shape}")

    if mask_2d.shape != (h_rs, w_rs):
        mask_2d = torch.nn.functional.interpolate(
            mask_2d.unsqueeze(0).unsqueeze(0), size=(h_rs, w_rs), mode="nearest"  # 1×1×h'×w'
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
            r = 6  # Increase point radius for better visibility
            if lbl == 1:  # positive
                draw.ellipse([(x - r, y - r), (x + r, y + r)], fill="green", outline="black")
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
    original_size: Optional[Tuple[int, int]] = None,
    grid_points: Optional[torch.Tensor] = None,
    threshold: float = 0.5,
    save_dir: str | Path = "./images",
    filename_info: str = "se_vis",
) -> Image.Image:
    """Overlay multiple masks with distinct colors."""

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    out_path = save_dir / f"{filename_info}.jpg"

    if original_size is not None:
        h_raw, w_raw = int(original_size[0]), int(original_size[1])
        resizer = ResizeLongestSide(max(image_tensor.shape[-2:]))
        new_h, new_w = resizer.get_preprocess_shape(h_raw, w_raw, resizer.target_length)
        image_tensor = image_tensor[:, :new_h, :new_w]
        image_tensor = torch.nn.functional.interpolate(
            image_tensor.unsqueeze(0),
            size=(h_raw, w_raw),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
        if masks.shape[-2:] != (h_raw, w_raw):
            masks = torch.nn.functional.interpolate(
                masks,
                size=(h_raw, w_raw),
                mode="nearest",
            )

    if image_tensor.dtype == torch.uint8:
        base = to_pil_image(image_tensor, mode="RGB")
    else:
        base = to_pil_image((image_tensor.clamp(0, 1) * 255).byte(), mode="RGB")

    base = base.convert("RGBA")
    draw = ImageDraw.Draw(base)

    if masks.ndim == 4:
        masks = masks.squeeze(1)
    colors = generate_distinct_colors(masks.shape[0])

    for idx, m in enumerate(masks):
        m_bin = (m > threshold).cpu().numpy().astype(np.uint8) * 255
        if m_bin.max() == 0:
            continue
        color = colors[idx % len(colors)]
        mask_pil = Image.fromarray(m_bin, mode="L")
        layer = Image.new("RGBA", base.size, color)
        base.paste(layer, mask=mask_pil)

    if grid_points is not None and grid_points.numel() > 0:
        pts = grid_points.float().cpu()
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

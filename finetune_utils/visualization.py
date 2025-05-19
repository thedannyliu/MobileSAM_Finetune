import torch
from torchvision.transforms.functional import to_pil_image
from PIL import Image, ImageDraw # 確保 Image 也被匯入
from pathlib import Path
import numpy as np # 用於顏色轉換

def overlay_mask_on_image(
    image_tensor: torch.Tensor,        # 輸入的影像張量 (CxHxW)
    mask_tensor: torch.Tensor,         # 預測的遮罩 (sigmoid 輸出, HxW or 1xHxW)
    bbox_tensor: torch.Tensor | None = None, # 可選的框提示 (4,)
    points_tensor: torch.Tensor | None = None, # 可選的點提示 (N, 2)
    # gt_mask_tensor: torch.Tensor | None = None, # <<-- 移除地面真實遮罩參數
    threshold: float = 0.5,
    save_dir: str = "./images",
    filename_info: str = "epoch_0_batch_0"
):
    """
    在影像上疊加預測遮罩、框提示(可選)和點提示(可選)。
    移除了地面真實遮罩的顯示。

    參數:
    - image_tensor (torch.Tensor): CxHxW 格式的原始影像張量 (通常是反正規化後的)。
    - mask_tensor (torch.Tensor): HxW 或 1xHxW 格式的預測遮罩張量 (通常是 sigmoid 輸出)。
    - bbox_tensor (torch.Tensor | None, optional): 4 個元素的框座標張量 [xmin, ymin, xmax, ymax]。預設為 None。
    - points_tensor (torch.Tensor | None, optional): Nx2 的點座標張量 [[x1,y1], [x2,y2], ...]。預設為 None。
    - threshold (float, optional): 應用於預測遮罩的閾值。預設為 0.5。
    - save_dir (str, optional): 儲存結果影像的目錄。預設為 "./images"。
    - filename_info (str, optional): 附加到儲存影像檔名的額外資訊。
    """

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    image_path = save_dir / f'{filename_info}.jpg'

    # 確保 mask_tensor 是 2D 的 (HxW)
    if mask_tensor.ndim == 3 and mask_tensor.shape[0] == 1:
        pred_mask_2d = mask_tensor.squeeze(0)
    elif mask_tensor.ndim == 2:
        pred_mask_2d = mask_tensor
    else:
        # 如果遮罩維度不正確，打印錯誤並返回，或者嘗試修復
        print(f"Warning: Unexpected mask_tensor shape: {mask_tensor.shape} in visualization. Expected HxW or 1xHxW.")
        # 可以選擇返回原始影像或引發錯誤
        if image_tensor.dtype == torch.uint8:
            to_pil_image(image_tensor).save(image_path)
        else:
            to_pil_image((image_tensor.clamp(0,1)*255).byte(), mode='RGB').save(image_path)
        return

    # 對預測遮罩應用閾值
    pred_binary_mask = (pred_mask_2d > threshold).float()

    # 將影像張量從 CxHxW 轉換為 PIL Image (RGB)
    if image_tensor.dtype == torch.uint8:
        pil_image = to_pil_image(image_tensor, mode='RGB')
    else:
        img_to_convert = image_tensor.clamp(0, 1) # 確保在 [0,1] 範圍
        pil_image = to_pil_image((img_to_convert * 255).byte(), mode='RGB')

    # 準備一個可以繪製的 RGBA 影像副本，以便疊加半透明遮罩
    overlay_image = pil_image.convert("RGBA")
    draw_context = ImageDraw.Draw(overlay_image)

    # 疊加預測遮罩 (例如：紅色，半透明)
    # 檢查預測遮罩尺寸是否與影像匹配，如果不匹配則進行插值 (作為保險)
    if pred_binary_mask.shape != (image_tensor.shape[1], image_tensor.shape[2]):
         pred_binary_mask = torch.nn.functional.interpolate(
             pred_binary_mask.unsqueeze(0).unsqueeze(0), # 增加批次和通道維度
             size=(image_tensor.shape[1], image_tensor.shape[2]), # 目標尺寸 H, W
             mode='nearest' # 使用最近鄰插值保持邊緣清晰
         ).squeeze() # 移除批次和通道維度

    pred_mask_np = pred_binary_mask.cpu().numpy().astype(np.uint8) * 255
    pred_mask_pil = Image.fromarray(pred_mask_np, mode="L") # 'L' 模式代表灰階圖
    # 創建一個紅色的顏色層，使用預測遮罩作為透明度
    # RGBA 中的 A 代表 Alpha (透明度)，0 是完全透明，255 是完全不透明
    # (255, 0, 0, 100) 中的 100 代表約 39% 的不透明度 (100/255)
    overlay_image.paste(Image.new("RGBA", overlay_image.size, (255, 0, 0, 100)), mask=pred_mask_pil)

    # --- 移除 GT Mask 相關的繪製邏輯 ---

    # 繪製邊界框 (如果提供)
    if bbox_tensor is not None and bbox_tensor.numel() == 4:
        bbox_list = bbox_tensor.tolist()
        # 只繪製真正有內容的框 (避免全是0的框)
        if not (bbox_list[0] == 0 and bbox_list[1] == 0 and bbox_list[2] == 0 and bbox_list[3] == 0):
             draw_context.rectangle(bbox_list, outline='lime', width=3)

    # 繪製點提示 (如果提供)
    if points_tensor is not None and points_tensor.numel() > 0:
        point_coords = points_tensor.tolist()
        radius = 5 # 點的半徑
        for (x, y) in point_coords:
            # 確保 x, y 是數值型態
            if isinstance(x, (int, float)) and isinstance(y, (int, float)):
                draw_context.ellipse(
                    [(x - radius, y - radius), (x + radius, y + radius)],
                    fill='yellow', outline='black' # 黃色填充，黑色邊框
                )
            else:
                print(f"Warning: Invalid coordinate type for point drawing: x={x}, y={y}")


    final_image_to_save = overlay_image.convert("RGB") # 轉換回 RGB 以儲存為 JPG
    final_image_to_save.save(image_path)

    return final_image_to_save
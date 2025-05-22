import torch
from torch import nn
from torch.nn import functional as F

from functools import partial
# --- 修改：從 typing 模組匯入 Optional 和 Tuple ---
from typing import Any, Dict, List, Union, Optional, Tuple # Tuple 也匯入以備用
# --- 修改結束 ---
import logging

from mobile_sam.modeling.tiny_vit_sam import TinyViT
from mobile_sam.modeling.image_encoder import ImageEncoderViT
from mobile_sam.modeling.mask_decoder import MaskDecoder
from mobile_sam.modeling.prompt_encoder import PromptEncoder
from mobile_sam.modeling import TwoWayTransformer

# logger = logging.getLogger(__name__) # 通常在主腳本中設定 logger

class Sam(nn.Module):
    mask_threshold: float = 0.0
    image_format: str = "RGB"

    def __init__(
        self,
        image_encoder: Union[ImageEncoderViT, TinyViT],
        prompt_encoder: PromptEncoder,
        mask_decoder: MaskDecoder,
        pixel_mean: List[float] = [123.675, 116.28, 103.53],
        pixel_std: List[float] = [58.395, 57.12, 57.375],
    ) -> None:
        super().__init__()
        self.image_encoder = image_encoder
        self.prompt_encoder = prompt_encoder
        self.mask_decoder = mask_decoder
        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), persistent=False)
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), persistent=False)

    @property
    def device(self) -> torch.device:
        return self.pixel_mean.device

    # VV VV VV VV VV VV VV VV VV VV VV VV VV VV VV VV VV VV VV
    # 主要修改 forward 方法的型別提示
    # VV VV VV VV VV VV VV VV VV VV VV VV VV VV VV VV VV VV VV
    def forward(self,
                image: torch.Tensor,
                # --- 修改型別提示 ---
                boxes: Optional[torch.Tensor] = None,
                points: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
                # --- 型別提示修改結束 ---
               ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        模型的前向傳播。

        參數:
          image (torch.Tensor): 已經過預處理和正規化的批次影像張量。
          boxes (Optional[torch.Tensor]): 批次化的框提示。如果沒有框提示，則為 None。
                                           形狀 (B, num_boxes_per_image, 4)，在此 finetuning 中通常是 (B, 1, 4)。
          points (Optional[Tuple[torch.Tensor, torch.Tensor]]): 批次化的點提示。如果沒有點提示，則為 None。
                                                               元組包含: (座標張量 (B, N, 2), 標籤張量 (B, N))。
        返回:
          Tuple[torch.Tensor, torch.Tensor]:
            - processed_masks (torch.Tensor): 預測的遮罩。
            - iou_predictions (torch.Tensor): 模型對預測遮罩的 IOU (品質) 的預估。
        """
        image_embeddings = self.image_encoder(image)

        sparse_embeddings, dense_embeddings = self.prompt_encoder(
            points=points,
            boxes=boxes,
            masks=None,
        )
        
        low_res_masks, iou_predictions = self.mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=self.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
        )

        processed_masks = self.postprocess_masks(
            masks=low_res_masks,
            input_size=(self.image_encoder.img_size, self.image_encoder.img_size),
            original_size=(image.shape[-2], image.shape[-1])
        )
        return processed_masks, iou_predictions
    # ^^ ^^ ^^ ^^ ^^ ^^ ^^ ^^ ^^ ^^ ^^ ^^ ^^ ^^ ^^ ^^ ^^ ^^ ^^
    # 主要修改 forward 方法的型別提示
    # ^^ ^^ ^^ ^^ ^^ ^^ ^^ ^^ ^^ ^^ ^^ ^^ ^^ ^^ ^^ ^^ ^^ ^^ ^^

    def postprocess_masks(
        self,
        masks: torch.Tensor,
        input_size: Tuple[int, int], # 注意這裡的 Tuple 來自 typing
        original_size: Tuple[int, int], # 注意這裡的 Tuple 來自 typing
    ) -> torch.Tensor:
        masks = F.interpolate(
            masks,
            size=(self.image_encoder.img_size, self.image_encoder.img_size),
            mode="bilinear",
            align_corners=False,
        )
        masks = masks[..., :input_size[0], :input_size[1]]
        masks = F.interpolate(
            masks,
            size=original_size,
            mode="bilinear",
            align_corners=False,
        )
        return masks

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        x = (x - self.pixel_mean) / self.pixel_std
        h, w = x.shape[-2:]
        padh = self.image_encoder.img_size - h
        padw = self.image_encoder.img_size - w
        x = F.pad(x, (0, padw, 0, padh))
        return x

def get_sam_vit_t(checkpoint: Optional[str] = None, resume: bool = False) -> Sam: # 也更新這裡的型別提示
    prompt_embed_dim = 256
    image_size = 1024
    vit_patch_size = 16
    image_embedding_size = image_size // vit_patch_size

    sam_model_instance = Sam(
            image_encoder=TinyViT(img_size=image_size, in_chans=3, num_classes=1000,
                embed_dims=[64, 128, 160, 320],
                depths=[2, 2, 6, 2],
                num_heads=[2, 4, 5, 10],
                window_sizes=[7, 7, 14, 7],
                mlp_ratio=4.,
                drop_rate=0.,
                drop_path_rate=0.0,
                use_checkpoint=False,
                mbconv_expand_ratio=4.0,
                local_conv_size=3,
                layer_lr_decay=0.8
            ),
            prompt_encoder=PromptEncoder(
                embed_dim=prompt_embed_dim,
                image_embedding_size=(image_embedding_size, image_embedding_size),
                input_image_size=(image_size, image_size),
                mask_in_chans=16,
            ),
            mask_decoder=MaskDecoder(
                    num_multimask_outputs=3,
                    transformer=TwoWayTransformer(
                        depth=2,
                        embedding_dim=prompt_embed_dim,
                        mlp_dim=2048,
                        num_heads=8,
                    ),
                    transformer_dim=prompt_embed_dim,
                    iou_head_depth=3,
                    iou_head_hidden_dim=256,
            ),
            pixel_mean=[123.675, 116.28, 103.53],
            pixel_std=[58.395, 57.12, 57.375],
        )

    if checkpoint is not None:
        if not resume:
            logging.info(f"Attempting to load pretrained weights from: {checkpoint}")
            try:
                with open(checkpoint, "rb") as f:
                    state_dict = torch.load(f, map_location='cpu')
                missing_keys, unexpected_keys = sam_model_instance.load_state_dict(state_dict, strict=False)
                if missing_keys:
                    logging.warning(f"Missing keys when loading SAM checkpoint '{checkpoint}': {missing_keys}")
                if unexpected_keys:
                    logging.warning(f"Unexpected keys when loading SAM checkpoint '{checkpoint}': {unexpected_keys}")
                if not missing_keys and not unexpected_keys:
                    logging.info(f"Successfully loaded all weights from checkpoint: {checkpoint}")
                else:
                    logging.info(f"Partially loaded weights from checkpoint: {checkpoint}")
            except Exception as e:
                logging.error(f"Error loading SAM checkpoint from {checkpoint}: {e}. Model parameters will remain as initialized.")
        else:
            logging.info(f"Resuming training. Checkpoint '{checkpoint}' loading will be handled by the main training script if needed.")
    else:
        logging.info("No checkpoint provided. Model parameters are randomly initialized.")

    return sam_model_instance
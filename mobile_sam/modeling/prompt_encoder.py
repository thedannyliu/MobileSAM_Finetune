# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch
from torch import nn

from typing import Any, Optional, Tuple, Type

from .common import LayerNorm2d


class PromptEncoder(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        image_embedding_size: Tuple[int, int], # <--- 注意這裡
        input_image_size: Tuple[int, int],
        mask_in_chans: int,
        activation: Type[nn.Module] = nn.GELU,
    ) -> None:
        """
        Encodes prompts for input to SAM's mask decoder.

        Arguments:
          embed_dim (int): The prompts' embedding dimension
          image_embedding_size (tuple(int, int)): The spatial size of the
            image embedding, as (H, W).
          input_image_size (int): The padded size of the image as input
            to the image encoder, as (H, W).
          mask_in_chans (int): The number of hidden channels used for
            encoding input masks.
          activation (nn.Module): The activation to use when encoding
            input masks.
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.input_image_size = input_image_size
        self.image_embedding_size = image_embedding_size
        self.pe_layer = PositionEmbeddingRandom(embed_dim // 2)

        self.num_point_embeddings: int = 4  # pos/neg point + 2 box corners
        point_embeddings = [nn.Embedding(1, embed_dim) for i in range(self.num_point_embeddings)]
        self.point_embeddings = nn.ModuleList(point_embeddings)
        self.not_a_point_embed = nn.Embedding(1, embed_dim)

        self.mask_input_size = (4 * image_embedding_size[0], 4 * image_embedding_size[1])
        self.mask_downscaling = nn.Sequential(
            nn.Conv2d(1, mask_in_chans // 4, kernel_size=2, stride=2),
            LayerNorm2d(mask_in_chans // 4),
            activation(),
            nn.Conv2d(mask_in_chans // 4, mask_in_chans, kernel_size=2, stride=2),
            LayerNorm2d(mask_in_chans),
            activation(),
            nn.Conv2d(mask_in_chans, embed_dim, kernel_size=1),
        )
        self.no_mask_embed = nn.Embedding(1, embed_dim)

    def get_dense_pe(self) -> torch.Tensor:
        """
        Returns the positional encoding used to encode point prompts,
        applied to a dense set of points the shape of the image encoding.

        Returns:
          torch.Tensor: Positional encoding with shape
            1x(embed_dim)x(embedding_h)x(embedding_w)
        """
        return self.pe_layer(self.image_embedding_size).unsqueeze(0)

    def _embed_points(
        self,
        points: torch.Tensor, # 傳入時可能是 (N, 2)
        labels: torch.Tensor, # 傳入時可能是 (N,)
        pad: bool,
    ) -> torch.Tensor:
        """Embeds point prompts."""
        points = points + 0.5  # Shift to center of pixel

        # ---- MODIFICATION START: Add batch dimension if missing ----
        if points.ndim == 2:
            points = points.unsqueeze(0)  # (N, 2) -> (1, N, 2)
        if labels.ndim == 1: # 假設 labels 的批次維度與 points 對應
            labels = labels.unsqueeze(0)  # (N,) -> (1, N)
        # ---- MODIFICATION END ----

        if pad:
            # padding_point 和 padding_label 的 shape[0] 應該是批次大小 B
            padding_point = torch.zeros((points.shape[0], 1, 2), device=points.device)
            padding_label = -torch.ones((labels.shape[0], 1), device=labels.device) # labels.shape[0] 現在是 B
            
            points = torch.cat([points, padding_point], dim=1) # 沿著 N 維度拼接
            labels = torch.cat([labels, padding_label], dim=1) # 沿著 N 維度拼接
        
        # points 現在的形狀是 (B, N_padded, 2)
        point_embedding = self.pe_layer.forward_with_coords(points, self.input_image_size)
        # point_embedding 的形狀是 (B, N_padded, C_embed)
        
        # labels 的形狀是 (B, N_padded)
        # 使用廣播機制進行賦值
        point_embedding[labels == -1] = 0.0
        point_embedding[labels == -1] += self.not_a_point_embed.weight
        point_embedding[labels == 0] += self.point_embeddings[0].weight # type: ignore
        point_embedding[labels == 1] += self.point_embeddings[1].weight # type: ignore
        return point_embedding

    def _embed_boxes(self, boxes: torch.Tensor) -> torch.Tensor:
        """Embeds box prompts."""
        boxes = boxes + 0.5  # Shift to center of pixel
        coords = boxes.reshape(-1, 2, 2)
        corner_embedding = self.pe_layer.forward_with_coords(coords, self.input_image_size)
        corner_embedding[:, 0, :] += self.point_embeddings[2].weight
        corner_embedding[:, 1, :] += self.point_embeddings[3].weight
        return corner_embedding

    def _embed_masks(self, masks: torch.Tensor) -> torch.Tensor:
        """Embeds mask inputs."""
        mask_embedding = self.mask_downscaling(masks)
        return mask_embedding

    def _get_batch_size(
        self,
        points: Optional[Tuple[torch.Tensor, torch.Tensor]],
        boxes: Optional[torch.Tensor],
        masks: Optional[torch.Tensor],
    ) -> int:
        """
        Gets the batch size of the output given the batch size of the input prompts.
        """
        if points is not None:
            return points[0].shape[0]
        elif boxes is not None:
            return boxes.shape[0]
        elif masks is not None:
            return masks.shape[0]
        else:
            return 1

    def _get_device(self) -> torch.device:
        return self.point_embeddings[0].weight.device

    def forward(
        self,
        points: Optional[Tuple[torch.Tensor, torch.Tensor]],
        boxes: Optional[torch.Tensor],
        masks: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Embeds different types of prompts, returning both sparse and dense
        embeddings.

        Arguments:
          points (tuple(torch.Tensor, torch.Tensor) or none): point coordinates
            and labels to embed.
          boxes (torch.Tensor or none): boxes to embed
          masks (torch.Tensor or none): masks to embed

        Returns:
          torch.Tensor: sparse embeddings for the points and boxes, with shape
            BxNxC.
          torch.Tensor: dense embeddings for the masks, with shape Bx(embed_dim)x(embed_h)x(embed_w)
        """
        bs = self._get_batch_size(points, boxes, masks)
        sparse_embeddings = torch.empty(
            (bs, 0, self.embed_dim), device=self._get_device(), dtype=next(self.parameters()).dtype
        )
        if points is not None:
            coords, labels = points
            point_embedding = self._embed_points(coords, labels, pad=(boxes is None))
            sparse_embeddings = torch.cat([sparse_embeddings, point_embedding], dim=1)

        if boxes is not None:
            # _embed_boxes needs to return (current_bs, K*2, embed_dim) or similar for concatenation
            # Original _embed_boxes returns (K, 2, embed_dim) if input boxes is (K,4)
            box_embeddings_raw = self._embed_boxes(boxes) # (K, 2, embed_dim)
            
            # ---- MODIFICATION START: Reshape box_embeddings for cat ----
            if box_embeddings_raw.ndim == 3 and bs == 1: # (K, 2, C) -> (1, K*2, C)
                num_boxes = box_embeddings_raw.shape[0]
                box_embeddings = box_embeddings_raw.reshape(bs, num_boxes * 2, self.embed_dim)
            elif box_embeddings_raw.ndim == 3 and box_embeddings_raw.shape[0] == bs : # (B, K_x_2, C) - if _embed_boxes was changed to be batch-aware
                box_embeddings = box_embeddings_raw # Assuming already (B, Seq, C)
            else:
                # Fallback or error for unexpected shapes
                # For now, assume bs=1 and reshape
                num_boxes = box_embeddings_raw.shape[0] # K
                box_embeddings = box_embeddings_raw.reshape(1, num_boxes * 2, self.embed_dim)
                if bs != 1:
                     print(f"Warning: box_embeddings batch size mismatch. Expected {bs}, got 1 after reshape. Expanding.")
                     box_embeddings = box_embeddings.expand(bs, -1, -1) # Try to expand if bs > 1
            # ---- MODIFICATION END ----
            sparse_embeddings = torch.cat([sparse_embeddings, box_embeddings], dim=1)

        # Add no_mask_embed if no other sparse prompts
        if sparse_embeddings.shape[1] == 0: # Check if any prompts were added
            # self.no_mask_embed.weight is (1, embed_dim)
            no_mask_embedding = self.no_mask_embed.weight.reshape(1, 1, self.embed_dim)
            if bs > 1:
                no_mask_embedding = no_mask_embedding.expand(bs, -1, -1)
            sparse_embeddings = torch.cat([sparse_embeddings, no_mask_embedding], dim=1)


        if masks is not None:
            dense_embeddings = self._embed_masks(masks)
        else:
            dense_embeddings = self.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(
                bs, 
                -1, 
                self.image_embedding_size[0], # 使用元組的第一個元素 (高度)
                self.image_embedding_size[1]  # 使用元組的第二個元素 (寬度)
            )
        
        return sparse_embeddings, dense_embeddings


class PositionEmbeddingRandom(nn.Module):
    """
    Positional encoding using random spatial frequencies.
    """

    def __init__(self, num_pos_feats: int = 64, scale: Optional[float] = None) -> None:
        super().__init__()
        if scale is None or scale <= 0.0:
            scale = 1.0
        self.register_buffer(
            "positional_encoding_gaussian_matrix",
            scale * torch.randn((2, num_pos_feats)),
        )

    def _pe_encoding(self, coords: torch.Tensor) -> torch.Tensor:
        """Positionally encode points that are normalized to [0,1]."""
        # assuming coords are in [0, 1]^2 square and have d_1 x ... x d_n x 2 shape
        coords = 2 * coords - 1
        coords = coords @ self.positional_encoding_gaussian_matrix
        coords = 2 * np.pi * coords
        # outputs d_1 x ... x d_n x C shape
        return torch.cat([torch.sin(coords), torch.cos(coords)], dim=-1)

    def forward(self, size: Tuple[int, int]) -> torch.Tensor:
        """Generate positional encoding for a grid of the specified size."""
        h, w = size
        device = self.positional_encoding_gaussian_matrix.device
        dtype = self.positional_encoding_gaussian_matrix.dtype
        grid = torch.ones((h, w), device=device, dtype=dtype)
        y_embed = grid.cumsum(dim=0) - 0.5
        x_embed = grid.cumsum(dim=1) - 0.5
        y_embed = y_embed / h
        x_embed = x_embed / w

        pe = self._pe_encoding(torch.stack([x_embed, y_embed], dim=-1))
        return pe.permute(2, 0, 1)  # C x H x W

    def forward_with_coords(
        self, coords_input: torch.Tensor, image_size: Tuple[int, int]
    ) -> torch.Tensor:
        """Positionally encode points that are not normalized to [0,1]."""
        coords = coords_input.clone()
        coords[:, :, 0] = coords[:, :, 0] / image_size[1]
        coords[:, :, 1] = coords[:, :, 1] / image_size[0]
        return self._pe_encoding(coords.to(torch.float))  # B x N x C

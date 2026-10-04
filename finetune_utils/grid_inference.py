"""Point-grid inference shared by training and validation."""

import torch
from mobile_sam.utils.amg import batch_iterator


def predict_from_grid(model, image, points, orig_size, input_size, batch_size=64, multimask_output=True):
    """Run the SAM model on a grid of points and return masks and IoU preds.
    input_size: (H_resized, W_resized) before padding, used to correctly crop padding.
    """
    device = image.device
    inp = model.preprocess(image.unsqueeze(0))
    embedding = model.image_encoder(inp)
    dense_pe = model.prompt_encoder.get_dense_pe()

    all_masks = []
    all_ious = []
    all_lowres = []
    for (pts,) in batch_iterator(batch_size, points):
        coords = torch.as_tensor(pts, dtype=torch.float, device=device)
        labels = torch.ones(coords.shape[0], dtype=torch.int, device=device)
        sparse, dense = model.prompt_encoder(
            points=(coords.unsqueeze(0), labels.unsqueeze(0)),
            boxes=None,
            masks=None,
        )
        low_res, iou_pred = model.mask_decoder(
            image_embeddings=embedding,
            image_pe=dense_pe,
            sparse_prompt_embeddings=sparse,
            dense_prompt_embeddings=dense,
            multimask_output=multimask_output,
        )
        masks = model.postprocess_masks(low_res, input_size, orig_size).squeeze(0)
        all_masks.append(masks)
        all_lowres.append(low_res.squeeze(0))
        all_ious.append(iou_pred.squeeze(0))
    return torch.cat(all_masks, dim=0), torch.cat(all_ious, dim=0), torch.cat(all_lowres, dim=0)

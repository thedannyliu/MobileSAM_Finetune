# Segment Everything Finetuning

This document outlines the added training mode for finetuning MobileSAM when the
target task is **segment everything**.  The approach mimics the behaviour of
`SamAutomaticMaskGenerator` by providing grid based prompts and supervising all
objects within an image simultaneously.

## Dataset
- Use the same folder structure as the original dataset:
  ```
  dataset/
    image/xxx.jpg
    mask/xxx/obj_0.png
    mask/xxx/obj_1.png
  ```
- For each image all object masks under `mask/<id>/` are loaded and stacked.
- A regular point grid is generated over the original image (default step
  size = 32 px).  Points are scaled to the network resolution (1024×1024).

## Training Changes
- New dataset class **`SegmentEverythingDataset`** provides the stacked ground
  truth masks and the grid prompts.
- When `dataset.mode` in the config is set to `"everything"` the training script
  switches to this dataset and calls a custom prediction routine that evaluates a
  batch of grid prompts for every image.
  - Each grid point produces three candidate masks (`multimask_output=True`).  For
    every candidate the IoU with each ground truth mask is computed.  A Hungarian
    assignment selects the best pairs and those with IoU ≥ 0.5 are considered foreground,
    the rest are background.
- Loss per candidate = **BCE + 0.5·Focal + Dice + w_cls·BCE(confidence,label)**.
  The IoU prediction head is supervised with MSE against the measured IoU.
  - For unmatched ground truth masks the candidate with highest IoU is also used
    for supervision.  Hungarian assignment is used in both training and validation
    to pair predictions and ground truths one-to-one.
- Distillation losses are disabled in this mode for simplicity, but the rest of
  the training pipeline (optimizer, scheduler, etc.) remains unchanged.

## Usage
In `configs/mobileSAM.json` set
```json
"dataset": {
  "mode": "everything",
  "grid_points": 32,
  "train_dataset": "./datasets/train",
  "val_dataset": "./datasets/val"
}
```
Running `python train.py --config configs/mobileSAM.json` will finetune MobileSAM
with segment-everything supervision.

# Finetuning MobileSAM for Segment Everything

This guide explains how the provided implementation trains MobileSAM in a segment-everything manner.  The goal is to mimic SAM's automatic mask generator so that the finetuned model can segment every object in an image using only grid prompts.

## Dataset Preparation

```
dataset/
  image/xxx.jpg
  mask/xxx/object_0.png
  mask/xxx/object_1.png
  ...
```

* Every image has a folder of ground-truth object masks under `mask/<id>/`.
* All masks are resized to the training resolution (default `1024×1024`).
* A regular grid of point prompts is generated on the original image. The value
  of `dataset.grid_points` sets the number of points **per side** (e.g. `32`
  creates a 32×32 grid regardless of image resolution).

## Training Strategy

1. **SegmentEverythingDataset** — loads all masks of an image and the grid prompts.
2. **Prediction** — for every grid point the model predicts three candidate masks (`multimask_output=True`).
3. **Matching** — the IoU of each candidate against every ground-truth mask is computed and a Hungarian assignment selects the best one-to-one pairs.  Assigned pairs with IoU ≥ `0.5` are treated as foreground while the rest are background.
4. **Loss**
   - Masks are downsampled to 256×256 so that loss is computed directly on the decoder's low resolution logits.
   - For each candidate: `BCE * w_bce + Focal * w_focal + Dice * w_dice`.
   - IoU prediction is supervised with MSE (`w_iou`).
   - Classification: every candidate is labelled as foreground or background. The
     predicted IoU acts as the confidence score and is trained with
     `BCE(pred_iou, label)` weighted by `w_cls`.
   - Distillation losses (encoder, decoder, attention, RKD) are weighted by the teacher specific weight and their own `weight` field then scaled by `lambda_coef`.
   - Losses are averaged over only the matched predictions so that unmatched background candidates do not dilute gradients.
5. **Evaluation** — the same Hungarian algorithm pairs predictions and ground-truth masks one-to-one before computing Dice and IoU.
6. **Dynamic λ** — if enabled, `lambda_coef` is adjusted with a plateau scheduler based on the validation score.

## Visualisation

Validation can optionally save visualisations.  When using segment-everything the function `overlay_masks_on_image` draws all matched predictions with distinct colours and also shows the grid points.  Images are saved under `visual.save_path/epoch_<n>/` whenever a new best score is achieved or every `visual.save_every_n_epochs` epochs.

All epoch losses are also appended to `training_log.txt` inside the same directory where model weights are saved.

## Running

Edit `configs/mobileSAM_se.json` and ensure `dataset.mode` is set to `"everything"`.  Adjust `grid_points`, loss weights, and `lambda_coef` as needed.  Then run

```bash
python train.py --config configs/mobileSAM_se.json
```

All model components are trained from the beginning (no frozen layers).  The configuration sets `freeze.*` flags to `false` and `unfreeze_epoch` to `0` so that the image encoder, prompt encoder and mask decoder are updated from epoch 0.

Training requires substantial GPU memory because thousands of candidate masks may be generated per image.  Increase `grid_points`, reduce the image size, or lower the prediction batch size in `predict_from_grid` if out-of-memory errors occur.


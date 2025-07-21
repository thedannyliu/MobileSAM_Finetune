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
2. **Prediction** — for every grid point the model predicts masks from the decoder.  
   • 預設會產生 **3 個候選遮罩** (`model.multimask_output: true`)；  
   • 若在 `configs/*.json` 的 `model` 區段加入
     ```json
     "multimask_output": false
     ```
     就會改為 **僅輸出 1 個遮罩** (最左側的 mask token)，可減少記憶體與運算量。
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

## Distillation (Optional)

The segment-everything pipeline現在與一般 MobileSAM 訓練共用相同蒸餾框架，支援四種子模組：

| 子模組 | config key | 說明 |
| ------- | ---------- | ---- |
| Encoder patch tokens | `encoder_patch` | 比對 image encoder 之 patch embeddings |
| Prompt-conditioned embeddings | `prompt_embed` | 比對 decoder transformer 中 prompt-aware features |
| Mask token logits | `decoder_mask_token` | 比對 decoder mask token 的 logits (KL) |
| Dense mask logits | `dense_mask_logits` | 於 **單遮罩** 與 **segment-everything** 皆可用，將學生選出之最佳 low-res logits 與教師對應 logits 作 KL+Focal 對齊 |

預設我們已在 `configs/mobileSAM_se.json` 開啟前三項，並將 `dense_mask_logits.enable` 設為 `false`，因此訓練日誌中的 `dense=0.000` 代表此蒸餾項目被關閉。欲啟用時，將 JSON 改為：
```json
"dense_mask_logits": { "enable": true, "w_kl": 0.6, "w_focal": 0.4, "gamma": 2.0 }
```

完整範例：
```json
"distillation": {
  "enable": true,
  "lambda_coef": 1.0,
  "encoder_patch":      { "enable": true,  "w_l2": 1.0, "w_cos": 1.0 },
  "prompt_embed":       { "enable": true,  "w_mse": 0.7, "w_cos": 0.3 },
  "decoder_mask_token": { "enable": true,  "w_kl": 1.0, "temperature": 0.5 },
  "dense_mask_logits":  { "enable": true,  "w_kl": 0.6, "w_focal": 0.4, "gamma": 2.0 }
}
```

> `dense_mask_logits` 會略增記憶體與運算，若 GPU 緊張可關閉 (保持 0)。


# Distillation Losses for Prompt - Based Fine-Tuning

This file describes the **four** knowledge-distillation objectives that can be
enabled during point / box / mixed prompt fine-tuning.

---
## 1 Encoder Patch Tokens
* **Hook**
  * Teacher ‑ ViT-H: `image_encoder.blocks[30]` (1024-C) → **1×1 Conv** → 256-C
  * Student ‑ MobileSAM: `image_encoder.mv2_blocks[-2]` (256-C)
* **Shape** `B × 256 × 64 × 64`
* **Loss**
  $$L_{enc} = \lambda_{l2}\,\text{MSE}(f_s, f_t) + \lambda_{cos}\,(1-\text{Cos}(f_s, f_t))$$

---
## 2 Prompt-Conditioned Image Embedding
* **Hook** `mask_decoder.twa_blocks[0]` (before cross-attention)
* **Shape** `B × 256 × 64 × 64`
* **Loss**
  $$L_{pe} = \alpha\,\text{MSE}(\hat f_s,\hat f_t) + (1-\alpha)(1-\text{Cos}(f_s,f_t))$$
  where \(\hat f=\text{L2-norm}(f)\).

---
## 3 Decoder Mask Tokens
* **Hook** `mask_decoder.mask_tokens`
* **Shape** `B × 4 × 256`
* **Loss**
  $$L_{tok}= \text{KL}(\text{softmax}(z_t/\tau)\;\Vert\;\text{softmax}(z_s/\tau))$$

---
## 4 Dense Mask Logits
* **Source** decoder output `low_res_logits` (`B × 1 × 256 × 256`)
* **Loss**
  $$L_{dense}=\beta\,\text{KL}(p_t\Vert p_s) + (1-\beta)\,\text{Focal}(p_s, p_t)$$
  with \(p=\sigma(\text{logits})\).

---
## Overall Distillation Loss
$$L_{dist} = w_{enc}L_{enc}+w_{pe}L_{pe}+w_{tok}L_{tok}+w_{dense}L_{dense}$$
The global coefficient `lambda_coef` further scales \(L_{dist}\) before adding
to the task loss.

---
## Configuration Snippet
```jsonc
"distillation": {
  "enable": true,
  "lambda_coef": 1.0,
  "encoder_patch":      { "enable": true,  "w_l2": 1.0, "w_cos": 1.0 },
  "prompt_embed":       { "enable": true,  "w_mse": 0.7, "w_cos": 0.3 },
  "decoder_mask_token": { "enable": true,  "w_kl": 1.0, "temperature": 0.5 },
  "dense_mask_logits":  { "enable": true,  "w_kl": 0.6, "w_focal": 0.4, "gamma": 2.0 }
},
"teachers": [
  { "name": "SAM-H",          "weight": 0.7, "checkpoint": "weights/sam_vit_h_4b8939.pth" },
  { "name": "MobileSAM_orig", "weight": 0.3, "checkpoint": "weights/mobile_sam.pt" }
]
```

Add this block (or adjust values) inside any existing JSON/YAML config to
activate distillation.  Disabling a sub-loss simply sets its `enable` flag to
`false`.

---
**Note**
* The teacher network(s) always run under `torch.no_grad()` and never receive
gradients; extra memory usage is limited to the forward activations.
* All loss components are logged individually in the training progress bar and
TensorBoard.

---
## Pre-computing Teacher Features (Optional)

Running the teacher online can be GPU-heavy.  You can **pre-extract** the three
features needed for distillation (encoder patch, prompt-embed, mask-token) with

```bash
python scripts/extract_teacher_features.py \
  --teacher_name SAM_vitH \
  --teacher_cfg  ./configs/sam_vith.yaml \
  --teacher_ckpt ./weights/sam_vit_h_4b8939.pth \
  --dataset_base_dir ./datasets \
  --output_base_dir ./precomputed/SAM_vitH \
  --splits train val \
  --fp16          # optional, halves disk size & GPU RAM
```

Key flags:

* `--mode everything` Add this when your training dataset mode is
  *segment-everything* (grid prompt).  Use `--grid_points N` to match your
  config.
* `--fp16` Stores FP16 `.npy` files (2× smaller) and runs teacher inference in
  half-precision.

Expected disk usage ≈ *4 MB per image* (see "Capacity Estimation" section of
the conversation).  If storage is tight, you may disable `dense_mask_logits`
and skip pre-computing that component (saves ~0.13 MB / image). 
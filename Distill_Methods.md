# Distillation Methods in MobileSAM Finetuning

This document explains the knowledge distillation techniques implemented in this repository. These methods allow the student MobileSAM model to learn from stronger teacher models by matching intermediate representations.

## 1. Encoder Matching
- **Purpose:** Align features extracted by the student and teacher image encoders.
- **Implementation:** `encoder_matching_loss` in `finetune_utils/distill_losses.py` computes Mean Squared Error (MSE) and Kullback–Leibler (KL) divergence between corresponding layers.
- **Formula:**
  \[
  L_{enc} = \frac{\lambda_{mse}}{N}\sum_{i} \|f_s^i - f_t^i\|_2^2 + \frac{\lambda_{kl}}{N}\sum_{i} KL(p_s^i\,\|\,p_t^i)\;,
  \]
  where $N$ is the number of matched layers and $p$ denotes softmax distributions with temperature.

## 2. Decoder Matching
- **Purpose:** Match the pre–logit features of the mask decoder.
- **Implementation:** `decoder_matching_loss` combines MSE, cosine distance and KL divergence between decoder features.

## 3. Attention Matching
- **Purpose:** Encourage similar attention maps in ViT layers.
- **Implementation:** `attention_matching_loss` computes KL divergence over attention matrices across selected layers.

## 4. Relational Knowledge Distillation
- **Purpose:** Preserve pairwise relations between image patches.
- **Implementation:** `rkd_loss` measures differences in pairwise distances and angles of embeddings from teacher and student.

## 5. Dynamic Lambda
- **Purpose:** Adjust distillation weight during training based on validation performance. When validation plateaus, the coefficient is reduced.

## 6. Precomputed Teacher Features
- Teacher features can be extracted ahead of time using `scripts/extract_teacher_features.py`. The training script loads these `.npy` files to speed up distillation.

All losses and their logging are implemented in `train.py`, which records individual distillation terms for monitoring during training.

## 7. Online Distillation

When `distillation.use_precomputed_features` is `false`, the teachers are instantiated during training and the same feature hooks are attached to them. Features are captured on the fly from each teacher's forward pass, enabling distillation without pre-extracting `.npy` files. This offers flexibility at the expense of additional compute time.

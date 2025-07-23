# Configuration Guide (`configs/`)

The `.json` files in this directory are configuration files for `train.py`, controlling all aspects of the training process, from model architecture and data paths to hyperparameters. You can copy `mobileSAM.json` and modify it to suit your specific needs.

## Main Configuration Parameters (`mobileSAM.json`)

The following is a detailed explanation of each parameter in `mobileSAM.json`.

### Project and Path Settings

*   `"project_name"`: (string) The name of the project. Used to organize outputs under `output_dir`, e.g., `output/MyProject/`.
*   `"run_name"`: (string) The name for the specific run, e.g., `output/MyProject/Run_01/`.
*   `"output_dir"`: (string) The root directory for saving training logs and model checkpoints.
*   `"checkpoint"`: (string) Path to the pre-trained MobileSAM checkpoint (`.pth`) file, which serves as the starting point for finetuning.
*   `"teacher_checkpoint"`: (string, optional) Path to the "teacher model" checkpoint (e.g., the original SAM ViT-H) used for knowledge distillation.
*   `"train_img_dir"` / `"train_mask_dir"`: (string) Paths to the training image and mask directories.
*   `"val_img_dir"` / `"val_mask_dir"`: (string) Paths to the validation image and mask directories.
*   `"train_img_dir_teacher_features"`: (string, optional) Path where pre-extracted feature maps from the teacher model are stored. This is required when `use_distill` is `true`.

### Model Architecture Settings

*   `"model_type"`: (string) The SAM model variant. For MobileSAM, this should be `"vit_t"`.
*   `"freeze"`: (object) A dictionary of booleans used to freeze specific parts of the model, preventing their weights from being updated during training.
    *   `"image_encoder"`: `true` to freeze the image encoder.
    *   `"prompt_encoder"`: `true` to freeze the prompt encoder.
    *   `"mask_decoder"`: `true` to freeze the mask decoder.
*   `"iou_head_depth"` / `"iou_head_hidden_dim"`: (integer) The depth and hidden dimension of the IoU prediction head in the mask decoder.

### Training Hyperparameters

*   `"num_epochs"`: (integer) The total number of training epochs.
*   `"batch_size"`: (integer) The batch size for each training step.
*   `"num_workers"`: (integer) The number of worker threads to use for data loading.
*   `"learning_rate"`: (float) The initial learning rate for the optimizer.
*   `"weight_decay"`: (float) The weight decay value in the AdamW optimizer.
*   `"img_size"`: (integer) The size to which images are resized during training (e.g., `1024`).
*   `"mask_threshold"`: (float) The threshold used to binarize the model's output logits into a mask during validation.

### Knowledge Distillation Settings

*   `"use_distill"`: (boolean) Whether to enable knowledge distillation. If `true`, an additional distillation loss will be computed during training.
*   `"distill_feature_level"`: (integer) Specifies which layer's feature maps to extract from the teacher model's image encoder for distillation (e.g., layer `8` or `11` for ViT-H).

---

## Multi-Stage Training (`stage_schedule`)

This project supports defining multiple training stages in a single run, such as starting with knowledge distillation and then switching to pure finetuning. This is achieved by adding a top-level `"stage_schedule"` array to the configuration file.

`train.py` checks the current training stage at the beginning of each epoch and dynamically overrides the relevant training parameters.

### `stage_schedule` Structure

`"stage_schedule"` is an array of objects, where each object represents a training stage and contains the following fields:

*   `"start_epoch"`: (integer) The starting epoch for this stage (inclusive).
*   `"end_epoch"`: (integer) The ending epoch for this stage (exclusive).
*   `"distillation"`: (boolean) Whether to enable the knowledge distillation loss in this stage.
*   `"lambda_coef"`: (float) The global weight coefficient for the distillation loss.
*   `"loss_weights"`: (object) Used to override the weights of various loss functions for this stage. You only need to set the losses you want to change; unset ones will use the global settings.
    *   `"focal"`: Weight for Focal Loss.
    *   `"dice"`: Weight for Dice Loss.
    *   `"iou"`: Weight for IoU Head Loss.

### Example Configurations

*   `distill_then_finetune.json`:
    *   **Stage 1 (Epochs 0-50):** Knowledge distillation only. `distillation` is `true`, while the weights for `focal`, `dice`, and `iou` losses are `0.0`.
    *   **Stage 2 (Epochs 50-100):** Finetuning only. `distillation` is `false`, and the weights for `focal`, `dice`, and `iou` losses are restored to `1.0`.

*   `finetune_then_distill.json`:
    *   The reverse of the above, starting with finetuning and then switching to knowledge distillation.

---

## Other Configuration Files

*   `mobileSAM_se.json`: An example configuration using a TinyViT variant with Squeeze-and-Excitation (SE) modules.
*   `.yaml` files (`sam_vith.yaml`, `mobile_sam_orig.yaml`, etc.): These are reference configuration files from the original SAM and MobileSAM projects, primarily used to define model architecture parameters. They are not read directly by `train.py`, but their contents may be integrated into `mobile_sam/build_sam.py`. 
# MobileSAM Finetuning: Training Method and Implementation Details

This document provides a detailed explanation of the training methodology and implementation specifics used in this MobileSAM finetuning project. The core training logic is orchestrated by `train.py`, leveraging utilities from the `finetune_utils` directory and model components from `mobile_sam`.

## 1. Overview of the Finetuning Process

The goal of finetuning is to adapt a pre-trained MobileSAM model to a specific downstream segmentation task or dataset. This is achieved by further training the model (or parts of it) on a custom dataset consisting of images and their corresponding ground truth segmentation masks. The process involves:

1.  **Configuration:** Defining training parameters, model paths, dataset locations, and model components to freeze/unfreeze.
2.  **Data Loading & Preprocessing:** Loading images and masks, applying necessary transformations, and generating prompts (bounding boxes from masks) for the SAM model.
3.  **Model Initialization:** Loading the pre-trained MobileSAM (TinyViT based) and optionally a teacher model (e.g., SAM ViT-H) if feature distillation is used.
4.  **Defining Training Objectives:**
    * **Primary Segmentation Loss:** A combination of Focal Loss and Dice Loss between the model's predicted masks and the ground truth masks.
    * **IoU Prediction Loss:** Training the IoU head of the mask decoder to accurately predict the quality of its output masks.
    * **Feature Distillation Loss (Optional):** Aligning intermediate features of MobileSAM's image encoder with those of a more powerful teacher model.
5.  **Optimization:** Updating model weights using the AdamW optimizer and a polynomial learning rate scheduler.
6.  **Evaluation & Checkpointing:** Periodically evaluating the model on a validation set (if provided) using Mean Intersection over Union (mIoU) and saving the best performing model checkpoints.

## 2. Configuration (`configs/mobileSAM.json`)

The training process is heavily guided by a JSON configuration file. Key parameters influencing the training method include:

* **`model_type`**: e.g., `"vit_t"` specifies the TinyViT backbone for MobileSAM.
* **`checkpoint`**: Path to the initial weights for MobileSAM. Finetuning starts from this state.
* **`freeze`**: A critical dictionary (`{"image_encoder": bool, "prompt_encoder": bool, "mask_decoder": bool}`). Setting a component to `true` means its parameters will not be updated during training. This allows for:
    * Full finetuning (all `false`).
    * Finetuning only the mask decoder (common for adapting to new mask types while keeping general image understanding).
    * Finetuning the image encoder and mask decoder but keeping the prompt encoder fixed.
* **`learning_rate`**: Initial learning rate for the optimizer.
* **`num_epochs`**: Total training iterations over the dataset.
* **`batch_size`**: Number of samples processed before the model is updated.
* **`img_size`**: Target resolution for images during training (e.g., 1024). Images are resized and padded.
* **Loss-related parameters**:
    * While not explicitly in the config for `SegLoss` weights (Focal/Dice), these are hardcoded in `finetune_utils/loss.py` (typically `focal_coeff=1.0`, `dice_coeff=1.0`).
* **Distillation-related parameters (if `use_distill: true`):**
    * **`teacher_checkpoint`**: Path to the teacher model's checkpoint (e.g., original SAM ViT-H).
    * **`distill_feature_level`**: Specifies the layer from the teacher's image encoder whose features are used for distillation (e.g., 8 or 11 for ViT-H). This corresponds to a specific block in the Vision Transformer.
    * **`train_img_dir_teacher_features`**: Path to the directory containing pre-extracted teacher features for the training images.

## 3. Data Handling (`finetune_utils/datasets.py`)

The `SAMDataset` class is responsible for loading and preprocessing data.

### 3.1. Initialization

* Takes paths to image and mask directories, configuration parameters, and a flag for training mode.
* Initializes `ResizeLongestSide` transform from `mobile_sam.utils.transforms` to resize images so their longest side matches `config.img_size`.
* Stores pixel mean (`PIXEL_MEAN`) and standard deviation (`PIXEL_STD`) for normalization, typically `[123.675, 116.28, 103.53]` and `[58.395, 57.12, 57.375]` respectively.

### 3.2. `__getitem__(self, index)` - Data Loading and Preprocessing per Sample

1.  **Load Image:** Reads an image using `cv2.imread` and converts it from BGR to RGB.
2.  **Load Mask:** Reads the corresponding mask using `cv2.imread(cv2.IMREAD_GRAYSCALE)`.
3.  **Teacher Features (if `use_distill` and `is_train`):**
    * Loads pre-extracted teacher features for the current image from the path specified by `config.train_img_dir_teacher_features`. These are typically stored as `.pt` files.
4.  **Transformations (Image and Mask):**
    * **Resizing:** Both image and mask are resized using `ResizeLongestSide(self.img_size).apply_image(image)` and `apply_image(mask)`. This maintains aspect ratio.
    * **Mask Binarization:** Ground truth masks are converted to binary format: pixels with value > 0 become 1, others remain 0. This ensures a consistent format for loss calculation.
    * **Normalization (Image only):** The image pixel values are normalized: `(image - PIXEL_MEAN) / PIXEL_STD`.
    * **Padding (Image and Mask):** Images and masks are padded to a square shape of `(self.img_size, self.img_size)` using `cv2.copyMakeBorder`. This is essential as ViTs expect fixed-size inputs.
    * **Channel Permutation (Image only):** Image tensor shape is changed from `(H, W, C)` to `(C, H, W)` as expected by PyTorch models.
5.  **Prompt Generation (from Ground Truth Mask):**
    * `get_boxes_from_masks(masks_torch, self.device)`: This crucial step generates bounding box prompts directly from the ground truth mask.
        * It finds the minimal bounding box enclosing the foreground pixels in the mask.
        * These bounding boxes serve as the primary prompt fed into the MobileSAM model during training, guiding it to segment the object defined by the ground truth.
6.  **Output Dictionary:** Returns a dictionary containing:
    * `image`: Preprocessed and normalized image tensor.
    * `original_size`: Original size of the image before resizing.
    * `boxes`: The generated bounding box prompt tensor.
    * `image_name`: Name of the image file.
    * `masks`: The preprocessed ground truth mask tensor.
    * `teacher_feature` (if applicable): Loaded teacher feature tensor.

This process ensures that the model receives consistently formatted inputs (image, prompt) and that the ground truth masks are ready for loss computation. Using bounding boxes derived from GT masks as prompts is a common strategy for training SAM-like models on datasets where explicit point/box prompts are not available.

## 4. Model Setup (`train.py` and `mobile_sam.build_sam`)

### 4.1. Student Model (MobileSAM)

* The MobileSAM model is instantiated using `sam_model_registry[config.model_type](checkpoint=config.checkpoint)`.
    * `sam_model_registry` (in `mobile_sam.build_sam.py`) is a dictionary mapping model type strings (e.g., `"vit_t"`) to functions that build the specific model architecture.
    * For `"vit_t"`, it calls `_build_sam` with `image_encoder_fn=TinyViT` and parameters for TinyViT (embedding dims, depths, num_heads, etc.) defined in `build_sam_vit_t`.
    * The `checkpoint` argument ensures that pre-trained weights are loaded into this architecture.
* The model is moved to the appropriate device (GPU or CPU).
* **Freezing Components:** Based on the `config.freeze` dictionary:
    * If `config.freeze['image_encoder']` is true, `model.image_encoder.eval()` is called, and `param.requires_grad = False` is set for all parameters in the image encoder. This prevents them from being updated.
    * Similar logic applies to `prompt_encoder` and `mask_decoder`.

### 4.2. Teacher Model (for Distillation)

* If `config.use_distill` is true:
    * A teacher SAM model is built, typically a larger ViT-H variant: `sam_model_registry["vit_h"](checkpoint=config.teacher_checkpoint)`.
    * The teacher model is set to evaluation mode (`teacher_model.eval()`) and all its parameters have `requires_grad = False`, as it's only used for providing target features, not for being trained itself.
    * The teacher model is also moved to the device.

## 5. Training Loop (`train.py`)

The core training logic iterates through epochs and batches.

### 5.1. Forward Pass

For each batch obtained from the `DataLoader`:
1.  **Data Preparation:**
    * `images = batch['image'].to(device)`
    * `gt_masks = batch['masks'].to(device)`
    * `boxes = batch['boxes'].to(device)`
    * If distilling: `teacher_features = batch['teacher_feature'].to(device)`
2.  **Model Input Construction:** The input to `model.forward()` is a list of dictionaries, where each dictionary corresponds to a sample in the batch and contains:
    * `'image'`: The image tensor.
    * `'original_size'`: Tuple `(original_height, original_width)`.
    * `'boxes'`: The bounding box prompt tensor for this image.
    * `'gt_masks'` (optional, but useful for direct loss calculation or internal model logic): The ground truth mask.
3.  **Student Model Forward Pass:**
    * `predicted_masks, iou_predictions = model(batched_input, multimask_output=True)`
        * `batched_input`: The list of dictionaries described above.
        * `multimask_output=True`: Instructs the model to return multiple mask proposals from the decoder. `predicted_masks` will typically have shape `(B, num_masks, H_lowres, W_lowres)`, e.g., `(B, 3, 256, 256)`. `iou_predictions` has shape `(B, num_masks)`.
4.  **Distillation Feature Extraction (if `config.use_distill`):**
    * `FeatureHookManager` (from `finetune_utils.feature_hooks`) is used. Before training starts, hooks are registered on both the student's image encoder and the (already feature-extracted) teacher's conceptual encoder path.
    * During the student's forward pass (`model(batched_input, ...)`), the hook on the student's image encoder automatically captures its intermediate features from the layer specified by `config.distill_student_feature_level` (this level would correspond to the teacher's `distill_feature_level`).
    * The `student_distill_features` are retrieved from the hook manager.

### 5.2. Loss Calculation

1.  **Segmentation Loss (`finetune_utils.loss.SegLoss`):**
    * `seg_loss_computer = SegLoss()`
    * `total_loss = seg_loss_computer(predicted_masks, gt_masks)`
        * This combines Focal Loss and Dice Loss. The `SegLoss` class internally calculates both and sums them.
        * It expects `predicted_masks` (logits from the model) and `gt_masks` (binary ground truth).
        * The losses are typically averaged over the batch.
2.  **IoU Prediction Loss:**
    * The `Sam` model's forward pass can internally compute a loss for its IoU prediction head if ground truth masks are available to calculate the true IoU of predicted masks. This is often an MSE or Smooth L1 loss.
    * The `train.py` script sums the segmentation loss and this internally calculated IoU loss: `total_loss = seg_loss + iou_loss_val` (where `iou_loss_val` comes from `model.iou_loss`).
3.  **Feature Distillation Loss (if `config.use_distill`):**
    * `distill_loss_computer = FeatureDistillationLoss(loss_type=config.get('distill_loss_type', 'mse'))`
    * `stu_feats_for_loss = student_distill_features[config.distill_student_feature_level]`
    * `distill_loss_val = distill_loss_computer(stu_feats_for_loss, teacher_features)`
        * This calculates a loss (e.g., MSE) between the student's intermediate features and the pre-computed teacher features.
    * `total_loss += distill_loss_weight * distill_loss_val` (where `distill_loss_weight` can be a hyperparameter, e.g., `config.get('distill_loss_weight', 1.0)`).

### 5.3. Backpropagation and Optimization

1.  **Zero Gradients:** `optimizer.zero_grad()`
2.  **Backward Pass:** `total_loss.backward()`
3.  **Optimizer Step:** `optimizer.step()`
4.  **Scheduler Step:** `scheduler.step()` (updates the learning rate based on the polynomial decay policy).

## 6. Optimizer and Scheduler

* **Optimizer:** `torch.optim.AdamW`
    * Configured with `lr=config.learning_rate` and `weight_decay=config.weight_decay`.
    * Only parameters with `requires_grad=True` are passed to the optimizer. This ensures frozen parts of the model are not updated.
* **Scheduler:** `finetune_utils.schedular.PolynomialLR`
    * `total_iters = len(train_loader) * config.num_epochs` (total training steps).
    * Decays the learning rate from `config.learning_rate` down to a `end_lr` (default 0) with a specified `power` (default 1.0, i.e., linear decay if `power=1`).

## 7. Evaluation and Checkpointing

### 7.1. Evaluation (`validate_model` function in `train.py`)

* Performed at the end of each epoch if validation data (`val_loader`) is provided.
* The model is set to evaluation mode: `model.eval()`.
* Iterates through the validation dataset:
    * For each batch, performs a forward pass (similar to training but without backpropagation).
    * Calculates segmentation loss and IoU loss.
    * Calculates Mean Intersection over Union (mIoU):
        * Predicted masks are binarized: `binary_masks = (predicted_masks > config.mask_threshold).byte()`.
        * Intersection: `(binary_masks & gt_masks).sum()`
        * Union: `(binary_masks | gt_masks).sum()`
        * IoU per sample: `intersection / union`.
        * mIoU is the average IoU over the validation set.
* The model is set back to training mode: `model.train()` (if not the end of training).
* Freezes components again after validation if they were unfrozen temporarily for full-model evaluation (though typically eval mode handles this).

### 7.2. Checkpointing (`finetune_utils.save_checkpoint.py`)

* Checkpoints are saved based on validation performance (mIoU).
* If the current epoch's validation mIoU is better than the `best_val_iou` seen so far:
    * The current model state dictionary (`model.state_dict()`) is saved.
    * The path is typically `output_dir/project_name/run_name/checkpoints/best_model.pth`.
* The latest model is also often saved at the end of each epoch or at regular intervals, regardless of best performance, e.g., `latest_model.pth`.

## 8. Feature Distillation Deep Dive (`scripts/extract_teacher_features.py` & `finetune_utils/distill_losses.py`)

### 8.1. Teacher Feature Extraction

The script `scripts/extract_teacher_features.py` is a prerequisite if `use_distill` is true.
1.  **Load Teacher Model:** Loads the specified teacher model (e.g., SAM ViT-H).
2.  **Setup Feature Hook:** Uses `FeatureHookManager` to register a forward hook on the teacher's image encoder at the layer specified by `config.distill_feature_level`. For ViT models, this is typically one of the transformer blocks. For example, if `distill_feature_level` is 8, it hooks into `teacher_model.image_encoder.blocks[8]`.
3.  **Iterate Through Dataset:**
    * For each image in the `image_dir` (which should be the same images used for student training):
        * Preprocess the image just like the student model would (resize, normalize, pad).
        * Pass the image through `teacher_model.image_encoder(image_tensor.unsqueeze(0))`.
        * The hook automatically captures the output features from the specified layer.
        * These features (a tensor) are detached from the computation graph, moved to CPU, and saved to the `output_dir` (e.g., as `image_name.pt`).

### 8.2. Distillation Loss Calculation During Student Training

1.  **Loading Teacher Features:** The `SAMDataset` loads the corresponding `.pt` file for each training image.
2.  **Student Feature Extraction:** `FeatureHookManager` (re-instantiated for the student model) captures features from the student's image encoder at a comparable layer (e.g., `student_model.image_encoder.neck` or a specific block if TinyViT structure allows direct layer indexing similar to the teacher). The `distill_student_feature_level` in the config would specify this. Often, if the teacher's features are from a late block, the student's features might be taken after its final image encoder stage (e.g., from the "neck" if it has one before features go to the decoder).
3.  **Loss Computation (`FeatureDistillationLoss`):**
    * The loss is typically Mean Squared Error (MSE) or L1 loss:
        * $L_{distill} = \frac{1}{N} \sum_{i=1}^{N} (F_{student,i} - F_{teacher,i})^2$ (for MSE)
        * Where $F_{student}$ and $F_{teacher}$ are the feature maps (flattened or not) from the student and teacher respectively, and $N$ is the number of elements in the feature map.
    * This forces the student's feature representation at that layer to become similar to the teacher's.

This detailed training methodology allows for flexible adaptation of MobileSAM, with options to control the extent of finetuning and leverage knowledge from larger models through feature distillation. The use of bounding box prompts derived from ground truth masks simplifies dataset preparation for segmentation tasks.
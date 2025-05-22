# Finetuning MobileSAM

This repository provides a comprehensive framework for finetuning the MobileSAM model, a lightweight version of the Segment Anything Model (SAM) designed for efficient execution on resource-constrained devices. The original MobileSAM leverages a TinyViT backbone, and this project enables further specialization of the model on custom datasets. It includes utilities for data loading, training, loss computation, and a Gradio-based demonstration application.

## Table of Contents

1.  [Project Overview](#project-overview)
2.  [Key Features](#key-features)
3.  [Model Architecture](#model-architecture)
    * [Image Encoder (TinyViT)](#image-encoder-tinyvit)
    * [Prompt Encoder](#prompt-encoder)
    * [Mask Decoder](#mask-decoder)
4.  [Finetuning Pipeline](#finetuning-pipeline)
    * [Configuration](#configuration)
    * [Dataset Preparation](#dataset-preparation)
    * [Training Script](#training-script)
    * [Loss Functions](#loss-functions)
    * [Optimizer and Scheduler](#optimizer-and-scheduler)
5.  [Installation](#installation)
6.  [Usage](#usage)
    * [Finetuning](#finetuning-1)
    * [Inference with Gradio App](#inference-with-gradio-app)
    * [Using the Predictor](#using-the-predictor)
    * [Automatic Mask Generation](#automatic-mask-generation)
    * [ONNX Export and Usage](#onnx-export-and-usage)
7.  [Directory Structure](#directory-structure)
8.  [Technical Deep Dive](#technical-deep-dive)
    * [Core Model Components](#core-model-components)
    * [Finetuning Utilities](#finetuning-utilities)
    * [Distillation (Optional)](#distillation-optional)
9.  [Contributing](#contributing)
10. [License](#license)

## Project Overview

The Segment Anything Model (SAM) has demonstrated remarkable zero-shot performance in image segmentation. MobileSAM adapts this power into a more compact architecture, primarily by replacing the original ViT image encoder with a TinyViT. This project focuses on enabling users to finetune MobileSAM on specific downstream tasks or custom datasets to potentially improve performance and tailor the model to particular domains.

The core idea is to update the weights of the MobileSAM model, particularly the image encoder and/or the mask decoder, using a custom dataset with corresponding ground truth masks. The finetuning process supports various configurations, loss functions (including focal and dice loss), and learning rate scheduling.

## Key Features

* **MobileSAM Implementation:** Core MobileSAM model architecture including TinyViT image encoder, prompt encoder, and mask decoder.
* **Finetuning Script:** A flexible `train.py` script for finetuning MobileSAM on custom datasets.
* **Customizable Configurations:** Training parameters, model paths, and dataset details can be managed via JSON configuration files.
* **Multiple Loss Functions:** Supports common segmentation losses like Focal Loss and Dice Loss, and their combination.
* **Knowledge Distillation (Implied):** Utilities like `extract_teacher_features.py` and `distill_losses.py` suggest capabilities for knowledge distillation from a larger teacher model (e.g., original SAM) to the MobileSAM student model, although direct implementation in `train.py` needs verification.
* **Learning Rate Scheduler:** Implements a polynomial learning rate scheduler.
* **Checkpointing:** Saves model checkpoints during training for later resumption or evaluation.
* **Gradio Web UI:** An interactive `app.py` for easy testing and visualization of the (finetuned) MobileSAM model.
* **ONNX Export:** Functionality to export the model to ONNX format for optimized inference.
* **Utility Scripts:** Includes scripts for automatic mask generation and ONNX export.
* **Modular Design:** Code is organized into modules for model components, finetuning utilities, and application logic.

## Model Architecture

MobileSAM, like the original SAM, consists of three main components: an image encoder, a prompt encoder, and a mask decoder.

### Image Encoder (TinyViT)

The primary modification in MobileSAM is the use of a Tiny Vision Transformer (TinyViT) as the image encoder. This significantly reduces the number of parameters and computational cost compared to the standard ViT used in SAM.
* **Implementation:** `mobile_sam.modeling.tiny_vit_sam.TinyViT`
* **Function:** Takes an input image (e.g., $1024 \times 1024 \times 3$) and processes it through several stages of transformer blocks and patch merging layers to produce image embeddings.
* **Key Aspects:**
    * `img_size`: Typically 1024.
    * `patch_size`: Defines the size of image patches (e.g., 16x16).
    * `in_chans`: Input channels (usually 3 for RGB).
    * `embed_dims`: A list specifying the embedding dimension at each of the 4 stages.
    * `depths`: A list specifying the number of transformer blocks in each stage.
    * `num_heads`: A list specifying the number of attention heads in each stage.
    * `window_sizes`: A list specifying the window sizes for windowed attention in each stage.
    * `mlp_ratio`: Ratio for MLP hidden dimension.
    * `out_indices`: Indices of stages from which to output features.
    * The image encoder outputs feature maps that are typically downsampled by a factor of 16 (e.g., $64 \times 64$ for a $1024 \times 1024$ input).

### Prompt Encoder

The prompt encoder processes various types of prompts (points, boxes, masks) and converts them into embeddings that can be combined with the image embeddings.
* **Implementation:** `mobile_sam.modeling.prompt_encoder.PromptEncoder`
* **Function:**
    * **Point Prompts:** Encodes sparse point coordinates (and associated labels indicating foreground/background) into positional encodings and learned embeddings.
    * **Box Prompts:** Encodes bounding box coordinates similarly, using positional encodings for the top-left corner and learned embeddings for "top-left" and "bottom-right" roles.
    * **Mask Prompts (Not explicitly finetuned here but part of SAM):** Can take a low-resolution mask and embed it using convolutions.
* **Key Aspects:**
    * `embed_dim`: The dimension of the output embeddings.
    * `image_embedding_size`: The spatial size of the image embeddings (e.g., $64 \times 64$).
    * `input_image_size`: The original input image size (e.g., $1024 \times 1024$).
    * Uses positional encodings for spatial information.
    * Produces a dense embedding (mask features) and sparse embeddings (point/box features).

### Mask Decoder

The mask decoder takes the image embeddings (from the image encoder) and prompt embeddings (from the prompt encoder) to predict segmentation masks.
* **Implementation:** `mobile_sam.modeling.mask_decoder.MaskDecoder`
* **Function:**
    * Combines image features and prompt embeddings using a two-way transformer architecture.
    * Upscales the features to produce mask predictions at a higher resolution (typically 1/4th of the input image size, e.g., $256 \times 256$).
    * Predicts multiple masks (usually 3) to handle ambiguity and an IoU score for each mask.
* **Key Aspects:**
    * `transformer_dim`: The feature dimension within the transformer.
    * `transformer`: The core two-way transformer module.
    * `num_multimask_outputs`: Number of ambiguous masks to output.
    * `iou_head`: A small MLP to predict the IoU of the generated masks.
    * The output masks are low-resolution and are typically upscaled to the original image size during post-processing.

## Finetuning Pipeline

### Configuration

Finetuning is primarily controlled by `train.py` and configured using a JSON file (e.g., `configs/mobileSAM.json`). Key configuration parameters include:

* `model_type`: Specifies the SAM model variant (e.g., "vit_t" for TinyViT based MobileSAM).
* `checkpoint`: Path to the pre-trained MobileSAM checkpoint (`.pth` file).
* `project_name`: Name for logging and output directories.
* `run_name`: Specific name for the training run.
* `train_img_dir`, `train_mask_dir`: Paths to training images and their corresponding masks.
* `val_img_dir`, `val_mask_dir`: Paths to validation images and masks.
* `output_dir`: Directory to save checkpoints and logs.
* `num_epochs`: Total number of training epochs.
* `batch_size`: Training batch size.
* `num_workers`: Number of data loading workers.
* `learning_rate`: Initial learning rate for the optimizer.
* `weight_decay`: Weight decay for regularization.
* `img_size`: Image size for training (e.g., 1024).
* `mask_threshold`: Threshold for binarizing predicted masks during evaluation.
* `iou_head_depth`, `iou_head_hidden_dim`: Parameters for the IoU prediction head in the mask decoder.
* `vit_dim`, `vit_depth`, `vit_mlp_dim`, `vit_num_heads`, `vit_patch_size`: Parameters defining the TinyViT architecture (can be overridden if not using a standard pre-configured TinyViT).
* `freeze`: A dictionary specifying which parts of the model to freeze (e.g., `image_encoder`, `prompt_encoder`, `mask_decoder`). This is crucial for controlling the extent of finetuning.
* `use_distill`: Boolean, if true, enables feature distillation (requires `teacher_checkpoint` and `distill_feature_level`).
* `teacher_checkpoint`: Path to the teacher model checkpoint (e.g., original SAM ViT-H).
* `distill_feature_level`: Specifies which layer's features from the teacher's image encoder to use for distillation (e.g., 8 or 11 for ViT-H).

### Dataset Preparation

The finetuning script expects datasets in a simple image-mask pair format:
* **Images:** Standard image files (e.g., JPG, PNG).
* **Masks:** Grayscale or binary image files where each pixel value represents a class or a binary segmentation. The `SAMDataset` in `finetune_utils/datasets.py` loads these masks and converts them to binary format if necessary (values > 0 become 1).
* The `SAMDataset` class handles loading images and masks, applying transformations (resizing, normalization), and generating bounding box prompts from the ground truth masks. These bounding boxes are then used as prompts during training.

### Training Script

The main training logic resides in `train.py`.
1.  **Setup:** Parses arguments, loads the configuration JSON.
2.  **Model Loading:**
    * Initializes the MobileSAM model using `build_sam_vit_t` (or other variants based on config) from `mobile_sam.build_sam.py`.
    * Loads pre-trained weights from the specified checkpoint.
    * Applies freezing to specified model components (e.g., `model.image_encoder.eval()` and `param.requires_grad = False`).
3.  **Data Loading:**
    * Creates `SAMDataset` instances for training and validation.
    * Uses `torch.utils.data.DataLoader` for batching and shuffling.
4.  **Optimizer & Scheduler:**
    * Uses AdamW optimizer (`torch.optim.AdamW`).
    * Implements a polynomial learning rate decay scheduler (`PolynomialLR` from `finetune_utils.schedular`).
5.  **Training Loop:**
    * Iterates over epochs and batches.
    * For each batch:
        * Moves data to the GPU.
        * Generates bounding box prompts from ground truth masks (`get_boxes_from_masks`).
        * Performs a forward pass through the model:
            * `model(batched_input, multimask_output=True)` where `batched_input` contains images, ground truth masks (for loss calculation if needed directly by model or for other purposes), and bounding box prompts.
        * Calculates loss (see [Loss Functions](#loss-functions)).
        * If using distillation, calculates distillation loss using `FeatureDistillationLoss` from `finetune_utils.distill_losses.py`. This loss compares intermediate features from the student (MobileSAM) and a pre-trained teacher model.
        * Performs backpropagation and optimizer step.
        * Updates learning rate.
    * Logs training metrics (loss, learning rate).
    * Performs validation at the end of each epoch:
        * Calculates validation loss and Mean IoU.
        * Saves the best model checkpoint based on validation IoU.
6.  **Logging:** Uses a custom logger (`load_logger` from `finetune_utils.load_logger`) to print and save logs.

### Loss Functions

The primary loss function used for segmentation is a combination of Focal Loss and Dice Loss.
* **Implementation:** `SegLoss` class in `finetune_utils.loss.py`.
* **Focal Loss:** Addresses class imbalance by down-weighting well-classified examples. It's a modification of cross-entropy loss.
    * $FL(p_t) = -\alpha_t (1 - p_t)^\gamma \log(p_t)$
* **Dice Loss:** Directly optimizes the Dice Coefficient (a measure of overlap).
    * $DL = 1 - \frac{2|X \cap Y|}{|X| + |Y|}$
* The combined loss is typically a weighted sum: $L_{total} = L_{focal} + L_{dice}$.
* **IoU Loss (for IoU head):** The mask decoder also predicts IoU scores for its mask outputs. The training involves a Smooth L1 loss or MSE loss between the predicted IoU and the actual IoU of the predicted mask with the ground truth. This is handled within the `Sam` model's forward pass if ground truth masks are provided.

If knowledge distillation is enabled (`use_distill: true` in config):
* **Feature Distillation Loss:** `FeatureDistillationLoss` in `finetune_utils.distill_losses.py`.
    * This loss aims to make the student model's intermediate features mimic those of a larger, more powerful teacher model.
    * It typically uses Mean Squared Error (MSE) or L1 loss between the student's and teacher's feature maps at specified layers of their respective image encoders.
    * The teacher model's features for the training dataset are pre-extracted using `scripts/extract_teacher_features.py` and saved to disk. These are then loaded by the `SAMDataset` during finetuning.

### Optimizer and Scheduler

* **Optimizer:** AdamW (`torch.optim.AdamW`) is used, which is Adam with decoupled weight decay. This often leads to better generalization.
    * Parameters: `lr` (learning rate), `weight_decay`.
* **Scheduler:** `PolynomialLR` (from `finetune_utils.schedular`) is used.
    * This scheduler decays the learning rate polynomially from the initial LR to a minimum LR over a specified number of steps or epochs.
    * Formula: $lr = (initial\_lr - end\_lr) \times (1 - \frac{current\_iter}{total\_iters})^{power} + end\_lr$
    * This helps in fine-tuning by starting with larger updates and gradually reducing them as training progresses.

## Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/thedannyliu/mobilesam_finetune.git](https://github.com/thedannyliu/mobilesam_finetune.git)
    cd mobilesam_finetune
    ```

2.  **Create a Python virtual environment (recommended):**
    ```bash
    python -m venv sam_env
    source sam_env/bin/activate  # On Windows: sam_env\Scripts\activate
    ```

3.  **Install dependencies:**
    The project uses PyTorch. Install it first, matching your CUDA version if GPU support is needed. Visit [pytorch.org](https://pytorch.org/) for specific instructions.
    Example (CUDA 11.8):
    ```bash
    pip install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu118](https://download.pytorch.org/whl/cu118)
    ```
    Then install other requirements:
    ```bash
    pip install -r requirements.txt
    ```
    The `requirements.txt` includes:
    * `torch`, `torchvision`, `torchaudio`
    * `numpy`
    * `opencv-python`
    * `pycocotools`
    * `matplotlib`
    * `onnx`, `onnxruntime`
    * `gradio`
    * `timm` (likely for TinyViT or general vision model utilities)
    * `segment_anything` (if using components directly from the original SAM package or for comparison)

4.  **Download Pre-trained Checkpoint:**
    You will need a pre-trained MobileSAM checkpoint (`.pth` file). The original MobileSAM authors provide one. Place it in a known location (e.g., `checkpoints/` directory, not included in the repo by default) and update the `checkpoint` path in your configuration file (e.g., `configs/mobileSAM.json`).
    * Original MobileSAM checkpoint (from their repository): `mobile_sam.pt`

## Usage

### Finetuning

1.  **Prepare your dataset:**
    * Organize your images and corresponding masks into separate directories (e.g., `dataset/train/images`, `dataset/train/masks`).
    * Ensure masks are single-channel images where non-zero pixels represent the object of interest.

2.  **Configure `configs/mobileSAM.json` (or create a new one):**
    * Set `train_img_dir` and `train_mask_dir` to your training data paths.
    * Set `val_img_dir` and `val_mask_dir` if you have a validation set.
    * Specify the `checkpoint` path to your pre-trained MobileSAM model.
    * Adjust `num_epochs`, `batch_size`, `learning_rate`, etc., as needed.
    * Configure the `freeze` dictionary to specify which parts of the model should not be updated. For example, to finetune only the mask decoder:
        ```json
        "freeze": {
            "image_encoder": true,
            "prompt_encoder": true,
            "mask_decoder": false
        }
        ```
    * If using feature distillation:
        * Set `use_distill: true`.
        * Provide `teacher_checkpoint` (e.g., path to SAM ViT-H checkpoint).
        * Run `scripts/extract_teacher_features.py` first (see below).
        * Set `train_img_dir_teacher_features` in the dataset config within `mobileSAM.json`.

3.  **(Optional) Extract Teacher Features for Distillation:**
    If `use_distill` is true, you need to pre-extract features from the teacher model:
    ```bash
    python scripts/extract_teacher_features.py \
        --config_file configs/mobileSAM.json \
        --image_dir path/to/your/train_images \
        --output_dir path/to/save/teacher_features
    ```
    Update `train_img_dir_teacher_features` in `configs/mobileSAM.json` to point to `path/to/save/teacher_features`.

4.  **Run the training script:**
    ```bash
    python train.py --config_file configs/mobileSAM.json
    ```
    Logs and checkpoints will be saved to the directory specified by `output_dir` in the config, under a subfolder named `project_name/run_name`.

### Inference with Gradio App

The project includes a Gradio application for interactive segmentation.
1.  **Ensure you have a trained (or pre-trained) MobileSAM checkpoint.**
2.  **Run the Gradio app:**
    ```bash
    python app/app.py --checkpoint path/to/your/mobilesam_checkpoint.pth
    ```
    Optional arguments for `app/app.py`:
    * `--model-type`: `vit_t` (default) or other SAM model types.
    * `--sam_checkpoint`: Path to the SAM model checkpoint.
    * `--port`: Port number for the Gradio app.
    * `--host`: Host address for the Gradio app.
    * `--img_path`: Optional path to an image to load by default.

    The app allows you to upload an image, click points (positive/negative), or draw bounding boxes to get segmentation masks.

### Using the Predictor

The `mobile_sam.predictor.SamPredictor` class provides a way to use the model programmatically.
See `notebooks/predictor_example.ipynb` for an example.
Key steps:
1.  Initialize the predictor:
    ```python
    from mobile_sam import sam_model_registry, SamPredictor
    model_type = "vit_t"
    sam_checkpoint = "path/to/checkpoint.pth"
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device='cuda') # or 'cpu'
    predictor = SamPredictor(sam)
    ```
2.  Set the image:
    ```python
    image = cv2.imread("path/to/image.jpg")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    predictor.set_image(image)
    ```
3.  Provide prompts (points, boxes):
    ```python
    input_point = np.array([[x, y]]) # e.g., [[500, 375]]
    input_label = np.array([1])      # 1 for foreground, 0 for background
    input_box = np.array([x1, y1, x2, y2]) # e.g., [425, 600, 700, 875]

    masks, scores, logits = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        box=input_box, # Optional
        multimask_output=True,
    )
    ```
    The `predict` method returns masks, their quality scores (predicted IoU), and raw logits.

### Automatic Mask Generation

The `mobile_sam.automatic_mask_generator.SamAutomaticMaskGenerator` can be used to segment all objects in an image.
See `notebooks/automatic_mask_generator_example.ipynb`.
Key steps:
1.  Initialize the generator:
    ```python
    from mobile_sam import sam_model_registry, SamAutomaticMaskGenerator
    model_type = "vit_t"
    sam_checkpoint = "path/to/checkpoint.pth"
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device='cuda')
    mask_generator = SamAutomaticMaskGenerator(sam)
    # Or, to use specific hyperparameters:
    # mask_generator = SamAutomaticMaskGenerator(
    #     model=sam,
    #     points_per_side=32,
    #     pred_iou_thresh=0.88,
    #     stability_score_thresh=0.95,
    #     crop_n_layers=0,
    #     crop_n_points_downscale_factor=1,
    #     min_mask_region_area=100, # Requires open-cv to run post-processing
    # )
    ```
2.  Generate masks:
    ```python
    image = cv2.imread("path/to/image.jpg")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    masks = mask_generator.generate(image)
    # `masks` is a list of dictionaries, each containing segmentation info.
    ```

### ONNX Export and Usage

The project supports exporting the MobileSAM model to ONNX for optimized inference.

1.  **Export the model:**
    ```bash
    python scripts/export_onnx_model.py \
        --checkpoint path/to/your/mobilesam_checkpoint.pth \
        --output path/to/save/mobilesam.onnx \
        --model-type vit_t \
        --quantize-out path/to/save/mobilesam_quantized.onnx # Optional: for int8 quantization
    ```
    Key arguments:
    * `--checkpoint`: Path to the PyTorch MobileSAM model.
    * `--output`: Path to save the ONNX model.
    * `--model-type`: Type of SAM model (e.g., `vit_t`).
    * `--quantize-out`: If provided, exports a quantized int8 ONNX model to this path. This can further speed up inference and reduce model size, but might require calibration data for optimal performance (not explicitly handled by this script, uses dynamic quantization).
    * `--return-single-mask`: If set, the ONNX model will only return the best mask.
    * `--opset`: ONNX opset version (default 13).

2.  **Use the ONNX model:**
    See `notebooks/onnx_model_example.ipynb`.
    This involves using `onnxruntime.InferenceSession`. The notebook demonstrates how to prepare inputs (image embeddings, point/box prompts) and run inference. The image encoder and the main SAM model (prompt/mask decoders) might be exported as separate ONNX models or a combined one depending on the export script's capabilities. The script `export_onnx_model.py` exports the prompt encoder and mask decoder part, assuming the image embeddings are pre-computed.
    The script also exports the image encoder separately:
    ```bash
    python scripts/export_onnx_model.py \
        --checkpoint path/to/your/mobilesam_checkpoint.pth \
        --output path/to/save/image_encoder.onnx \
        --model-type vit_t \
        --export-encoder # Add this flag
    ```

## Directory Structure
mobilesam_finetune/
├── app/                      # Gradio application
│   ├── README.md
│   ├── app.py                # Main Gradio app script
│   ├── requirements.txt      # App-specific requirements
│   └── utils/                # Utility functions for the app
│       ├── tools.py
│       └── tools_gradio.py
├── configs/                  # Configuration files
│   ├── mobileSAM.json        # Example finetuning configuration
│   ├── mobile_sam_orig.yaml  # Original MobileSAM config (reference)
│   └── sam_vith.yaml         # SAM ViT-H config (reference for teacher)
├── finetune_utils/           # Utilities for finetuning
│   ├── datasets.py           # Custom PyTorch Dataset for SAM
│   ├── distill_losses.py     # Feature distillation loss
│   ├── feature_hooks.py      # Hooks for extracting intermediate features
│   ├── load_checkpoint.py    # Functions for loading model checkpoints
│   ├── load_config.py        # Function for loading JSON configs
│   ├── load_logger.py        # Logging setup
│   ├── loss.py               # Segmentation loss (Focal + Dice)
│   ├── save_checkpoint.py    # Functions for saving model checkpoints
│   ├── schedular.py          # Learning rate schedulers (PolynomialLR)
│   └── visualization.py      # Visualization utilities (not extensively used in train.py)
├── mobile_sam/               # Core MobileSAM model code (adapted from official SAM)
│   ├── init.py
│   ├── automatic_mask_generator.py
│   ├── build_sam.py          # Functions to build SAM models (e.g., sam_model_registry)
│   ├── modeling/             # Model architecture components
│   │   ├── init.py
│   │   ├── common.py         # Common layers (MLP, LayerNorm2d)
│   │   ├── image_encoder.py  # Original SAM ViT image encoder (reference)
│   │   ├── mask_decoder.py   # Mask decoder module
│   │   ├── prompt_encoder.py # Prompt encoder module
│   │   ├── sam.py            # Main Sam class orchestrating encoders and decoder
│   │   ├── tiny_vit_sam.py   # TinyViT image encoder implementation
│   │   └── transformer.py    # Transformer and Attention blocks
│   ├── predictor.py          # SamPredictor class for inference
│   └── utils/                # Utility functions for SAM
│       ├── init.py
│       ├── amg.py            # Utilities for automatic mask generation
│       ├── onnx.py           # ONNX conversion helper (SamOnnxModel)
│       └── transforms.py     # Image transformation utilities
├── notebooks/                # Jupyter notebooks with examples
│   ├── automatic_mask_generator_example.ipynb
│   ├── onnx_model_example.ipynb
│   └── predictor_example.ipynb
├── scripts/                  # Helper scripts
│   ├── amg.py                # Script for running automatic mask generation
│   ├── export_onnx_model.py  # Script to export model to ONNX format
│   └── extract_teacher_features.py # Script to pre-compute teacher model features for distillation
├── .gitignore
├── CODE_OF_CONDUCT.md
├── CONTRIBUTING.md
├── LICENSE                   # (Assumed Apache 2.0 based on original SAM)
├── README.md                 # This file
├── requirements.txt          # Project-level Python dependencies
├── setup.cfg
├── setup.py                  # For package installation (if developed as a library)
└── train.py                  # Main script for finetuning MobileSAM

## Technical Deep Dive

### Core Model Components (Recap from Model Architecture)

* **`mobile_sam.modeling.tiny_vit_sam.TinyViT`**: The heart of MobileSAM's efficiency. It's a compact Vision Transformer. Key parameters like `embed_dims`, `depths`, `num_heads`, `window_sizes` define its architecture across its multiple stages. It outputs image embeddings, typically of shape `(B, C, H_emb, W_emb)` (e.g., `B, 256, 64, 64`).
* **`mobile_sam.modeling.prompt_encoder.PromptEncoder`**: Converts sparse (points, boxes) and potentially dense (masks) prompts into embeddings. Uses learnable embeddings for prompt types and positional encodings for spatial locations. Outputs `sparse_embeddings` and `dense_embeddings`.
* **`mobile_sam.modeling.mask_decoder.MaskDecoder`**: A transformer-based decoder that attends to both image embeddings and prompt embeddings to produce multiple output masks and their predicted IoU scores. The core is a `TwoWayTransformer` which allows bidirectional attention flow between tokens representing image features and tokens representing prompt queries.
* **`mobile_sam.modeling.sam.Sam`**: The main model class that integrates the image encoder, prompt encoder, and mask decoder. Its `forward` method orchestrates the flow:
    1.  Processes the input image with `image_encoder` to get image embeddings.
    2.  Processes prompts (points, boxes, masks) with `prompt_encoder` to get prompt embeddings.
    3.  Feeds image embeddings and prompt embeddings to `mask_decoder` to get low-resolution masks and IoU predictions.
    4.  Upscales masks to the original image resolution if required.

### Finetuning Utilities (`finetune_utils/`)

* **`datasets.py:SAMDataset`**:
    * Loads images and masks from specified directories.
    * Handles image resizing to `img_size` (e.g., 1024x1024) and normalization using `transforms.ResizeLongestSide` and pixel mean/std deviation.
    * Crucially, for training with box prompts, it extracts bounding boxes from the ground truth masks using `get_boxes_from_masks`. These boxes then serve as prompts to the model during training.
    * If distillation is used, it also loads pre-computed teacher features and aligns them with the student's input.
* **`loss.py:SegLoss`**: Combines Focal Loss and Dice Loss. This is standard for segmentation tasks where pixel-wise classification can be imbalanced and direct optimization of overlap is beneficial.
* **`distill_losses.py:FeatureDistillationLoss`**:
    * Calculates MSE (or other L_p norm) loss between student and teacher feature maps.
    * Requires `feature_hooks.py:FeatureHookManager` to register hooks on the student and teacher image encoders to extract intermediate layer activations during their forward passes.
    * The teacher's features are pre-computed and loaded by the dataset to avoid repeated forward passes of the heavy teacher model during student training.
* **`load_checkpoint.py`**: Contains `load_mobile_sam_checkpoint` which carefully loads weights from a pre-trained checkpoint, potentially ignoring mismatched keys or freezing parts of the model as per the configuration.
* **`schedular.py:PolynomialLR`**: A learning rate scheduler that decays the LR polynomially. Helps in stabilizing training in later stages.
* **`feature_hooks.py:FeatureHookManager`**: A utility to attach forward hooks to specific modules (layers) within a PyTorch model. This is used by `FeatureDistillationLoss` to grab the intermediate activations from the image encoders of both the student (MobileSAM) and the teacher model (e.g., SAM ViT-H) at specified layers. The `extract_teacher_features.py` script also uses this to save these teacher activations.

### Distillation (Optional)

Knowledge Distillation (KD) is a technique where a smaller "student" model learns from a larger, more performant "teacher" model. In this project, it appears to be implemented as feature-map distillation:
1.  **Teacher Feature Extraction (`scripts/extract_teacher_features.py`):**
    * A powerful teacher model (e.g., original SAM with ViT-H encoder) processes the training images.
    * `FeatureHookManager` is used to capture the output of a specific intermediate layer (e.g., the 8th or 11th block) of the teacher's image encoder.
    * These feature maps are saved to disk (e.g., as `.pt` or `.npy` files) for each training image.
2.  **Student Training (`train.py` with `use_distill: true`):**
    * The `SAMDataset` loads the pre-computed teacher features corresponding to each training image.
    * During the student's (MobileSAM) forward pass, `FeatureHookManager` captures the features from the equivalent (or chosen) layer of MobileSAM's TinyViT encoder.
    * The `FeatureDistillationLoss` calculates a loss (e.g., MSE) between the student's features and the loaded teacher's features. This loss term is added to the primary segmentation loss.
    * The intuition is that the student learns to produce intermediate representations similar to those of the more powerful teacher, which can guide the student to better solutions, especially when the student model has much lower capacity.

This distillation approach can be particularly useful for transferring the rich representations learned by large vision transformers into smaller, more efficient models like MobileSAM.

## Contributing

Please refer to `CONTRIBUTING.md` for guidelines on contributing to this project. Ensure that any contributions align with the `CODE_OF_CONDUCT.md`.

## License

The original Segment Anything Model (SAM) and MobileSAM are typically released under the Apache 2.0 License. This finetuning repository, if it builds upon that work, would likely also fall under a compatible open-source license. Please check the `LICENSE` file for specific details. (Note: A `LICENSE` file was not explicitly provided in the uploaded project structure, but it's standard practice).

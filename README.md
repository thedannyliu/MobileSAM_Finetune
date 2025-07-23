# Finetuning MobileSAM

This repository provides a comprehensive framework for finetuning the MobileSAM model, a lightweight version of the Segment Anything Model (SAM) optimized for performance on resource-constrained devices. It includes a complete suite for data processing, training, inference, and interactive demonstration.

## Key Features

*   **MobileSAM Architecture:** Full implementation of the MobileSAM model, including the TinyViT image encoder, prompt encoder, and mask decoder.
*   **Flexible Finetuning:** A robust `train.py` script for finetuning MobileSAM on custom datasets.
*   **Configuration-Driven:** Manage all training parameters, model paths, and dataset details via JSON configuration files.
*   **Multi-Stage Training:** Define complex training schedules with different phases (e.g., distillation followed by finetuning) in a single run.
*   **Advanced Loss Functions:** Supports a combination of Focal Loss and Dice Loss for segmentation, with an MSE loss for the IoU head.
*   **Knowledge Distillation:** Utilities to distill knowledge from a larger teacher model (like the original SAM) to improve MobileSAM's performance.
*   **Optimizers and Schedulers:** Integrated with AdamW optimizer and a Polynomial Learning Rate Scheduler for stable convergence.
*   **Checkpoint Management:** Automatically saves the best-performing model checkpoints based on validation metrics.
*   **Gradio Web UI:** An interactive `app.py` for easy testing and visualization of segmentation results.
*   **ONNX Export:** Functionality to export the model to ONNX format for optimized, cross-platform inference.
*   **Comprehensive Utilities:** Includes scripts for automatic mask generation, programmatic prediction, and feature extraction.

## ❗️ Important: Image Preprocessing

As of the June 2025 update, the training and inference pipelines have been standardized to simplify data handling.

> **Your data loading pipeline should ONLY convert images to a PyTorch Tensor and scale them to the `0-255` range.** Do not apply any other normalization (e.g., subtracting mean or dividing by standard deviation).

**Reasoning:**
1.  **Internal Normalization:** The `Sam.preprocess()` method within the model already normalizes the input using ImageNet statistics (`mean=[123.675, 116.28, 103.53]`, `std=[58.395, 57.12, 57.375]`). Applying normalization twice will corrupt the input data and prevent the model from converging.
2.  **Workflow Alignment:** Keeping the `0-255` range ensures 1:1 compatibility with the official `SamPredictor` and `SamAutomaticMaskGenerator` workflows, which is critical for correct prompt-to-image alignment and mask post-processing.

A correct `torchvision.transforms` pipeline should look like this:

```python
T.Compose([
    T.ToTensor(),             # Converts image to a [0, 1] float tensor
    T.Lambda(lambda x: x*255.0) # Scales tensor to the [0, 255] range
])
```

Please ensure any custom `Dataset` you write follows this logic.

## Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/thedannyliu/MobileSAM-fast-finetuning.git
    cd MobileSAM-fast-finetuning
    ```

2.  **Create a Python virtual environment (recommended):**
    ```bash
    python -m venv sam_env
    source sam_env/bin/activate  # On Windows use: sam_env\Scripts\activate
    ```

3.  **Install dependencies:**
    First, install PyTorch, ensuring it matches your system's CUDA version for GPU support. Visit [pytorch.org](https://pytorch.org/) for specific instructions.
    
    Example for CUDA 11.8:
    ```bash
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    ```
    
    Then, install the remaining project requirements:
    ```bash
    pip install -r requirements.txt
    ```

4.  **Download Pre-trained Checkpoint:**
    You need a pre-trained MobileSAM checkpoint to start finetuning. You can download the official `mobile_sam.pt` from the original authors' repository. Place it in a convenient location (e.g., a `checkpoints/` directory) and update the `checkpoint` path in your JSON configuration file.

## Directory Structure

```
mobilesam_finetune/
├── app/                      # Gradio application
│   └── app.py                # Main Gradio app script
├── configs/                  # Configuration files (e.g., mobileSAM.json)
├── docs/                     # Documentation (e.g., stage_schedule.md)
├── finetune_utils/           # Utilities for finetuning
│   ├── datasets.py           # Custom PyTorch Dataset for SAM
│   ├── distill_losses.py     # Feature distillation loss
│   ├── loss.py               # Segmentation loss (Focal + Dice)
│   └── schedular.py          # Learning rate schedulers
├── mobile_sam/               # Core MobileSAM model code
│   ├── build_sam.py          # Model registry
│   ├── modeling/             # Model architecture components
│   │   ├── sam.py            # Main Sam class
│   │   └── tiny_vit_sam.py   # TinyViT image encoder
│   ├── predictor.py          # SamPredictor for programmatic inference
│   └── automatic_mask_generator.py # SamAutomaticMaskGenerator
├── notebooks/                # Jupyter notebooks with examples
├── scripts/                  # Helper scripts
│   ├── export_onnx_model.py  # Script to export model to ONNX
│   └── extract_teacher_features.py # Script for knowledge distillation
└── train.py                  # Main script for finetuning
```

## Usage

### 1. Finetuning

1.  **Prepare Your Dataset:**
    *   Organize your images and masks into separate directories (e.g., `data/train/images`, `data/train/masks`).
    *   Masks should be single-channel images (e.g., grayscale PNG) where non-zero pixels represent the object of interest.

2.  **Create a Configuration File:**
    *   Copy `configs/mobileSAM.json` and customize it for your project.
    *   Set `train_img_dir`, `train_mask_dir`, `val_img_dir`, and `val_mask_dir` to your dataset paths.
    *   Set `checkpoint` to the path of your pre-trained MobileSAM model.
    *   Adjust hyperparameters like `num_epochs`, `batch_size`, and `learning_rate`.
    *   Use the `freeze` dictionary to control which parts of the model are trained. For example, to finetune only the mask decoder:
        ```json
        "freeze": {
            "image_encoder": true,
            "prompt_encoder": true,
            "mask_decoder": false
        }
        ```

3.  **(Optional) Prepare for Knowledge Distillation:**
    *   If you want to use knowledge distillation (`use_distill: true`), you must first pre-extract features from a teacher model (e.g., the original SAM ViT-H).
    *   Run the extraction script:
    ```bash
    python scripts/extract_teacher_features.py \
            --config_file configs/your_config.json \
            --image_dir /path/to/your/train_images \
            --output_dir /path/to/save/teacher_features
    ```
    *   Update your config to point `train_img_dir_teacher_features` to the output directory.

4.  **Run Training:**
    ```bash
    python train.py --config_file configs/your_config.json
    ```
    Logs and checkpoints will be saved under the `output_dir` specified in your configuration.

### 2. Multi-Stage Training

This repository supports defining a sequence of training stages within a single run. This is useful for complex training regimes, such as starting with distillation and then switching to pure finetuning.

To enable this, add a `stage_schedule` array to your JSON config. `train.py` will dynamically adjust training parameters at the beginning of each epoch based on this schedule.

*   `start_epoch`, `end_epoch`: The epoch range for the stage.
*   `distillation`: A boolean to enable or disable the distillation loss.
*   `lambda_coef`: A global weight for the distillation loss component.
*   `loss_weights`: Per-stage overrides for other loss components.

**Example `stage_schedule` in `config.json`:**
```json
"stage_schedule": [
    {
        "start_epoch": 0,
        "end_epoch": 10,
        "distillation": true,
        "lambda_coef": 1.0,
        "loss_weights": { "focal": 0.0, "dice": 0.0, "iou": 0.0 }
    },
    {
        "start_epoch": 10,
        "end_epoch": 20,
        "distillation": false,
        "lambda_coef": 0.0,
        "loss_weights": { "focal": 1.0, "dice": 1.0, "iou": 1.0 }
    }
]
```
For a detailed schema, see **[`docs/stage_schedule.md`](docs/stage_schedule.md)**.

### 3. Inference with the Gradio App

The interactive Gradio app is the easiest way to test your model.

1.  **Launch the app:**
    ```bash
    python app/app.py --checkpoint /path/to/your/finetuned_checkpoint.pth
    ```
2.  **Use the UI:**
    *   Upload an image.
    *   Add foreground/background points by clicking on the image.
    *   Draw bounding boxes to specify the object of interest.
    *   The model will generate and display the segmentation mask in real-time.

### 4. Programmatic Inference

For integration into other applications, use the `SamPredictor` class. See `notebooks/predictor_example.ipynb` for a complete example.

**Key Steps:**
    ```python
import cv2
import numpy as np
    from mobile_sam import sam_model_registry, SamPredictor

# 1. Initialize the model and predictor
sam_checkpoint = "/path/to/your/checkpoint.pth"
    model_type = "vit_t"
device = "cuda"

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)
    predictor = SamPredictor(sam)

# 2. Set the image
image = cv2.imread("image.jpg")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    predictor.set_image(image)

# 3. Provide prompts
input_point = np.array([[500, 375]])  # [[x, y]]
input_label = np.array([1])           # 1 for foreground, 0 for background
input_box = np.array([425, 600, 700, 875]) # [x1, y1, x2, y2]

# 4. Predict
    masks, scores, logits = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
    box=input_box,
        multimask_output=True,
    )
    ```

### 5. Automatic Mask Generation

To segment all objects in an image automatically, use the `SamAutomaticMaskGenerator`. See `notebooks/automatic_mask_generator_example.ipynb`.

    ```python
    from mobile_sam import sam_model_registry, SamAutomaticMaskGenerator

# 1. Initialize the generator
mask_generator = SamAutomaticMaskGenerator(sam) # Use the 'sam' model from the previous example

# 2. Generate masks
masks = mask_generator.generate(image) # 'masks' is a list of dicts, each with segmentation info
    ```

### 6. ONNX Export and Usage

Export the model to ONNX for faster inference and deployment.

1.  **Export the decoder and prompt encoder:**
    ```bash
    python scripts/export_onnx_model.py \
        --checkpoint /path/to/your/checkpoint.pth \
        --output models/mobilesam_decoder.onnx \
        --model-type vit_t \
        --quantize-out models/mobilesam_decoder_quant.onnx  # Optional: for int8 quantization
    ```

2.  **Export the image encoder separately:**
    ```bash
    python scripts/export_onnx_model.py \
        --checkpoint /path/to/your/checkpoint.pth \
        --output models/mobilesam_encoder.onnx \
        --model-type vit_t \
        --export-encoder
    ```

For an example of how to run inference with the exported ONNX models, refer to `notebooks/onnx_model_example.ipynb`.

## Technical Deep Dive

### Model Architecture

*   **`TinyViT` Image Encoder:** The core of MobileSAM's efficiency. It's a compact Vision Transformer that generates image embeddings (e.g., `B, 256, 64, 64`) from a 1024x1024 input.
*   **`PromptEncoder`:** Converts sparse (points, boxes) and dense (masks) prompts into embeddings that the decoder can understand.
*   **`MaskDecoder`:** A transformer-based decoder that uses two-way attention to combine image embeddings and prompt embeddings, predicting segmentation masks and their quality (IoU scores).
*   **`Sam` Class:** The main model that integrates the three components and orchestrates the forward pass.

### Finetuning Internals

*   **`SAMDataset` (`finetune_utils/datasets.py`):**
    *   Loads images and masks.
    *   **Coordinate System:** This is a critical detail. The dataset is responsible for transforming prompt coordinates (points and boxes) to match the model's input space. For a 1024x1024 model input, all prompts are scaled and padded to align with the resized image. **Raw, original image coordinates are NOT passed to the model during training or validation**; they are only used for visualization purposes.
    *   Extracts bounding box prompts from ground truth masks on the fly.
    *   If distillation is used, it loads the corresponding pre-computed teacher features for each image.

*   **Loss Functions (`finetune_utils/loss.py`):**
    *   **`SegLoss`:** A combination of Focal Loss (to handle class imbalance) and Dice Loss (to directly optimize mask overlap). The total segmentation loss is a weighted sum of the two.
    *   **IoU Loss:** The model's IoU head is trained with an MSE loss between its predicted IoU and the actual IoU of the predicted mask vs. the ground truth.

*   **Knowledge Distillation (`finetune_utils/distill_losses.py`):**
    *   The `FeatureDistillationLoss` computes an MSE loss between intermediate feature maps from the student's TinyViT encoder and a powerful teacher's encoder (e.g., ViT-H from the original SAM).
    *   This encourages the smaller student model to learn the richer feature representations of the larger teacher, often improving performance.
    *   This process relies on `FeatureHookManager` to extract activations from specific model layers without altering the model definitions.

## Contributing

Please refer to `CONTRIBUTING.md` for guidelines on contributing to this project. All contributions should adhere to the `CODE_OF_CONDUCT.md`.

## License

This project is built upon the original MobileSAM and Segment Anything models, which are released under the Apache 2.0 License. This finetuning repository is likely also covered by a compatible open-source license. Please check the `LICENSE` file for specific details.

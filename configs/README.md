# Training configuration

`train.py --config <file.json>` reads a nested JSON object. Paths are relative
to the working directory; run commands from the repository root.

| Section | Controls |
| --- | --- |
| `dataset` | `train_dataset`, `val_dataset`, `num_workers`, prompt mode and point counts |
| `model` | `type`, `checkpoint_path`, `image_size`, output `save_path`, `multimask_output` |
| `train` | Epochs, learning rate, batch size, gradient accumulation, warmup, validation and early stopping |
| `freeze` | Image/prompt encoder and mask decoder freezing; `unfreeze_epoch` |
| `distillation` | Teacher feature source and enabled encoder, prompt, token and logit losses |
| `teachers` | Each teacher's name, YAML config, checkpoint and mixture weight |
| `stage_schedule` | Optional epoch ranges and per-stage distillation/task-loss overrides |
| `visual` | Validation overlays, output directory and save frequency |

## Data and prompts

Each split uses `image/<stem>.(jpg|jpeg|png)` with binary masks under
`mask/<stem>/*.png`. The default `ComponentDataset` creates one sample per mask.
`dataset.mode: "everything"` selects `SegmentEverythingDataset` and point-grid
matching instead. Prompt modes and coordinate conversion are implemented in
[`finetune_utils/datasets.py`](../finetune_utils/datasets.py).

Images are converted back to the 0–255 range before SAM preprocessing.
Do not insert a second ImageNet normalization into the data transform.

## Choose a recipe

- `mobileSAM.json`: point/box prompts; distillation is **enabled** by default.
- `mobileSAM_se.json`: segment-everything recipe.
- `distill_then_finetune.json`: distillation in epochs 0–99, supervised losses in 100–299.
- `finetune_then_distill.json`: see `stage_schedule` for the reverse curriculum.

For supervised-only training, set `distillation.enable` to `false` and use a
recipe without a stage that re-enables it. For distillation, provide every
teacher checkpoint listed in `teachers`; none is downloaded by the trainer.

`use_precomputed_features: true` loads `.npy` tensors from
`<precomputed_root>/<teacher>/<split>/`. Feature names must match
`load_cached_npy_features` and the extractor; this path currently uses CUDA.
Online teacher inference is selected when that setting is false.

## Outputs and resume

Set a distinct `model.save_path` for each run and `visual.save_path` for overlays.
The training driver writes checkpoints and TensorBoard events under the run
path. Inspect the `train.resume` handling in `train.py` before resuming a changed
config: architecture, teacher features and optimizer state must remain compatible.

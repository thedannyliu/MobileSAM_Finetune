# Multi-Stage Training with `stage_schedule`

A common research question is whether it is more effective to (a) let the student **first mimic the teacher** via knowledge-distillation and then fine-tune on the ground-truth labels, or (b) do the opposite, or (c) keep both objectives on during the whole training.

Starting from **v2025-07-21** the training script supports an *arbitrary number* of consecutive stages, each with its own

* epoch range (inclusive start, exclusive end)
* on/off switch for distillation
* global distillation coefficient `lambda_coef`
* individual weights for task losses (BCE / Focal / Dice / IoU / cls)

This is controlled by a top-level array `"stage_schedule"` in the JSON config.

---
## JSON Schema
```jsonc
// Root level of your config (siblings to "dataset", "model", ...)
"stage_schedule": [
  {
    "name": "(optional description)",
    "start_epoch": 0,            // inclusive
    "end_epoch": 100,            // exclusive – if omitted defaults to total epochs
    "distillation": true,        // enable teacher-student loss in this stage
    "lambda_coef": 1.0,          // scales the sum of all distill losses
    "loss_weights": {            // overrides train.loss_weights only in this stage
      "bce": 1.0,
      "focal": 15,
      "dice": 1.0,
      "iou": 1.0,
      "cls": 1.0
    }
  }
  /* , { … next stage … } */
]
```
If `stage_schedule` is **absent** the behaviour is identical to previous versions (single stage, flags taken directly from `train` / `distillation` blocks).

---
## Example 1 Distill → Finetune
```jsonc
"stage_schedule": [
  {
    "name": "distill_only",
    "start_epoch": 0,
    "end_epoch": 100,
    "distillation": true,
    "lambda_coef": 1.0,
    "loss_weights": { "bce": 0, "focal": 0, "dice": 0, "iou": 0, "cls": 0 }
  },
  {
    "name": "finetune_only",
    "start_epoch": 100,
    "end_epoch": 300,
    "distillation": false,
    "lambda_coef": 0.0,
    "loss_weights": { "bce": 1.0, "focal": 15, "dice": 1.0, "iou": 1.0, "cls": 1.0 }
  }
]
```

---
## Example 2 Finetune → Distill
Swap the order and adjust `loss_weights` accordingly:
```jsonc
"stage_schedule": [
  { "name": "finetune_first",  "start_epoch": 0,   "end_epoch": 100, "distillation": false, "lambda_coef": 0.0,
    "loss_weights": { "bce": 1.0, "focal": 15, "dice": 1.0, "iou": 1.0, "cls": 1.0 }},
  { "name": "distill_second",  "start_epoch": 100, "end_epoch": 300, "distillation": true,  "lambda_coef": 1.0,
    "loss_weights": { "bce": 0,   "focal": 0,  "dice": 0,   "iou": 0,   "cls": 0   }}
]
```

---
## Interaction with the existing `distillation` block
* The **base** `distillation.enable` flag still decides whether teacher models & hooks are prepared **by default**.
* If *any* stage sets `"distillation": true`, teachers will be loaded even when the base flag is `false`.
* `lambda_coef` in each stage overrides the global value for that epoch range.

---
## Logging & TensorBoard
The active stage name and the current values of
`lambda_coef`, `use_distillation`, and the five task-loss weights are logged at every epoch start. TensorBoard scalars (loss/metrics) remain unchanged, so existing dashboards continue to work.

---
## Backward Compatibility
Older config files without `stage_schedule` require **no change**. The new logic simply reverts to the single-stage behaviour. 
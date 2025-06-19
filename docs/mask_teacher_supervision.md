# Mask-Teacher Supervision for Segment-Everything fine-tuning

This document explains the two new optional supervision strategies that can be enabled from the YAML/JSON **config** file when training in *segment-everything* mode (`dataset.mode: "everything"`).

---
## 1 replace_gt  ‑  Teacher-only hard labels

```
mask_teacher:
  enable: true               # turn the feature on / off
  method: "replace_gt"       # <- choose this method
  teacher_type: "vit_h"      # "vit_h" (SAM-H) or "vit_t" (original MobileSAM)
  checkpoint_path: "/path/to/teacher/checkpoint.pth"
  iou_threshold: 0.3         # Hungarian threshold used for matching
```

Training flow
1.  For every image we run the **teacher** (e.g.
    SAM-H).
2.  Hungarian matching between *teacher masks* and *ground-truth masks*.
    ‑ Only pairs whose IoU ≥ `iou_threshold` are kept.  
    ‑ For the kept pairs we **replace** the teacher mask with the exact GT
      mask → this yields an *improved teacher set* that has the same mask
      ordering as the teacher but the pixel accuracy of GT.
3.  A second Hungarian is done between *student* and this improved teacher
   set.  Losses (BCE / Focal / Dice / IoU / cls) are computed exactly as
   before, but the **targets are the teacher masks** (GT has been fully
   replaced).
4.  The normal feature-distillation block (encoder / decoder / attention /
   RKD) continues to work unchanged.

When this method is active the original GT no longer contributes to the
loss; you are effectively fine-tuning the student to imitate the
(corrected) teacher masks.

---
## 2 dual_supervision  ‑  Hard GT +  Soft Teacher

```
mask_teacher:
  enable: true
  method: "dual_supervision"
  teacher_type: "vit_h"
  checkpoint_path: "/path/to/teacher/checkpoint.pth"
  iou_threshold: 0.3
  soft_loss:
    type: "l2"              # "l2" or "kl"
    weight: 0.5             # λ_soft, multiplies the soft loss when added
```

Training flow
1.  The normal student-vs-GT Hungarian and task losses are kept **as-is**
    (hard labels).
2.  In parallel we compute teacher predictions and perform a Hungarian
   between *student* and *teacher*; pairs with IoU ≥ `iou_threshold` are
   considered.
3.  For every matched pair a pixel-wise **soft loss** is added:
    * **L2**  `MSE(student_logit, teacher_logit)`  (default)
    * **KL**  `KL( sigmoid(student) || sigmoid(teacher) )`
4.  The soft loss is averaged over all matched pairs in the mini-batch and
   multiplied by `soft_loss.weight` before being added to the main task
   loss.

This method preserves the fidelity of original GT while gently pushing the
student's logits towards the (usually smoother) teacher logits, resulting
in better boundary quality without sacrificing recall.

---
## Choosing the teacher

`teacher_type` maps directly to the keys in `mobile_sam.sam_model_registry`:

* `vit_h`   → official SAM-H
* `vit_t`   → original MobileSAM-T (fast, tiny)

Any custom checkpoint that is compatible with the chosen backbone can be
used by pointing `checkpoint_path` to the *.pth* file.

---
## Compatibility notes

* The feature works **only** in `dataset.mode == "everything"` because
  Hungarian matching requires multi-mask targets.
* The teacher network is always executed under `torch.no_grad()` and its
  parameters never receive gradients; the extra GPU memory is therefore
  limited to the forward pass activations.
* All new metrics are visible in the progress bar (`soft`) and TensorBoard
  (`train/task_loss` already includes the soft term).

---
## Example JSON snippet

```json
{
  "dataset": {
    "mode": "everything",
    ...
  },
  "model": { ... },
  "train":  { ... },

  "mask_teacher": {
    "enable": true,
    "method": "dual_supervision",
    "teacher_type": "vit_h",
    "checkpoint_path": "/models/sam_hq_vit_h.pth",
    "iou_threshold": 0.3,
    "soft_loss": {
      "type": "kl",
      "weight": 0.3
    }
  }
}
```

Add this block to any existing config to activate the new supervision
mechanism. If the block is omitted or `enable: false`, training proceeds
exactly as before. 
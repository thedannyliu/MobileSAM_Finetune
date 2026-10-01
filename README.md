# MobileSAM fine-tuning

Fine-tune MobileSAM for point- and box-prompted segmentation, with optional
multi-teacher distillation and staged training. The implementation builds on
[MobileSAM](https://github.com/ChaoningZhang/MobileSAM) and
[Segment Anything](https://github.com/facebookresearch/segment-anything).

## Start training

Use Python 3.10+ and install a PyTorch/torchvision pair for your CUDA runtime.
From the repository root:

```bash
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
python train.py --config configs/mobileSAM.json
```

**Before running:** edit the dataset and checkpoint paths in the JSON config.
The supplied `mobileSAM.json` enables distillation and needs both configured
teacher checkpoints. For supervised fine-tuning alone, set
`distillation.enable` to `false`. The files in `configs/` are recipes, not
completed experiments or automatically downloaded assets.

A dataset split contains `image/<stem>.jpg` and one or more binary masks at
`mask/<stem>/*.png`. `dataset.train_dataset` and `dataset.val_dataset` point
to the split roots. Images and masks must agree in original resolution.

| Recipe | Purpose |
| --- | --- |
| [`mobileSAM.json`](configs/mobileSAM.json) | Prompted segmentation with optional distillation |
| [`mobileSAM_se.json`](configs/mobileSAM_se.json) | Point-grid / segment-everything training |
| [`distill_then_finetune.json`](configs/distill_then_finetune.json) | Distillation followed by supervised fine-tuning |
| [`finetune_then_distill.json`](configs/finetune_then_distill.json) | Supervised fine-tuning followed by distillation |

All recipes use `python train.py --config <file>`.
See the [configuration reference](configs/README.md) for stages, losses and outputs.

## Code map

| Path | Responsibility |
| --- | --- |
| [`train.py`](train.py) | Training and validation orchestration |
| [`finetune_utils/`](finetune_utils) | Datasets, distillation losses, feature hooks, cache and scheduler |
| [`mobile_sam/`](mobile_sam) | SAM-compatible model, predictor and automatic mask generator |
| [`scripts/`](scripts) | Data preparation, feature extraction, evaluation and ONNX export |
| [`notebooks/`](notebooks) | Interactive prediction and ONNX examples |
| [`app/`](app) | Demo application; separate requirements |

Teacher feature extraction uses `scripts/extract_teacher_features.py`;
ONNX export uses `scripts/export_onnx_model.py`. Run each with `--help`
for its input contract. Keep datasets, teacher features and training outputs
outside source control.

## Scope and validation

Training requires the configured datasets and weights; CUDA is required for
the precomputed teacher-feature cache. There is no published accuracy or speed
claim established by the example configs. Report the dataset split, prompt
protocol, checkpoint and hardware with any measured result.

See [CONTRIBUTING.md](CONTRIBUTING.md). Code is provided under the
[Apache 2.0 license](LICENSE); upstream notices are retained.

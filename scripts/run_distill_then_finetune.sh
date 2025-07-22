#!/usr/bin/env bash
# Run two-stage training: distillation → finetune
# Usage: bash scripts/run_distill_then_finetune.sh [additional-args]

python train.py --config configs/distill_then_finetune.json "$@" 
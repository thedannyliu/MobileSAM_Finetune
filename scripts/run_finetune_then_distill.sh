#!/usr/bin/env bash
# Run two-stage training: finetune → distillation
# Usage: bash scripts/run_finetune_then_distill.sh [additional-args]

python train.py --config configs/finetune_then_distill.json "$@" 
# Contributing

Keep changes focused and explain the dataset/prompt behavior they affect.
Open a GitHub issue for reproducible bugs or a pull request with the fix.

Install development tools with `python -m pip install -r requirements-dev.txt`.
For training changes, record the config, dataset split, checkpoint, seed,
hardware and a small before/after run. Never commit datasets, credentials,
teacher caches or generated checkpoints.

Keep model code under `mobile_sam`, training helpers under `finetune_utils`,
and command-line utilities under `scripts`. Prefer the existing config-driven
training entry point over a new shell wrapper for each experiment.

Preserve upstream copyright notices and license terms.

The old config/logger/checkpoint/loss helpers and `schedular.py` had no
in-repository callers and were removed. The actual training path remains in
`train.py` with shared helpers in `finetune_utils`; use the model registry in
`mobile_sam` rather than the removed duplicate model loader. Old implementations
remain available in git history.

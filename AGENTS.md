# Repository Guidelines

## Project Structure

This unified repo coordinates the SkyBall jersey workflow:

- `jersey_gallery/`: builds player crop galleries from unified SkyBall split files.
- `clip_reid/`: trains/evaluates the CLIP-ReID player re-identification model from 15-crop galleries.
- `jersey_sdg/`: labels jersey visibility with Qwen and trains uncertainty-aware jersey-number recognition.
- `workflows/`: canonical cross-repo procedures and result tracking.

Large datasets live under `/mnt/t/data/vball/skyball/...`; training and labeling outputs live under `/mnt/t/output/...`.

## Canonical Dataset Workflow

When given a new split such as `/mnt/t/data/vball/skyball/unified_dataset/splits/v2.json`, follow `workflows/skyball_jersey_dataset_workflow.md`. The required order is:

1. Build `/mnt/t/data/vball/skyball/jersey/gallery/v2` with `target-count 15`.
2. Train/evaluate `clip_reid` on `gallery/v2`.
3. Build `/mnt/t/data/vball/skyball/jersey/gallery_jersey_recognition/v2_4x` with `target-count 60`.
4. Label visibility in `jersey_sdg` with Qwen.
5. Train jersey recognition in `jersey_sdg`.
6. Append metrics to `workflows/results.jsonl` and update `workflows/results_summary.md`.

## Development Commands

Run repo-local commands from their owning directory unless a command uses `python -m jersey_sdg...`, which should be run from this repo root’s parent package directory: `/home/atao/vsdevel/skyball/jersey`.

```bash
cd jersey_gallery && PYTHONPATH=. conda run -n pt5090new python -m pytest tests -q
conda run -n pt5090new python -m pytest jersey_sdg/tests
```

Use `pt5090new` for training/tests and `vllm5090` for Qwen/vLLM labeling.

## Operational Rules

Prefer `tmux` for long jobs. Use GPUs `2,3` for jersey recognition or Qwen when available; leave `0,1` for detector/ReID jobs when they are active. Do not overwrite completed outputs under `/mnt/t/output`; create versioned run names. Record command lines, output paths, dataset version, and metrics for every final run.

## Git Hygiene

Track source, docs, scripts, and lightweight configs. Do not commit checkpoints, generated crops, TensorBoard logs, zip datasets, caches, or debug images. Use concise imperative commit messages such as `Add v2 jersey workflow results`.
